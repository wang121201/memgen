#!/usr/bin/env python3
"""Run P128/P256/P512 hybrid sparse full-inference samples in parallel."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(
    "/home/xmu/nvidiagds/codex-runs/llm-footprint-v1/memgen/"
    "hyfiss-prefill-matrix-20260913-01a06837-r1"
)
DRIVER = ROOT / "llm-range-matrix-driver"
MODEL = Path("/home/xmu/nvidiagds/models/qwen2.5-1.5b-instruct-q8_0.gguf")
TRACER = ROOT / "sample-tracer-r5/memory_tracer_sample_r5.so"
RUNS = (
    (128, "GPU-7a22d253-7921-4e49-992e-0199eebd6f86", "p128d32-r1"),
    (256, "GPU-53a2015e-19cf-83d7-cbc8-fac96aab7d38", "p256d32-r1"),
    (512, "GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13", "p512d32-r1"),
)
EXPECTED = {
    DRIVER: "66c757df426e848cfd0dd5feb44d9faa3c890ba812f2b62087c0f940217ea72f",
    MODEL: "d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8",
    TRACER: "209ddbe97bfd10b04af9367958d0088738094c83cbb43a1ac7901b4e63ee4481",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_apps() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    frozen = {}
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"input digest differs: {path}: {actual}")
        frozen[path.name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    if shutil.disk_usage(ROOT).free < 100 * (1 << 30):
        raise SystemExit("less than 100 GiB free before parallel sample capture")
    active = gpu_apps()
    targets = []
    for prompt, gpu_uuid, leaf in RUNS:
        if any(line.split(",", 1)[0].strip() == gpu_uuid for line in active):
            raise SystemExit(f"target GPU is not idle: {gpu_uuid}")
        output = ROOT / "samples" / leaf
        plan = ROOT / "sample-plans-r3" / f"p{prompt}d32.tsv"
        plan_receipt_path = ROOT / "sample-plans-r3" / f"p{prompt}d32.json"
        if output.exists():
            raise SystemExit(f"refusing to reuse output: {output}")
        receipt = json.loads(plan_receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "PASS_PLAN_COVERS_EVERY_KERNEL_NOT_MEMORY_TRACE":
            raise SystemExit(f"plan receipt did not pass: P{prompt}")
        targets.append((prompt, gpu_uuid, output, plan, plan_receipt_path))

    manifest_path = ROOT / "samples/parallel-p128-p512-r1.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite: {manifest_path}")
    manifest = {
        "schema": "hyfiss_parallel_full_inference_sparse_samples_v1",
        "status": "STARTED",
        "started_utc": utc_now(),
        "decode_steps": 32,
        "inputs": frozen,
        "runs": [],
        "claim_boundary": "Sparse memory-SASS inputs only; no profiles, generated full traces, cache results, or NCU accuracy",
    }
    processes = []
    for prompt, gpu_uuid, output, plan, plan_receipt_path in targets:
        output.mkdir(parents=True, exist_ok=False)
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("root_sudo", "LD_PRELOAD", "GGML_BACKEND_PATH")
            and not key.startswith("HYFISS_")
        }
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "LD_PRELOAD": str(TRACER),
                "PATH": "/usr/local/cuda-12.8/bin:/usr/bin:/bin",
                "NVDISASM": "/usr/local/cuda-12.8/bin/nvdisasm",
                "ACK_CTX_INIT_LIMITATION": "1",
                "GGML_CUDA_DISABLE_GRAPHS": "1",
                "HYFISS_FAST_TRACE_MODE": "sample",
                "HYFISS_FAST_LANE_FORMAT": "memc",
                "HYFISS_FAST_CHANNEL_SIZE": "8388608",
                "HYFISS_FAST_TRACE_ROTATE_SIZE": "536870912",
                "HYFISS_EXTERNAL_PHASE_CONTROL": "1",
                "HYFISS_FAST_COLLECTION_ENABLED": "0",
                "HYFISS_MAX_KERNELS": "0",
                "HYFISS_SAMPLED_CTA_PLAN": str(plan),
                "LLMGPUSIM_CAPTURE_RUN_ID": f"p{prompt}d32-full-inference-sparse-sample-r1",
            }
        )
        command = [str(DRIVER), str(MODEL), str(prompt), "32"]
        stdout = (output / "stdout.log").open("xb")
        stderr = (output / "stderr.log").open("xb")
        process = subprocess.Popen(command, cwd=output, env=env, stdout=stdout, stderr=stderr)
        row = {
            "prompt_tokens": prompt,
            "gpu_uuid": gpu_uuid,
            "output": str(output),
            "plan": {"path": str(plan), "sha256": sha256(plan)},
            "plan_receipt": {"path": str(plan_receipt_path), "sha256": sha256(plan_receipt_path)},
            "pid": process.pid,
            "started_utc": utc_now(),
        }
        manifest["runs"].append(row)
        processes.append((process, stdout, stderr, row))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    failed = False
    for process, stdout, stderr, row in processes:
        started = time.monotonic()
        returncode = process.wait()
        row["wait_seconds"] = time.monotonic() - started
        stdout.close()
        stderr.close()
        row["returncode"] = returncode
        row["finished_utc"] = utc_now()
        failed |= returncode != 0
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["status"] = "PASS_PROCESSES_EXITED_ZERO_NOT_YET_VALIDATED" if not failed else "FAIL"
    manifest["finished_utc"] = utc_now()
    manifest["gpu_apps_after"] = gpu_apps()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "status": manifest["status"]}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
