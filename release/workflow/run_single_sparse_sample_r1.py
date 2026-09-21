#!/usr/bin/env python3
"""Run one collision-safe full-inference HyFiSS sparse sample."""

from __future__ import annotations

import argparse
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
EXPECTED_KERNELS = {64: 19258, 128: 19258, 256: 19204, 512: 19121, 1024: 20119}
EXPECTED_SHA256 = {
    DRIVER: "66c757df426e848cfd0dd5feb44d9faa3c890ba812f2b62087c0f940217ea72f",
    MODEL: "d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8",
    TRACER: "209ddbe97bfd10b04af9367958d0088738094c83cbb43a1ac7901b4e63ee4481",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=int, choices=sorted(EXPECTED_KERNELS), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    need_label = f"p{args.prompt}d32-"
    if not args.run_label.startswith(need_label) or "/" in args.run_label or "\\" in args.run_label:
        raise SystemExit(f"run label must start with {need_label!r} and be one path component")

    plan = ROOT / f"sample-plans-r3/p{args.prompt}d32.tsv"
    plan_receipt_path = ROOT / f"sample-plans-r3/p{args.prompt}d32.json"
    output = ROOT / "samples" / args.run_label
    if output.exists():
        raise SystemExit(f"refusing to reuse output: {output}")
    inputs = {}
    for path in (DRIVER, MODEL, TRACER, plan, plan_receipt_path):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
        digest = sha256(path)
        if path in EXPECTED_SHA256 and digest != EXPECTED_SHA256[path]:
            raise SystemExit(f"input digest differs: {path}: {digest}")
        inputs[path.name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
    plan_receipt = json.loads(plan_receipt_path.read_text(encoding="utf-8"))
    if plan_receipt.get("status") != "PASS_PLAN_COVERS_EVERY_KERNEL_NOT_MEMORY_TRACE":
        raise SystemExit("sample plan receipt did not pass")
    if int(plan_receipt.get("kernel_count", -1)) != EXPECTED_KERNELS[args.prompt]:
        raise SystemExit("sample plan kernel count differs from frozen placement census")
    active = [line for line in gpu_apps() if line.split(",", 1)[0].strip() == args.gpu_uuid]
    if active:
        raise SystemExit(f"GPU is not idle: {active}")
    if shutil.disk_usage(ROOT).free < 100 * (1 << 30):
        raise SystemExit("less than 100 GiB free before sample capture")

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "hyfiss_full_inference_sparse_sample_run_v1",
        "status": "STARTED",
        "workload": {"prompt_tokens": args.prompt, "decode_steps": 32, "batch": 1},
        "gpu_uuid": args.gpu_uuid,
        "inputs": inputs,
        "started_utc": utc_now(),
        "claim_boundary": (
            "Hybrid complete CTA placement plus sampled memory-SASS input only; no profile, "
            "generated full trace, cache result, or NCU accuracy claim"
        ),
    }
    manifest_path = output / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("root_sudo", "LD_PRELOAD", "GGML_BACKEND_PATH")
        and not key.startswith("HYFISS_")
    }
    env.update({
        "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
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
        "LLMGPUSIM_CAPTURE_RUN_ID": f"{args.run_label}-full-inference-sparse-sample",
    })
    command = [str(DRIVER), str(MODEL), str(args.prompt), "32"]
    (output / "command.json").write_text(json.dumps({"argv": command}, indent=2) + "\n")
    began = time.monotonic()
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        process = subprocess.run(command, cwd=output, env=env, stdout=stdout, stderr=stderr)
    manifest.update({
        "finished_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - began,
        "returncode": process.returncode,
        "status": "PASS_PROCESS_EXITED_ZERO_NOT_YET_VALIDATED" if process.returncode == 0 else "FAIL",
        "gpu_apps_after": gpu_apps(),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": manifest["status"], "elapsed_seconds": manifest["elapsed_seconds"]}))
    return 0 if process.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
