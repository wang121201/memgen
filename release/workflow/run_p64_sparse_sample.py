#!/usr/bin/env python3
"""Run the first full-inference hybrid sparse-sample pilot (P64D32)."""

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
TRACER = ROOT / "sample-tracer-r3/memory_tracer_sample_r3.so"
PLAN = ROOT / "sample-plans-r3/p64d32.tsv"
PLAN_RECEIPT = ROOT / "sample-plans-r3/p64d32.json"
OUTPUT = ROOT / "samples/p64d32-r1"
GPU_UUID = "GPU-18ace299-5348-e6e4-d48c-1ee5a602859b"
EXPECTED_SHA256 = {
    DRIVER: "66c757df426e848cfd0dd5feb44d9faa3c890ba812f2b62087c0f940217ea72f",
    MODEL: "d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8",
    TRACER: "05c7f107e3d2ba5c64dc602f25e00f9ba87dc1ad0b77d7989011dc1ef6ca3251",
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
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to reuse output: {OUTPUT}")
    inputs = {}
    for path in (DRIVER, MODEL, TRACER, PLAN, PLAN_RECEIPT):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
        digest = sha256(path)
        if path in EXPECTED_SHA256 and digest != EXPECTED_SHA256[path]:
            raise SystemExit(f"input digest differs: {path}: {digest}")
        inputs[path.name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
    plan_receipt = json.loads(PLAN_RECEIPT.read_text(encoding="utf-8"))
    if plan_receipt.get("status") != "PASS_PLAN_COVERS_EVERY_KERNEL_NOT_MEMORY_TRACE":
        raise SystemExit("sample plan receipt did not pass")
    if int(plan_receipt.get("kernel_count", -1)) != 19258:
        raise SystemExit("P64 sample plan kernel count differs")
    active = [line for line in gpu_apps() if line.split(",", 1)[0].strip() == GPU_UUID]
    if active:
        raise SystemExit(f"GPU is not idle: {active}")
    if shutil.disk_usage(ROOT).free < 100 * (1 << 30):
        raise SystemExit("less than 100 GiB free before sample capture")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "hyfiss_full_inference_sparse_sample_run_v1",
        "status": "STARTED",
        "workload": {"prompt_tokens": 64, "decode_steps": 32, "batch": 1},
        "gpu_uuid": GPU_UUID,
        "inputs": inputs,
        "started_utc": utc_now(),
        "claim_boundary": (
            "Hybrid CTA placement plus sampled memory-SASS input only; no profile, "
            "generated full trace, cache result, or NCU accuracy claim"
        ),
    }
    manifest_path = OUTPUT / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("root_sudo", "LD_PRELOAD", "GGML_BACKEND_PATH")
        and not key.startswith("HYFISS_")
    }
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
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
            "HYFISS_SAMPLED_CTA_PLAN": str(PLAN),
            "LLMGPUSIM_CAPTURE_RUN_ID": "p64d32-full-inference-sparse-sample-r1",
        }
    )
    command = [str(DRIVER), str(MODEL), "64", "32"]
    (OUTPUT / "command.json").write_text(
        json.dumps({"argv": command}, indent=2) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
        process = subprocess.run(command, cwd=OUTPUT, env=env, stdout=stdout, stderr=stderr)
    manifest.update(
        {
            "finished_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "returncode": process.returncode,
            "status": "PASS_PROCESS_EXITED_ZERO_NOT_YET_VALIDATED" if process.returncode == 0 else "FAIL",
            "gpu_apps_after": gpu_apps(),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "status": manifest["status"], "elapsed_seconds": manifest["elapsed_seconds"]}))
    return 0 if process.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
