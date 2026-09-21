#!/usr/bin/env python3
"""Run isolated HBServe placement-only captures on one GPU per workload.

This records one placement tuple per CTA and deliberately does not collect
memory addresses.  Every output directory must be absent before launch.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(
    "/home/xmu/nvidiagds/codex-runs/llm-footprint-v1/memgen/"
    "hyfiss-prefill-matrix-20260913-01a06837-r1"
)
DRIVER = EXPERIMENT_ROOT / "llm-range-matrix-driver"
TRACER = EXPERIMENT_ROOT / "placement-tracer-r1/memory_tracer_placement_r3.so"
MODEL = Path("/home/xmu/nvidiagds/models/qwen2.5-1.5b-instruct-q8_0.gguf")
DECODE_STEPS = 32
RUNS = (
    (128, "GPU-18ace299-5348-e6e4-d48c-1ee5a602859b", "p128d32-r3"),
    (256, "GPU-7a22d253-7921-4e49-992e-0199eebd6f86", "p256d32-r3"),
    (512, "GPU-53a2015e-19cf-83d7-cbc8-fac96aab7d38", "p512d32-r3"),
    (1024, "GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13", "p1024d32-r3"),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for required in (DRIVER, TRACER, MODEL):
        if not required.is_file():
            raise SystemExit(f"missing required file: {required}")

    placement_root = EXPERIMENT_ROOT / "placement"
    targets = []
    for prompt, gpu_uuid, leaf in RUNS:
        output = placement_root / leaf
        if output.exists():
            raise SystemExit(f"refusing to reuse output directory: {output}")
        targets.append((prompt, gpu_uuid, output))

    manifest = {
        "schema": "hyfiss_parallel_placement_matrix_v1",
        "claim_boundary": (
            "CTA-to-SM placement metadata only; no memory-address trace and no "
            "cache or hardware-accuracy result"
        ),
        "started_utc": utc_now(),
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "tracer": {"path": str(TRACER), "sha256": sha256(TRACER)},
        "model": {"path": str(MODEL), "sha256": sha256(MODEL)},
        "decode_steps": DECODE_STEPS,
        "runs": [],
    }

    processes = []
    for prompt, gpu_uuid, output in targets:
        output.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "LD_PRELOAD": str(TRACER),
                "NVDISASM": "/usr/local/cuda-12.8/bin/nvdisasm",
                "ACK_CTX_INIT_LIMITATION": "1",
                "GGML_CUDA_DISABLE_GRAPHS": "1",
                "HYFISS_FAST_TRACE_MODE": "placement",
                "HYFISS_EXTERNAL_PHASE_CONTROL": "1",
                "HYFISS_FAST_COLLECTION_ENABLED": "0",
                "HYFISS_FAST_CHANNEL_SIZE": "8388608",
                "HYFISS_MAX_KERNELS": "0",
                "LLMGPUSIM_CAPTURE_RUN_ID": f"p{prompt}d{DECODE_STEPS}-placement-r3",
            }
        )
        env["PATH"] = "/usr/local/cuda-12.8/bin:" + env.get("PATH", "")
        command = [str(DRIVER), str(MODEL), str(prompt), str(DECODE_STEPS)]
        stdout = (output / "stdout.log").open("xb")
        stderr = (output / "stderr.log").open("xb")
        (output / "started.utc").write_text(utc_now() + "\n", encoding="utf-8")
        proc = subprocess.Popen(
            command,
            cwd=output,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        row = {
            "prompt_tokens": prompt,
            "decode_steps": DECODE_STEPS,
            "gpu_uuid": gpu_uuid,
            "output": str(output),
            "pid": proc.pid,
            "command": command,
            "started_utc": utc_now(),
        }
        manifest["runs"].append(row)
        processes.append((proc, stdout, stderr, output, row))

    manifest_path = EXPERIMENT_ROOT / "placement/parallel-p128-p1024-r3.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    failed = False
    for proc, stdout, stderr, output, row in processes:
        returncode = proc.wait()
        stdout.close()
        stderr.close()
        row["returncode"] = returncode
        row["finished_utc"] = utc_now()
        (output / "exit_code.txt").write_text(str(returncode) + "\n", encoding="utf-8")
        (output / "finished.utc").write_text(row["finished_utc"] + "\n", encoding="utf-8")
        failed |= returncode != 0
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    manifest["finished_utc"] = utc_now()
    manifest["status"] = "PASS_PROCESSES_EXITED_ZERO_NOT_YET_VALIDATED" if not failed else "FAIL"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), "status": manifest["status"]}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
