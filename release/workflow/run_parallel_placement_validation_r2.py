#!/usr/bin/env python3
"""Validate all completed r3 placement captures concurrently."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(
    "/home/xmu/nvidiagds/codex-runs/llm-footprint-v1/memgen/"
    "hyfiss-prefill-matrix-20260913-01a06837-r1"
)
VALIDATOR = ROOT / "source/validate_placement_capture.py"
DECODE_STEPS = 32
PROMPTS = (128, 256, 512, 1024)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> int:
    output_root = ROOT / "validation-r1"
    output_root.mkdir(parents=True, exist_ok=True)
    targets = []
    for prompt in PROMPTS:
        capture = ROOT / f"placement/p{prompt}d32-r3"
        if prompt == 1024:
            observer = Path(
                "/home/xmu/nvidiagds/codex-runs/llm-footprint-v1/memgen/"
                "hyfiss-accuracy-20260911-01a08d56/llm-range-r2/"
                "p1024d32-source/observer/kernels.tsv"
            )
        else:
            observer = ROOT / f"observer/p{prompt}d32/kernels.tsv"
        output = output_root / f"p{prompt}d32.json"
        stdout_path = output_root / f"p{prompt}d32.stdout.log"
        stderr_path = output_root / f"p{prompt}d32.stderr.log"
        for required in (capture, observer):
            if not required.exists():
                raise SystemExit(f"missing input: {required}")
        for fresh in (output, stdout_path, stderr_path):
            if fresh.exists():
                raise SystemExit(f"refusing to overwrite: {fresh}")
        targets.append((prompt, capture, observer, output, stdout_path, stderr_path))

    manifest_path = output_root / "parallel-p128-p1024-r1.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite: {manifest_path}")
    manifest = {
        "schema": "hyfiss_parallel_placement_validation_v1",
        "started_utc": utc_now(),
        "runs": [],
    }
    processes = []
    for prompt, capture, observer, output, stdout_path, stderr_path in targets:
        command = [
            "python3",
            "-B",
            str(VALIDATOR),
            "--capture-root",
            str(capture),
            "--observer-tsv",
            str(observer),
            "--expected-prompt",
            str(prompt),
            "--expected-decode",
            str(DECODE_STEPS),
            "--output",
            str(output),
        ]
        stdout = stdout_path.open("xb")
        stderr = stderr_path.open("xb")
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        row = {
            "prompt_tokens": prompt,
            "capture_root": str(capture),
            "observer_tsv": str(observer),
            "output": str(output),
            "command": command,
            "pid": proc.pid,
            "started_utc": utc_now(),
        }
        manifest["runs"].append(row)
        processes.append((proc, stdout, stderr, row))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    failed = False
    for proc, stdout, stderr, row in processes:
        returncode = proc.wait()
        stdout.close()
        stderr.close()
        row["returncode"] = returncode
        row["finished_utc"] = utc_now()
        failed |= returncode != 0
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    manifest["finished_utc"] = utc_now()
    manifest["status"] = "PASS_ALL_VALIDATORS_EXITED_ZERO" if not failed else "FAIL"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "status": manifest["status"]}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
