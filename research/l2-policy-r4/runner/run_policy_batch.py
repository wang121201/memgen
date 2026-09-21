#!/usr/bin/env python3
"""Run one immutable full-inference L2 policy shard and seal a small receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--profile-index", type=Path, required=True)
    parser.add_argument("--app-config", type=Path, required=True)
    parser.add_argument("--issue-config", type=Path, required=True)
    parser.add_argument("--hw-config", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--workload", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.job_root.mkdir(parents=True, exist_ok=False)
    inputs = [args.binary, args.profile_index, args.app_config, args.issue_config,
              args.hw_config, args.population]
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError([str(path) for path in inputs if not path.is_file()])
    os.sched_setaffinity(0, {args.cpu})
    command = [
        str(args.binary), "--mode", "memgen",
        "--profile-index", str(args.profile_index),
        "--app-config", str(args.app_config),
        "--issue-config", str(args.issue_config),
        "--hw-config", str(args.hw_config),
        "--stats", str(args.job_root / "stats.json"),
        "--output-dir", str(args.job_root / "model"),
        "--include-local", "false", "--observe-cache", "false",
    ]
    input_pins = {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in inputs}
    write_new(args.job_root / "command.json", {
        "argv": command,
        "cpu": args.cpu,
        "policies": args.policies.split(","),
        "workload": args.workload,
        "environment": {"FULL_POLICY_MATRIX": args.policies},
        "inputs": input_pins,
    })
    started = time.time()
    write_new(args.job_root / "start.json", {
        "pid": os.getpid(), "started_unix": started, "cpu": args.cpu,
        "workload": args.workload, "policies": args.policies.split(","),
    })
    result: dict[str, object] = {
        "status": "FAIL", "returncode": None, "workload": args.workload,
        "policies": args.policies.split(","), "cpu": args.cpu,
    }
    try:
        environment = dict(os.environ, FULL_POLICY_MATRIX=args.policies)
        with (args.job_root / "stdout.log").open("xb") as stdout, \
             (args.job_root / "stderr.log").open("xb") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr,
                                       env=environment, check=False)
        result["returncode"] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(f"policy replay returned {completed.returncode}")
        expected = ["dirty-policies.csv", "policy-manifest.csv", "ownership.csv",
                    "versions.csv", "stats.json", "model/kernel_summary.csv"]
        missing = [name for name in expected if not (args.job_root / name).is_file()]
        if missing:
            raise RuntimeError(f"missing outputs: {missing}")
        rows = sum(1 for _ in (args.job_root / "dirty-policies.csv").open(
            "r", encoding="utf-8")) - 1
        population = json.loads(args.population.read_text(encoding="utf-8"))
        selected = 1 + len(set(args.policies.split(",")) - {"disabled"})
        if rows != len(population) * selected:
            raise RuntimeError(
                f"row population mismatch: {rows} != {len(population)}*{selected}")
        outputs = {}
        for name in expected:
            path = args.job_root / name
            outputs[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        result.update(status="COMPLETE_CAUSAL_FULL_INFERENCE_POLICY_SHARD_NOT_HARDWARE_ACCEPTANCE",
                      kernels=len(population), policy_count=selected,
                      rows=rows, outputs=outputs)
    except BaseException:
        result["error"] = traceback.format_exc()
    result["elapsed_seconds"] = time.time() - started
    write_new(args.job_root / "finish.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if str(result["status"]).startswith("COMPLETE_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
