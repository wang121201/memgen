#!/usr/bin/env python3
"""Launch disjoint CPU-pinned shards for one full-inference workload."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--profile-index", type=Path, required=True)
    parser.add_argument("--app-config", type=Path, required=True)
    parser.add_argument("--issue-config", type=Path, required=True)
    parser.add_argument("--hw-config", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--cpus", required=True,
                        help="Comma-separated physical CPU identifiers")
    parser.add_argument("--policies", default="all",
                        help="Comma-separated registry IDs, or all")
    return parser.parse_args()


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    cpus = [int(value) for value in args.cpus.split(",")]
    if len(cpus) != len(set(cpus)):
        raise ValueError("duplicate CPU identifiers")
    registry = list(csv.DictReader(args.policy_manifest.open(encoding="utf-8")))
    registered = [row["policy"] for row in registry]
    requested = ([name for name in registered if name != "disabled"]
                 if args.policies == "all"
                 else [name for name in args.policies.split(",") if name != "disabled"])
    unknown = sorted(set(requested) - set(registered))
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    requested = list(dict.fromkeys(requested))
    batches = [requested[index:index + args.batch_size]
               for index in range(0, len(requested), args.batch_size)]
    if len(batches) > len(cpus):
        raise ValueError(f"need {len(batches)} CPUs, only {len(cpus)} supplied")
    args.matrix_root.mkdir(parents=True, exist_ok=False)
    logs = args.matrix_root / "launcher-logs"
    logs.mkdir()
    common = [
        "--binary", str(args.binary),
        "--profile-index", str(args.profile_index),
        "--app-config", str(args.app_config),
        "--issue-config", str(args.issue_config),
        "--hw-config", str(args.hw_config),
        "--population", str(args.population),
        "--workload", args.workload,
    ]
    jobs = []
    for index, policies in enumerate(batches):
        name = f"batch-{index:02d}"
        command = ["python3", str(args.runner),
                   "--job-root", str(args.matrix_root / name),
                   "--policies", ",".join(policies),
                   "--cpu", str(cpus[index]), *common]
        stdout = (logs / f"{name}.stdout").open("xb")
        stderr = (logs / f"{name}.stderr").open("xb")
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        stdout.close()
        stderr.close()
        jobs.append({"name": name, "pid": process.pid, "cpu": cpus[index],
                     "policies": policies, "command": command})
    manifest = {
        "status": "RUNNING", "started_unix": time.time(),
        "workload": args.workload, "batch_size": args.batch_size,
        "policy_count_excluding_repeated_disabled": len(requested),
        "jobs": jobs,
    }
    write_new(args.matrix_root / "launch.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
