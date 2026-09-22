#!/usr/bin/env python3
"""Integrate constant service costs over Memgen's exclusive outcomes.

This diagnostic intentionally has no scheduler, dependency model, bandwidth
limit, backpressure, or compute overlap.  Its result is serial memory work,
not predicted GPU elapsed time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = (
    "kernel_id",
    "l1_requests",
    "l1_hits",
    "l1_pending_hits",
    "l1_misses",
    "l2_requests",
    "l2_hits",
    "l2_pending_hits",
    "l2_misses",
    "dram_requests",
    "dram_load_requests",
    "dram_store_requests",
    "dram_load_sectors",
    "dram_store_sectors",
)
EVENTS = ("l1_hit", "l2_hit", "dram_read_sector", "dram_write_sector")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(row: dict[str, str], name: str, row_number: int) -> int:
    try:
        value = int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {name} is not an integer") from exc
    if value < 0:
        raise ValueError(f"row {row_number}: {name} is negative")
    return value


def load_latencies(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(EVENTS):
        raise ValueError(f"latency config must contain exactly: {', '.join(EVENTS)}")
    result: dict[str, float] = {}
    for event in EVENTS:
        latency = float(value[event])
        if not math.isfinite(latency) or latency < 0:
            raise ValueError(f"latency for {event} must be finite and nonnegative")
        result[event] = latency
    return result


def load_phase_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["kernel_id", "phase"]:
            raise ValueError("phase map header must be: kernel_id,phase")
        result: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            kernel_id = row["kernel_id"].strip()
            phase = row["phase"].strip()
            if not kernel_id or not phase:
                raise ValueError(f"phase map row {row_number} has an empty field")
            if kernel_id in result:
                raise ValueError(f"phase map repeats kernel_id {kernel_id}")
            result[kernel_id] = phase
    return result


def counts_for(row: dict[str, str], row_number: int) -> dict[str, int]:
    values = {name: integer(row, name, row_number) for name in REQUIRED_COLUMNS}
    if values["l1_requests"] != (
        values["l1_hits"] + values["l1_pending_hits"] + values["l1_misses"]
    ):
        raise ValueError(f"row {row_number}: L1 outcome conservation failed")
    if values["l2_requests"] != (
        values["l2_hits"] + values["l2_pending_hits"] + values["l2_misses"]
    ):
        raise ValueError(f"row {row_number}: L2 outcome conservation failed")
    if values["dram_requests"] != (
        values["dram_load_requests"] + values["dram_store_requests"]
    ):
        raise ValueError(f"row {row_number}: DRAM request conservation failed")
    return {
        "l1_hit": values["l1_hits"] + values["l1_pending_hits"],
        "l2_hit": values["l2_hits"] + values["l2_pending_hits"],
        "dram_read_sector": values["dram_load_sectors"],
        "dram_write_sector": values["dram_store_sectors"],
    }


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for event in EVENTS:
        target[event] += source[event]


def summarize(counts: dict[str, int], latencies: dict[str, float]) -> dict[str, Any]:
    work = {event: counts[event] * latencies[event] for event in EVENTS}
    if not all(math.isfinite(v) for v in work.values()) or not math.isfinite(sum(work.values())):
        raise ValueError("serial work exceeds finite numeric range")
    return {
        "event_counts": dict(counts),
        "serial_work_ns_by_event": work,
        "serial_memory_work_ns": sum(work.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-summary", required=True, type=Path)
    parser.add_argument("--latency-config", required=True, type=Path)
    parser.add_argument("--phase-map", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    latencies = load_latencies(args.latency_config)
    phase_map = load_phase_map(args.phase_map)
    totals = defaultdict(int)
    by_phase: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen_kernels: set[str] = set()

    with args.kernel_summary.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"kernel summary is missing columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            kernel_id = row["kernel_id"].strip()
            if not kernel_id or kernel_id in seen_kernels:
                raise ValueError(f"row {row_number}: empty or repeated kernel_id {kernel_id!r}")
            if args.phase_map is not None and kernel_id not in phase_map:
                raise ValueError(f"phase map omits kernel_id {kernel_id}")
            seen_kernels.add(kernel_id)
            counts = counts_for(row, row_number)
            add_counts(totals, counts)
            add_counts(by_phase[phase_map.get(kernel_id, "unassigned")], counts)

    unused = sorted(set(phase_map) - seen_kernels)
    if unused:
        raise ValueError(f"phase map contains unknown kernel IDs: {', '.join(unused[:8])}")
    if not seen_kernels:
        raise ValueError("kernel summary has no data rows")

    result = {
        "status": "PASS_SIMPLE_LATENCY_SERIAL_MEMORY_WORK",
        "claim_boundary": (
            "Constant-cost serial memory work only; no scheduling, bandwidth, "
            "stall propagation, compute overlap, or GPU elapsed-time claim"
        ),
        "inputs": {
            "kernel_summary": str(args.kernel_summary.resolve()),
            "kernel_summary_sha256": sha256(args.kernel_summary),
            "latency_config": str(args.latency_config.resolve()),
            "latency_config_sha256": sha256(args.latency_config),
            "phase_map": None if args.phase_map is None else str(args.phase_map.resolve()),
            "phase_map_sha256": None if args.phase_map is None else sha256(args.phase_map),
        },
        "latencies_ns": latencies,
        "kernel_count": len(seen_kernels),
        "whole": summarize(totals, latencies),
        "by_phase": {
            phase: summarize(counts, latencies) for phase, counts in sorted(by_phase.items())
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["scope", *EVENTS, "serial_memory_work_ns"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for scope, summary in [("whole", result["whole"]), *result["by_phase"].items()]:
            writer.writerow({
                "scope": scope,
                **summary["event_counts"],
                "serial_memory_work_ns": summary["serial_memory_work_ns"],
            })
    print(json.dumps({
        "status": result["status"],
        "kernel_count": result["kernel_count"],
        "serial_memory_work_ns": result["whole"]["serial_memory_work_ns"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
