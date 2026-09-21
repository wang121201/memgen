#!/usr/bin/env python3
"""Validate a HyFiSS placement-only capture against a CUPTI launch census."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "hyfiss_placement_capture_validation_v1"
APP_RE = re.compile(r"^-kernel_([0-9]+)_([^ ]+)\s+(.*)$")
ISSUE_HEADER_RE = re.compile(r"^-trace_issued_sm_id_([0-9]+)\s+([0-9]+),([0-9]+),(.*)$")
ISSUE_TUPLE_RE = re.compile(r"\(([0-9]+),([0-9]+),([0-9a-fA-F]+)\),")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def parse_app(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            match = APP_RE.match(raw.rstrip("\n"))
            if not match:
                continue
            kernel, field, value = int(match.group(1)), match.group(2), match.group(3)
            need(field not in rows.setdefault(kernel, {}), f"duplicate app field: kernel {kernel} {field}")
            rows[kernel][field] = value
    need(rows, "app.config contains no kernel rows")
    need(sorted(rows) == list(range(1, len(rows) + 1)), "app.config kernel IDs are not contiguous from 1")
    required = {
        "kernel_name", "num_registers", "shared_mem_bytes", "grid_size", "block_size",
        "grid_dim_x", "grid_dim_y", "grid_dim_z", "tb_dim_x", "tb_dim_y", "tb_dim_z",
        "llama_phase",
    }
    for kernel, row in rows.items():
        need(required.issubset(row), f"kernel {kernel} lacks app fields {sorted(required.difference(row))}")
    return rows


def parse_observer(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    need(bool(rows), "observer TSV contains no kernels")
    return rows


def validate_resources(app: dict[int, dict[str, str]], observer: list[dict[str, str]]) -> None:
    need(len(app) == len(observer), f"kernel count differs: app={len(app)} observer={len(observer)}")
    mapping = {
        "grid_dim_x": "grid_x", "grid_dim_y": "grid_y", "grid_dim_z": "grid_z",
        "tb_dim_x": "block_x", "tb_dim_y": "block_y", "tb_dim_z": "block_z",
        "num_registers": "registers",
    }
    for kernel, observed in enumerate(observer, 1):
        row = app[kernel]
        for app_field, observer_field in mapping.items():
            need(int(row[app_field]) == int(observed[observer_field]),
                 f"kernel {kernel} resource mismatch: {app_field}")
        grid = int(observed["grid_x"]) * int(observed["grid_y"]) * int(observed["grid_z"])
        block = int(observed["block_x"]) * int(observed["block_y"]) * int(observed["block_z"])
        shared = int(observed["static_shared"]) + int(observed["dynamic_shared"])
        need(int(row["grid_size"]) == grid, f"kernel {kernel} grid_size mismatch")
        need(int(row["block_size"]) == block, f"kernel {kernel} block_size mismatch")
        need(int(row["shared_mem_bytes"]) == shared, f"kernel {kernel} shared memory mismatch")


def validate_issue(path: Path, app: dict[int, dict[str, str]]) -> dict[str, Any]:
    grids = [0] + [int(app[kernel]["grid_size"]) for kernel in range(1, len(app) + 1)]
    seen = [bytearray()] + [bytearray(size) for size in grids[1:]]
    counts = [0] * len(grids)
    timestamps_min: list[int | None] = [None] * len(grids)
    timestamps_max: list[int | None] = [None] * len(grids)
    declared_sm_count: int | None = None
    declared_sms: set[int] = set()
    sm_line_count = 0
    tuples = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if line.startswith("-trace_issued_sms_num "):
                need(declared_sm_count is None, "duplicate issued SM count")
                declared_sm_count = int(line.split()[1])
                continue
            match = ISSUE_HEADER_RE.match(line)
            if not match:
                continue
            sm_from_key, declared_tuples, sm_from_value, payload = (
                int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)
            )
            need(sm_from_key == sm_from_value, f"issue SM key/value mismatch at line {line_number}")
            need(sm_from_key not in declared_sms, f"duplicate issue SM line {sm_from_key}")
            declared_sms.add(sm_from_key)
            sm_line_count += 1
            found = ISSUE_TUPLE_RE.findall(payload)
            need(len(found) == declared_tuples,
                 f"issue tuple count mismatch for SM {sm_from_key}: declared={declared_tuples} parsed={len(found)}")
            for kernel_text, block_text, timestamp_text in found:
                kernel, block, timestamp = int(kernel_text), int(block_text), int(timestamp_text, 16)
                need(1 <= kernel < len(grids), f"issue kernel out of range: {kernel}")
                need(0 <= block < grids[kernel], f"kernel {kernel} CTA out of range: {block}")
                need(seen[kernel][block] == 0, f"duplicate issue tuple: kernel {kernel} CTA {block}")
                seen[kernel][block] = 1
                counts[kernel] += 1
                timestamps_min[kernel] = timestamp if timestamps_min[kernel] is None else min(timestamps_min[kernel], timestamp)
                timestamps_max[kernel] = timestamp if timestamps_max[kernel] is None else max(timestamps_max[kernel], timestamp)
                tuples += 1
    need(declared_sm_count is not None, "issue.config lacks issued SM count")
    need(declared_sm_count == sm_line_count == len(declared_sms), "issued SM count differs from SM rows")
    for kernel in range(1, len(grids)):
        need(counts[kernel] == grids[kernel],
             f"kernel {kernel} CTA coverage mismatch: expected={grids[kernel]} observed={counts[kernel]}")
    return {
        "declared_sm_count": declared_sm_count,
        "sm_ids": sorted(declared_sms),
        "tuple_count": tuples,
        "kernel_count": len(grids) - 1,
        "all_kernel_cta_sets_exact": True,
        "timestamp_nonempty_kernel_count": sum(value is not None for value in timestamps_min[1:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--observer-tsv", type=Path, required=True)
    parser.add_argument("--expected-prompt", type=int, required=True)
    parser.add_argument("--expected-decode", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.capture_root.resolve()
    output = args.output.resolve()
    need(not output.exists(), f"refusing to overwrite {output}")
    need(not (root / "memory_traces").exists(), "placement capture unexpectedly contains memory_traces")
    app_path = root / "configs" / "app.config"
    issue_path = root / "configs" / "issue.config"
    receipt_path = root / "configs" / "capture_receipt.json"
    app = parse_app(app_path)
    observer = parse_observer(args.observer_tsv.resolve())
    validate_resources(app, observer)
    issue = validate_issue(issue_path, app)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_records = sum(int(row["grid_size"]) for row in app.values())
    need(receipt.get("status") == "PASS", "capture receipt did not pass")
    need(receipt.get("trace_mode") == "placement", "capture receipt is not placement mode")
    need(int(receipt["kernel_count"]) == len(app), "receipt kernel count mismatch")
    need(
        int(receipt["device_pushed_records"])
        == int(receipt["host_received_records"])
        == int(receipt["host_persisted_records"])
        == issue["tuple_count"]
        == expected_records,
        "placement record conservation mismatch",
    )
    for field in ("dropped_records", "malformed_packets", "sequence_errors", "io_errors"):
        need(int(receipt[field]) == 0, f"non-zero receipt error field: {field}")
    phases = Counter(row["llama_phase"] for row in app.values())
    need(phases["prefill"] > 0, "missing prefill phase")
    for step in range(args.expected_decode):
        need(phases[f"decode_step_{step}"] > 0, f"missing decode_step_{step}")
    need(sum(phases.values()) == len(app), "phase counts do not cover every kernel")
    result = {
        "schema": SCHEMA,
        "status": "PASS_PLACEMENT_CONFIG_EXACT_CTA_COVERAGE_NOT_MEMORY_TRACE",
        "workload": {
            "model": "Qwen2.5-1.5B-Instruct Q8_0 via llama.cpp",
            "batch": 1,
            "prompt_tokens": args.expected_prompt,
            "decode_steps": args.expected_decode,
        },
        "definition": {
            "placement_record": "one tuple (kernel ID, linear CTA ID, physical SM ID, first instrumented kernel-entry clock)",
            "validation": "every kernel has exactly CTA IDs 0..grid_size-1 once; CUDA launch resources equal the independent CUPTI observer row at the same ordinal",
            "excluded_claims": [
                "memory instruction or lane address trace",
                "sampled memory-SASS template availability",
                "HBServe full-inference trace generation",
                "naive-cache or Memgen accuracy",
                "hardware performance timing",
            ],
        },
        "kernel_count": len(app),
        "phase_kernel_counts": dict(sorted(phases.items())),
        "expected_placement_records": expected_records,
        "issue": issue,
        "resource_sequence_matches_cupti": True,
        "memory_traces_present": False,
        "inputs": {
            "app_config": artifact(app_path),
            "issue_config": artifact(issue_path),
            "capture_receipt": artifact(receipt_path),
            "observer_tsv": artifact(args.observer_tsv.resolve()),
        },
        "validation_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "kernel_count": result["kernel_count"],
        "placement_records": expected_records,
        "phase_kernel_counts": result["phase_kernel_counts"],
        "validation_seconds": result["validation_seconds"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
