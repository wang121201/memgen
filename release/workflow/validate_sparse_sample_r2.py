#!/usr/bin/env python3
"""Fail-closed validation for one full-inference HyFiSS sparse sample.

The capture contains one placement record for every CTA and memory-SASS
records only for the CTA IDs named by a frozen train+holdout plan.  This
validator separates capture integrity from compatibility with the current
HBServe profile builder; a valid capture can still require a profile adapter
when it contains mixed global/shared async-copy records.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time
from types import ModuleType
from typing import Any


SCHEMA = "hyfiss_full_inference_sparse_sample_validation_v1"
MEMC_RE = re.compile(r"^kernel_([0-9]+)\.memc(?:\.part([0-9]+))?$")
XFER_RE = re.compile(r"^kernel_([0-9]+)\.xfer$")
STRUCTURAL_FIELDS = (
    "kernel_name",
    "num_registers",
    "shared_mem_bytes",
    "grid_size",
    "block_size",
    "grid_dim_x",
    "grid_dim_y",
    "grid_dim_z",
    "tb_dim_x",
    "tb_dim_y",
    "tb_dim_z",
    "llama_phase",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_module(path: Path, name: str) -> ModuleType:
    resolved = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, resolved)
    need(spec is not None and spec.loader is not None, f"cannot import {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_plan(path: Path, app: dict[int, dict[str, str]]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        need(reader.fieldnames == ["kernel_id", "cta_ids"], "sample plan header differs")
        for line_number, row in enumerate(reader, 2):
            kernel = int(row["kernel_id"])
            need(kernel not in result, f"duplicate sample-plan kernel at line {line_number}")
            raw_ids = [int(value) for value in row["cta_ids"].split(",")]
            need(raw_ids == sorted(set(raw_ids)) and raw_ids, f"invalid CTA set at plan line {line_number}")
            need(kernel in app, f"sample-plan kernel outside app.config: {kernel}")
            grid = int(app[kernel]["grid_size"])
            need(raw_ids[-1] < grid, f"sample-plan CTA escapes grid at kernel {kernel}")
            result[kernel] = set(raw_ids)
    need(sorted(result) == list(range(1, len(app) + 1)), "sample plan is not dense over every kernel")
    return result


def discover_all_memc(memory_dir: Path, kernel_count: int) -> dict[int, list[Path]]:
    files: dict[int, list[tuple[int, Path]]] = {}
    for path in memory_dir.iterdir():
        if not path.is_file():
            continue
        match = MEMC_RE.fullmatch(path.name)
        if match is None and XFER_RE.fullmatch(path.name):
            continue
        need(match is not None, f"unexpected memory-trace file: {path.name}")
        kernel = int(match.group(1))
        part = int(match.group(2) or 0)
        need(1 <= kernel <= kernel_count, f"MEMC kernel outside app.config: {kernel}")
        files.setdefault(kernel, []).append((part, path))
    need(sorted(files) == list(range(1, kernel_count + 1)), "MEMC base-file census does not cover every kernel")
    result: dict[int, list[Path]] = {}
    for kernel in range(1, kernel_count + 1):
        rows = sorted(files[kernel])
        need([part for part, _ in rows] == list(range(len(rows))), f"non-contiguous MEMC parts at kernel {kernel}")
        result[kernel] = [path for _, path in rows]
    return result


def validate_transfer_files(memory_dir: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    declared = receipt.get("transfer_files")
    need(isinstance(declared, list), "capture receipt has no transfer-file list")
    by_kernel: dict[int, dict[str, Any]] = {}
    for item in declared:
        kernel = int(item["kernel"])
        need(kernel not in by_kernel, f"duplicate transfer-file receipt for kernel {kernel}")
        by_kernel[kernel] = item
    actual = {
        int(match.group(1)): path
        for path in memory_dir.iterdir()
        if path.is_file() and (match := XFER_RE.fullmatch(path.name)) is not None
    }
    need(set(actual) == set(by_kernel), "transfer-file census differs from capture receipt")
    total_bytes = 0
    total_records = 0
    for kernel, path in actual.items():
        item = by_kernel[kernel]
        size = path.stat().st_size
        need(size == int(item["bytes"]), f"transfer-file byte count differs at kernel {kernel}")
        need(sha256_file(path) == item["sha256"], f"transfer-file SHA-256 differs at kernel {kernel}")
        total_bytes += size
        total_records += int(item["records"])
    need(total_records == int(receipt["transfer_records"]), "transfer-record total differs")
    return {
        "files": len(actual),
        "bytes": total_bytes,
        "records": total_records,
        "source_lane_reads": int(receipt["transfer_source_lane_reads"]),
        "all_file_hashes_match_receipt": True,
    }


def phase_events(path: Path) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    finished = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            if not raw.startswith("{"):
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if value.get("schema") != "hyfiss_llm_pair_v1":
                continue
            if value.get("event") == "phase_end":
                phase = str(value["phase"])
                need(phase not in events, f"duplicate phase_end event: {phase}")
                events[phase] = value
            elif value.get("event") == "finish":
                need(value.get("status") == "PASS", "driver finish event did not pass")
                finished = True
    need(finished, "driver stdout has no PASS finish event")
    return events


def compare_phase_events(sample: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    need(set(sample) == set(baseline), "sample/baseline phase sets differ")
    max_abs = {"minimum": 0.0, "maximum": 0.0, "sum": 0.0}
    for phase in sorted(sample):
        left, right = sample[phase], baseline[phase]
        for field in ("logits", "argmax"):
            need(int(left[field]) == int(right[field]), f"{phase} differs in {field}")
        for field in max_abs:
            delta = abs(float(left[field]) - float(right[field]))
            max_abs[field] = max(max_abs[field], delta)
            scale = max(1.0, abs(float(right[field])))
            need(delta <= 1e-6 * scale, f"{phase} differs in {field}: {delta}")
    return {"phase_count": len(sample), "maximum_absolute_numeric_delta": max_abs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--placement-root", type=Path, required=True)
    parser.add_argument("--placement-validation", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-receipt", type=Path, required=True)
    parser.add_argument("--placement-validator", type=Path, required=True)
    parser.add_argument("--sample-tool", type=Path, required=True)
    parser.add_argument("--expected-prompt", type=int, required=True)
    parser.add_argument("--expected-decode", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    capture = args.capture_root.resolve(strict=True)
    placement = args.placement_root.resolve(strict=True)
    output = args.output.resolve()
    need(not output.exists(), f"refusing to overwrite {output}")
    placement_tools = load_module(args.placement_validator, "placement_validation_api")
    sample_tools = load_module(args.sample_tool, "sample_trace_api")

    app_path = capture / "configs/app.config"
    issue_path = capture / "configs/issue.config"
    receipt_path = capture / "configs/capture_receipt.json"
    baseline_app_path = placement / "configs/app.config"
    app = placement_tools.parse_app(app_path)
    baseline_app = placement_tools.parse_app(baseline_app_path)
    need(len(app) == len(baseline_app), "sample/placement kernel counts differ")
    for kernel in range(1, len(app) + 1):
        for field in STRUCTURAL_FIELDS:
            need(app[kernel][field] == baseline_app[kernel][field], f"kernel {kernel} differs in {field}")

    placement_validation = json.loads(args.placement_validation.resolve(strict=True).read_text(encoding="utf-8"))
    need(
        placement_validation.get("status") == "PASS_PLACEMENT_CONFIG_EXACT_CTA_COVERAGE_NOT_MEMORY_TRACE",
        "placement baseline validation did not pass",
    )
    need(int(placement_validation["kernel_count"]) == len(app), "placement-validation kernel count differs")

    plan_path = args.plan.resolve(strict=True)
    plan_receipt_path = args.plan_receipt.resolve(strict=True)
    plan = parse_plan(plan_path, app)
    plan_receipt = json.loads(plan_receipt_path.read_text(encoding="utf-8"))
    need(plan_receipt.get("status") == "PASS_PLAN_COVERS_EVERY_KERNEL_NOT_MEMORY_TRACE", "plan receipt did not pass")
    need(int(plan_receipt["kernel_count"]) == len(app), "plan receipt kernel count differs")
    need(plan_receipt["plan"]["sha256"] == sha256_file(plan_path), "plan receipt hash differs")
    selected_total = sum(len(values) for values in plan.values())
    need(int(plan_receipt["total_sampled_ctas"]) == selected_total, "plan selected-CTA total differs")

    issue = placement_tools.validate_issue(issue_path, app)
    grid_total = sum(int(row["grid_size"]) for row in app.values())
    need(issue["tuple_count"] == grid_total, "issue tuple total differs from app grids")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    need(receipt.get("status") == "PASS", "capture receipt did not pass")
    need(receipt.get("trace_mode") == "sample", "capture is not sample mode")
    need(receipt.get("lane_format") == "memc_v3", "capture is not MEMCv3")
    need(int(receipt["kernel_count"]) == len(app), "capture-receipt kernel count differs")
    need(int(receipt["sampled_plan_kernel_count"]) == len(app), "receipt plan kernel count differs")
    need(int(receipt["sampled_plan_applied_kernel_count"]) == len(app), "receipt applied-plan count differs")
    for field in ("dropped_records", "malformed_packets", "sequence_errors", "io_errors"):
        need(int(receipt[field]) == 0, f"capture receipt has non-zero {field}")
    pushed = int(receipt["device_pushed_records"])
    received = int(receipt["host_received_records"])
    persisted = int(receipt["host_persisted_records"])
    need(pushed == received == persisted, "capture receipt record conservation differs")

    transfer = validate_transfer_files(capture / "memory_traces", receipt)

    memory_files = discover_all_memc(capture / "memory_traces", len(app))
    memory_records = 0
    memory_bytes = 0
    previous_sequence: int | None = None
    missing_selected_total = 0
    missing_global_total = 0
    missing_examples: list[dict[str, Any]] = []
    non_global_records = 0
    records_with_global_group = 0
    group_spaces: Counter[str] = Counter()
    opcode_counts: Counter[str] = Counter()
    sampled_records_by_phase: Counter[str] = Counter()
    sampled_ctas_seen_total = 0
    for kernel in range(1, len(app) + 1):
        selected = plan[kernel]
        observed: set[int] = set()
        observed_global: set[int] = set()
        for path in memory_files[kernel]:
            memory_bytes += path.stat().st_size
            for record in sample_tools.memc_records(path):
                memory_records += 1
                sampled_records_by_phase[app[kernel]["llama_phase"]] += 1
                block = int(record["block"])
                need(block in selected, f"kernel {kernel} contains unplanned memory CTA {block}")
                observed.add(block)
                sequence = int(record["source_sequence"])
                need(0 <= sequence < persisted, f"MEMC sequence outside receipt range: {sequence}")
                if previous_sequence is not None:
                    need(sequence > previous_sequence, f"MEMC global sequence is not strictly increasing at kernel {kernel}")
                previous_sequence = sequence
                mask = int(record["mask"], 16)
                has_global = False
                all_global = True
                for group in record["groups"]:
                    global_mask = int(group["global_mask"])
                    local_mask = int(group["local_mask"])
                    shared_mask = int(group["shared_mask"])
                    unknown_mask = int(group["unknown_mask"])
                    if global_mask:
                        group_spaces["global"] += 1
                        has_global = True
                    if local_mask:
                        group_spaces["local"] += 1
                    if shared_mask:
                        group_spaces["shared"] += 1
                    if unknown_mask:
                        group_spaces["unknown"] += 1
                    all_global &= global_mask == mask and local_mask == 0 and shared_mask == 0 and unknown_mask == 0
                if has_global:
                    records_with_global_group += 1
                    observed_global.add(block)
                if not all_global:
                    non_global_records += 1
                opcode_counts[str(record["opcode"])] += 1
        sampled_ctas_seen_total += len(observed)
        missing = sorted(selected - observed)
        missing_global = sorted(selected - observed_global)
        missing_selected_total += len(missing)
        missing_global_total += len(missing_global)
        if (missing or missing_global) and len(missing_examples) < 32:
            missing_examples.append({
                "kernel": kernel,
                "phase": app[kernel]["llama_phase"],
                "missing_any_record": missing[:32],
                "missing_global_record": missing_global[:32],
            })

    need(memory_records + issue["tuple_count"] == persisted, "MEMC plus placement records do not equal receipt total")
    need(memory_records > 0 and previous_sequence is not None, "sample capture contains no memory records")
    need(sampled_ctas_seen_total <= selected_total, "observed sampled CTA total exceeds plan")

    phases = Counter(row["llama_phase"] for row in app.values())
    need(phases["prefill"] > 0, "capture has no prefill kernels")
    for step in range(args.expected_decode):
        need(phases[f"decode_step_{step}"] > 0, f"capture lacks decode_step_{step}")
    need(sum(phases.values()) == len(app), "phase labels do not cover every kernel")

    input_token_path = capture / "input_tokens.i32"
    baseline_input_token_path = placement / "input_tokens.i32"
    need(sha256_file(input_token_path) == sha256_file(baseline_input_token_path), "sample/baseline input tokens differ")
    phase_comparison = compare_phase_events(
        phase_events(capture / "stdout.log"),
        phase_events(placement / "stdout.log"),
    )

    current_profile_ready = missing_selected_total == 0 and non_global_records == 0
    global_cache_profile_ready = missing_global_total == 0
    result = {
        "schema": SCHEMA,
        "status": (
            "PASS_CAPTURE_INTEGRITY_AND_CURRENT_PROFILE_INPUT"
            if current_profile_ready else
            "PASS_CAPTURE_INTEGRITY_PROFILE_ADAPTER_REQUIRED"
        ),
        "workload": {
            "model": "Qwen2.5-1.5B-Instruct Q8_0 via llama.cpp",
            "batch": 1,
            "prompt_tokens": args.expected_prompt,
            "decode_steps": args.expected_decode,
        },
        "definitions": {
            "capture_integrity": "every launched CTA has one placement tuple, every memory record belongs to the frozen sparse CTA plan, records are lossless and phases/output match an independently validated placement-only run",
            "current_profile_input": "every selected CTA has a record and every captured address group is entirely global, matching the unmodified HBServe MEMCv3 profile reader",
            "global_cache_profile_input": "every selected CTA has at least one global address group; mixed global/shared async-copy records may require an explicit global-only lowering adapter",
            "excluded_claims": [
                "generated full-inference memory-SASS",
                "naive-cache or Memgen metrics",
                "NCU hardware accuracy",
                "performance timing",
            ],
        },
        "kernel_count": len(app),
        "phase_kernel_counts": dict(sorted(phases.items())),
        "grid_ctas": grid_total,
        "planned_sample_ctas": selected_total,
        "sampled_ctas_with_any_record": sampled_ctas_seen_total,
        "missing_selected_ctas": missing_selected_total,
        "missing_selected_ctas_with_global_record": missing_global_total,
        "missing_examples": missing_examples,
        "memory": {
            "files": sum(len(paths) for paths in memory_files.values()),
            "bytes": memory_bytes,
            "records": memory_records,
            "records_with_global_group": records_with_global_group,
            "records_not_all_global": non_global_records,
            "address_group_space_nonzero_counts": dict(sorted(group_spaces.items())),
            "top_opcodes": opcode_counts.most_common(32),
            "records_by_phase": dict(sorted(sampled_records_by_phase.items())),
            "source_sequences_strictly_increasing": True,
        },
        "async_global_to_shared_transfer": transfer,
        "issue": issue,
        "record_conservation": {
            "device_pushed": pushed,
            "host_received": received,
            "host_persisted": persisted,
            "decoded_memory_plus_issue": memory_records + issue["tuple_count"],
        },
        "sample_matches_placement_workload": True,
        "phase_output_comparison": phase_comparison,
        "current_profile_input_ready": current_profile_ready,
        "global_cache_profile_input_ready_after_explicit_space_lowering": global_cache_profile_ready,
        "inputs": {
            "app_config": artifact(app_path),
            "issue_config": artifact(issue_path),
            "capture_receipt": artifact(receipt_path),
            "sample_plan": artifact(plan_path),
            "sample_plan_receipt": artifact(plan_receipt_path),
            "placement_validation": artifact(args.placement_validation.resolve(strict=True)),
            "sample_tool": artifact(args.sample_tool.resolve(strict=True)),
            "placement_validator": artifact(args.placement_validator.resolve(strict=True)),
            "input_tokens": artifact(input_token_path),
        },
        "validation_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "kernel_count": result["kernel_count"],
        "memory_records": memory_records,
        "missing_selected_ctas": missing_selected_total,
        "records_not_all_global": non_global_records,
        "validation_seconds": result["validation_seconds"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
