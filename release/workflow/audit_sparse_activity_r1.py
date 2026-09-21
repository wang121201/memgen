#!/usr/bin/env python3
"""Aggregate observed and empty sampled CTA classes by kernel structure."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


STRUCTURAL_FIELDS = (
    "kernel_name", "num_registers", "shared_mem_bytes", "grid_size", "block_size",
    "grid_dim_x", "grid_dim_y", "grid_dim_z", "tb_dim_x", "tb_dim_y", "tb_dim_z",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path.resolve(strict=True))
    need(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sample-tool", type=Path, required=True)
    parser.add_argument("--validation-tool", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    need(not output.exists(), f"refusing to overwrite {output}")
    capture = args.capture_root.resolve(strict=True)
    sample = load(args.sample_tool, "sample_api")
    validation = load(args.validation_tool, "validation_api")
    app_path = capture / "configs/app.config"
    app = validation.parse_app(app_path)
    plan_path = args.plan.resolve(strict=True)
    plan = validation.parse_plan(plan_path, app)
    files = validation.discover_all_memc(capture / "memory_traces", len(app))

    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    phase_missing: Counter[str] = Counter()
    kernels_with_missing = 0
    kernels_with_no_sampled_records = 0
    total_observed = 0
    total_missing = 0
    for kernel in range(1, len(app) + 1):
        observed: set[int] = set()
        records = 0
        for path in files[kernel]:
            for record in sample.memc_records(path):
                block = int(record["block"])
                need(block in plan[kernel], f"kernel {kernel} contains unplanned CTA {block}")
                observed.add(block)
                records += 1
        missing = plan[kernel] - observed
        total_observed += len(observed)
        total_missing += len(missing)
        if missing:
            kernels_with_missing += 1
            phase_missing[app[kernel]["llama_phase"]] += len(missing)
        if not observed:
            kernels_with_no_sampled_records += 1
        key = tuple(app[kernel][field] for field in STRUCTURAL_FIELDS)
        row = groups.setdefault(key, {
            "structure": {field: app[kernel][field] for field in STRUCTURAL_FIELDS},
            "kernel_count": 0,
            "kernel_ids": [],
            "sampled_ctas": 0,
            "observed_ctas": 0,
            "missing_ctas": 0,
            "kernels_with_missing": 0,
            "kernels_with_no_sampled_records": 0,
            "memory_records": 0,
            "activity_patterns": Counter(),
        })
        row["kernel_count"] += 1
        if len(row["kernel_ids"]) < 32:
            row["kernel_ids"].append(kernel)
        row["sampled_ctas"] += len(plan[kernel])
        row["observed_ctas"] += len(observed)
        row["missing_ctas"] += len(missing)
        row["kernels_with_missing"] += int(bool(missing))
        row["kernels_with_no_sampled_records"] += int(not observed)
        row["memory_records"] += records
        pattern = json.dumps({"observed": sorted(observed), "missing": sorted(missing)}, separators=(",", ":"))
        row["activity_patterns"][pattern] += 1

    rows = []
    for row in groups.values():
        patterns = []
        for encoded, count in row.pop("activity_patterns").most_common():
            value = json.loads(encoded)
            patterns.append({"kernel_count": count, **value})
        row["activity_patterns"] = patterns
        rows.append(row)
    rows.sort(key=lambda row: (-row["missing_ctas"], row["structure"]["kernel_name"]))
    result = {
        "schema": "hyfiss_sparse_cta_activity_census_v1",
        "status": "COMPLETE_ACTIVITY_CENSUS_NOT_FULL_GRID_ACTIVITY",
        "label": args.label,
        "definition": {
            "observed_cta": "a CTA named in the frozen sample plan that produced at least one persisted MEMCv3 instruction",
            "missing_cta": "a CTA named in the plan with no persisted memory instruction; this is not a transport loss because capture conservation is validated separately",
            "boundary": "unselected CTAs remain unknown; repeated sampled patterns do not prove a full-grid activity map",
        },
        "kernel_count": len(app),
        "unique_structures": len(rows),
        "planned_sample_ctas": sum(len(values) for values in plan.values()),
        "observed_sample_ctas": total_observed,
        "missing_sample_ctas": total_missing,
        "kernels_with_missing": kernels_with_missing,
        "kernels_with_no_sampled_records": kernels_with_no_sampled_records,
        "missing_ctas_by_phase": dict(sorted(phase_missing.items())),
        "structures_with_missing": [row for row in rows if row["missing_ctas"]],
        "inputs": {
            "app_config": artifact(app_path),
            "capture_receipt": artifact(capture / "configs/capture_receipt.json"),
            "sample_plan": artifact(plan_path),
            "sample_tool": artifact(args.sample_tool.resolve(strict=True)),
            "validation_tool": artifact(args.validation_tool.resolve(strict=True)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "kernel_count": result["kernel_count"],
        "kernels_with_missing": kernels_with_missing,
        "kernels_with_no_sampled_records": kernels_with_no_sampled_records,
        "missing_sample_ctas": total_missing,
        "structures_with_missing": len(result["structures_with_missing"]),
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
