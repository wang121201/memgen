#!/usr/bin/env python3
"""Build one fail-closed train+holdout CTA plan per full inference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census-tool", type=Path, required=True)
    parser.add_argument("--target", action="append", nargs=2, metavar=("LABEL", "APP_CONFIG"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    census_path = args.census_tool.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("full_phase_census", census_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load census tool: {census_path}")
    census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(census)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, raw_app in args.target:
        app_path = Path(raw_app).resolve(strict=True)
        metadata = census.app_metadata(app_path)
        expected = list(range(1, len(metadata) + 1))
        if sorted(metadata) != expected:
            raise RuntimeError(f"{label}: app kernel IDs are not dense 1..N")
        plan_path = output_root / f"{label}.tsv"
        phase_selected: dict[str, int] = defaultdict(int)
        phase_grid: dict[str, int] = defaultdict(int)
        mode_counts: Counter[str] = Counter()
        selected_counts = []
        total_grid = 0
        total_selected = 0
        with plan_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("kernel_id\tcta_ids\n")
            for kernel in expected:
                item = metadata[kernel]
                dims = tuple(int(item[f"grid_dim_{axis}"]) for axis in "xyz")
                plan = census.cta_plan(dims, item["kernel_name"])
                selected = sorted(set(plan["train"]) | set(plan["holdout"]))
                grid = dims[0] * dims[1] * dims[2]
                if not selected or len(selected) > 1024 or selected[-1] >= grid:
                    raise RuntimeError(
                        f"{label}: invalid CTA plan at kernel {kernel}: "
                        f"selected={len(selected)} grid={grid}"
                    )
                stream.write(f"{kernel}\t{','.join(str(value) for value in selected)}\n")
                phase = item.get("llama_phase", "unknown")
                phase_selected[phase] += len(selected)
                phase_grid[phase] += grid
                mode_counts[plan["mode"]] += 1
                selected_counts.append(len(selected))
                total_selected += len(selected)
                total_grid += grid
        receipt = {
            "schema": "hyfiss_full_inference_sparse_cta_plan_v1",
            "status": "PASS_PLAN_COVERS_EVERY_KERNEL_NOT_MEMORY_TRACE",
            "label": label,
            "kernel_count": len(metadata),
            "total_grid_ctas": total_grid,
            "total_sampled_ctas": total_selected,
            "sampled_fraction_of_grid_ctas": total_selected / total_grid,
            "sampled_ctas_per_kernel": {
                "minimum": min(selected_counts),
                "maximum": max(selected_counts),
                "mean": total_selected / len(selected_counts),
            },
            "plan_mode_kernel_counts": dict(sorted(mode_counts.items())),
            "phase_totals": {
                phase: {
                    "grid_ctas": phase_grid[phase],
                    "sampled_ctas": phase_selected[phase],
                    "sampled_fraction": phase_selected[phase] / phase_grid[phase],
                }
                for phase in sorted(phase_grid)
            },
            "inputs": {
                "app_config": artifact(app_path),
                "census_tool": artifact(census_path),
            },
            "plan": artifact(plan_path),
            "claim_boundary": (
                "Train plus disjoint-holdout CTA selection only; no address capture, "
                "profile admission, generated trace, cache result, or hardware accuracy"
            ),
        }
        receipt_path = output_root / f"{label}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({
            "label": label,
            "kernels": len(metadata),
            "grid_ctas": total_grid,
            "sampled_ctas": total_selected,
            "sampled_fraction": total_selected / total_grid,
            "plan": str(plan_path),
            "receipt": str(receipt_path),
        })
    print(json.dumps({"status": "PASS", "rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
