#!/usr/bin/env python3
"""Build a fail-closed 566-kernel sampled-SASS profile census.

The source phase is profiled once by ordinal.  Later full-phase generation may
retarget each admitted source profile only to the structurally identical
ordinal in another endpoint.  Grids with one or two CTAs retain the complete
grid and explicitly make no CTA-extrapolation claim; all larger grids require
a disjoint holdout.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any, Iterable


SCHEMA = "hyfiss_full_phase_profile_census_v1"
DECODES = (2, 4, 8, 16, 32, 64, 128)
APP_RE = re.compile(r"^-kernel_([0-9]+)_([^ ]+)\s+(.*)$")
STRUCTURAL_FIELDS = (
    "kernel_name",
    "grid_size",
    "block_size",
    "num_registers",
    "shared_mem_bytes",
    "grid_dim_x",
    "grid_dim_y",
    "grid_dim_z",
    "tb_dim_x",
    "tb_dim_y",
    "tb_dim_z",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    need(resolved.is_file(), f"not a file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def app_metadata(path: Path) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            match = APP_RE.fullmatch(raw.rstrip("\n"))
            if match is None:
                continue
            kernel, field, value = int(match.group(1)), match.group(2), match.group(3)
            row = result.setdefault(kernel, {})
            need(field not in row, f"duplicate app.config field: kernel {kernel} {field}")
            row[field] = value
    need(result, "app.config has no kernel metadata")
    return result


def unique(values: Iterable[int]) -> list[int]:
    return sorted(set(values))


def block_id(coordinates: tuple[int, int, int], dims: tuple[int, int, int]) -> int:
    x, y, z = coordinates
    gx, gy, gz = dims
    need(0 <= x < gx and 0 <= y < gy and 0 <= z < gz, "CTA coordinates escape grid")
    return x + gx * (y + gy * z)


def positions(extent: int, fractions: tuple[float, ...]) -> list[int]:
    return unique(round((extent - 1) * fraction) for fraction in fractions)


def cta_plan(dims: tuple[int, int, int], kernel_name: str | None = None) -> dict[str, Any]:
    grid_size = math.prod(dims)
    need(grid_size > 0, "empty CTA grid")
    # The 36-CTA quantizer has a captured boundary CTA with a distinct ranked
    # instruction structure.  Retain its whole small grid so class selection is
    # exact by CTA ID; never infer that the final CTA is generically the tail.
    if grid_size <= 2 or (grid_size == 36 and kernel_name == "quantize_q8_1"):
        return {
            "mode": "complete_grid_no_cta_extrapolation",
            "train": list(range(grid_size)),
            "holdout": [],
        }

    active_axes = [axis for axis, extent in enumerate(dims) if extent > 1]
    if len(active_axes) == 1:
        axis = active_axes[0]
        extent = dims[axis]

        def one_dimensional_block(position: int) -> int:
            coordinate = [0, 0, 0]
            coordinate[axis] = position
            return block_id(tuple(coordinate), dims)

        train_positions = positions(extent, (0.0, 0.01, 0.25, 0.5, 0.75, 1.0))
        holdout_positions = positions(extent, (0.02, 0.125, 1.0 / 3.0, 2.0 / 3.0, 0.875, 0.98))
        train = unique(one_dimensional_block(value) for value in train_positions)
        holdout = unique(
            one_dimensional_block(value)
            for value in holdout_positions
            if one_dimensional_block(value) not in set(train)
        )
        if not holdout:
            return {
                "mode": "complete_grid_no_cta_extrapolation",
                "train": list(range(grid_size)),
                "holdout": [],
            }
        return {"mode": "disjoint_cta_holdout", "train": train, "holdout": holdout}

    gx, gy, gz = dims
    train_coordinates: set[tuple[int, int, int]] = {(0, 0, 0)}
    # Retain a complete y table at the anchor x/z so the existing categorical-y
    # fallback remains exact for grouped-query-attention head steps.
    for y in range(gy):
        train_coordinates.add((0, y, 0))
    if gx > 1:
        train_coordinates.update({(1, 0, 0), (gx - 1, 0, 0)})
    if gz > 1:
        train_coordinates.update({(0, 0, 1), (0, 0, gz - 1)})

    x_values = positions(gx, (0.01, 0.25, 0.5, 0.75, 0.99))
    y_values = positions(gy, (0.0, 0.25, 0.5, 0.75, 1.0))
    z_values = positions(gz, (0.0, 0.5, 1.0))
    holdout_coordinates = [
        (x, y, z)
        for z in z_values
        for y in y_values
        for x in x_values
        if (x, y, z) not in train_coordinates
    ]
    train = unique(block_id(value, dims) for value in train_coordinates)
    holdout = unique(block_id(value, dims) for value in holdout_coordinates)[:32]
    need(holdout, f"multi-axis grid has no disjoint holdout: {dims}")
    return {"mode": "disjoint_cta_holdout", "train": train, "holdout": holdout}


def run_process(stage: Path, argv: list[str], timeout_seconds: int) -> dict[str, Any]:
    stage.mkdir(parents=True)
    save_json(stage / "command.json", {"argv": argv, "timeout_seconds": timeout_seconds})
    started = time.monotonic()
    with (stage / "stdout.log").open("x", encoding="utf-8") as stdout, (
        stage / "stderr.log"
    ).open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            argv,
            stdout=stdout,
            stderr=stderr,
            env={
                key: value
                for key, value in os.environ.items()
                if key not in ("root_sudo", "LD_PRELOAD")
            },
            start_new_session=True,
        )
        save_json(stage / "started.json", {"pid": process.pid, "process_group": process.pid})
        stop_reason = None
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stop_reason = "timeout"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    result = {
        "returncode": process.returncode,
        "seconds": time.monotonic() - started,
        "stop_reason": stop_reason,
        "command": artifact(stage / "command.json"),
        "stdout": artifact(stage / "stdout.log"),
        "stderr": artifact(stage / "stderr.log"),
    }
    save_json(stage / "process.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--sample-tool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-decode", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--minimum-free-gib", type=int, default=64)
    args = parser.parse_args()
    need(args.source_decode in DECODES, "source decode is outside the frozen endpoint set")
    need(1 <= args.workers <= 16, "workers outside bounded range")

    pair_root = args.pair_root.resolve(strict=True)
    capture_root = pair_root / "capture"
    app_path = capture_root / "configs" / "app.config"
    issue_path = capture_root / "configs" / "issue.config"
    structure_path = args.structure.resolve(strict=True)
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    need(
        structure.get("status") == "PASS_ENDPOINT_STRUCTURE_CENSUS_NOT_TRACE_OR_CACHE_ACCURACY",
        "endpoint structure census did not pass",
    )
    endpoint_rows = {
        int(row["decode_tokens"]): row
        for row in structure["rows"]
        if int(row.get("decode_tokens", -1)) in DECODES
    }
    need(tuple(sorted(endpoint_rows)) == DECODES, "seven endpoint rows are incomplete")
    source_row = endpoint_rows[args.source_decode]
    source_kernel_ids = [int(value) for value in source_row["kernel_ids"]]
    need(len(source_kernel_ids) == 566, "source phase does not contain 566 kernels")

    metadata = app_metadata(app_path)
    for ordinal, source_kernel in enumerate(source_kernel_ids):
        source = metadata[source_kernel]
        for decode in DECODES:
            target_kernel = int(endpoint_rows[decode]["kernel_ids"][ordinal])
            target = metadata[target_kernel]
            for field in STRUCTURAL_FIELDS:
                need(
                    source.get(field) == target.get(field),
                    f"D{decode} ordinal {ordinal} structural mismatch in {field}",
                )

    sample_tool = args.sample_tool.resolve(strict=True)
    output_root = args.output_root.resolve()
    need(not output_root.exists(), f"output collision: {output_root}")
    need(
        shutil.disk_usage(pair_root).free >= args.minimum_free_gib * (1 << 30),
        "insufficient initial free space",
    )
    output_root.mkdir(parents=True)
    (output_root / "profiles").mkdir()
    (output_root / "logs").mkdir()
    start = {
        "schema": SCHEMA,
        "status": "STARTED",
        "source_decode": args.source_decode,
        "source_phase": source_row["phase"],
        "kernel_count": len(source_kernel_ids),
        "workers": args.workers,
        "inputs": {
            "pair_finish": artifact(pair_root / "finish.json"),
            "structure": artifact(structure_path),
            "app_config": artifact(app_path),
            "issue_config": artifact(issue_path),
            "sample_tool": artifact(sample_tool),
            "runner": artifact(Path(__file__)),
        },
        "claim_boundary": (
            "Profiles only: no generated full phase, cache replay, hardware accuracy, or "
            "cross-kernel writeback claim. Complete-grid profiles make no CTA extrapolation claim."
        ),
    }
    save_json(output_root / "start.json", start)

    def profile_one(item: tuple[int, int]) -> dict[str, Any]:
        ordinal, kernel = item
        row = metadata[kernel]
        dims = tuple(int(row[f"grid_dim_{axis}"]) for axis in "xyz")
        plan = cta_plan(dims, row["kernel_name"])
        destination = output_root / "profiles" / f"ordinal-{ordinal:03d}-k{kernel}"
        destination.mkdir()
        profile_path = destination / "profile.json"
        argv = [
            "python3", "-B", str(sample_tool), "profile",
            "--trace-root", str(capture_root),
            "--kernel", str(kernel),
            "--input-format", "memc",
            "--train-ctas", ",".join(str(value) for value in plan["train"]),
            "--workload-id", (
                f"qwen25-15b-q8-p512-d{args.source_decode}-full-phase-"
                f"ordinal-{ordinal}-k{kernel}"
            ),
            "--output", str(profile_path),
        ]
        if plan["mode"] == "complete_grid_no_cta_extrapolation":
            argv.append("--complete-grid")
        else:
            argv.extend([
                "--holdout-ctas",
                ",".join(str(value) for value in plan["holdout"]),
            ])
        process = run_process(output_root / "logs" / f"profile-{ordinal:03d}-k{kernel}", argv, args.timeout_seconds)
        result: dict[str, Any] = {
            "ordinal": ordinal,
            "kernel_id": kernel,
            "kernel_name": row["kernel_name"],
            "grid_dims": list(dims),
            "grid_size": math.prod(dims),
            "plan": plan,
            "process": process,
        }
        if process["returncode"] != 0 or process["stop_reason"] is not None:
            error_text = (output_root / "logs" / f"profile-{ordinal:03d}-k{kernel}" / "stderr.log").read_text(
                encoding="utf-8", errors="replace"
            )
            result.update(
                status="REJECTED_PROFILE",
                error_tail=error_text[-2000:],
            )
            return result
        profile_value = json.loads(profile_path.read_text(encoding="utf-8"))
        if plan["mode"] == "complete_grid_no_cta_extrapolation":
            need(
                profile_value.get("status") in {
                    "PASS_COMPLETE_GRID_NO_CTA_EXTRAPOLATION",
                    "PASS_COMPLETE_GRID_STRUCTURAL_CLASSES_NO_CTA_EXTRAPOLATION",
                },
                f"kernel {kernel} complete-grid profile status differs",
            )
            need(
                profile_value.get("schema") in ({
                    "name": "hbserve.hyfiss_sampled_sass_profile", "version": 5,
                }, {
                    "name": "hbserve.hyfiss_sampled_sass_profile", "version": 6,
                }),
                f"kernel {kernel} complete-grid profile schema differs",
            )
        else:
            need(
                profile_value.get("status") == "PASS_EXACT_COORDINATE_RULE_TRAIN_AND_DISJOINT_HOLDOUT",
                f"kernel {kernel} holdout profile status differs",
            )
            need(profile_value.get("schema") == {
                "name": "hbserve.hyfiss_sampled_sass_profile", "version": 4,
            }, f"kernel {kernel} holdout profile schema differs")
        need(int(profile_value["kernel"]["id"]) == kernel, f"kernel {kernel} profile identity differs")
        result.update(
            status="PASS_PROFILE",
            profile=artifact(profile_path),
            holdout=profile_value["holdout"],
        )
        return result

    began = time.monotonic()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(profile_one, item): item
            for item in enumerate(source_kernel_ids)
        }
        for future in concurrent.futures.as_completed(futures):
            ordinal, kernel = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "ordinal": ordinal,
                    "kernel_id": kernel,
                    "kernel_name": metadata[kernel].get("kernel_name"),
                    "status": "INFRASTRUCTURE_FAILURE",
                    "error": repr(error),
                }
            results.append(result)
            print(json.dumps({
                "ordinal": ordinal,
                "kernel": kernel,
                "status": result["status"],
            }, sort_keys=True), flush=True)
    results.sort(key=lambda row: row["ordinal"])
    need([row["kernel_id"] for row in results] == source_kernel_ids, "profile census order differs")
    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    passed = counts.get("PASS_PROFILE", 0)
    finish = {
        **start,
        "status": (
            "PASS_ALL_566_PROFILES"
            if passed == 566 else
            "COMPLETE_PROFILE_CENSUS_WITH_REJECTIONS"
        ),
        "seconds": time.monotonic() - began,
        "counts": counts,
        "results": results,
        "next_gate": (
            "Generate and full-semantic-audit D2 only after all 566 profiles pass."
            if passed == 566 else
            "Inspect explicit profile rejections; do not generate a partial phase."
        ),
    }
    save_json(output_root / "finish.json", finish)
    print(json.dumps({
        "status": finish["status"],
        "counts": counts,
        "finish": str(output_root / "finish.json"),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
