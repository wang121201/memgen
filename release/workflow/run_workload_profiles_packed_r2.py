#!/usr/bin/env python3
"""Build a fail-closed packed HBServe profile for every kernel in a sparse run."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any


PASS_CAPTURE = "PASS_CAPTURE_INTEGRITY_PROFILE_ADAPTER_REQUIRED"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_profile_tool(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("hbserve_sampled_sass", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load profile tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["kernel_id", "mode", "train_ctas", "holdout_ctas"]:
            raise RuntimeError(f"unexpected profile-plan header: {reader.fieldnames}")
        rows = list(reader)
    kernel_ids = [int(row["kernel_id"]) for row in rows]
    if kernel_ids != list(range(1, len(rows) + 1)):
        raise RuntimeError("profile plan kernel IDs are not dense ordered 1..N")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-tool", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--sample-plan", type=Path, required=True)
    parser.add_argument("--profile-plan", type=Path, required=True)
    parser.add_argument("--capture-validation", type=Path, required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()

    started = time.monotonic()
    profile_tool_path = args.profile_tool.resolve(strict=True)
    capture_root = args.capture_root.resolve(strict=True)
    sample_plan = args.sample_plan.resolve(strict=True)
    profile_plan = args.profile_plan.resolve(strict=True)
    validation_path = args.capture_validation.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to reuse output root: {output_root}")
    output_root.mkdir(parents=True)
    manifest_path = output_root / "manifest.json"
    pack_path = output_root / "profiles.pack.jsonl"
    index_path = output_root / "profiles.index.jsonl"

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != PASS_CAPTURE:
        raise SystemExit(f"capture validation is not admitted: {validation.get('status')}")
    rows = read_plan(profile_plan)
    if int(validation.get("kernel_count", -1)) != len(rows):
        raise SystemExit("capture-validation and profile-plan kernel counts differ")

    tool = load_profile_tool(profile_tool_path)
    # Populate immutable whole-workload caches once.  This avoids hashing the
    # 0.5--0.7 GB issue file and scanning app.config once per kernel.
    app_config = capture_root / "configs/app.config"
    issue_config = capture_root / "configs/issue.config"
    tool.sha256_file(app_config)
    tool.sha256_file(issue_config)
    tool.app_kernel_lines(app_config, 1)
    tool.sample_plan_ctas(sample_plan, 1)

    def cached_artifact(path: Path) -> dict[str, Any]:
        resolved = path.resolve(strict=True)
        return {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": tool.sha256_file(resolved),
        }

    manifest: dict[str, Any] = {
        "schema": "hbserve_full_inference_profile_census_v1",
        "status": "STARTED",
        "workload_id": args.workload_id,
        "kernel_count": len(rows),
        "completed_kernels": 0,
        "started_utc": utc_now(),
        "inputs": {
            "profile_tool": artifact(profile_tool_path),
            "capture_validation": artifact(validation_path),
            "sample_plan": artifact(sample_plan),
            "profile_plan": artifact(profile_plan),
            "app_config": cached_artifact(app_config),
            "issue_config": cached_artifact(issue_config),
        },
        "claim_boundary": (
            "Per-kernel HBServe memory-SASS profiles only; cache replay and hardware "
            "accuracy are not claimed by this receipt"
        ),
    }
    write_manifest(manifest_path, manifest)

    pack_digest = hashlib.sha256()
    index_digest = hashlib.sha256()
    try:
        with pack_path.open("xb") as pack_stream, index_path.open("xb") as index_stream:
            for position, row in enumerate(rows, 1):
                kernel = int(row["kernel_id"])
                complete_grid = row["mode"] == "complete_grid_no_cta_extrapolation"
                if complete_grid and row["holdout_ctas"]:
                    raise RuntimeError(f"kernel {kernel}: complete-grid row has holdout CTAs")
                if not complete_grid and not row["holdout_ctas"]:
                    raise RuntimeError(f"kernel {kernel}: sampled row has no disjoint holdout")
                result = tool.profile(SimpleNamespace(
                    trace_root=str(capture_root),
                    kernel=kernel,
                    input_format="memc",
                    train_ctas=row["train_ctas"],
                    holdout_ctas=row["holdout_ctas"] or None,
                    complete_grid=complete_grid,
                    sample_plan=str(sample_plan),
                    workload_id=args.workload_id,
                    output=None,
                ))
                status = result.get("status", "")
                if not status.startswith("PASS_"):
                    raise RuntimeError(f"kernel {kernel}: profile did not pass: {status}")
                profile_encoded = (
                    json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                profile_offset = pack_stream.tell()
                pack_stream.write(profile_encoded)
                pack_stream.flush()
                pack_digest.update(profile_encoded)
                index_row = {
                    "kernel_id": kernel,
                    "mode": row["mode"],
                    "schema": result["schema"],
                    "status": status,
                    "profile_payload_sha256": result["profile_payload_sha256"],
                    "path": str(pack_path),
                    "offset": profile_offset,
                    "bytes": len(profile_encoded),
                    "sha256": hashlib.sha256(profile_encoded).hexdigest(),
                }
                encoded = (json.dumps(index_row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                index_stream.write(encoded)
                index_stream.flush()
                index_digest.update(encoded)
                manifest["completed_kernels"] = position
                if position % args.progress_interval == 0 or position == len(rows):
                    manifest["last_kernel_id"] = kernel
                    manifest["elapsed_seconds"] = time.monotonic() - started
                    write_manifest(manifest_path, manifest)
    except BaseException as error:
        manifest.update({
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "finished_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        })
        write_manifest(manifest_path, manifest)
        raise

    manifest.update({
        "status": "PASS_ALL_KERNEL_PROFILES_ADMITTED_NOT_CACHE_REPLAYED",
        "finished_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "profile_pack": {
            "path": str(pack_path),
            "bytes": pack_path.stat().st_size,
            "sha256": pack_digest.hexdigest(),
            "records": len(rows),
            "format": "one compact JSON profile plus newline per indexed byte range",
        },
        "profile_index": {
            "path": str(index_path),
            "bytes": index_path.stat().st_size,
            "sha256": index_digest.hexdigest(),
        },
        "profile_files_created": 2,
    })
    write_manifest(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "kernel_count": len(rows),
        "output": str(output_root),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
