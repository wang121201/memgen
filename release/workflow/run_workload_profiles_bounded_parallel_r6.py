#!/usr/bin/env python3
"""Build a fail-closed packed HBServe profile for every kernel in a sparse run."""

from __future__ import annotations

import argparse
import concurrent.futures
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
_WORKER_CONTEXT: dict[str, Any] = {}


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


def initialize_profile_worker(
    profile_tool_path: str,
    capture_root: str,
    sample_plan: str,
    workload_id: str,
) -> None:
    tool = load_profile_tool(Path(profile_tool_path))
    capture = Path(capture_root)
    plan = Path(sample_plan)
    app_config = capture / "configs/app.config"
    issue_config = capture / "configs/issue.config"
    # Each worker pays these immutable whole-workload scans once, rather than
    # once per kernel.  All later accesses use the profiler's identity caches.
    tool.sha256_file(app_config)
    tool.sha256_file(issue_config)
    tool.app_kernel_lines(app_config, 1)
    tool.sample_plan_ctas(plan, 1)
    tool.discover_memc_files(capture, 1)
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update({
        "tool": tool,
        "capture_root": capture_root,
        "sample_plan": sample_plan,
        "workload_id": workload_id,
    })


def build_profile_row(row: dict[str, str]) -> dict[str, Any]:
    tool = _WORKER_CONTEXT["tool"]
    kernel = int(row["kernel_id"])
    try:
        complete_grid = row["mode"] == "complete_grid_no_cta_extrapolation"
        if complete_grid and row["holdout_ctas"]:
            raise RuntimeError("complete-grid row has holdout CTAs")
        if not complete_grid and not row["holdout_ctas"]:
            raise RuntimeError("sampled row has no disjoint holdout")
        train_ctas = tool.parse_ids(row["train_ctas"])
        holdout_ctas = (
            tool.parse_ids(row["holdout_ctas"])
            if row["holdout_ctas"] else
            []
        )
        promoted_ctas: list[int] = []
        if not complete_grid:
            if len(holdout_ctas) < 2:
                raise RuntimeError("cannot promote while retaining a holdout")
            promoted_ctas = [holdout_ctas[0]]
            train_ctas = sorted(train_ctas + promoted_ctas)
            holdout_ctas = holdout_ctas[1:]
        result = tool.profile(SimpleNamespace(
            trace_root=_WORKER_CONTEXT["capture_root"],
            kernel=kernel,
            input_format="memc",
            train_ctas=",".join(str(value) for value in train_ctas),
            holdout_ctas=(
                ",".join(str(value) for value in holdout_ctas)
                if holdout_ctas else None
            ),
            complete_grid=complete_grid,
            sample_plan=_WORKER_CONTEXT["sample_plan"],
            workload_id=_WORKER_CONTEXT["workload_id"],
            output=None,
            sequence_gaps_validated=True,
        ))
        status = result.get("status", "")
        if not status.startswith("PASS_"):
            raise RuntimeError(f"profile did not pass: {status}")
        return {
            "kernel": kernel,
            "mode": row["mode"],
            "result": result,
            "status": status,
            "promoted_ctas": promoted_ctas,
            "training_cta_count": len(train_ctas),
            "holdout_cta_count": len(holdout_ctas),
        }
    except BaseException as error:
        raise RuntimeError(f"kernel {kernel}: {error}") from error


def ordered_parallel_profiles(
    executor: concurrent.futures.ProcessPoolExecutor,
    rows: list[dict[str, str]],
    workers: int,
) -> Any:
    """Yield process results in kernel order with a bounded look-ahead window.

    This prevents a slow early complete-grid kernel from allowing thousands of
    later JSON profiles to accumulate in IPC buffers.  Any completed future
    that failed is raised immediately, even if an earlier kernel is unfinished.
    """
    window = max(workers, workers * 2)
    next_submit = 0
    next_emit = 0
    pending: dict[concurrent.futures.Future[Any], int] = {}
    completed: dict[int, dict[str, Any]] = {}

    def fill() -> None:
        nonlocal next_submit
        while next_submit < len(rows) and len(pending) + len(completed) < window:
            future = executor.submit(build_profile_row, rows[next_submit])
            pending[future] = next_submit
            next_submit += 1

    fill()
    while next_emit < len(rows):
        while next_emit not in completed:
            done, _not_done = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                position = pending.pop(future)
                completed[position] = future.result()
            fill()
        yield completed.pop(next_emit)
        next_emit += 1
        fill()


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
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be in 1..32")
    if args.progress_interval < 1:
        parser.error("--progress-interval must be positive")

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

    app_config = capture_root / "configs/app.config"
    issue_config = capture_root / "configs/issue.config"

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
            "app_config": artifact(app_config),
            "issue_config": artifact(issue_config),
        },
        "execution": {
            "workers": args.workers,
            "profile_computation": "process_parallel" if args.workers > 1 else "serial",
            "pack_emission": "single_writer_dense_kernel_order",
        },
        "claim_boundary": (
            "Per-kernel HBServe memory-SASS profiles only; cache replay and hardware "
            "accuracy are not claimed by this receipt"
        ),
        "sampling_adjustment": {
            "policy": (
                "For each disjoint-holdout row, promote the smallest frozen holdout CTA "
                "into training before fitting; retain every other frozen holdout CTA as "
                "an exact independent check"
            ),
            "selected_cta_union_unchanged": True,
            "minimum_remaining_holdouts": 1,
        },
    }
    write_manifest(manifest_path, manifest)

    pack_digest = hashlib.sha256()
    index_digest = hashlib.sha256()
    executor: concurrent.futures.ProcessPoolExecutor | None = None
    try:
        worker_arguments = (
            str(profile_tool_path),
            str(capture_root),
            str(sample_plan),
            args.workload_id,
        )
        if args.workers == 1:
            initialize_profile_worker(*worker_arguments)
            generated_profiles = (build_profile_row(row) for row in rows)
        else:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=initialize_profile_worker,
                initargs=worker_arguments,
            )
            generated_profiles = ordered_parallel_profiles(
                executor,
                rows,
                args.workers,
            )
        with pack_path.open("xb") as pack_stream, index_path.open("xb") as index_stream:
            for position, built in enumerate(generated_profiles, 1):
                kernel = built["kernel"]
                result = built["result"]
                status = built["status"]
                profile_encoded = (
                    json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                profile_offset = pack_stream.tell()
                pack_stream.write(profile_encoded)
                pack_stream.flush()
                pack_digest.update(profile_encoded)
                index_row = {
                    "kernel_id": kernel,
                    "mode": built["mode"],
                    "schema": result["schema"],
                    "status": status,
                    "profile_payload_sha256": result["profile_payload_sha256"],
                    "promoted_training_ctas": built["promoted_ctas"],
                    "training_cta_count": built["training_cta_count"],
                    "holdout_cta_count": built["holdout_cta_count"],
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
                manifest["last_kernel_id"] = kernel
                if position % args.progress_interval == 0 or position == len(rows):
                    manifest["elapsed_seconds"] = time.monotonic() - started
                    write_manifest(manifest_path, manifest)
        if executor is not None:
            executor.shutdown(wait=True)
            executor = None
    except BaseException as error:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
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
