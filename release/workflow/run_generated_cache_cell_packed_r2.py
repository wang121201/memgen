#!/usr/bin/env python3
"""Replay one complete HBServe-generated workload through one cache model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


PROFILE_PASS = "PASS_ALL_KERNEL_PROFILES_ADMITTED_NOT_CACHE_REPLAYED"
MEMGEN_COLUMNS = (
    "mem_insts", "lane_accesses", "sector_requests", "read_sector_requests",
    "write_sector_requests", "atomic_sector_requests", "l1_requests", "l1_hits",
    "l1_pending_hits", "l1_misses", "l1_line_misses", "l1_sector_misses",
    "l2_requests", "l2_hits", "l2_pending_hits", "l2_misses",
    "l2_line_misses", "l2_sector_misses", "dram_requests",
    "dram_load_requests", "dram_store_requests", "dram_load_sectors",
    "dram_store_sectors", "dram_load_bytes", "dram_store_bytes",
    "l2_writeback_events", "l2_writeback_dirty_sectors",
    "l2_dirty_drain_events", "l2_dirty_drain_sectors", "reads", "writes",
    "atomics", "write_full_sector_requests", "write_partial_sector_requests",
    "write_covered_bytes", "l2_read_requests", "l2_read_hits",
    "l2_read_pending_hits", "l2_read_misses", "l2_read_line_misses",
    "l2_read_sector_misses", "l2_write_requests", "l2_write_hits",
    "l2_write_pending_hits", "l2_write_misses", "l2_write_line_misses",
    "l2_write_sector_misses", "l2_atomic_requests", "l2_atomic_hits",
    "l2_atomic_pending_hits", "l2_atomic_misses", "l2_atomic_line_misses",
    "l2_atomic_sector_misses",
)


def now() -> str:
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_digest(path: Path, expected: str, label: str) -> dict[str, Any]:
    value = artifact(path)
    if value["sha256"] != expected:
        raise RuntimeError(f"{label} SHA-256 differs: {value['sha256']}")
    return value


def validate_profile_root(profile_root: Path, workload_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = profile_root / "manifest.json"
    index_path = profile_root / "profiles.index.jsonl"
    manifest = load_json(manifest_path)
    if manifest.get("status") != PROFILE_PASS:
        raise RuntimeError(f"profile manifest is not admitted: {manifest.get('status')}")
    if manifest.get("workload_id") != workload_id:
        raise RuntimeError("profile workload identity differs")
    expected_index = manifest.get("profile_index", {})
    if expected_index.get("sha256") != sha256(index_path):
        raise RuntimeError("profile index SHA-256 differs from its manifest")
    rows: list[dict[str, Any]] = []
    with index_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if not rows or [int(row["kernel_id"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError("profile index is not dense ordered 1..N")
    if len(rows) != int(manifest.get("kernel_count", -1)):
        raise RuntimeError("profile index and manifest kernel counts differ")
    expected_pack = manifest.get("profile_pack")
    if expected_pack is not None:
        pack_path = Path(expected_pack["path"]).resolve(strict=True)
        if expected_pack.get("sha256") != sha256(pack_path):
            raise RuntimeError("profile pack SHA-256 differs from its manifest")
        if int(expected_pack.get("records", -1)) != len(rows):
            raise RuntimeError("profile pack record count differs")
    for row in rows:
        path = Path(row["path"]).resolve(strict=True)
        if not str(row.get("status", "")).startswith("PASS_"):
            raise RuntimeError(f"kernel {row['kernel_id']} profile is not PASS")
        digest = str(row.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"kernel {row['kernel_id']} has invalid profile SHA-256")
        if "offset" in row:
            offset, size = int(row["offset"]), int(row["bytes"])
            if offset < 0 or size <= 0 or offset + size > path.stat().st_size:
                raise RuntimeError(f"kernel {row['kernel_id']} packed byte range differs")
        elif int(row.get("bytes", path.stat().st_size)) != path.stat().st_size:
            raise RuntimeError(f"kernel {row['kernel_id']} profile size differs")
    return manifest, rows


def phase_map(app_config: Path, kernel_count: int) -> dict[int, str]:
    pattern = re.compile(r"^-kernel_([0-9]+)_llama_phase\s+(\S+)\s*$")
    result: dict[int, str] = {}
    with app_config.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = pattern.match(line)
            if match:
                result[int(match.group(1))] = match.group(2)
    if sorted(result) != list(range(1, kernel_count + 1)):
        raise RuntimeError("app.config phase labels do not cover every kernel")
    return result


def make_full_address_objects(path: Path) -> None:
    # Compact requests store address[47:32] as object_index and address[31:0]
    # as object_offset.  This dense, non-overlapping table reconstructs every
    # canonical 48-bit device address without inventing allocation ownership.
    with path.open("x", encoding="ascii", buffering=1 << 20) as stream:
        extent = 1 << 32
        for index in range(1 << 16):
            stream.write(f"{index}\t{index * extent}\t{extent}\n")


def clean_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key != "root_sudo" and not key.startswith("HYFISS_")
    }


def memgen_totals(path: Path, phases: dict[int, str]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    totals = {key: 0 for key in MEMGEN_COLUMNS}
    phase_totals: dict[str, dict[str, int]] = defaultdict(lambda: {key: 0 for key in MEMGEN_COLUMNS})
    kernels: list[int] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(key not in reader.fieldnames for key in MEMGEN_COLUMNS):
            raise RuntimeError("Memgen kernel summary lacks required columns")
        for row in reader:
            kernel = int(row["kernel_id"])
            kernels.append(kernel)
            for key in MEMGEN_COLUMNS:
                value = int(row[key])
                totals[key] += value
                phase_totals[phases[kernel]][key] += value
    if kernels != list(range(1, len(phases) + 1)):
        raise RuntimeError("Memgen kernel summary is not dense ordered 1..N")
    return totals, dict(phase_totals)


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("memgen", "naive"), required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--stream-engine", type=Path, required=True)
    parser.add_argument("--stream-engine-sha256", required=True)
    parser.add_argument("--naive-engine", type=Path)
    parser.add_argument("--naive-engine-sha256")
    parser.add_argument("--hw-config", type=Path, required=True)
    parser.add_argument("--hw-config-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    profile_root = args.profile_root.resolve(strict=True)
    capture_root = args.capture_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to reuse output root: {output_root}")
    if args.arm == "naive" and (args.naive_engine is None or args.naive_engine_sha256 is None):
        raise SystemExit("naive arm requires its engine and SHA-256")

    profile_manifest, profile_rows = validate_profile_root(profile_root, args.workload_id)
    index_path = profile_root / "profiles.index.jsonl"
    app_config = capture_root / "configs/app.config"
    issue_config = capture_root / "configs/issue.config"
    for key, path in (("app_config", app_config), ("issue_config", issue_config)):
        expected = profile_manifest["inputs"][key]
        if expected["sha256"] != sha256(path):
            raise RuntimeError(f"capture {key} differs from profile manifest")
    phases = phase_map(app_config, len(profile_rows))

    inputs = {
        "profile_manifest": artifact(profile_root / "manifest.json"),
        "profile_index": artifact(index_path),
        "app_config": artifact(app_config),
        "issue_config": artifact(issue_config),
        "stream_engine": require_digest(
            args.stream_engine, args.stream_engine_sha256, "stream engine"),
        "hw_config": require_digest(args.hw_config, args.hw_config_sha256, "hardware config"),
    }
    if args.arm == "naive":
        inputs["naive_engine"] = require_digest(
            args.naive_engine, args.naive_engine_sha256, "naive engine")

    output_root.mkdir(parents=True, exist_ok=False)
    source_stats_path = output_root / "source-stats.json"
    manifest_path = output_root / "finish.json"
    manifest: dict[str, Any] = {
        "schema": "hbserve_generated_full_inference_cache_cell_v1",
        "status": "STARTED",
        "arm": args.arm,
        "workload_id": args.workload_id,
        "kernel_count": len(profile_rows),
        "inputs": inputs,
        "started_utc": now(),
        "cache_contract": (
            {
                "architecture": "per-SM L1 plus shared L2",
                "l1_size_bytes_per_sm": 32768,
                "l2_size_bytes": 41943040,
                "l1_reset_at_kernel_boundary": True,
                "l2_preserved_across_full_inference": True,
                "write_policy": "write-back",
                "terminal_dirty_drain": False,
            }
            if args.arm == "memgen" else
            {
                "architecture": "one shared named cache; no independent L1/L2",
                "capacity_bytes": 41943040,
                "line_bytes": 128,
                "sector_bytes": 32,
                "associativity": 16,
                "set_index": "linear",
                "write_policy": "write-back",
                "write_allocate": True,
                "write_miss_fetch": True,
                "terminal_dirty_drain": False,
            }
        ),
        "claim_boundary": (
            "Full-inference address stream dynamically generated from HBServe sampled profiles, "
            "then replayed through the named cache model; not a captured full SASS trace or hardware result"
        ),
    }
    write_manifest(manifest_path, manifest)

    base_command = [
        str(args.stream_engine.resolve()), "--mode", args.arm if args.arm == "memgen" else "compact",
        "--profile-index", str(index_path), "--app-config", str(app_config),
        "--issue-config", str(issue_config), "--hw-config", str(args.hw_config.resolve()),
        "--stats", str(source_stats_path),
    ]
    env = clean_environment()
    try:
        if args.arm == "memgen":
            model_output = output_root / "model"
            command = base_command + ["--output-dir", str(model_output)]
            manifest["commands"] = {"stream_and_cache": command}
            write_manifest(manifest_path, manifest)
            with (output_root / "stdout.log").open("xb") as stdout, \
                 (output_root / "stderr.log").open("xb") as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"Memgen stream replay exited {result.returncode}")
            source = load_json(source_stats_path)
            if source.get("status") != "PASS" or source.get("mode") != "memgen":
                raise RuntimeError("Memgen source receipt did not pass")
            summary_path = model_output / "kernel_summary.csv"
            totals, by_phase = memgen_totals(summary_path, phases)
            if totals["mem_insts"] != int(source["generated_memory_instructions"]):
                raise RuntimeError("Memgen/source memory-instruction totals differ")
            if totals["lane_accesses"] != int(source["generated_lane_addresses"]):
                raise RuntimeError("Memgen/source lane-address totals differ")
            run_summary = (model_output / "run_summary.txt").read_text(encoding="utf-8")
            required_summary = (
                "num_sms=48", "sector_size=32", "l1_size_bytes=32768",
                "l2_size_bytes=41943040", "preserve_l2=1", "preserve_l1=0",
                "l1_store_policy=bypass", "write_sector_policy=line-miss-only",
                "dram_store_policy=writeback", "l2_dirty_drain=0",
            )
            if any(token not in run_summary for token in required_summary):
                raise RuntimeError("Memgen runtime cache contract differs")
            metrics = {
                "l1_hit_rate": ratio(totals["l1_hits"], totals["sector_requests"]),
                "l1_hit_denominator_source_sectors": totals["sector_requests"],
                "l2_hit_rate": ratio(totals["l2_hits"], totals["l2_requests"]),
                "l2_hit_denominator_requests": totals["l2_requests"],
                "dram_read_bytes": totals["dram_load_bytes"],
                "dram_write_bytes": totals["dram_store_bytes"],
                "totals": totals,
                "by_phase": by_phase,
            }
            outputs = {
                "source_stats": artifact(source_stats_path),
                "kernel_summary": artifact(summary_path),
                "run_summary": artifact(model_output / "run_summary.txt"),
            }
            status = "PASS_HBSERVE_GENERATED_FULL_INFERENCE_MEMGEN_REPLAY"
        else:
            objects_path = output_root / "objects.tsv"
            make_full_address_objects(objects_path)
            cache_stats_path = output_root / "cache-stats.json"
            cache_command = [
                str(args.naive_engine.resolve()), "--objects", str(objects_path),
                "--stats", str(cache_stats_path), "--capacity-bytes", "41943040",
                "--line-bytes", "128", "--sector-bytes", "32",
                "--associativity", "16", "--write-policy", "write-back",
                "--write-allocate", "true", "--write-miss-fetch", "true",
                "--final-drain", "false",
            ]
            manifest["commands"] = {"stream": base_command, "cache": cache_command}
            write_manifest(manifest_path, manifest)
            with (output_root / "stream.stderr.log").open("xb") as stream_stderr, \
                 (output_root / "cache.stderr.log").open("xb") as cache_stderr:
                cache = subprocess.Popen(
                    cache_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=cache_stderr, env=env)
                assert cache.stdin is not None
                generator = subprocess.Popen(
                    base_command, stdout=cache.stdin, stderr=stream_stderr, env=env)
                cache.stdin.close()
                generator_code = generator.wait()
                cache_code = cache.wait()
            if generator_code != 0 or cache_code != 0:
                raise RuntimeError(
                    f"naive pipeline exited generator={generator_code}, cache={cache_code}")
            source = load_json(source_stats_path)
            cache_result = load_json(cache_stats_path)
            if source.get("status") != "PASS" or source.get("mode") != "compact":
                raise RuntimeError("compact source receipt did not pass")
            if cache_result.get("status") != "PASS":
                raise RuntimeError("naive cache receipt did not pass")
            counts = cache_result["counts"]
            cache_stats = cache_result["cache_stats"]
            if int(counts["input_requests"]) != int(source["sector_requests"]):
                raise RuntimeError("naive/source sector-request totals differ")
            if int(counts["input_r_bytes"]) != 32 * int(source["read_sector_requests"]):
                raise RuntimeError("naive/source read-sector totals differ")
            if int(counts["input_w_bytes"]) != 32 * int(source["write_sector_requests"]):
                raise RuntimeError("naive/source write-sector totals differ")
            hits = int(cache_stats["read_hits"]) + int(cache_stats["write_hits"])
            metrics = {
                "l1_hit_rate": None,
                "l1_reason": "single-cache architecture has no independent L1",
                "l2_hit_rate": None,
                "l2_reason": "single-cache denominator is unfiltered source sectors, unlike NCU L2",
                "single_cache_hit_rate": ratio(hits, int(counts["input_requests"])),
                "single_cache_hit_denominator": int(counts["input_requests"]),
                "dram_read_bytes": int(counts["output_r_bytes"]),
                "dram_write_bytes": int(counts["output_w_bytes"]),
                "cache_statistics": cache_stats,
                "counts": counts,
                "final_state": cache_result["final_state"],
            }
            outputs = {
                "source_stats": artifact(source_stats_path),
                "cache_stats": artifact(cache_stats_path),
                "objects": artifact(objects_path),
            }
            status = "PASS_HBSERVE_GENERATED_FULL_INFERENCE_NAIVE_REPLAY"

        if int(source["kernel_count"]) != len(profile_rows):
            raise RuntimeError("source/profile kernel counts differ")
        if int(source["materialized_raw_sass_bytes"]) != 0:
            raise RuntimeError("stream bridge unexpectedly materialized raw SASS")
        manifest.update({
            "status": status,
            "finished_utc": now(),
            "elapsed_seconds": time.monotonic() - started,
            "source_semantic_digest": [source["semantic_digest_a"], source["semantic_digest_b"]],
            "source_generated_memory_instructions": source["generated_memory_instructions"],
            "source_generated_lane_addresses": source["generated_lane_addresses"],
            "materialized_raw_sass_bytes": 0,
            "metrics": metrics,
            "outputs": outputs,
        })
        write_manifest(manifest_path, manifest)
        print(json.dumps({
            "status": status, "workload_id": args.workload_id, "arm": args.arm,
            "elapsed_seconds": manifest["elapsed_seconds"], "output": str(output_root),
        }, sort_keys=True))
        return 0
    except BaseException as error:
        manifest.update({
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "finished_utc": now(),
            "elapsed_seconds": time.monotonic() - started,
        })
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    sys.exit(main())
