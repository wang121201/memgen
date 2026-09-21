#!/usr/bin/env python3
"""Fit and replay a sampled affine HyFiSS memory-SASS kernel profile.

This bridge is intentionally narrower than HBServe's compact address format.
It retains the SASS opcode, active-lane mask, per-lane stride encoding and CTA
identity required by HyFiSS memgen.  ``profile`` reads one captured kernel,
retains only explicitly named training CTAs, and normally scores disjoint
holdout CTAs.  A grid with too few CTAs for a disjoint holdout can instead be
retained completely and is then labeled no-extrapolation rather than sampled.
``generate`` never reads the captured ``.mem`` body: it combines the frozen
profile with the observed app/issue configuration and emits a new raw HyFiSS
kernel trace.

Only an exact integer-affine/coordinate address rule is accepted.  Missing CTA
issue entries, changing instruction structure, non-affine training points, or
any holdout address/sector mismatch fail closed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import heapq
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Iterable


PROFILE_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 1}
MEMCV3_PROFILE_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 2}
MEMCV3_XYZ_PROFILE_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 3}
MEMCV3_COORDINATE_PROFILE_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 4}
MEMCV3_COMPLETE_GRID_PROFILE_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 5}
MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 6}
MEMCV3_COORDINATE_GLOBAL_LOWERED_PROFILE_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 7}
MEMCV3_COMPLETE_GRID_GLOBAL_LOWERED_PROFILE_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 8}
MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 9}
GENERATION_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_generation", "version": 1}
COMPARISON_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_comparison", "version": 2}
AUDIT_SCHEMA = {"name": "hbserve.hyfiss_sampled_sass_semantic_audit", "version": 1}
ISSUE_TUPLE_RE = re.compile(r"\((\d+),(\d+),([0-9a-fA-Fx]+)\)")
MEMC_PART_RE = re.compile(r"^kernel_(\d+)\.memc\.part(\d+)$")
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
_CACHE_LOCK = threading.Lock()
_SHA256_CACHE: dict[tuple[str, int, int], str] = {}
_APP_CONFIG_CACHE: dict[
    tuple[str, int, int],
    dict[int, tuple[tuple[str, ...], dict[str, str]]],
] = {}
_SAMPLE_PLAN_CACHE: dict[tuple[str, int, int], dict[int, tuple[int, ...]]] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def file_cache_key(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return str(resolved), stat.st_size, stat.st_mtime_ns


def sha256_file(path: Path) -> str:
    before = file_cache_key(path)
    with _CACHE_LOCK:
        cached = _SHA256_CACHE.get(before)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    require(file_cache_key(path) == before, f"file changed while hashing: {path}")
    value = digest.hexdigest()
    with _CACHE_LOCK:
        _SHA256_CACHE[before] = value
    return value


def write_json_new(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def parse_ids(raw: str) -> list[int]:
    values = [int(token, 0) for token in raw.split(",") if token.strip()]
    require(values and len(values) == len(set(values)), "CTA list must be nonempty and unique")
    require(all(value >= 0 for value in values), "CTA IDs must be nonnegative")
    return sorted(values)


def parse_hex(token: str) -> int:
    return int(token[2:], 16) if token.lower().startswith("0x") else int(token, 16)


def parse_stride(token: str) -> tuple[int, int]:
    stride, count = token.split(":", 1)
    value = int(stride, 10)
    repetitions = int(count, 10)
    require(repetitions > 0, f"invalid stride repetition: {token}")
    return value, repetitions


def parse_memory_line(line: str, source_sequence: int) -> dict[str, Any]:
    fields = line.split()
    require(len(fields) >= 8, f"short raw memory line: {line[:160]}")
    cursor = 0
    block = parse_hex(fields[cursor]); cursor += 1
    pc = fields[cursor]; cursor += 1
    opcode = fields[cursor]; cursor += 1
    mask = fields[cursor]; cursor += 1
    timestamp = parse_hex(fields[cursor]); cursor += 1
    group_count = parse_hex(fields[cursor]); cursor += 1
    groups = []
    for _ in range(group_count):
        require(cursor + 1 < len(fields), "truncated address group")
        base = parse_hex(fields[cursor]); cursor += 1
        pair_count = parse_hex(fields[cursor]); cursor += 1
        require(cursor + pair_count <= len(fields), "truncated stride pairs")
        pairs = fields[cursor:cursor + pair_count]
        cursor += pair_count
        for pair in pairs:
            parse_stride(pair)
        groups.append({"base": base, "pairs": pairs})
    require(cursor == len(fields), f"unparsed raw memory fields: {fields[cursor:]}")
    return {
        "block": block,
        "pc": pc,
        "opcode": opcode,
        "mask": mask,
        "timestamp": timestamp,
        "groups": groups,
        "source_sequence": source_sequence,
    }


def active_lane_offsets(mask_token: str, pairs: list[str]) -> tuple[tuple[int, int], ...]:
    mask = parse_hex(mask_token)
    strides: list[int] = []
    for token in pairs:
        stride, count = parse_stride(token)
        strides.extend([stride] * count)
    require(len(strides) <= 31, "more than 31 lane strides")
    address = 0
    result = []
    for lane in range(32):
        if lane and lane - 1 < len(strides):
            address += strides[lane - 1]
        if mask & (1 << lane):
            result.append((lane, address))
    return tuple(result)


def signature(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["pc"], record["opcode"], record["mask"],
        tuple(active_lane_offsets(record["mask"], group["pairs"]) for group in record["groups"]),
    )


def address_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(group["base"] for group in record["groups"])


def indexed_records(records: list[dict[str, Any]]) -> tuple[dict[int, tuple[Any, int]], dict[tuple[Any, int], dict[str, Any]]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[signature(record)].append(record)
    by_sequence: dict[int, tuple[Any, int]] = {}
    by_slot: dict[tuple[Any, int], dict[str, Any]] = {}
    for sig, values in buckets.items():
        ordered = sorted(values, key=lambda item: (address_key(item), item["timestamp"], item["source_sequence"]))
        for rank, record in enumerate(ordered):
            slot = (sig, rank)
            by_sequence[record["source_sequence"]] = slot
            by_slot[slot] = record
    return by_sequence, by_slot


def collect_ctas(memory_path: Path, selected: set[int]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    require(memory_path.is_file(), f"missing memory trace: {memory_path}")
    found: dict[int, list[dict[str, Any]]] = {block: [] for block in selected}
    digest = hashlib.sha256()
    lines = 0
    with memory_path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            lines += 1
            first = raw.split(maxsplit=1)[0]
            block = parse_hex(first.decode("ascii"))
            if block in selected:
                found[block].append(parse_memory_line(raw.decode("ascii"), lines - 1))
    missing = sorted(block for block, records in found.items() if not records)
    require(not missing, f"selected CTAs absent from source trace: {missing}")
    return found, {
        "sha256": digest.hexdigest(), "bytes": memory_path.stat().st_size,
        "lines": lines,
    }


def read_uvar(stream: Any, *, eof: bool = False) -> int | None:
    value = 0
    for index in range(10):
        raw = stream.read(1)
        if not raw:
            if index == 0 and eof:
                return None
            raise ValueError("truncated MEMCv3 varint")
        byte = raw[0]
        require(index < 9 or byte <= 1, "MEMCv3 uint64 varint overflow")
        value |= (byte & 127) << (7 * index)
        if byte < 128:
            return value
    raise ValueError("MEMCv3 varint too long")


def read_svar(stream: Any) -> int:
    value = read_uvar(stream)
    require(value is not None, "missing MEMCv3 signed varint")
    return (value >> 1) ^ -(value & 1)


def discover_memc_files(trace_root: Path, kernel: int) -> list[Path]:
    memory_dir = trace_root / "memory_traces"
    base = memory_dir / f"kernel_{kernel}.memc"
    require(base.is_file(), f"missing MEMCv3 base file: {base}")
    parts: list[tuple[int, Path]] = []
    for path in memory_dir.glob(f"kernel_{kernel}.memc.part*"):
        match = MEMC_PART_RE.fullmatch(path.name)
        require(match is not None and int(match.group(1)) == kernel, f"invalid MEMCv3 part name: {path.name}")
        parts.append((int(match.group(2)), path))
    parts.sort()
    require([index for index, _ in parts] == list(range(1, len(parts) + 1)), "non-contiguous MEMCv3 parts")
    return [base] + [path for _, path in parts]


def memc_records(path: Path) -> Iterable[dict[str, Any]]:
    """Decode raw MEMCv3 without inferring spaces or physical ordering."""
    with path.open("rb", buffering=1024 * 1024) as stream:
        require(stream.read(9) == b"HYFMEMC1\n", "not a HyFiSS MEMC file")
        require(read_uvar(stream) == 3, "sampled profiling requires MEMCv3")
        opcode_count = read_uvar(stream)
        require(opcode_count is not None and 1 <= opcode_count <= 10000, "MEMCv3 opcode count")
        opcodes = []
        for _ in range(opcode_count):
            length = read_uvar(stream)
            require(length is not None and 1 <= length <= 4096, "MEMCv3 opcode length")
            raw = stream.read(length)
            require(len(raw) == length, "truncated MEMCv3 opcode")
            opcodes.append(raw.decode("ascii"))
        cta = pc = relative_clock = base = 0
        while (sequence := read_uvar(stream, eof=True)) is not None:
            cta = (cta + read_svar(stream)) & MASK64
            pc = (pc + read_svar(stream)) & MASK64
            opcode_id = read_uvar(stream)
            mask = read_uvar(stream)
            require(opcode_id is not None and opcode_id < len(opcodes), "MEMCv3 opcode id")
            require(mask is not None and 0 < mask <= MASK32, "MEMCv3 active mask")
            relative_clock = (relative_clock + read_svar(stream)) & MASK64
            ref_count = read_uvar(stream)
            require(ref_count in (1, 2), "MEMCv3 reference count")
            sm = read_uvar(stream)
            cta_warp = read_uvar(stream)
            function = read_uvar(stream)
            full_clock = read_uvar(stream)
            require(
                cta <= MASK32 and pc <= MASK32 and relative_clock <= MASK32
                and sm is not None and sm <= MASK32
                and cta_warp is not None and cta_warp <= MASK32
                and function is not None and 0 < function <= MASK32
                and full_clock is not None,
                "MEMCv3 identity range",
            )
            groups = []
            for _ in range(ref_count):
                tag = read_uvar(stream)
                require(tag is not None and tag <= 4, "MEMCv3 space tag")
                if tag == 4:
                    global_mask = read_uvar(stream)
                    local_mask = read_uvar(stream)
                    shared_mask = read_uvar(stream)
                    require(None not in (global_mask, local_mask, shared_mask), "truncated MEMCv3 space masks")
                else:
                    global_mask = mask if tag == 1 else 0
                    local_mask = mask if tag == 2 else 0
                    shared_mask = mask if tag == 3 else 0
                require(
                    ((global_mask | local_mask | shared_mask) & ~mask) == 0
                    and not (global_mask & local_mask)
                    and not (global_mask & shared_mask)
                    and not (local_mask & shared_mask),
                    "MEMCv3 overlapping/out-of-mask spaces",
                )
                unknown_mask = mask & ~(global_mask | local_mask | shared_mask)
                base = (base + read_svar(stream)) & MASK64
                pair_count = read_uvar(stream)
                require(pair_count is not None and 1 <= pair_count <= 31, "MEMCv3 stride-pair count")
                pairs = []
                lanes = 1
                for _ in range(pair_count):
                    stride = read_svar(stream)
                    run = read_uvar(stream)
                    require(run is not None and 1 <= run <= 31 and lanes + run <= 32, "MEMCv3 stride run")
                    pairs.append(f"{stride}:{run}")
                    lanes += run
                require(lanes == 32, "MEMCv3 stride runs do not reconstruct 32 lanes")
                groups.append({
                    "base": base,
                    "pairs": pairs,
                    "global_mask": global_mask,
                    "local_mask": local_mask,
                    "shared_mask": shared_mask,
                    "unknown_mask": unknown_mask,
                })
            yield {
                "block": cta,
                "pc": f"{pc:x}",
                "opcode": opcodes[opcode_id],
                "mask": f"{mask:x}",
                "timestamp": full_clock,
                "groups": groups,
                "source_sequence": sequence,
                "sm": sm,
                "cta_warp": cta_warp,
                "function": function,
                "relative_clock": relative_clock,
            }


def lower_memc_record_to_global(record: dict[str, Any]) -> tuple[dict[str, Any], bool, int]:
    """Lower only established MEMCv3 address-space roles to cache-visible global refs.

    Ordinary records must be wholly global.  The sole mixed-space exception is
    SM89 LDGSTS async global-to-shared copy: exactly one full-mask shared
    destination and one full-mask global source.  The shared destination is
    deliberately removed because L1/L2/DRAM replay models only global traffic.
    """
    mask = parse_hex(record["mask"])
    groups = record["groups"]
    all_global = all(
        group["global_mask"] == mask
        and group["local_mask"] == 0
        and group["shared_mask"] == 0
        and group["unknown_mask"] == 0
        for group in groups
    )
    if all_global:
        lowered = dict(record)
        lowered["groups"] = [
            {key: value for key, value in group.items() if key not in (
                "global_mask", "local_mask", "shared_mask", "unknown_mask",
            )}
            for group in groups
        ]
        return lowered, False, 0

    require(record["opcode"].startswith("LDGSTS"), "unsupported mixed-space MEMCv3 opcode")
    require(len(groups) == 2, "LDGSTS lowering requires exactly two address groups")
    global_groups = [
        group for group in groups
        if group["global_mask"] == mask
        and group["local_mask"] == 0
        and group["shared_mask"] == 0
        and group["unknown_mask"] == 0
    ]
    shared_groups = [
        group for group in groups
        if group["shared_mask"] == mask
        and group["global_mask"] == 0
        and group["local_mask"] == 0
        and group["unknown_mask"] == 0
    ]
    require(
        len(global_groups) == 1 and len(shared_groups) == 1,
        "LDGSTS lowering requires one global source and one shared destination",
    )
    lowered = dict(record)
    lowered["groups"] = [{
        key: value
        for key, value in global_groups[0].items()
        if key not in ("global_mask", "local_mask", "shared_mask", "unknown_mask")
    }]
    return lowered, True, 1


def collect_ctas_memc(
    paths: list[Path],
    selected: set[int],
    *,
    allow_empty: bool = False,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    found: dict[int, list[dict[str, Any]]] = {block: [] for block in selected}
    artifacts = []
    records = 0
    lowered_async_copy_records = 0
    discarded_shared_address_groups = 0
    previous_sequence = None
    for path in paths:
        artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for record in memc_records(path):
            if previous_sequence is not None:
                require(record["source_sequence"] == previous_sequence + 1, "MEMCv3 sequence discontinuity within kernel")
            previous_sequence = record["source_sequence"]
            records += 1
            if record["block"] not in selected:
                continue
            record, was_lowered, discarded = lower_memc_record_to_global(record)
            lowered_async_copy_records += int(was_lowered)
            discarded_shared_address_groups += discarded
            found[record["block"]].append(record)
    missing = sorted(block for block, values in found.items() if not values)
    require(
        allow_empty or not missing,
        f"selected CTAs absent from MEMCv3 source: {missing}",
    )
    return found, {
        "format": "memc_v3",
        "memory_files": artifacts,
        "bytes": sum(item["bytes"] for item in artifacts),
        "lines": records,
        "global_only_lowering": {
            "policy": "strict all-global, or established LDGSTS one-global-source plus one-shared-destination",
            "lowered_async_copy_records": lowered_async_copy_records,
            "discarded_shared_address_groups": discarded_shared_address_groups,
            "local_or_unknown_groups_accepted": 0,
        },
        "empty_selected_ctas": missing,
        "empty_selected_cta_count": len(missing),
        "empty_selected_ctas_accepted": allow_empty,
    }


def expand_lane_addresses(record: dict[str, Any], bases: Iterable[int] | None = None) -> list[int]:
    result: list[int] = []
    mask = parse_hex(record["mask"])
    base_values = list(bases) if bases is not None else [group["base"] for group in record["groups"]]
    require(len(base_values) == len(record["groups"]), "base/group count mismatch")
    for base, group in zip(base_values, record["groups"]):
        strides: list[int] = []
        for token in group["pairs"]:
            stride, count = parse_stride(token)
            strides.extend([stride] * count)
        require(len(strides) <= 31, "more than 31 lane strides")
        address = base
        for lane in range(32):
            if lane and lane - 1 < len(strides):
                address += strides[lane - 1]
            if mask & (1 << lane):
                result.append(address)
    return result


def app_kernel_lines(path: Path, kernel: int) -> tuple[list[str], dict[str, str]]:
    cache_key = file_cache_key(path)
    with _CACHE_LOCK:
        cached = _APP_CONFIG_CACHE.get(cache_key)
    if cached is None:
        parsed_lines: dict[int, list[str]] = defaultdict(list)
        parsed_values: dict[int, dict[str, str]] = defaultdict(dict)
        pattern = re.compile(r"^-kernel_(\d+)_([^\s]+)(?:\s+(.*))?$")
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                match = pattern.match(line)
                if match is None:
                    continue
                kernel_id = int(match.group(1))
                parsed_lines[kernel_id].append(line)
                parsed_values[kernel_id][match.group(2)] = (match.group(3) or "").strip()
        cached = {
            kernel_id: (tuple(parsed_lines[kernel_id]), dict(parsed_values[kernel_id]))
            for kernel_id in parsed_lines
        }
        require(file_cache_key(path) == cache_key, f"app.config changed while indexing: {path}")
        with _CACHE_LOCK:
            _APP_CONFIG_CACHE[cache_key] = cached
    lines_value, values_value = cached.get(kernel, ((), {}))
    lines = list(lines_value)
    values = dict(values_value)
    require(lines, f"kernel {kernel} absent from app.config")
    for key in ("kernel_name", "grid_size", "block_size", "llama_phase"):
        require(values.get(key), f"kernel {kernel} missing app.config field {key}")
    return lines, values


def sample_plan_ctas(path: Path, kernel: int) -> tuple[int, ...]:
    cache_key = file_cache_key(path)
    with _CACHE_LOCK:
        cached = _SAMPLE_PLAN_CACHE.get(cache_key)
    if cached is None:
        parsed: dict[int, tuple[int, ...]] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(reader.fieldnames == ["kernel_id", "cta_ids"], "unexpected sample-plan header")
            for row in reader:
                kernel_id = int(row["kernel_id"])
                require(kernel_id not in parsed, f"duplicate sample-plan kernel {kernel_id}")
                ctas = tuple(parse_ids(row["cta_ids"]))
                parsed[kernel_id] = ctas
        require(file_cache_key(path) == cache_key, f"sample plan changed while indexing: {path}")
        with _CACHE_LOCK:
            _SAMPLE_PLAN_CACHE[cache_key] = parsed
        cached = parsed
    require(kernel in cached, f"kernel {kernel} absent from sample plan")
    return cached[kernel]


def fit_affine(points: list[tuple[int, int]], label: str) -> dict[str, int]:
    require(len(points) >= 2, f"{label}: at least two training CTAs required")
    x0, y0 = points[0]
    x1, y1 = next(((x, y) for x, y in points[1:] if x != x0), (None, None))
    require(x1 is not None and y1 is not None, f"{label}: distinct CTA IDs required")
    dx = int(x1) - x0
    dy = int(y1) - y0
    require(dy % dx == 0, f"{label}: non-integral affine slope")
    slope = dy // dx
    intercept = y0 - slope * x0
    require(all(intercept + slope * x == y for x, y in points), f"{label}: non-affine training addresses")
    return {"intercept": intercept, "cta_x_stride": slope}


def block_coordinates(block: int, grid_dims: tuple[int, int, int]) -> tuple[int, int, int]:
    grid_x, grid_y, grid_z = grid_dims
    require(grid_x > 0 and grid_y > 0 and grid_z > 0, "grid dimensions must be positive")
    require(0 <= block < grid_x * grid_y * grid_z, f"CTA {block} escapes grid dimensions")
    x = block % grid_x
    quotient = block // grid_x
    y = quotient % grid_y
    z = quotient // grid_y
    return x, y, z


def fit_affine_xyz(
    points: list[tuple[int, int]],
    grid_dims: tuple[int, int, int],
    label: str,
) -> dict[str, int]:
    require(len(points) >= 2, f"{label}: at least two training CTAs required")
    coordinates = [(block_coordinates(block, grid_dims), address) for block, address in points]
    anchor_coordinates, anchor_address = coordinates[0]
    strides = [0, 0, 0]
    for axis, extent in enumerate(grid_dims):
        if extent == 1:
            continue
        candidate = next((
            (coordinate, address)
            for coordinate, address in coordinates[1:]
            if coordinate[axis] != anchor_coordinates[axis]
            and all(coordinate[other] == anchor_coordinates[other] for other in range(3) if other != axis)
        ), None)
        require(candidate is not None, f"{label}: missing axis-aligned training CTA for axis {axis}")
        coordinate, address = candidate
        delta_coordinate = coordinate[axis] - anchor_coordinates[axis]
        delta_address = address - anchor_address
        require(delta_address % delta_coordinate == 0, f"{label}: non-integral affine slope on axis {axis}")
        strides[axis] = delta_address // delta_coordinate
    intercept = anchor_address - sum(
        stride * coordinate for stride, coordinate in zip(strides, anchor_coordinates)
    )
    require(
        all(
            intercept + sum(stride * value for stride, value in zip(strides, coordinate)) == address
            for coordinate, address in coordinates
        ),
        f"{label}: non-affine training addresses in CTA xyz coordinates",
    )
    return {
        "intercept": intercept,
        "cta_x_stride": strides[0],
        "cta_y_stride": strides[1],
        "cta_z_stride": strides[2],
    }


def fit_affine_x_y_table_z(
    points: list[tuple[int, int]],
    grid_dims: tuple[int, int, int],
    label: str,
) -> dict[str, Any]:
    """Fit linear x/z plus a complete categorical y-offset table.

    This represents head/group steps without pretending that flattened CTA IDs
    are globally affine.  Every y coordinate must be present in training at
    the anchor x/z; disjoint holdouts still test unseen coordinate combinations.
    """
    require(len(points) >= 2, f"{label}: at least two training CTAs required")
    coordinates = [(block_coordinates(block, grid_dims), address) for block, address in points]
    anchor_coordinates, anchor_address = coordinates[0]
    x_stride = z_stride = 0
    for axis, extent in ((0, grid_dims[0]), (2, grid_dims[2])):
        if extent == 1:
            continue
        candidate = next((
            (coordinate, address)
            for coordinate, address in coordinates[1:]
            if coordinate[axis] != anchor_coordinates[axis]
            and all(coordinate[other] == anchor_coordinates[other] for other in range(3) if other != axis)
        ), None)
        require(candidate is not None, f"{label}: missing axis-aligned training CTA for axis {axis}")
        coordinate, address = candidate
        delta_coordinate = coordinate[axis] - anchor_coordinates[axis]
        delta_address = address - anchor_address
        require(delta_address % delta_coordinate == 0, f"{label}: non-integral affine slope on axis {axis}")
        if axis == 0:
            x_stride = delta_address // delta_coordinate
        else:
            z_stride = delta_address // delta_coordinate
    y_addresses: list[int] = []
    for y in range(grid_dims[1]):
        candidate = next((
            address
            for coordinate, address in coordinates
            if coordinate == (anchor_coordinates[0], y, anchor_coordinates[2])
        ), None)
        require(candidate is not None, f"{label}: categorical y rule requires training CTA at y={y}")
        y_addresses.append(candidate)
    y_offsets = [address - y_addresses[anchor_coordinates[1]] for address in y_addresses]
    intercept = (
        anchor_address
        - x_stride * anchor_coordinates[0]
        - z_stride * anchor_coordinates[2]
        - y_offsets[anchor_coordinates[1]]
    )
    require(
        all(
            intercept + x_stride * coordinate[0] + y_offsets[coordinate[1]] + z_stride * coordinate[2] == address
            for coordinate, address in coordinates
        ),
        f"{label}: non-separable training addresses in CTA coordinates",
    )
    return {
        "intercept": intercept,
        "cta_x_stride": x_stride,
        "cta_y_offsets": y_offsets,
        "cta_z_stride": z_stride,
    }


def fit_affine_x_y_table_z_half_partition(
    points: list[tuple[int, int]],
    grid_dims: tuple[int, int, int],
    label: str,
) -> dict[str, Any]:
    """Fit x-linear/y-categorical plus one geometry-fixed z-half step.

    CUTLASS split-K kernels observed here partition the z grid into two equal
    groups.  The partition is fixed from grid geometry, not searched from
    holdout addresses.  Training must independently identify x, lower-half z,
    upper-half z and every y category; all supplied training points must fit.
    """
    require(grid_dims[2] >= 4, f"{label}: z-half partition requires z extent >= 4")
    coordinates = [(block_coordinates(block, grid_dims), address) for block, address in points]
    anchor_coordinates, anchor_address = coordinates[0]
    require(anchor_coordinates[2] == 0, f"{label}: z-half partition requires a z=0 anchor")
    partition = (grid_dims[2] + 1) // 2

    x_stride = 0
    if grid_dims[0] > 1:
        candidate = next((
            (coordinate, address)
            for coordinate, address in coordinates[1:]
            if coordinate[0] != anchor_coordinates[0]
            and coordinate[1] == anchor_coordinates[1]
            and coordinate[2] == anchor_coordinates[2]
        ), None)
        require(candidate is not None, f"{label}: missing axis-aligned x training CTA")
        coordinate, address = candidate
        delta = coordinate[0] - anchor_coordinates[0]
        require((address - anchor_address) % delta == 0, f"{label}: non-integral x stride")
        x_stride = (address - anchor_address) // delta

    lower = next((
        (coordinate, address)
        for coordinate, address in coordinates[1:]
        if 0 < coordinate[2] < partition
        and coordinate[0] == anchor_coordinates[0]
        and coordinate[1] == anchor_coordinates[1]
    ), None)
    require(lower is not None, f"{label}: missing lower-half z training CTA")
    lower_coordinate, lower_address = lower
    require(
        (lower_address - anchor_address) % lower_coordinate[2] == 0,
        f"{label}: non-integral lower-half z stride",
    )
    z_stride = (lower_address - anchor_address) // lower_coordinate[2]

    upper = next((
        (coordinate, address)
        for coordinate, address in coordinates
        if coordinate[2] >= partition
        and coordinate[0] == anchor_coordinates[0]
        and coordinate[1] == anchor_coordinates[1]
    ), None)
    require(upper is not None, f"{label}: missing upper-half z training CTA")
    upper_coordinate, upper_address = upper
    z_partition_stride = upper_address - anchor_address - z_stride * upper_coordinate[2]

    y_addresses: list[int] = []
    for y in range(grid_dims[1]):
        candidate = next((
            address
            for coordinate, address in coordinates
            if coordinate == (anchor_coordinates[0], y, anchor_coordinates[2])
        ), None)
        require(candidate is not None, f"{label}: categorical y rule requires training CTA at y={y}")
        y_addresses.append(candidate)
    y_offsets = [address - y_addresses[anchor_coordinates[1]] for address in y_addresses]
    intercept = anchor_address - x_stride * anchor_coordinates[0] - y_offsets[anchor_coordinates[1]]

    def predict(coordinate: tuple[int, int, int]) -> int:
        return (
            intercept
            + x_stride * coordinate[0]
            + y_offsets[coordinate[1]]
            + z_stride * coordinate[2]
            + z_partition_stride * int(coordinate[2] >= partition)
        )

    require(all(predict(coordinate) == address for coordinate, address in coordinates),
            f"{label}: training addresses do not fit geometry-fixed z-half partition")
    return {
        "kind": "coordinate_z_half_partition",
        "intercept": intercept,
        "cta_x_stride": x_stride,
        "cta_y_offsets": y_offsets,
        "cta_z_stride": z_stride,
        "cta_z_partition": partition,
        "cta_z_partition_stride": z_partition_stride,
    }


def fit_coordinate_rule(
    points: list[tuple[int, int]],
    grid_dims: tuple[int, int, int],
    label: str,
) -> dict[str, Any]:
    if len(points) == 1:
        require(
            grid_dims == (1, 1, 1),
            f"{label}: one training CTA is valid only for a one-CTA complete grid",
        )
        return {
            "intercept": points[0][1],
            "cta_x_stride": 0,
            "cta_y_stride": 0,
            "cta_z_stride": 0,
        }
    try:
        return fit_affine_xyz(points, grid_dims, label)
    except ValueError as affine_error:
        try:
            return fit_affine_x_y_table_z(points, grid_dims, label)
        except ValueError as table_error:
            try:
                return fit_affine_x_y_table_z_half_partition(points, grid_dims, label)
            except ValueError as partition_error:
                raise ValueError(
                    f"{affine_error}; categorical-y fallback failed: {table_error}; "
                    f"geometry-fixed z-half fallback failed: {partition_error}"
                ) from partition_error


def predict_bases(
    entry: dict[str, Any],
    block: int,
    grid_dims: tuple[int, int, int] | None = None,
) -> list[int]:
    result = []
    coordinates = None
    for rule in entry["address_rules"]:
        if rule.get("kind") == "exact_cta_base_table":
            key = str(block)
            require(key in rule["bases_by_cta"], f"CTA {block} absent from exact address table")
            result.append(int(rule["bases_by_cta"][key]))
            continue
        if "cta_y_stride" in rule or "cta_y_offsets" in rule or "cta_z_stride" in rule:
            require(grid_dims is not None, "CTA xyz address rule requires grid dimensions")
            if coordinates is None:
                coordinates = block_coordinates(block, grid_dims)
            result.append(
                rule["intercept"]
                + rule["cta_x_stride"] * coordinates[0]
                + (
                    rule["cta_y_offsets"][coordinates[1]]
                    if "cta_y_offsets" in rule else
                    rule["cta_y_stride"] * coordinates[1]
                )
                + rule["cta_z_stride"] * coordinates[2]
                + (
                    rule["cta_z_partition_stride"]
                    if "cta_z_partition" in rule
                    and coordinates[2] >= rule["cta_z_partition"] else
                    0
                )
            )
            continue
        result.append(rule["intercept"] + rule["cta_x_stride"] * block)
    return result


def structure_sha256(slots: Iterable[tuple[Any, int]]) -> str:
    """Hash the complete ranked-slot multiset used to define one CTA class."""
    payload = []
    for sig, rank in sorted(slots):
        pc, opcode, mask, group_offsets = sig
        payload.append({
            "pc": pc,
            "opcode": opcode,
            "mask": mask,
            "group_lane_offsets": [list(offsets) for offsets in group_offsets],
            "signature_rank": rank,
        })
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def fit_profile_rule(
    points: list[tuple[int, int]],
    grid_dims: tuple[int, int, int],
    label: str,
    *,
    allow_exact_table: bool,
) -> dict[str, Any]:
    try:
        return fit_coordinate_rule(points, grid_dims, label)
    except ValueError:
        if not allow_exact_table:
            raise
        return {
            "kind": "exact_cta_base_table",
            "bases_by_cta": {str(block): address for block, address in points},
            "scope": "complete_grid_structural_class_only",
        }


def build_template(
    sampled: dict[int, list[dict[str, Any]]],
    indexed: dict[int, tuple[dict[int, tuple[Any, int]], dict[tuple[Any, int], dict[str, Any]]]],
    ctas: list[int],
    grid_dims: tuple[int, int, int],
    *,
    coordinate_rules: bool,
    allow_exact_table: bool,
) -> list[dict[str, Any]]:
    require(ctas, "empty structural class")
    anchor = ctas[0]
    if not sampled[anchor]:
        require(
            all(not sampled[block] for block in ctas),
            "empty structural class contains a nonempty CTA",
        )
        return []
    anchor_by_sequence, _anchor_by_slot = indexed[anchor]
    minimum_by_cta = {
        block: min(record["timestamp"] for record in sampled[block])
        for block in ctas
    }
    template = []
    for ordinal, record in enumerate(sorted(sampled[anchor], key=lambda item: item["source_sequence"])):
        slot = anchor_by_sequence[record["source_sequence"]]
        _sig, rank = slot
        rules = []
        for group_index in range(len(record["groups"])):
            points = [
                (block, indexed[block][1][slot]["groups"][group_index]["base"])
                for block in ctas
            ]
            if coordinate_rules:
                rules.append(fit_profile_rule(
                    points,
                    grid_dims,
                    f"slot {ordinal} group {group_index}",
                    allow_exact_table=allow_exact_table,
                ))
            else:
                rules.append(fit_affine(points, f"slot {ordinal} group {group_index}"))
        entry = {
            "ordinal": ordinal,
            "pc": record["pc"], "opcode": record["opcode"], "mask": record["mask"],
            "groups": [{"pairs": group["pairs"]} for group in record["groups"]],
            "signature_rank": rank,
            "address_rules": rules,
            "sampled_timestamp_delta": record["timestamp"] - minimum_by_cta[anchor],
        }
        if allow_exact_table:
            entry["sampled_timestamp_delta_by_cta"] = {
                str(block): indexed[block][1][slot]["timestamp"] - minimum_by_cta[block]
                for block in ctas
            }
        template.append(entry)
    return template


def profile(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    trace_root = Path(args.trace_root).resolve()
    app_path = trace_root / "configs" / "app.config"
    issue_path = trace_root / "configs" / "issue.config"
    memory_path = trace_root / "memory_traces" / f"kernel_{args.kernel}.mem"
    train = parse_ids(args.train_ctas)
    complete_grid = bool(args.complete_grid)
    if complete_grid:
        require(not args.holdout_ctas, "complete-grid profile cannot also declare holdout CTAs")
        holdout: list[int] = []
    else:
        require(args.holdout_ctas, "disjoint-holdout profile requires --holdout-ctas")
        holdout = parse_ids(args.holdout_ctas)
    require(not set(train) & set(holdout), "training and holdout CTAs overlap")
    sample_plan_arg = getattr(args, "sample_plan", None)
    sample_plan_path = Path(sample_plan_arg).resolve() if sample_plan_arg else None
    if complete_grid:
        require(sample_plan_path is not None, "complete-grid profile requires --sample-plan provenance")
    if sample_plan_path is not None:
        selected_by_plan = sample_plan_ctas(sample_plan_path, args.kernel)
        require(
            selected_by_plan == tuple(sorted(train + holdout)),
            "profile CTA selection differs from frozen capture sample plan",
        )
    app_lines, metadata = app_kernel_lines(app_path, args.kernel)
    grid_size = int(metadata["grid_size"])
    grid_dims = tuple(int(metadata[key]) for key in ("grid_dim_x", "grid_dim_y", "grid_dim_z"))
    require(len(grid_dims) == 3 and grid_dims[0] * grid_dims[1] * grid_dims[2] == grid_size, "app.config grid dimensions disagree with grid size")
    require(all(block < grid_size for block in train + holdout), "sample CTA escapes kernel grid")
    if complete_grid:
        require(train == list(range(grid_size)), "complete-grid training CTAs must exactly cover 0..grid_size-1")
    if args.input_format == "raw":
        require(not complete_grid, "complete-grid profile currently requires MEMCv3 input")
        sampled, source = collect_ctas(memory_path, set(train + holdout))
        profile_schema = PROFILE_SCHEMA
        source_paths = {"memory_path": str(memory_path)}
    else:
        memc_paths = discover_memc_files(trace_root, args.kernel)
        sampled, source = collect_ctas_memc(
            memc_paths,
            set(train + holdout),
            allow_empty=complete_grid,
        )
        globally_lowered = source["global_only_lowering"]["lowered_async_copy_records"] > 0
        profile_schema = (
            MEMCV3_COMPLETE_GRID_GLOBAL_LOWERED_PROFILE_SCHEMA
            if complete_grid and globally_lowered else
            MEMCV3_COMPLETE_GRID_PROFILE_SCHEMA
            if complete_grid else
            MEMCV3_COORDINATE_GLOBAL_LOWERED_PROFILE_SCHEMA
            if globally_lowered else
            MEMCV3_COORDINATE_PROFILE_SCHEMA
        )
        memory_path = memc_paths[0]
        source_paths = {
            "memory_path": str(memory_path),
            "memory_paths": [str(path) for path in memc_paths],
        }
    indexed = {block: indexed_records(records) for block, records in sampled.items()}
    structural_groups: dict[frozenset[tuple[Any, int]], list[int]] = defaultdict(list)
    for block, (_by_sequence, slots) in indexed.items():
        structural_groups[frozenset(slots)].append(block)
    if len(structural_groups) > 1:
        require(
            complete_grid,
            "sampled/disjoint-holdout profile found heterogeneous CTA structures; "
            "use an explicitly complete grid rather than extrapolating a class selector",
        )
        profile_schema = (
            MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA
            if profile_schema in (
                MEMCV3_COORDINATE_GLOBAL_LOWERED_PROFILE_SCHEMA,
                MEMCV3_COMPLETE_GRID_GLOBAL_LOWERED_PROFILE_SCHEMA,
            ) else
            MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA
        )

    template: list[dict[str, Any]] = []
    structural_classes: list[dict[str, Any]] = []
    cta_class_by_id: list[str] = []
    if profile_schema in (
        MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
        MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
    ):
        cta_class_by_id = [""] * grid_size
        ordered_groups = sorted(structural_groups.items(), key=lambda item: min(item[1]))
        for class_index, (slots, class_ctas_value) in enumerate(ordered_groups):
            class_ctas = sorted(class_ctas_value)
            class_id = f"class-{class_index:02d}"
            class_template = build_template(
                sampled,
                indexed,
                class_ctas,
                grid_dims,
                coordinate_rules=True,
                allow_exact_table=True,
            )
            for block in class_ctas:
                require(not cta_class_by_id[block], f"CTA {block} belongs to multiple structural classes")
                cta_class_by_id[block] = class_id
            structural_classes.append({
                "class_id": class_id,
                "ctas": class_ctas,
                "anchor_cta": class_ctas[0],
                "structure_sha256": structure_sha256(slots),
                "template": class_template,
                "empty_no_memory_activity": not class_template,
            })
        require(all(cta_class_by_id), "structural class map does not cover the complete grid")
    else:
        anchor_slots = set(indexed[train[0]][1])
        for block, (_by_sequence, slots) in indexed.items():
            require(set(slots) == anchor_slots, f"CTA {block} instruction/address-rank structure differs")
        template = build_template(
            sampled,
            indexed,
            train,
            grid_dims,
            coordinate_rules=profile_schema in (
                MEMCV3_COORDINATE_PROFILE_SCHEMA,
                MEMCV3_COMPLETE_GRID_PROFILE_SCHEMA,
                MEMCV3_COORDINATE_GLOBAL_LOWERED_PROFILE_SCHEMA,
                MEMCV3_COMPLETE_GRID_GLOBAL_LOWERED_PROFILE_SCHEMA,
            ),
            allow_exact_table=False,
        )

    address_groups = lane_addresses = exact_address_groups = exact_lane_addresses = 0
    actual_sectors: set[int] = set()
    predicted_sectors: set[int] = set()
    for block in holdout:
        slots = indexed[block][1]
        for entry in template:
            template_record = {
                "pc": entry["pc"], "opcode": entry["opcode"], "mask": entry["mask"],
                "groups": entry["groups"],
            }
            sig = signature(template_record)
            actual = slots[(sig, entry["signature_rank"])]
            predicted = predict_bases(entry, block, grid_dims)
            for expected, observed in zip(predicted, [group["base"] for group in actual["groups"]]):
                address_groups += 1
                exact_address_groups += int(expected == observed)
            actual_lanes = expand_lane_addresses(actual)
            predicted_lanes = expand_lane_addresses(template_record, predicted)
            require(len(actual_lanes) == len(predicted_lanes), "lane expansion changed")
            lane_addresses += len(actual_lanes)
            exact_lane_addresses += sum(a == b for a, b in zip(actual_lanes, predicted_lanes))
            actual_sectors.update(address // 32 for address in actual_lanes)
            predicted_sectors.update(address // 32 for address in predicted_lanes)
    if complete_grid:
        holdout_result = {
            "ctas": 0,
            "instructions": 0,
            "address_groups": 0,
            "exact_address_groups": 0,
            "address_group_exact_fraction": None,
            "lane_addresses": 0,
            "exact_lane_addresses": 0,
            "lane_address_exact_fraction": None,
            "sector_set_jaccard": None,
            "not_applicable_reason": (
                "every CTA is retained as training data; this profile performs no CTA extrapolation"
            ),
        }
    else:
        union = actual_sectors | predicted_sectors
        intersection = actual_sectors & predicted_sectors
        holdout_result = {
            "ctas": len(holdout), "instructions": len(holdout) * len(template),
            "address_groups": address_groups, "exact_address_groups": exact_address_groups,
            "address_group_exact_fraction": exact_address_groups / address_groups,
            "lane_addresses": lane_addresses, "exact_lane_addresses": exact_lane_addresses,
            "lane_address_exact_fraction": exact_lane_addresses / lane_addresses,
            "sector_set_jaccard": len(intersection) / len(union) if union else 1.0,
        }
        require(exact_address_groups == address_groups, "holdout address bases are not exact")
        require(exact_lane_addresses == lane_addresses, "holdout lane addresses are not exact")
        require(actual_sectors == predicted_sectors, "holdout sector set differs")

    result = {
        "schema": profile_schema,
        "status": (
            "PASS_COMPLETE_GRID_STRUCTURAL_CLASSES_NO_CTA_EXTRAPOLATION"
            if profile_schema in (
                MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
                MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
            ) else
            "PASS_COMPLETE_GRID_NO_CTA_EXTRAPOLATION"
            if complete_grid else
            "PASS_EXACT_COORDINATE_RULE_TRAIN_AND_DISJOINT_HOLDOUT"
            if profile_schema in (
                MEMCV3_COORDINATE_PROFILE_SCHEMA,
                MEMCV3_COORDINATE_GLOBAL_LOWERED_PROFILE_SCHEMA,
            ) else
            "PASS_EXACT_AFFINE_TRAIN_AND_DISJOINT_HOLDOUT"
        ),
        "workload": args.workload_id,
        "kernel": {
            "id": args.kernel, "name": metadata["kernel_name"],
            "phase": metadata["llama_phase"], "grid_size": grid_size,
            "grid_dims": list(grid_dims),
            "block_size": int(metadata["block_size"]),
            "app_config_lines": app_lines,
        },
        "source": {
            "trace_root": str(trace_root), **source_paths, **source,
            "app_config": str(app_path), "app_config_sha256": sha256_file(app_path),
            "issue_config": str(issue_path), "issue_config_sha256": sha256_file(issue_path),
            "sample_plan": str(sample_plan_path) if sample_plan_path is not None else None,
            "sample_plan_sha256": sha256_file(sample_plan_path) if sample_plan_path is not None else None,
            "read_scope": "full kernel byte/hash scan; only named CTA records retained",
        },
        "sampling": {
            "training_ctas": train, "holdout_ctas": holdout,
            "training_fraction_of_grid": len(train) / grid_size,
            "holdout_fraction_of_grid": len(holdout) / grid_size,
            "captured_instruction_lines_per_cta": (
                len(template)
                if profile_schema not in (
                    MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
                    MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
                ) else
                None
            ),
            "captured_instruction_lines_by_structural_class": (
                {
                    item["class_id"]: len(item["template"])
                    for item in structural_classes
                }
                if structural_classes else
                None
            ),
            "validation_mode": (
                "complete_grid_exact_cta_structure_map_no_extrapolation"
                if profile_schema in (
                    MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
                    MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
                ) else
                "complete_grid_no_cta_extrapolation"
                if complete_grid else
                "disjoint_cta_holdout"
            ),
        },
        "model": {
            "address_rule": (
                "per-structural-class exact coordinate rule, with exact per-CTA base tables only where the complete grid cannot be fit"
                if profile_schema in (
                    MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
                    MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
                ) else
                "exact integer affine in CTA (x,y,z) per structural-signature rank and address group"
                if profile_schema in (
                    MEMCV3_COORDINATE_PROFILE_SCHEMA,
                    MEMCV3_COMPLETE_GRID_PROFILE_SCHEMA,
                    MEMCV3_COORDINATE_GLOBAL_LOWERED_PROFILE_SCHEMA,
                    MEMCV3_COMPLETE_GRID_GLOBAL_LOWERED_PROFILE_SCHEMA,
                ) else
                "exact integer affine in flattened CTA ID per structural-signature rank and address group"
            ),
            "generated_order": "observed issue.config CTA start plus sampled intra-CTA timestamp delta",
            "retained_fields": ["pc", "opcode", "active_lane_mask", "lane_stride_runs", "cta_id"],
            "not_modeled": ["production cross-warp issue order", "production per-request timestamp", "warp_id"],
            "source_format": args.input_format,
            "memcv3_lowering": (
                "cache-visible global groups only; ordinary records must be wholly global; "
                "established SM89 LDGSTS dual-reference records retain their full-mask global "
                "source and discard only the full-mask shared destination; local, unknown, and "
                "other mixed-space records fail closed"
            ),
            "structure_selection": (
                "exact_complete_grid_cta_id_map"
                if profile_schema in (
                    MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
                    MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
                ) else
                "single_structure_for_every_cta"
            ),
        },
        "holdout": holdout_result,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if profile_schema in (
        MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
        MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
    ):
        result["structural_classes"] = structural_classes
        result["cta_class_by_id"] = cta_class_by_id
    else:
        result["template"] = template
    result["profile_payload_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    write_json_new(Path(args.output).resolve(), result)
    return result


def parse_issue_config(path: Path, kernel: int) -> tuple[int, dict[int, tuple[int, int]]]:
    sm_count = None
    mapping: dict[int, tuple[int, int]] = {}
    prefix = "-trace_issued_sm_id_"
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("-trace_issued_sms_num "):
                sm_count = int(line.split()[1])
                continue
            if not line.startswith(prefix):
                continue
            key = line.split(None, 1)[0]
            sm = int(key[len(prefix):])
            for match in ISSUE_TUPLE_RE.finditer(line):
                kid, block, start = int(match.group(1)), int(match.group(2)), parse_hex(match.group(3))
                if kid != kernel:
                    continue
                require(block not in mapping, f"duplicate issue mapping for CTA {block}")
                mapping[block] = (sm, start)
    require(sm_count is not None and sm_count > 0, "issue.config has no SM count")
    return sm_count, mapping


def format_generated_line(
    block: int,
    timestamp: int,
    entry: dict[str, Any],
    grid_dims: tuple[int, int, int] | None = None,
) -> str:
    fields = [format(block, "x"), entry["pc"], entry["opcode"], entry["mask"], format(timestamp, "x"), format(len(entry["groups"]), "x")]
    bases = predict_bases(entry, block, grid_dims)
    for base, group in zip(bases, entry["groups"]):
        require(base >= 0, f"negative generated address for CTA {block}")
        pairs = group["pairs"]
        fields.extend([f"0x{base:x}", format(len(pairs), "x"), *pairs])
    return " ".join(fields) + " \n"


def generate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    profile_path = Path(args.profile).resolve()
    value = json.loads(profile_path.read_text(encoding="utf-8"))
    require(
        value.get("schema") in (
            PROFILE_SCHEMA,
            MEMCV3_PROFILE_SCHEMA,
            MEMCV3_XYZ_PROFILE_SCHEMA,
            MEMCV3_COORDINATE_PROFILE_SCHEMA,
            MEMCV3_COMPLETE_GRID_PROFILE_SCHEMA,
            MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
            MEMCV3_COORDINATE_GLOBAL_LOWERED_PROFILE_SCHEMA,
            MEMCV3_COMPLETE_GRID_GLOBAL_LOWERED_PROFILE_SCHEMA,
            MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
        ),
        "unsupported sampled profile schema",
    )
    require(
        value.get("status") in (
            "PASS_EXACT_AFFINE_TRAIN_AND_DISJOINT_HOLDOUT",
            "PASS_EXACT_COORDINATE_RULE_TRAIN_AND_DISJOINT_HOLDOUT",
            "PASS_COMPLETE_GRID_NO_CTA_EXTRAPOLATION",
            "PASS_COMPLETE_GRID_STRUCTURAL_CLASSES_NO_CTA_EXTRAPOLATION",
        ),
        "profile did not pass holdout",
    )
    source = value["source"]
    app_path = Path(args.app_config or source["app_config"]).resolve()
    issue_path = Path(args.issue_config or source["issue_config"]).resolve()
    require(sha256_file(app_path) == source["app_config_sha256"], "app.config identity changed")
    require(sha256_file(issue_path) == source["issue_config_sha256"], "issue.config identity changed")
    source_kernel = int(value["kernel"]["id"])
    kernel = int(args.target_kernel) if args.target_kernel is not None else source_kernel
    source_grid_size = int(value["kernel"]["grid_size"])
    source_grid_dims_value = value["kernel"].get("grid_dims")
    source_grid_dims = (
        tuple(int(item) for item in source_grid_dims_value)
        if source_grid_dims_value is not None else None
    )
    if value.get("schema") in (
        MEMCV3_XYZ_PROFILE_SCHEMA,
        MEMCV3_COORDINATE_PROFILE_SCHEMA,
        MEMCV3_COMPLETE_GRID_PROFILE_SCHEMA,
        MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
        MEMCV3_COORDINATE_GLOBAL_LOWERED_PROFILE_SCHEMA,
        MEMCV3_COMPLETE_GRID_GLOBAL_LOWERED_PROFILE_SCHEMA,
        MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
    ):
        require(source_grid_dims is not None and len(source_grid_dims) == 3, "v3 profile has no grid dimensions")
    if kernel == source_kernel:
        target_app_lines = value["kernel"]["app_config_lines"]
        _current_lines, target_metadata = app_kernel_lines(app_path, kernel)
        require(target_metadata["kernel_name"] == value["kernel"]["name"], "source kernel name changed")
        require(int(target_metadata["grid_size"]) == source_grid_size, "source grid size changed")
        require(int(target_metadata["block_size"]) == int(value["kernel"]["block_size"]), "source block size changed")
    else:
        target_app_lines, target_metadata = app_kernel_lines(app_path, kernel)
        require(target_metadata["kernel_name"] == value["kernel"]["name"], "retarget kernel name differs")
        require(int(target_metadata["grid_size"]) == source_grid_size, "retarget grid size differs")
        require(int(target_metadata["block_size"]) == int(value["kernel"]["block_size"]), "retarget block size differs")
        source_fields = {
            line.split(None, 1)[0].split(f"-kernel_{source_kernel}_", 1)[1]: line.split(None, 1)[1]
            for line in value["kernel"]["app_config_lines"]
            if line.startswith(f"-kernel_{source_kernel}_") and " " in line
        }
        for field in ("num_registers", "shared_mem_bytes", "grid_dim_x", "grid_dim_y", "grid_dim_z", "tb_dim_x", "tb_dim_y", "tb_dim_z"):
            require(source_fields.get(field) == target_metadata.get(field), f"retarget structural field differs: {field}")
    grid_size = int(target_metadata["grid_size"])
    target_grid_dims = tuple(int(target_metadata[key]) for key in ("grid_dim_x", "grid_dim_y", "grid_dim_z"))
    require(target_grid_dims[0] * target_grid_dims[1] * target_grid_dims[2] == grid_size, "target grid dimensions disagree with grid size")
    if source_grid_dims is not None:
        require(target_grid_dims == source_grid_dims, "retarget grid dimensions differ")
    sm_count, mapping = parse_issue_config(issue_path, kernel)
    require(set(mapping) == set(range(grid_size)), f"issue mapping does not exactly cover 0..{grid_size - 1}")
    stream_stdout = bool(getattr(args, "stdout", False))
    output_root = None if stream_stdout else Path(args.output_root).resolve()
    if output_root is not None:
        require(not output_root.exists(), f"refusing to replace output root: {output_root}")
        config_dir = output_root / "configs"
        memory_dir = output_root / "memory_traces"
        config_dir.mkdir(parents=True)
        memory_dir.mkdir()
        (config_dir / "app.config").write_text(
            "####################################################################################\n\n" +
            "\n".join(target_app_lines) + "\n", encoding="utf-8",
        )
    by_sm: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for block, (sm, cta_start) in mapping.items():
        require(0 <= sm < sm_count, f"CTA {block} maps outside SM range")
        by_sm[sm].append((block, cta_start))
    if output_root is not None:
        with (config_dir / "issue.config").open("w", encoding="utf-8") as handle:
            handle.write("####################################################################################\n\n")
            handle.write(f"-trace_issued_sms_num {sm_count}\n")
            for sm in range(sm_count):
                rows = sorted(by_sm.get(sm, []), key=lambda item: (item[1], item[0]))
                tuples = ",".join(f"({kernel},{block},{start:x})" for block, start in rows)
                handle.write(f"-trace_issued_sm_id_{sm} {len(rows)}")
                if tuples:
                    handle.write("," + tuples)
                handle.write("\n")

    template_by_block: dict[int, list[dict[str, Any]]] = {}
    if value.get("schema") in (
        MEMCV3_STRUCTURAL_CLASS_PROFILE_SCHEMA,
        MEMCV3_STRUCTURAL_CLASS_GLOBAL_LOWERED_PROFILE_SCHEMA,
    ):
        raw_classes = value.get("structural_classes")
        raw_map = value.get("cta_class_by_id")
        require(isinstance(raw_classes, list) and raw_classes, "missing structural classes")
        require(isinstance(raw_map, list) and len(raw_map) == grid_size, "invalid CTA structural class map")
        classes: dict[str, dict[str, Any]] = {}
        assigned: set[int] = set()
        for item in raw_classes:
            class_id = item.get("class_id")
            ctas = item.get("ctas")
            class_template = item.get("template")
            require(isinstance(class_id, str) and class_id and class_id not in classes, "invalid/duplicate structural class id")
            require(isinstance(ctas, list) and ctas, f"structural class {class_id} has no CTAs")
            require(isinstance(class_template, list), f"structural class {class_id} has no template list")
            classes[class_id] = item
            for raw_block in ctas:
                block = int(raw_block)
                require(0 <= block < grid_size and block not in assigned, "structural class CTA overlap/range failure")
                require(raw_map[block] == class_id, f"CTA {block} class map disagrees with class membership")
                assigned.add(block)
                template_by_block[block] = sorted(
                    class_template,
                    key=lambda entry: (
                        int(entry.get("sampled_timestamp_delta_by_cta", {}).get(str(block), entry["sampled_timestamp_delta"])),
                        entry["ordinal"],
                    ),
                )
        require(assigned == set(range(grid_size)), "structural classes do not exactly cover target grid")
        require(set(raw_map) == set(classes), "CTA class map names do not match structural classes")
    else:
        template = sorted(value["template"], key=lambda item: (item["sampled_timestamp_delta"], item["ordinal"]))
        require(template, "empty sampled template")
        template_by_block = {block: template for block in range(grid_size)}

    def timestamp_delta(entry: dict[str, Any], block: int) -> int:
        table = entry.get("sampled_timestamp_delta_by_cta")
        if table is None:
            return int(entry["sampled_timestamp_delta"])
        key = str(block)
        require(key in table, f"CTA {block} absent from exact timestamp table")
        return int(table[key])

    heap: list[tuple[int, int, int, int]] = []
    for block, (sm, start) in mapping.items():
        block_template = template_by_block[block]
        if not block_template:
            continue
        heap.append((start + timestamp_delta(block_template[0], block), sm, block, 0))
    heapq.heapify(heap)
    memory_path = None if output_root is None else memory_dir / f"kernel_{kernel}.mem"
    digest = hashlib.sha256()
    lines = byte_count = 0
    handle = sys.stdout.buffer if stream_stdout else memory_path.open("xb")
    try:
        while heap:
            _absolute_time, sm, block, ordinal = heapq.heappop(heap)
            block_template = template_by_block[block]
            entry = block_template[ordinal]
            encoded = format_generated_line(
                block,
                timestamp_delta(entry, block),
                entry,
                target_grid_dims,
            ).encode("ascii")
            handle.write(encoded)
            digest.update(encoded)
            lines += 1
            byte_count += len(encoded)
            next_ordinal = ordinal + 1
            if next_ordinal < len(block_template):
                start = mapping[block][1]
                next_time = start + timestamp_delta(block_template[next_ordinal], block)
                heapq.heappush(heap, (next_time, sm, block, next_ordinal))
        handle.flush()
    finally:
        if not stream_stdout:
            handle.close()
    expected_lines = sum(len(template_by_block[block]) for block in range(grid_size))
    require(lines == expected_lines, "generated instruction count differs from per-CTA templates")
    manifest = {
        "schema": GENERATION_SCHEMA,
        "status": (
            "PASS_STREAMED_WITHOUT_SOURCE_MEM_BODY"
            if stream_stdout else
            "PASS_GENERATED_WITHOUT_SOURCE_MEM_BODY"
        ),
        "profile": str(profile_path), "profile_sha256": sha256_file(profile_path),
        "output_root": str(output_root) if output_root is not None else None,
        "source_kernel": source_kernel,
        "kernel": kernel, "target_phase": target_metadata["llama_phase"],
        "retargeted": kernel != source_kernel, "grid_size": grid_size,
        "sm_count": sm_count, "issue_entries": len(mapping),
        "generated_memory_path": str(memory_path) if memory_path is not None else None,
        "generated_memory_sha256": digest.hexdigest(),
        "generated_memory_bytes": byte_count, "generated_instruction_lines": lines,
        "expected_generated_instruction_lines": expected_lines,
        "structural_class_count": len(value.get("structural_classes", [])) or 1,
        "ordering": "global merge of issue.config CTA start plus sampled intra-CTA timestamp delta",
        "source_mem_body_opened": False,
        "target_mem_body_opened": False,
        "streamed_to_stdout": stream_stdout,
        "claim_boundary": "sampled affine memory-SASS-equivalent lane trace; not a production issue-order reconstruction",
        "elapsed_seconds": time.perf_counter() - started,
    }
    if output_root is not None:
        write_json_new(output_root / "generation.json", manifest)
    return manifest


def csv_row(path: Path, kernel: int) -> dict[str, float]:
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if int(raw["kernel_id"]) == kernel:
                return {key: float(value) for key, value in raw.items() if key != "kernel_id"} | {"kernel_id": float(kernel)}
    raise ValueError(f"kernel {kernel} absent from {path}")


def memgen_metrics(row: dict[str, float]) -> dict[str, float | None]:
    def ratio(numerator: float, denominator: float) -> float | None:
        return None if denominator == 0 else numerator / denominator
    return {
        "mem_insts": row["mem_insts"], "lane_accesses": row["lane_accesses"],
        "sector_requests": row["sector_requests"],
        "l1_requests": row["l1_requests"],
        "l1_lower_traffic_avoidance": ratio(row["l1_hits"] + row["l1_pending_hits"], row["l1_requests"]),
        "l2_read_requests": row["l2_read_requests"],
        "l2_read_lower_traffic_avoidance": ratio(row["l2_read_hits"] + row["l2_read_pending_hits"], row["l2_read_requests"]),
        "dram_read_sectors": row["dram_load_sectors"], "dram_write_sectors": row["dram_store_sectors"],
    }


def relative(predicted: float | None, observed: float | None) -> float | None:
    if predicted is None or observed in (None, 0):
        return None
    return (predicted - observed) / observed


def metric_errors(predicted: dict[str, float | None], observed: dict[str, float | None]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("mem_insts", "lane_accesses", "sector_requests", "l1_requests", "l2_read_requests", "dram_read_sectors", "dram_write_sectors"):
        result[f"{metric}_relative_error"] = relative(predicted[metric], observed[metric])
    for metric in ("l1_lower_traffic_avoidance", "l2_read_lower_traffic_avoidance"):
        lhs, rhs = predicted[metric], observed[metric]
        result[f"{metric}_percentage_point_error"] = None if lhs is None or rhs is None else 100 * (lhs - rhs)
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    profile_value = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    require(profile_value.get("schema") == PROFILE_SCHEMA, "unsupported profile")
    kernel = int(profile_value["kernel"]["id"])
    paths = {
        "generated_cold": Path(args.generated_summary).resolve(),
        "captured_cold": Path(args.captured_summary).resolve(),
        "captured_continuous": Path(args.continuous_summary).resolve(),
    }
    require(bool(args.generated_sorted_summary) == bool(args.captured_sorted_summary),
            "provide both generated and captured timestamp-sorted summaries")
    if args.generated_sorted_summary:
        paths.update({
            "generated_cold_timestamp_sorted": Path(args.generated_sorted_summary).resolve(),
            "captured_cold_timestamp_sorted": Path(args.captured_sorted_summary).resolve(),
        })
    rows = {name: memgen_metrics(csv_row(path, kernel)) for name, path in paths.items()}
    spec = importlib.util.spec_from_file_location("ncu_alignment", Path(args.alignment_analyzer).resolve())
    require(spec is not None and spec.loader is not None, "cannot load NCU alignment analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ncu = None
    with module.NcuRows(Path(args.ncu_csv).resolve()) as source:
        for row in source:
            if int(row["kernel_id"]) == kernel:
                ncu = row
                break
    require(ncu is not None, f"kernel {kernel} absent from NCU report")
    ncu_metrics = {
        "l1_load_requests": ncu["l1_load_requests"],
        "l1_load_hit_rate": None if ncu["l1_load_requests"] == 0 else ncu["l1_load_hits"] / ncu["l1_load_requests"],
        "l2_read_requests": ncu["l2_read_requests"],
        "l2_read_hit_rate": None if ncu["l2_read_hits"] + ncu["l2_read_misses"] == 0 else ncu["l2_read_hits"] / (ncu["l2_read_hits"] + ncu["l2_read_misses"]),
        "dram_read_sectors": ncu["dram_read_sectors"], "dram_write_sectors": ncu["dram_write_sectors"],
        "gpu_time_ms_diagnostic": ncu["gpu_time_ms"],
    }
    cold_errors = metric_errors(rows["generated_cold"], rows["captured_cold"])
    sorted_errors = None
    ordering_effect = None
    if args.generated_sorted_summary:
        sorted_errors = metric_errors(
            rows["generated_cold_timestamp_sorted"],
            rows["captured_cold_timestamp_sorted"],
        )
        ordering_effect = {
            "captured_no_sort_vs_timestamp_sort": metric_errors(
                rows["captured_cold"], rows["captured_cold_timestamp_sorted"]
            ),
            "generated_no_sort_vs_timestamp_sort": metric_errors(
                rows["generated_cold"], rows["generated_cold_timestamp_sorted"]
            ),
        }
    continuous_vs_ncu = {
        "l1_request_relative_error": relative(rows["captured_continuous"]["l1_requests"], ncu_metrics["l1_load_requests"]),
        "l1_rate_percentage_point_error": 100 * (rows["captured_continuous"]["l1_lower_traffic_avoidance"] - ncu_metrics["l1_load_hit_rate"]),
        "l2_read_request_relative_error": relative(rows["captured_continuous"]["l2_read_requests"], ncu_metrics["l2_read_requests"]),
        "l2_read_rate_percentage_point_error": 100 * (rows["captured_continuous"]["l2_read_lower_traffic_avoidance"] - ncu_metrics["l2_read_hit_rate"]),
        "dram_read_relative_error": relative(rows["captured_continuous"]["dram_read_sectors"], ncu_metrics["dram_read_sectors"]),
        "dram_write_relative_error": relative(rows["captured_continuous"]["dram_write_sectors"], ncu_metrics["dram_write_sectors"]),
    }
    result = {
        "schema": COMPARISON_SCHEMA,
        "status": "COMPLETE_SAMPLED_SASS_COLD_REPLAY_AND_CONTINUOUS_HARDWARE_CONTEXT",
        "kernel": profile_value["kernel"], "holdout": profile_value["holdout"],
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()} | {
            "profile": {"path": str(Path(args.profile).resolve()), "sha256": sha256_file(Path(args.profile).resolve())},
            "ncu_csv": {"path": str(Path(args.ncu_csv).resolve()), "sha256": sha256_file(Path(args.ncu_csv).resolve())},
        },
        "memgen": rows, "ncu": ncu_metrics,
        "generated_vs_captured_same_cold_start": cold_errors,
        "generated_vs_captured_same_cold_start_timestamp_sorted": sorted_errors,
        "ordering_effect": ordering_effect,
        "captured_continuous_vs_ncu": continuous_vs_ncu,
        "claim_boundary": {
            "generated_vs_captured": "same isolated cold cache and same memgen; measures sampled trace generation plus modeled order",
            "timestamp_sorted_localization": "both traces ordered by issue.config CTA start plus their trace timestamps; isolates address/template timing from capture-file interleaving",
            "captured_continuous_vs_ncu": "full-workload cache history; NCU is hardware-observed but replay timing is diagnostic",
            "generated_vs_ncu_not_scored": "isolated cold cache does not share the continuous hardware cache state",
        },
    }
    write_json_new(Path(args.output).resolve(), result)
    return result


def semantic_event_bytes(record: dict[str, Any]) -> bytes:
    fields = [record["pc"], record["opcode"], record["mask"]]
    for group in record["groups"]:
        fields.append(f"base={group['base']:x}")
        fields.extend(f"lane={lane}:offset={offset}" for lane, offset in active_lane_offsets(record["mask"], group["pairs"]))
        fields.append("group-end")
    return "|".join(fields).encode("ascii")


def semantic_accumulate(record: dict[str, Any], by_block: dict[int, list[int]]) -> int:
    modulus = 1 << 128
    event = semantic_event_bytes(record)
    first = int.from_bytes(hashlib.blake2b(b"sum\0" + event, digest_size=16).digest(), "big")
    second = int.from_bytes(hashlib.blake2b(b"xor\0" + event, digest_size=16).digest(), "big")
    state = by_block.setdefault(record["block"], [0, 0, 0])
    state[0] += 1
    state[1] = (state[1] + first) % modulus
    state[2] ^= second
    return sum(len(active_lane_offsets(record["mask"], group["pairs"])) for group in record["groups"])


def semantic_finish(by_block: dict[int, list[int]]) -> str:
    final = hashlib.sha256()
    for block, (count, summed, xored) in sorted(by_block.items()):
        final.update(block.to_bytes(8, "little"))
        final.update(count.to_bytes(8, "little"))
        final.update(summed.to_bytes(16, "big"))
        final.update(xored.to_bytes(16, "big"))
    return final.hexdigest()


def semantic_census(path: Path) -> dict[str, Any]:
    # Per-CTA sum and xor of independent event digests form a deterministic,
    # order-independent multiset certificate.  Count is retained separately.
    by_block: dict[int, list[int]] = {}
    file_digest = hashlib.sha256()
    lines = lanes = 0
    with path.open("rb") as handle:
        for raw in handle:
            file_digest.update(raw)
            record = parse_memory_line(raw.decode("ascii"), lines)
            lanes += semantic_accumulate(record, by_block)
            lines += 1
    return {
        "path": str(path), "file_sha256": file_digest.hexdigest(),
        "bytes": path.stat().st_size, "instruction_lines": lines,
        "active_lane_addresses": lanes, "ctas": len(by_block),
        "cta_semantic_multiset_sha256": semantic_finish(by_block),
    }


def semantic_census_memc(path: Path, kernel: int) -> dict[str, Any]:
    trace_root = path.parent.parent
    paths = discover_memc_files(trace_root, kernel)
    by_block: dict[int, list[int]] = {}
    lanes = records = 0
    files = []
    previous_sequence = None
    for source in paths:
        files.append({"path": str(source), "bytes": source.stat().st_size, "sha256": sha256_file(source)})
        for record in memc_records(source):
            if previous_sequence is not None:
                require(record["source_sequence"] == previous_sequence + 1, "MEMCv3 sequence discontinuity in audit")
            previous_sequence = record["source_sequence"]
            record, _was_lowered, _discarded = lower_memc_record_to_global(record)
            lanes += semantic_accumulate(record, by_block)
            records += 1
    return {
        "path": str(path),
        "files": files,
        "file_sha256": hashlib.sha256(canonical_json(files)).hexdigest(),
        "bytes": sum(item["bytes"] for item in files),
        "instruction_lines": records,
        "active_lane_addresses": lanes,
        "ctas": len(by_block),
        "cta_semantic_multiset_sha256": semantic_finish(by_block),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    captured_path = Path(args.captured).resolve()
    generated_path = Path(args.generated).resolve()
    require(captured_path.is_file() and generated_path.is_file(), "missing audit input")
    captured = semantic_census_memc(captured_path, args.kernel) if args.captured_format == "memc" else semantic_census(captured_path)
    generated = semantic_census(generated_path)
    compared = ("instruction_lines", "active_lane_addresses", "ctas", "cta_semantic_multiset_sha256")
    equality = {field: captured[field] == generated[field] for field in compared}
    require(all(equality.values()), f"full semantic census differs: {equality}")
    result = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS_FULL_CTA_ACTIVE_LANE_SEMANTIC_MULTISET_EQUAL",
        "kernel": args.kernel, "captured": captured, "generated": generated,
        "equality": equality,
        "digest_contract": {
            "included": ["CTA ID", "PC", "opcode", "active-lane mask", "address-group identity", "every active-lane address"],
            "excluded": ["file order", "timestamp", "inactive-lane compression strides"],
            "method": "per-CTA count plus modulo-2^128 sum and xor of domain-separated BLAKE2b-128 event digests; SHA-256 over CTA-ordered certificates",
        },
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_new(Path(args.output).resolve(), result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("profile")
    fit.add_argument("--trace-root", required=True)
    fit.add_argument("--kernel", type=int, required=True)
    fit.add_argument("--input-format", choices=("raw", "memc"), default="raw")
    fit.add_argument("--train-ctas", required=True)
    fit.add_argument("--holdout-ctas")
    fit.add_argument("--complete-grid", action="store_true")
    fit.add_argument("--sample-plan", help="frozen tracer CTA plan; required for complete-grid admission")
    fit.add_argument("--workload-id", required=True)
    fit.add_argument("--output", required=True)
    emit = commands.add_parser("generate")
    emit.add_argument("--profile", required=True)
    emit.add_argument("--app-config")
    emit.add_argument("--issue-config")
    emit.add_argument("--target-kernel", type=int)
    emit_destination = emit.add_mutually_exclusive_group(required=True)
    emit_destination.add_argument("--output-root")
    emit_destination.add_argument(
        "--stdout",
        action="store_true",
        help="stream raw generated SASS-memory lines to stdout without materializing a trace",
    )
    score = commands.add_parser("compare")
    score.add_argument("--profile", required=True)
    score.add_argument("--generated-summary", required=True)
    score.add_argument("--captured-summary", required=True)
    score.add_argument("--continuous-summary", required=True)
    score.add_argument("--generated-sorted-summary")
    score.add_argument("--captured-sorted-summary")
    score.add_argument("--ncu-csv", required=True)
    score.add_argument("--alignment-analyzer", required=True)
    score.add_argument("--output", required=True)
    verify = commands.add_parser("audit")
    verify.add_argument("--captured", required=True)
    verify.add_argument("--captured-format", choices=("raw", "memc"), default="raw")
    verify.add_argument("--generated", required=True)
    verify.add_argument("--kernel", type=int, required=True)
    verify.add_argument("--output", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "profile":
        result = profile(args)
    elif args.command == "generate":
        result = generate(args)
    elif args.command == "compare":
        result = compare(args)
    else:
        result = audit(args)
    summary_stream = sys.stderr if args.command == "generate" and args.stdout else sys.stdout
    print(
        json.dumps({key: result[key] for key in ("status",) if key in result}, sort_keys=True),
        file=summary_stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
