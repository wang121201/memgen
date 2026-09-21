#!/usr/bin/env python3
"""Audit whether a full MEMCv3 capture covers target kernel structures.

This is a metadata gate only.  A matching launch structure permits a later
independent audit of CTA activity/structural classes; it does not establish
address equality or authorize reuse of absolute addresses.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any


APP_RE = re.compile(r"^-kernel_([0-9]+)_([^ ]+)\s+(.*)$")
FIELDS = (
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
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def parse_app(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            match = APP_RE.fullmatch(raw.rstrip("\n"))
            if match is None:
                continue
            kernel, field, value = int(match.group(1)), match.group(2), match.group(3)
            need(field not in rows.setdefault(kernel, {}), f"duplicate app field {kernel}:{field}")
            rows[kernel][field] = value
    need(sorted(rows) == list(range(1, len(rows) + 1)), "app kernel IDs are not dense")
    for kernel, row in rows.items():
        need(set(FIELDS).issubset(row), f"kernel {kernel} lacks structural fields")
    return rows


def structural_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in FIELDS)


def key_payload(key: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(FIELDS, key))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, action="append", required=True)
    parser.add_argument("--target", action="append", nargs=2, metavar=("LABEL", "CAPTURE_ROOT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    need(not output.exists(), f"refusing to overwrite {output}")
    by_key: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    oracles = []
    for oracle_index, raw_root in enumerate(args.oracle_root):
        oracle_root = raw_root.resolve(strict=True)
        oracle_app_path = oracle_root / "configs/app.config"
        oracle_receipt_path = oracle_root / "configs/capture_receipt.json"
        oracle_receipt = json.loads(oracle_receipt_path.read_text(encoding="utf-8"))
        need(oracle_receipt.get("status") == "PASS", "oracle capture receipt did not pass")
        oracle = parse_app(oracle_app_path)
        for kernel, row in oracle.items():
            by_key[structural_key(row)].append((oracle_index, kernel))
        oracles.append({
            "root": str(oracle_root),
            "kernel_count": len(oracle),
            "unique_structures": len({structural_key(row) for row in oracle.values()}),
            "inputs": {"app_config": artifact(oracle_app_path), "capture_receipt": artifact(oracle_receipt_path)},
        })

    targets = []
    overall_unmatched = 0
    for label, raw_root in args.target:
        root = Path(raw_root).resolve(strict=True)
        app_path = root / "configs/app.config"
        receipt_path = root / "configs/capture_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        need(receipt.get("status") == "PASS", f"{label} target capture receipt did not pass")
        target = parse_app(app_path)
        key_counts = Counter(structural_key(row) for row in target.values())
        unmatched = []
        matched_kernels = 0
        for key, count in key_counts.items():
            candidates = by_key.get(key, [])
            if candidates:
                matched_kernels += count
            else:
                unmatched.append({"kernel_count": count, "structure": key_payload(key)})
        overall_unmatched += len(target) - matched_kernels
        targets.append({
            "label": label,
            "kernel_count": len(target),
            "unique_structures": len(key_counts),
            "matched_kernels": matched_kernels,
            "matched_fraction": matched_kernels / len(target),
            "unmatched_kernels": len(target) - matched_kernels,
            "unmatched_unique_structures": len(unmatched),
            "unmatched_structures": unmatched,
            "inputs": {"app_config": artifact(app_path), "capture_receipt": artifact(receipt_path)},
        })

    value = {
        "schema": "hyfiss_structural_oracle_coverage_v1",
        "status": "PASS_ALL_TARGET_STRUCTURES_COVERED_NOT_ADDRESS_OR_ACTIVITY_EQUIVALENCE" if overall_unmatched == 0 else "COMPLETE_WITH_UNMATCHED_STRUCTURES",
        "definition": {
            "structural_key_fields": list(FIELDS),
            "match": "exact equality of kernel name, registers, shared memory, grid dimensions, block dimensions, grid size, and block size",
            "permitted_next_step": "read only matching oracle kernels to test stability of CTA activity and instruction-structure classes",
            "not_proven": ["absolute address equality", "CTA activity equality", "memory instruction equality", "cache accuracy"],
        },
        "oracles": oracles,
        "union_unique_structures": len(by_key),
        "targets": targets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": value["status"],
        "targets": [{"label": row["label"], "matched_fraction": row["matched_fraction"], "unmatched_unique_structures": row["unmatched_unique_structures"]} for row in targets],
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
