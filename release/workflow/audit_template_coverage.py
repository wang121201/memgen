#!/usr/bin/env python3
"""Audit structural template coverage for the fixed-D32 prefill matrix.

This is deliberately a fail-closed *structural* audit.  A matching row means
that an existing captured launch has the same kernel symbol and CUDA launch
resource geometry.  It does not by itself prove that the sampled memory-SASS
profile transfers across workload inputs or that target CTA-to-SM placement is
known.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "hbserve_prefill_template_structural_coverage_v1"
SIGNATURE_FIELDS = (
    "name",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "registers",
    "static_shared",
    "dynamic_shared",
    "shared_executed_raw",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(SIGNATURE_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing signature fields {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no launch rows")
    return rows


def signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in SIGNATURE_FIELDS)


def split_phases(rows: list[dict[str, str]], decode_steps: int, kernels_per_decode: int) -> dict[str, list[dict[str, str]]]:
    decode_total = decode_steps * kernels_per_decode
    if len(rows) <= decode_total:
        raise ValueError(f"launch count {len(rows)} does not contain a non-empty prefill plus {decode_total} decode launches")
    prefill_count = len(rows) - decode_total
    result = {"prefill": rows[:prefill_count]}
    for step in range(decode_steps):
        begin = prefill_count + step * kernels_per_decode
        result[f"decode_{step}"] = rows[begin : begin + kernels_per_decode]
    return result


def sequence_equal(left: Iterable[dict[str, str]], right: Iterable[dict[str, str]]) -> bool:
    return [signature(row) for row in left] == [signature(row) for row in right]


def top_unmatched(rows: list[dict[str, str]], source_signatures: set[tuple[str, ...]], limit: int = 12) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter(signature(row) for row in rows if signature(row) not in source_signatures)
    result = []
    for sig, count in counts.most_common(limit):
        values = dict(zip(SIGNATURE_FIELDS, sig))
        result.append({"launches": count, **values})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-p512", type=Path, required=True)
    parser.add_argument("--source-p1024", type=Path, required=True)
    parser.add_argument("--target", action="append", nargs=2, metavar=("LABEL", "TSV"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--kernels-per-decode", type=int, default=566)
    args = parser.parse_args()

    source_p512 = read_rows(args.source_p512.resolve())
    source_p1024 = read_rows(args.source_p1024.resolve())
    source_rows = source_p512 + source_p1024
    source_signatures = {signature(row) for row in source_rows}
    source_by_signature: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for label, rows in (("P512D1", source_p512), ("P1024D32", source_p1024)):
        for ordinal, row in enumerate(rows, 1):
            source_by_signature[signature(row)].append(f"{label}:kernel_ordinal_{ordinal}")

    p512_phases = split_phases(source_p512, 1, args.kernels_per_decode)
    p1024_phases = split_phases(source_p1024, args.decode_steps, args.kernels_per_decode)

    targets = []
    seen_labels: set[str] = set()
    for label, raw_path in args.target:
        if label in seen_labels:
            raise ValueError(f"duplicate target label {label}")
        seen_labels.add(label)
        path = Path(raw_path).resolve()
        rows = read_rows(path)
        phases = split_phases(rows, args.decode_steps, args.kernels_per_decode)
        covered = [row for row in rows if signature(row) in source_signatures]
        unique = {signature(row) for row in rows}
        unique_covered = unique.intersection(source_signatures)
        phase_rows = []
        for phase_name, phase in phases.items():
            phase_covered = sum(signature(row) in source_signatures for row in phase)
            ordinal_matches: dict[str, bool | None] = {
                "P512D1_same_phase": None,
                "P1024D32_same_phase": None,
            }
            if phase_name in p512_phases:
                ordinal_matches["P512D1_same_phase"] = sequence_equal(phase, p512_phases[phase_name])
            if phase_name in p1024_phases:
                ordinal_matches["P1024D32_same_phase"] = sequence_equal(phase, p1024_phases[phase_name])
            phase_rows.append({
                "phase": phase_name,
                "launches": len(phase),
                "structurally_covered_launches": phase_covered,
                "structural_coverage_fraction": phase_covered / len(phase),
                "ordinal_sequence_matches": ordinal_matches,
                "top_unmatched_signatures": top_unmatched(phase, source_signatures),
            })
        targets.append({
            "label": label,
            "path": str(path),
            "sha256": sha256_file(path),
            "launches": len(rows),
            "prefill_launches": len(phases["prefill"]),
            "decode_launches": len(rows) - len(phases["prefill"]),
            "structurally_covered_launches": len(covered),
            "structural_coverage_fraction": len(covered) / len(rows),
            "unique_signatures": len(unique),
            "structurally_covered_unique_signatures": len(unique_covered),
            "phases": phase_rows,
        })

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_STRUCTURAL_COVERAGE_NOT_GENERATION_ADMISSION",
        "definition": {
            "match": "same kernel symbol, grid dimensions, block dimensions, register count, static/dynamic/executed shared memory",
            "scope": "candidate source-template coverage only",
            "excluded_claims": [
                "sampled memory-SASS semantic transfer",
                "target app.config or issue.config availability",
                "target CTA-to-SM placement",
                "generated full-inference trace completion",
                "cache or hardware accuracy",
            ],
            "signature_fields": list(SIGNATURE_FIELDS),
            "decode_steps": args.decode_steps,
            "kernels_per_decode": args.kernels_per_decode,
        },
        "sources": [
            {"label": "P512D1", "path": str(args.source_p512.resolve()), "sha256": sha256_file(args.source_p512.resolve()), "launches": len(source_p512)},
            {"label": "P1024D32", "path": str(args.source_p1024.resolve()), "sha256": sha256_file(args.source_p1024.resolve()), "launches": len(source_p1024)},
        ],
        "source_unique_signatures": len(source_signatures),
        "targets": targets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "targets": [
            {
                "label": row["label"],
                "launches": row["launches"],
                "prefill_launches": row["prefill_launches"],
                "coverage": row["structural_coverage_fraction"],
            }
            for row in targets
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
