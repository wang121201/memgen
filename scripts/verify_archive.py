#!/usr/bin/env python3
"""Read-only checks for the archived source and accuracy tables."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".gguf",
    ".mem",
    ".memc",
    ".ncu-rep",
    ".nsys-rep",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    required = [
        ROOT / "release/source/tools/hbserve_profile_stream_cache_semantic_r17.cpp",
        ROOT / "release/config/RTX4000Ada.paper-v1.config",
        ROOT / "release/release-manifest.json",
        ROOT / "integrations/sglang/compact-sources/upstream/template_adapter_r4/hbserve_adapter.py",
        ROOT / "integrations/sglang/memgen-adapter/run_memgen.py",
        ROOT / "evidence/historical-prefill-d32/finish.json",
        ROOT / "evidence/historical-prefill-d32/tables.md",
        ROOT / "evidence/sglang/L2_CACHE_STRATEGY_ACCURACY_REPORT.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    require(not missing, f"missing required archive files: {missing}")

    historical = rows(ROOT / "validation/historical_prefill_d32.csv")
    expected = {
        "P64D32": (0.48, 4.55, 1.02, 22.71, 0.06, 4.55),
        "P128D32": (0.90, 3.64, 1.09, 12.53, 0.05, 3.63),
        "P256D32": (1.85, 2.66, 1.48, 8.31, 0.05, 2.65),
        "P512D32": (3.85, 25.13, 2.32, 5.62, 0.21, 25.13),
        "P1024D32": (7.81, 23.07, 4.10, 1.14, 1.08, 23.07),
    }
    require([row["workload"] for row in historical] == list(expected), "historical workload order changed")
    keys = (
        "naive_dram_read_error_pct",
        "naive_dram_write_error_pct",
        "memgen_l1_hit_error_pct",
        "memgen_l2_hit_error_pct",
        "memgen_dram_read_error_pct",
        "memgen_dram_write_error_pct",
    )
    for row in historical:
        actual = tuple(float(row[key]) for key in keys)
        require(actual == expected[row["workload"]], f"historical values changed: {row['workload']} {actual}")
        require(row["framework"] == "llama.cpp" and row["dtype"] == "Q8_0", "historical evidence relabeled")

    current = rows(ROOT / "validation/sglang_current.csv")
    require(len(current) == 6, "current SGLang table must contain two workloads x three ranges")
    require({row["workload"] for row in current} == {"P128D2", "P128D16"}, "unexpected SGLang workload")
    require(all(row["framework"] == "SGLang-0.4.10" and row["dtype"] == "BF16" for row in current), "SGLang contract changed")
    require({row["range"] for row in current} == {"whole", "prefill", "decode"}, "SGLang ranges incomplete")

    forbidden = []
    oversized = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        lower = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or lower.endswith(".memc.zst"):
            forbidden.append(str(path.relative_to(ROOT)))
        if path.stat().st_size > 10 * 1024 * 1024:
            oversized.append((str(path.relative_to(ROOT)), path.stat().st_size))
    require(not forbidden, f"forbidden large/raw artifacts present: {forbidden}")
    require(not oversized, f"files larger than 10 MiB present: {oversized}")
    print(
        "PASS_ARCHIVE_CONTRACT: stable source present; historical and SGLang evidence remain distinct; "
        "no raw trace/model/NCU database or >10 MiB file"
    )


if __name__ == "__main__":
    main()

