#!/usr/bin/env python3
"""Print one sampled profile slot's exact per-CTA bases for diagnosis."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-tool", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--kernel", type=int, required=True)
    parser.add_argument("--ctas", required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("sample_trace", args.sample_tool.resolve(strict=True))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sample tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = {int(value) for value in args.ctas.split(",")}
    paths = module.discover_memc_files(args.trace_root.resolve(strict=True), args.kernel)
    sampled, source = module.collect_ctas_memc(paths, selected)
    indexed = {block: module.indexed_records(records) for block, records in sampled.items()}
    anchor_records = sorted(sampled[args.anchor], key=lambda item: item["source_sequence"])
    record = anchor_records[args.ordinal]
    slot = indexed[args.anchor][0][record["source_sequence"]]
    rows = []
    for block in sorted(selected):
        actual = indexed[block][1][slot]
        rows.append({
            "cta": block,
            "bases": [group["base"] for group in actual["groups"]],
            "timestamp": actual["timestamp"],
            "source_sequence": actual["source_sequence"],
        })
    print(json.dumps({
        "kernel": args.kernel,
        "ordinal": args.ordinal,
        "anchor_record": {
            "pc": record["pc"],
            "opcode": record["opcode"],
            "mask": record["mask"],
            "signature_rank": slot[1],
        },
        "rows": rows,
        "source_lowering": source.get("global_only_lowering"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
