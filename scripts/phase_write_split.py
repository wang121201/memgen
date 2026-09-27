#!/usr/bin/env python3
"""Split a subset replay's kernel summary by hardware range.

The archived comparison measures DRAM write traffic per *range* (Prefill, Decode1,
Decode2, Decode whole). A kernel summary row is per kernel, so the range is
recovered from the subset's own app config: `-kernel_<n>_llama_phase` names the
phase each renumbered kernel belongs to. Kernel order is the replay order, and
`kernel_id` is 1-based in the same order.

This reports the numbers the write-path acceptance has to match, plus the
intermediate counters that decide them, so a residual can be attributed instead
of guessed:

  dram_store_bytes          hardware write traffic for the range
  l2_write_requests         store sectors accepted by L2 (hardware merges 30.6%)
  l2_writeback_dirty_sectors  dirty sectors evicted inside the range
  write_sector_requests     the store stream the model was given (must not move)

Usage:
  python3 scripts/phase_write_split.py KERNEL_SUMMARY.csv APP_CONFIG [--json]
"""

import argparse
import csv
import json
import re
import sys


def read_phases(path):
    """Return {kernel_id: phase} from -kernel_<n>_llama_phase keys."""
    phases = {}
    with open(path) as handle:
        for line in handle:
            match = re.match(r"\s*-kernel_(\d+)_llama_phase\s+(\S+)", line)
            if match:
                phases[int(match.group(1))] = match.group(2)
    return phases


def totals(rows):
    out = {}
    for name in rows[0]:
        try:
            out[name] = sum(int(float(row[name])) for row in rows if row.get(name))
        except ValueError:
            continue
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("app_config")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    phases = read_phases(args.app_config)
    if not phases:
        raise SystemExit("no -kernel_<n>_llama_phase keys in " + args.app_config)
    with open(args.summary) as handle:
        rows = list(csv.DictReader(handle))
    known = [row for row in rows if int(row["kernel_id"]) in phases]
    if len(known) != len(rows):
        raise SystemExit("summary has kernels without a phase label; refusing to guess")

    grouped = {}
    for row in known:
        grouped.setdefault(phases[int(row["kernel_id"])], []).append(row)
    grouped["all"] = known

    fields = ("write_sector_requests", "l2_write_requests", "l2_write_hits",
              "l2_write_misses", "l2_writeback_events", "l2_writeback_dirty_sectors",
              "l2_dirty_drain_sectors", "dram_store_requests", "dram_store_sectors",
              "dram_store_bytes", "dram_load_bytes")
    result = {}
    for phase in sorted(grouped, key=lambda name: (name == "all", name)):
        totals_phase = totals(grouped[phase])
        result[phase] = {"kernels": len(grouped[phase])}
        result[phase].update({name: totals_phase.get(name, 0) for name in fields})

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    header = ("range", "kernels") + fields
    print(" ".join("%-26s" % name for name in header))
    for phase, values in result.items():
        cells = [phase, str(values["kernels"])] + [str(values[name]) for name in fields]
        print(" ".join("%-26s" % cell for cell in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
