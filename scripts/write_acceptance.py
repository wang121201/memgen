#!/usr/bin/env python3
"""Compare a replay's write traffic with the archived hardware protocol.

The acceptance question is per range, because the archived comparison records the
hardware ranges as independent protocols (`Decode1`, `Decode2`, `Decode`,
`Prefill`) and explicitly forbids merging them into one denominator. This tool
joins three things that already exist:

  model side     a kernel summary plus the app config that labels its kernels
                 (see phase_write_split.py)
  hardware side  the archived comparison's median and min/max per range
  mapping        the recorded `model_field` -> NCU metric pairs

The model side is a continuous replay, so its `Decode` range is the union of the
`Decode1` and `Decode2` kernels; the hardware `Decode` value is a separate
measurement and is compared against that union rather than against the sum of the
hardware steps.

It prints the residual per range and, for the write path, the two factors the
residual decomposes into: how many store sectors the model sent relative to
hardware, and how many of them it evicted dirty.

Usage:
  python3 scripts/write_acceptance.py --summary OUT/model/kernel_summary.csv \\
      --app-config OUT/app.config [--phases Decode1,Decode2,Decode] [--json]
"""

import argparse
import json
import sys

from phase_write_split import read_phases, totals

DEFAULT_COMPARISON = ("evidence/sglang/p32d2-p64d2-traffic-comparison-20260923/"
                      "traffic-comparison/comparison.json")

# model CSV column -> NCU metric in the archived comparison
WRITE_FIELDS = {
    "dram_store_bytes": "dram__bytes_write.sum",
    "l2_write_requests": "lts__t_sectors_srcunit_tex_op_write.sum",
    "write_sector_requests": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "dram_load_bytes": "dram__bytes_read.sum",
}


def hardware_rows(path, workload):
    with open(path) as handle:
        data = json.load(handle)
    rows = {}
    for row in data["rows"]:
        if row["workload"] != workload or row["model"] != "r4-small-shared":
            continue
        key = (row["metric"], row["phase"])
        rows[key] = row
    if not rows:
        raise SystemExit("no %s rows for %s in %s" % ("r4-small-shared", workload, path))
    return data, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--app-config", required=True)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    parser.add_argument("--workload", default="qwen1p5b-P32D2")
    parser.add_argument("--phases", default="Decode1,Decode2,Decode")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    import csv as _csv
    phases = read_phases(args.app_config)
    with open(args.summary) as handle:
        rows = list(_csv.DictReader(handle))
    unknown = [row for row in rows if int(row["kernel_id"]) not in phases]
    if unknown:
        raise SystemExit("summary has kernels without a phase label; refusing to guess")
    grouped = {}
    for row in rows:
        grouped.setdefault(phases[int(row["kernel_id"])], []).append(row)
    model = {phase: totals(grouped[phase]) for phase in grouped}
    # The replay is continuous, so its combined Decode range is the union of the
    # two step kernels; hardware measures that range as its own protocol.
    steps = [phase for phase in ("Decode1", "Decode2") if phase in grouped]
    if len(steps) == 2:
        model["Decode"] = totals(grouped["Decode1"] + grouped["Decode2"])

    data, hw = hardware_rows(args.comparison, args.workload)
    report = {}
    for phase in args.phases.split(","):
        if phase not in model:
            print("range %s: no kernels in this summary" % phase, file=sys.stderr)
            continue
        report[phase] = {}
        for field, metric in WRITE_FIELDS.items():
            row = hw.get((metric, phase))
            if row is None:
                continue
            simulated = model[phase].get(field)
            if simulated is None:
                continue
            median = row["hardware_median"]
            error = (simulated - median) / median * 100.0
            report[phase][field] = {
                "model": simulated,
                "hardware_median": median,
                "hardware_min": row["hardware_min"],
                "hardware_max": row["hardware_max"],
                "error_percent": error,
                # Judged on this model's value, not the archived model's verdict.
                "within_observed_range": row["hardware_min"] <= simulated <= row["hardware_max"],
                "hardware_spread_percent": row["hardware_relative_range_percent"],
            }

    if args.json:
        print(json.dumps(report, indent=1))
        return 0

    print("archived comparison: %s (%s)" % (args.comparison, data.get("schema", "?")))
    print("workload: %s" % args.workload)
    print()
    print("%-9s %-21s %14s %14s %10s %8s" %
          ("range", "quantity", "model", "hardware", "error", "in range"))
    for phase, fields in report.items():
        for field, values in fields.items():
            print("%-9s %-21s %14d %14d %9.2f%% %8s" %
                  (phase, field, values["model"], values["hardware_median"],
                   values["error_percent"], "yes" if values["within_observed_range"] else "no"))

    print()
    print("write residual decomposition (model / hardware):")
    for phase in report:
        if "dram_store_bytes" not in report[phase]:
            continue
        stream = report[phase]["write_sector_requests"]
        dirty = model[phase]["l2_writeback_dirty_sectors"]
        hw_store = report[phase]["dram_store_bytes"]["hardware_median"] / 32.0
        hw_stream = stream["hardware_median"]
        if not hw_store or not hw_stream:
            continue
        print("  %-9s store sectors x%.4f, dirty fraction x%.4f, product x%.4f "
              "(dirty sectors %d vs %.0f)" %
              (phase, stream["model"] / hw_stream,
               (dirty / stream["model"]) / (hw_store / hw_stream),
               report[phase]["dram_store_bytes"]["model"] /
               report[phase]["dram_store_bytes"]["hardware_median"], dirty, hw_store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
