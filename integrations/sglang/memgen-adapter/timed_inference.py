#!/usr/bin/env python3
"""Measure uninstrumented SGLang CUDA execution for one declared memgen case.

The script deliberately does not load NVBit or the native observer.  It runs
one complete warmup pass, then records CUDA-event elapsed time for the declared
Prefill and Decode phases.  If ``--traffic-csv`` names a memgen semantic
traffic file, only the non-warmup phases are paired with the measured time to
produce an effective-bandwidth diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matrix_common as reference  # noqa: E402
import matrix_workload as workload  # noqa: E402


def reset_state(runner) -> None:
    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()


def traffic_totals(path: Path) -> dict[str, int]:
    totals = {"dram_read_bytes": 0, "dram_write_bytes": 0}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            # The CSV contains separate warmup/* and measurement phases.  Pair
            # only the measurement rows with the CUDA-event measurement pass.
            if row["service_phase"].startswith("warmup/"):
                continue
            for key in totals:
                totals[key] += int(row[key])
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    workload.add_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--traffic-csv", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    frozen = workload.from_args(args)
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    import sglang.bench_one_batch as bo

    started = time.monotonic()
    server = bo.ServerArgs(
        model_path=frozen["model"], dtype="bfloat16", load_format="safetensors",
        device="cuda", tp_size=1, pp_size=1, attention_backend="flashinfer",
        disable_cuda_graph=True, cuda_graph_max_bs=1, enable_torch_compile=False,
        disable_overlap_schedule=True, disable_radix_cache=True,
        mem_fraction_static=.90, max_total_tokens=frozen["max_total_tokens"],
        max_running_requests=1, random_seed=0, cpu_offload_gb=0)
    bo._set_envs_and_config(server)
    load_start = time.monotonic()
    runner, _ = bo.load_model(server, bo.PortArgs.init_new(server), 0)
    load_seconds = time.monotonic() - load_start
    fixed = [torch.tensor([v], dtype=torch.int64, device=runner.device)
             for v in frozen["decode_input_ids"]]
    samples = {phase: [] for phase in frozen["phases"]}
    for repeat in range(args.repeats):
        reset_state(runner)
        with torch.no_grad():
            # Warmup is a state-preconditioning pass and is not reported as a
            # measured phase.  The measured pass starts from the same reset
            # allocator/cache state for every repeat.
            batch = None
            for i in range(len(frozen["phases"])):
                if i == 0:
                    _, _, batch = bo.extend(reference.make_request(bo, frozen), runner)
                else:
                    _, _ = bo.decode(fixed[i - 1], batch, runner)
            torch.cuda.synchronize()
            reset_state(runner)
            for i, phase in enumerate(frozen["phases"]):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                if i == 0:
                    _, _, batch = bo.extend(reference.make_request(bo, frozen), runner)
                else:
                    _, _ = bo.decode(fixed[i - 1], batch, runner)
                end.record()
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
                samples[phase].append(elapsed_ms)
            torch.cuda.synchronize()

    phase_stats = {}
    for phase, values in samples.items():
        phase_stats[phase] = dict(
            repeats_ms=values,
            median_ms=statistics.median(values),
            mean_ms=statistics.fmean(values),
        )
    total_ms = sum(v["median_ms"] for v in phase_stats.values())
    result = dict(
        schema="SGLANG_MEMGEN_EXECUTION_TIMING_V1",
        case_id=frozen["case_id"],
        model=frozen["model_key"],
        prefill_length=frozen["prefill_length"],
        decode_steps=frozen["decode_steps"],
        repeats=args.repeats,
        warmup_executed=True,
        timing_source="CUDA_EVENT_UNINSTRUMENTED_SGLANG",
        profiling_seconds_are_hardware_runtime=True,
        script_sha256=reference.sha256(Path(__file__)),
        model_load_seconds=load_seconds,
        phases=phase_stats,
        median_total_ms=total_ms,
        median_total_seconds=total_ms / 1000.0,
        wall_seconds=time.monotonic() - started,
    )
    if args.traffic_csv:
        traffic = traffic_totals(args.traffic_csv)
        seconds = result["median_total_seconds"]
        result["paired_traffic"] = traffic
        result["effective_bandwidth_gb_per_s"] = {
            "dram_read": traffic["dram_read_bytes"] / seconds / 1e9,
            "dram_write": traffic["dram_write_bytes"] / seconds / 1e9,
            "dram_total": (traffic["dram_read_bytes"] + traffic["dram_write_bytes"]) / seconds / 1e9,
        }
    reference.write_json(args.output / "timing.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
