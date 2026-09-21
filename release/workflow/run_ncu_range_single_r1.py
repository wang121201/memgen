#!/usr/bin/env python3
"""Collect and admit one cold-start continuous NCU application range."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(
    "/home/xmu/nvidiagds/codex-runs/llm-footprint-v1/memgen/"
    "hyfiss-prefill-matrix-20260913-01a06837-r1"
)
DRIVER = ROOT / "llm-range-matrix-driver"
MODEL = Path("/home/xmu/nvidiagds/models/qwen2.5-1.5b-instruct-q8_0.gguf")
NCU = Path("/usr/local/cuda-12.8/bin/ncu")
METRICS = (
    "l1tex__t_sectors.sum",
    "l1tex__t_sectors_lookup_hit.sum",
    "lts__t_sectors.sum",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)
EXPECTED_SHA256 = {
    DRIVER: "66c757df426e848cfd0dd5feb44d9faa3c890ba812f2b62087c0f940217ea72f",
    MODEL: "d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8",
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def save_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def gpu_snapshot() -> dict[str, list[str]]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    return {
        "gpus": [line.strip() for line in gpu.stdout.splitlines() if line.strip()],
        "compute_apps": [line.strip() for line in apps.stdout.splitlines() if line.strip()],
    }


def phase_files(root: Path) -> dict[str, dict[str, Any]]:
    names = ["input_tokens.i32", "prefill.logits.f32"] + [f"decode_step_{index}.logits.f32" for index in range(32)]
    return {name: artifact(root / name) for name in names}


def compare_outputs(profiled: Path, baseline: Path) -> dict[str, Any]:
    lhs, rhs = phase_files(profiled), phase_files(baseline)
    for name in lhs:
        need(lhs[name]["bytes"] == rhs[name]["bytes"], f"output size differs: {name}")
        need(lhs[name]["sha256"] == rhs[name]["sha256"], f"output SHA-256 differs: {name}")
    return {"file_count": len(lhs), "all_bytes_and_sha256_exact": True, "profiled": lhs, "baseline": rhs}


def parse_raw_csv(path: Path, logs: str) -> dict[str, Any]:
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    starts = [index for index, row in enumerate(rows) if row and row[0] == "ID"]
    need(len(starts) == 1, "NCU import does not contain exactly one raw header")
    header, units, *values = rows[starts[0]:]
    need(len(values) == 1, f"NCU import expected one range row, got {len(values)}")
    need(len(header) == len(set(header)) == len(units) == len(values[0]), "NCU raw table shape differs")
    row = dict(zip(header, values[0]))
    unit = dict(zip(header, units))
    need(row["ID"] == "0" and row["Kernel Name"] == "range", "NCU row is not application range 0")
    need(row["CC"] == "8.9" and row["Stream"] == "n/a", "NCU target identity differs")

    def integer(name: str) -> int:
        value = Decimal(row[name].replace(",", ""))
        need(value.is_finite() and value >= 0 and value == int(value), f"invalid NCU integer: {name}")
        return int(value)

    need(
        unit["profiler__replayer_passes"] == unit["profiler__replayer_passes_type_warmup"] == "pass",
        "NCU replay-pass units differ",
    )
    need(integer("profiler__replayer_passes") == 1, "NCU range did not use exactly one pass")
    need(integer("profiler__replayer_passes_type_warmup") == 0, "NCU inserted a warmup pass")
    values_by_metric = {}
    units_by_metric = {}
    for name in METRICS:
        expected_unit = "byte" if name.startswith("dram") else "sector"
        need(unit[name] == expected_unit, f"NCU metric unit differs: {name}")
        values_by_metric[name] = integer(name)
        units_by_metric[name] = unit[name]
    need(0 < values_by_metric[METRICS[1]] <= values_by_metric[METRICS[0]], "NCU L1 hit/request bounds differ")
    need(0 < values_by_metric[METRICS[3]] <= values_by_metric[METRICS[2]], "NCU L2 hit/request bounds differ")
    need(values_by_metric[METRICS[-2]] % 32 == values_by_metric[METRICS[-1]] % 32 == 0, "NCU DRAM bytes are not sector aligned")
    need(re.findall(r'Profiling "range" - (\d+): Application replay pass (\d+)', logs) == [("0", "1")], "NCU did not report exactly one range/pass")
    need("==ERROR==" not in logs and "No ranges were profiled" not in logs, "NCU reported an error or no range")
    return {
        "range_id": 0,
        "process_id": int(row["Process ID"]),
        "metrics": values_by_metric,
        "units": units_by_metric,
        "l1_hit_rate": values_by_metric[METRICS[1]] / values_by_metric[METRICS[0]],
        "l2_hit_rate": values_by_metric[METRICS[3]] / values_by_metric[METRICS[2]],
        "l2_hit_plus_miss_minus_total": values_by_metric[METRICS[3]] + values_by_metric[METRICS[4]] - values_by_metric[METRICS[2]],
        "single_application_replay_pass": True,
        "warmup_passes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=int, choices=(64, 128, 256, 512, 1024), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--placement-root", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    need(args.run_label == f"p{args.prompt}d32", "run label must be the canonical p<P>d32 value")
    password = os.environ.pop("root_sudo", None)
    need(bool(password), "authorized root_sudo credential unavailable")
    placement = args.placement_root.resolve(strict=True)
    output = ROOT / "ncu-range-r1" / args.run_label
    need(not output.exists(), f"refusing to overwrite {output}")
    inputs = {
        "driver": artifact(DRIVER),
        "model": artifact(MODEL),
        "ncu": artifact(NCU),
        "placement_app_config": artifact(placement / "configs/app.config"),
        "placement_capture_receipt": artifact(placement / "configs/capture_receipt.json"),
        "runner": artifact(Path(__file__)),
    }
    for path, expected in EXPECTED_SHA256.items():
        need(inputs["driver" if path == DRIVER else "model"]["sha256"] == expected, f"frozen input differs: {path}")
    before = gpu_snapshot()
    need(any(args.gpu_uuid in row for row in before["gpus"]), "target GPU UUID is absent")
    need(not [row for row in before["compute_apps"] if row.split(",", 1)[0].strip() == args.gpu_uuid], "target GPU is busy")

    output.mkdir(parents=True, exist_ok=False)
    start = {
        "schema": "hyfiss_prefill_matrix_ncu_range_v1",
        "status": "STARTED",
        "workload": {"prompt_tokens": args.prompt, "decode_steps": 32, "batch": 1, "context_capacity": 8192},
        "gpu_uuid": args.gpu_uuid,
        "metrics": list(METRICS),
        "inputs": inputs,
        "gpu_before": before,
        "started_utc": utc_now(),
        "scope": "one cold-start application range covering prefill then all 32 decode forwards; no per-kernel cache reset",
    }
    save_new(output / "start.json", start)
    env = {
        key: value for key, value in os.environ.items()
        if key not in ("root_sudo", "LD_PRELOAD", "CUDA_INJECTION64_PATH", "GGML_BACKEND_PATH")
        and not key.startswith("HYFISS_")
    }
    env.update({
        "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "PATH": "/usr/local/cuda-12.8/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    command = [
        "sudo", "-S", "-p", "", "/usr/bin/timeout", "--kill-after=20", str(args.timeout_seconds),
        "/usr/bin/env", "-u", "LD_PRELOAD", "-u", "CUDA_INJECTION64_PATH",
        "CUDA_VISIBLE_DEVICES=" + args.gpu_uuid,
        "GGML_CUDA_DISABLE_GRAPHS=1",
        "PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin",
        str(NCU), "--config-file", "off", "--replay-mode", "app-range", "--cache-control", "all",
        "--clock-control", "none", "--metrics", ",".join(METRICS), "--export", str(output / "report"),
        str(DRIVER), str(MODEL), str(args.prompt), "32",
    ]
    save_new(output / "command.json", {"argv": command, "credential_transport": "sudo stdin; value neither logged nor persisted"})
    began = time.monotonic()
    with (output / "stdout.log").open("x", encoding="utf-8") as stdout, (output / "stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=output, env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True, start_new_session=True)
        save_new(output / "started.json", {"pid": process.pid, "process_group": process.pid, "started_utc": utc_now()})
        stop_reason = None
        try:
            process.communicate(password + "\n", timeout=args.timeout_seconds + 60)
        except subprocess.TimeoutExpired:
            stop_reason = "timeout"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    need(process.returncode == 0 and stop_reason is None, f"NCU process failed: rc={process.returncode} stop={stop_reason}")
    logs = (output / "stdout.log").read_text(encoding="utf-8", errors="replace") + "\n" + (output / "stderr.log").read_text(encoding="utf-8", errors="replace")
    import_root = output / "import"
    import_root.mkdir()
    with (import_root / "stdout.csv").open("x", encoding="utf-8") as stdout, (import_root / "stderr.log").open("x", encoding="utf-8") as stderr:
        imported = subprocess.run(
            [str(NCU), "--import", str(output / "report.ncu-rep"), "--page", "raw", "--csv", "--print-units", "base"],
            env=env, stdout=stdout, stderr=stderr, text=True, timeout=300,
        )
    need(imported.returncode == 0, "NCU report import failed")
    counters = parse_raw_csv(import_root / "stdout.csv", logs)
    outputs = compare_outputs(output, placement)
    finish = {
        **start,
        "status": "PASS_SINGLE_RANGE_SINGLE_PASS_EXACT_LLM_OUTPUTS_HARDWARE_COUNTER_ORACLE",
        "finished_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - began,
        "process": {"returncode": process.returncode, "stop_reason": stop_reason},
        "counters": counters,
        "output_equivalence": outputs,
        "gpu_after": gpu_snapshot(),
        "artifacts": [artifact(path) for path in (
            output / "command.json", output / "started.json", output / "stdout.log", output / "stderr.log",
            output / "report.ncu-rep", import_root / "stdout.csv", import_root / "stderr.log", output / "loaded_maps.txt",
        )],
        "claim_boundary": (
            "Hardware cache/DRAM counters for one exact continuous workload range. NCU replayed execution time is diagnostic; "
            "this receipt alone does not validate HBServe generation or either cache model."
        ),
    }
    save_new(output / "finish.json", finish)
    print(json.dumps({
        "status": finish["status"],
        "prompt": args.prompt,
        "l1_hit_rate": counters["l1_hit_rate"],
        "l2_hit_rate": counters["l2_hit_rate"],
        "dram_read_bytes": counters["metrics"]["dram__bytes_read.sum"],
        "dram_write_bytes": counters["metrics"]["dram__bytes_write.sum"],
        "elapsed_seconds": finish["elapsed_seconds"],
        "finish": str(output / "finish.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": repr(error)}), file=sys.stderr)
        raise
