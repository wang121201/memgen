#!/usr/bin/env python3
"""Join completed full-inference policy shards to the matching NCU oracle."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


READ = "dram__bytes_read.sum"
WRITE = "dram__bytes_write.sum"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--hardware-summary", type=Path, required=True)
    parser.add_argument("--policy-registry", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def error(model: int, hardware: int) -> dict[str, Any]:
    delta = model - hardware
    signed = None if hardware == 0 else 100.0 * delta / hardware
    return {
        "model_B": model,
        "hardware_median_B": hardware,
        "delta_B": delta,
        "absolute_delta_B": abs(delta),
        "signed_error_pct": signed,
        "absolute_error_pct": None if signed is None else abs(signed),
        "strictly_below_10_percent": model == 0 if hardware == 0 else abs(signed) < 10.0,
    }


def hardware_rows(path: Path, decode_steps: int) -> dict[str, dict[str, Any]]:
    selected = [row for row in read_json(path) if row["decode_steps"] == decode_steps]
    result = {}
    for row in selected:
        if row["scope"] == "whole":
            key = "whole"
        elif row["scope"] in ("Prefill", "Decode"):
            key = row["scope"]
        elif row["scope"] == "steps" and row["phase"].startswith("Decode"):
            key = row["phase"]
        else:
            continue
        if key in result:
            raise ValueError(f"duplicate hardware scope {key}")
        result[key] = row
    expected = {"whole", "Prefill", "Decode"} | {
        f"Decode{index}" for index in range(1, decode_steps + 1)}
    if set(result) != expected:
        raise ValueError(f"hardware scopes differ: {sorted(set(result) ^ expected)}")
    return result


def row_key(row: dict[str, str]) -> tuple[str, int]:
    return row["policy"], int(row["kernel_id"])


def load_policy_rows(roots: list[Path]) -> tuple[dict[tuple[str, int], dict[str, str]], list[dict[str, Any]]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    failures = []
    ignored = {"policy", "read_state_equal"}
    for root in roots:
        for batch in sorted(root.glob("batch-*")):
            finish_path = batch / "finish.json"
            if not finish_path.is_file():
                failures.append({"batch": str(batch), "status": "RUNNING_OR_UNSEALED"})
                continue
            finish = read_json(finish_path)
            if not str(finish.get("status", "")).startswith("COMPLETE_"):
                failures.append({"batch": str(batch), **finish})
                continue
            source = list(csv.DictReader((batch / "dirty-policies.csv").open(encoding="utf-8")))
            if len(source) != finish["rows"]:
                raise ValueError(f"sealed row count drift: {batch}")
            for row in source:
                key = row_key(row)
                if key in rows:
                    old = rows[key]
                    delta = {field: (old[field], row[field]) for field in row
                             if field not in ignored and old[field] != row[field]}
                    if delta:
                        raise ValueError(f"duplicate policy row differs {key}: {delta}")
                else:
                    rows[key] = row
    return rows, failures


def scope_names(phase: str) -> list[str]:
    names = ["whole", phase]
    if phase.startswith("Decode"):
        names.append("Decode")
    return names


def fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}%"


def mib(value: int) -> str:
    return f"{value / (1 << 20):.2f} MiB"


def gib(value: int) -> str:
    return f"{value / (1 << 30):.2f} GiB"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], scope: str) -> str:
    header = ("| 策略 | NCU read | 模拟 read | read 相对误差 | "
              "NCU write | 模拟 write | write 绝对误差 | write 相对误差 |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|\n")
    body = []
    for row in sorted((item for item in rows if item["scope"] == scope),
                      key=lambda item: (item["write_absolute_error_pct"] is None,
                                        item["write_absolute_error_pct"] or 0.0,
                                        item["policy"])):
        body.append(
            f"| `{row['policy']}` | {gib(row['hardware_read_B'])} | "
            f"{gib(row['model_read_B'])} | {fmt_pct(row['read_signed_error_pct'])} | "
            f"{mib(row['hardware_write_B'])} | {mib(row['model_write_B'])} | "
            f"{mib(row['write_absolute_delta_B'])} | {fmt_pct(row['write_signed_error_pct'])} |"
        )
    return header + "\n".join(body) + "\n"


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    population = read_json(args.population)
    members = {int(row["kernel_id"]): row for row in population}
    if sorted(members) != list(range(1, len(population) + 1)):
        raise ValueError("population is not dense and ordered")
    registry_rows = list(csv.DictReader(args.policy_registry.open(encoding="utf-8")))
    registry = {row["policy"]: row for row in registry_rows}
    policy_rows, failures = load_policy_rows(args.matrix_root)
    completed = sorted({policy for policy, _ in policy_rows})
    missing = sorted(set(registry) - set(completed))
    unknown = sorted(set(completed) - set(registry))
    if unknown:
        raise ValueError(f"unregistered completed policies: {unknown}")
    if missing and not args.allow_partial:
        raise ValueError(f"missing completed policies: {missing}")
    by_policy = collections.defaultdict(dict)
    for (policy, kernel), row in policy_rows.items():
        by_policy[policy][kernel] = row
    for policy, rows in by_policy.items():
        if set(rows) != set(members):
            raise ValueError(f"{policy} kernel coverage differs")
    hardware = hardware_rows(args.hardware_summary, args.decode_steps)
    aggregates: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(
        lambda: {"kernels": 0, "read_B": 0, "write_B": 0,
                 "incomplete_victim_sectors": 0, "missing_victim_bytes": 0})
    for policy, rows in by_policy.items():
        for kernel, row in rows.items():
            member = members[kernel]
            if member["role"] != "measurement":
                continue
            for scope in scope_names(member["phase"]):
                target = aggregates[(policy, scope)]
                target["kernels"] += 1
                target["read_B"] += int(row["read_B"])
                target["write_B"] += int(row["write_B"])
                target["incomplete_victim_sectors"] += int(row["incomplete_victim_sectors"])
                target["missing_victim_bytes"] += int(row["missing_victim_bytes"])
    scopes = ["whole", "Prefill", "Decode"] + [
        f"Decode{index}" for index in range(1, args.decode_steps + 1)]
    result_rows = []
    for policy in completed:
        for scope in scopes:
            model = aggregates[(policy, scope)]
            hw = hardware[scope]
            re = error(model["read_B"], int(hw["metrics"][READ]["median"]))
            we = error(model["write_B"], int(hw["metrics"][WRITE]["median"]))
            result_rows.append({
                "workload": args.workload, "policy": policy,
                "family": registry[policy]["family"], "scope": scope,
                "model_kernels": model["kernels"],
                "hardware_read_B": re["hardware_median_B"],
                "model_read_B": re["model_B"],
                "read_delta_B": re["delta_B"],
                "read_signed_error_pct": re["signed_error_pct"],
                "read_absolute_error_pct": re["absolute_error_pct"],
                "hardware_write_B": we["hardware_median_B"],
                "model_write_B": we["model_B"],
                "write_delta_B": we["delta_B"],
                "write_absolute_delta_B": we["absolute_delta_B"],
                "write_signed_error_pct": we["signed_error_pct"],
                "write_absolute_error_pct": we["absolute_error_pct"],
                "read_below_10_pct": re["strictly_below_10_percent"],
                "write_below_10_pct": we["strictly_below_10_percent"],
                "incomplete_victim_sectors": model["incomplete_victim_sectors"],
                "missing_victim_bytes": model["missing_victim_bytes"],
            })
    lookup = {(row["policy"], row["scope"]): row for row in result_rows}
    summaries = []
    for policy in completed:
        decode_steps = [lookup[(policy, f"Decode{index}")]
                        for index in range(1, args.decode_steps + 1)]
        denominator = sum(row["hardware_write_B"] for row in decode_steps)
        decode_write_wape = (100.0 * sum(row["write_absolute_delta_B"] for row in decode_steps)
                             / denominator if denominator else None)
        core = [lookup[(policy, scope)] for scope in ("whole", "Prefill", "Decode")]
        summaries.append({
            "workload": args.workload, "policy": policy,
            "family": registry[policy]["family"],
            "whole_read_error_pct": core[0]["read_signed_error_pct"],
            "whole_write_error_pct": core[0]["write_signed_error_pct"],
            "prefill_read_error_pct": core[1]["read_signed_error_pct"],
            "prefill_write_error_pct": core[1]["write_signed_error_pct"],
            "decode_read_error_pct": core[2]["read_signed_error_pct"],
            "decode_write_error_pct": core[2]["write_signed_error_pct"],
            "decode_write_absolute_delta_B": core[2]["write_absolute_delta_B"],
            "decode_step_write_wape_pct": decode_write_wape,
            "max_decode_step_write_absolute_error_pct": max(
                row["write_absolute_error_pct"] for row in decode_steps
                if row["write_absolute_error_pct"] is not None),
            "whole_prefill_decode_joint_below_10_pct": all(
                row[direction + "_below_10_pct"] for row in core
                for direction in ("read", "write")),
            "model_domain_complete": all(
                row["incomplete_victim_sectors"] == 0 and row["missing_victim_bytes"] == 0
                for row in core),
        })
    summaries.sort(key=lambda row: (
        abs(row["decode_write_error_pct"]) if row["decode_write_error_pct"] is not None else float("inf"),
        row["policy"]))
    flat_rows = []
    for row in result_rows:
        flat_rows.append({key: row[key] for key in row})
    write_csv(args.output_root / "scope-results.csv", flat_rows)
    write_csv(args.output_root / "policy-summary.csv", summaries)
    decode_step_rows = [row for row in result_rows if row["scope"].startswith("Decode") and row["scope"] != "Decode"]
    write_csv(args.output_root / "decode-step-results.csv", decode_step_rows)
    aggregate = {
        "status": "COMPLETE_FULL_POLICY_MATRIX_NOT_HARDWARE_ACCEPTANCE" if not missing
                  else "PARTIAL_FULL_POLICY_MATRIX_NOT_HARDWARE_ACCEPTANCE",
        "workload": args.workload, "decode_steps": args.decode_steps,
        "source_kernels": len(population),
        "measurement_kernels": sum(row["role"] == "measurement" for row in population),
        "completed_policies": completed, "missing_policies": missing,
        "failed_or_unsealed_shards": failures, "summaries": summaries,
        "hardware_accuracy_accepted": False,
        "limitations": [
            "All policies are replayed on the reconstructed HBServe request order, not a measured global GPU arrival frontier.",
            "NCU values are three-repeat medians; model values are deterministic single replays.",
            "Incomplete partial-dirty victims are reported and prevent strict semantic admission.",
            "A numerical traffic match does not identify NVIDIA replacement or write-service mechanisms.",
        ],
        "inputs": {
            "population": {"path": str(args.population), "sha256": sha256(args.population)},
            "hardware_summary": {"path": str(args.hardware_summary), "sha256": sha256(args.hardware_summary)},
            "policy_registry": {"path": str(args.policy_registry), "sha256": sha256(args.policy_registry)},
        },
    }
    write_json_new(args.output_root / "aggregate.json", aggregate)
    report = [
        f"# {args.workload} 全量推理 L2 策略对比\n",
        "本报告中的 NCU（NVIDIA Nsight Compute）值是同一工作负载三次硬件测量的中位数。"
        "模拟值来自 HBServe 生成的完整 memory-SASS 请求流和 Memgen L2 在线因果重放。"
        "`Decode` 表示一次请求中连续完整 Decode 区间；它是本文 write traffic 的主验收范围。"
        "逐步 `DecodeN` 只用于诊断，不能替代连续区间。所有流量为字节口径；MiB/GiB 均为二进制单位。\n",
        f"- 源 kernel：{len(population):,}\n"
        f"- measurement kernel：{aggregate['measurement_kernels']:,}\n"
        f"- 已完成策略：{len(completed)}/{len(registry)}\n"
        f"- 缺失策略：{', '.join(missing) if missing else '无'}\n"
        f"- 三范围读写全部严格小于 10% 的策略："
        f"{sum(row['whole_prefill_decode_joint_below_10_pct'] for row in summaries)}\n",
        "## 连续 Decode（主表）\n",
        markdown_table(result_rows, "Decode"),
        "## Whole inference\n",
        markdown_table(result_rows, "whole"),
        "## Prefill\n",
        markdown_table(result_rows, "Prefill"),
        "## 解释边界\n",
        "- 策略按连续 Decode write 相对误差的绝对值排序。\n"
        "- 当硬件 write 分母很小时，单步相对误差会被放大，因此同时保留 MiB 绝对误差与逐步 WAPE。\n"
        "- `model_domain_complete=false` 表示存在模型无法完整覆盖旧字节的 partial-dirty victim；即使数值接近，也不能升级为严格硬件精度验收。\n"
        "- timestamp sort、MSHR、WTB、地址映射等改变上游顺序或时序的方案不属于本同源 L2 策略矩阵。\n",
    ]
    (args.output_root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    finish = {
        "status": aggregate["status"], "completed_policies": len(completed),
        "expected_policies": len(registry), "missing_policies": missing,
        "aggregate_sha256": sha256(args.output_root / "aggregate.json"),
        "scope_results_sha256": sha256(args.output_root / "scope-results.csv"),
        "report_sha256": sha256(args.output_root / "REPORT.md"),
    }
    write_json_new(args.output_root / "finish.json", finish)
    print(json.dumps(finish, sort_keys=True))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
