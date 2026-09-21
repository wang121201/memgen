#!/usr/bin/env python3
"""Combine sealed per-workload full-inference policy comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True,
                        help="WORKLOAD=/absolute/result/root")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        workload, separator, raw = value.partition("=")
        if not separator or not workload or not raw:
            raise ValueError(f"invalid --result {value!r}")
        if workload in roots:
            raise ValueError(f"duplicate workload {workload}")
        roots[workload] = Path(raw)
    if len(roots) < 2:
        raise ValueError("at least two workload results are required")
    return roots


def signed(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}%"


def mib(value: int) -> str:
    return f"{value / (1 << 20):.2f} MiB"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("empty combined rows")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def workload_table(workload: str, rows: list[dict[str, Any]]) -> str:
    header = (
        "| 策略 | 策略族 | Whole read误差 | Whole write（NCU / 模拟 / 绝对差 / 误差） | "
        "Prefill write误差 | Decode read误差 | Decode write（NCU / 模拟 / 绝对差 / 误差） | "
        "逐step write WAPE | 支持域完整 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = []
    for row in rows:
        prefix = workload + "_"
        body.append(
            f"| `{row['policy']}` | `{row['family']}` | "
            f"{signed(row[prefix + 'whole_read_error_pct'])} | "
            f"{mib(row[prefix + 'whole_hardware_write_B'])} / "
            f"{mib(row[prefix + 'whole_model_write_B'])} / "
            f"{mib(row[prefix + 'whole_write_absolute_delta_B'])} / "
            f"{signed(row[prefix + 'whole_write_error_pct'])} | "
            f"{signed(row[prefix + 'prefill_write_error_pct'])} | "
            f"{signed(row[prefix + 'decode_read_error_pct'])} | "
            f"{mib(row[prefix + 'decode_hardware_write_B'])} / "
            f"{mib(row[prefix + 'decode_model_write_B'])} / "
            f"{mib(row[prefix + 'decode_write_absolute_delta_B'])} / "
            f"{signed(row[prefix + 'decode_write_error_pct'])} | "
            f"{signed(row[prefix + 'decode_step_write_wape_pct'])} | "
            f"{'是' if row[prefix + 'model_domain_complete'] else '否'} |"
        )
    return header + "\n".join(body) + "\n"


def main() -> int:
    args = parse_args()
    roots = parse_roots(args.result)
    args.output_root.mkdir(parents=True, exist_ok=False)
    workloads: dict[str, dict[str, Any]] = {}
    policy_sets: dict[str, set[str]] = {}
    pins: dict[str, Any] = {}
    for workload, root in roots.items():
        finish = read_json(root / "finish.json")
        if finish.get("status") != "COMPLETE_FULL_POLICY_MATRIX_NOT_HARDWARE_ACCEPTANCE":
            raise ValueError(f"{workload} is not a complete sealed matrix: {finish}")
        aggregate = read_json(root / "aggregate.json")
        summaries = {row["policy"]: row for row in aggregate["summaries"]}
        scopes = {(row["policy"], row["scope"]): row for row in
                  csv.DictReader((root / "scope-results.csv").open(encoding="utf-8"))}
        policy_sets[workload] = set(summaries)
        workloads[workload] = {"aggregate": aggregate, "summaries": summaries,
                               "scopes": scopes}
        pins[workload] = {
            "root": str(root),
            "finish_sha256": sha256(root / "finish.json"),
            "aggregate_sha256": sha256(root / "aggregate.json"),
            "scope_results_sha256": sha256(root / "scope-results.csv"),
        }
    first = next(iter(policy_sets.values()))
    if any(value != first for value in policy_sets.values()):
        raise ValueError({name: sorted(value ^ first) for name, value in policy_sets.items()})

    combined = []
    for policy in sorted(first):
        families = {workloads[name]["summaries"][policy]["family"] for name in workloads}
        if len(families) != 1:
            raise ValueError(f"family drift for {policy}: {families}")
        row: dict[str, Any] = {"policy": policy, "family": families.pop()}
        errors = []
        all_gates = []
        for workload, data in workloads.items():
            summary = data["summaries"][policy]
            whole = data["scopes"][(policy, "whole")]
            decode = data["scopes"][(policy, "Decode")]
            prefix = workload + "_"
            fields = {
                "whole_read_error_pct": summary["whole_read_error_pct"],
                "whole_write_error_pct": summary["whole_write_error_pct"],
                "whole_hardware_write_B": int(whole["hardware_write_B"]),
                "whole_model_write_B": int(whole["model_write_B"]),
                "whole_write_absolute_delta_B": int(whole["write_absolute_delta_B"]),
                "prefill_write_error_pct": summary["prefill_write_error_pct"],
                "decode_read_error_pct": summary["decode_read_error_pct"],
                "decode_write_error_pct": summary["decode_write_error_pct"],
                "decode_hardware_write_B": int(decode["hardware_write_B"]),
                "decode_model_write_B": int(decode["model_write_B"]),
                "decode_write_absolute_delta_B": int(decode["write_absolute_delta_B"]),
                "decode_step_write_wape_pct": summary["decode_step_write_wape_pct"],
                "model_domain_complete": summary["model_domain_complete"],
                "whole_prefill_decode_joint_below_10_pct":
                    summary["whole_prefill_decode_joint_below_10_pct"],
            }
            for key, value in fields.items():
                row[prefix + key] = value
            errors.append(abs(float(summary["decode_write_error_pct"])))
            all_gates.append(bool(summary["whole_prefill_decode_joint_below_10_pct"]))
        row["worst_decode_write_absolute_error_pct"] = max(errors)
        row["all_workloads_joint_below_10_pct"] = all(all_gates)
        combined.append(row)
    combined.sort(key=lambda row: (row["worst_decode_write_absolute_error_pct"], row["policy"]))
    write_csv(args.output_root / "combined-policy-summary.csv", combined)

    ranking_header = (
        "| 排名 | 策略 | 策略族 | " + " | ".join(
            f"{name} Decode write误差" for name in workloads) +
        " | 跨workload最坏误差 | 全范围均<10% |\n" +
        "|---:|---|---|" + "---:|" * len(workloads) + "---:|---:|\n"
    )
    ranking_body = []
    for index, row in enumerate(combined, 1):
        values = " | ".join(signed(row[name + "_decode_write_error_pct"])
                            for name in workloads)
        ranking_body.append(
            f"| {index} | `{row['policy']}` | `{row['family']}` | {values} | "
            f"{row['worst_decode_write_absolute_error_pct']:.2f}% | "
            f"{'是' if row['all_workloads_joint_below_10_pct'] else '否'} |"
        )
    reports = [
        "# 全量推理 L2 策略跨 workload 对比\n",
        "本报告只比较在线因果、同一 HBServe 请求源上可直接重放的 L2（Level 2 Cache，二级缓存）策略。"
        "NCU（NVIDIA Nsight Compute）是真实 RTX 4000 Ada 的三次测量中位数；模型是 HBServe + Memgen。"
        "Whole 表示一次完整请求，Prefill 表示提示阶段，Decode 表示该请求中连续全部解码步骤。"
        "WAPE（Weighted Absolute Percentage Error，加权绝对百分比误差）按逐 Decode step 的绝对字节差之和除以硬件字节和。"
        "MiB 为 2^20 B。主排序键是各 workload 中连续 Decode write 相对误差绝对值的最大值；"
        "数值接近不等于恢复了 NVIDIA 的真实 replacement/writeback 机制。\n",
        "## 跨 workload Decode write 排名\n",
        ranking_header + "\n".join(ranking_body) + "\n",
    ]
    for workload, data in workloads.items():
        reports.extend([
            f"## {workload} 全量推理\n",
            f"- 源 kernel：{data['aggregate']['source_kernels']:,}\n"
            f"- measurement kernel：{data['aggregate']['measurement_kernels']:,}\n"
            f"- 策略数：{len(data['summaries'])}\n",
            workload_table(workload, combined),
        ])
    reports.extend([
        "## 不进入本排名的探索\n",
        "timestamp sort（时间戳重排）、CTA/warp arrival frontier、MSHR（Miss Status Holding Register，未完成缺失状态表）、"
        "WTB（Write Transaction Buffer，写事务缓冲）、地址映射、partition 数和 HBFSim 时序会改变上游请求、并发或物理服务口径，"
        "不能伪装成同一请求流上的 L2 replacement/writeback 策略。kernel-boundary drain 和无法恢复精确源码的 producer-stream drain 也不进入部署候选排名。\n",
        "## 结论边界\n",
        "- 连续 Decode write 是主验收范围；Whole 可能被大体量 Prefill 掩盖。\n"
        "- 单步硬件 write 很小时，相对误差会被放大，因此表中同时保留 MiB 绝对差与逐步 WAPE。\n"
        "- `支持域完整=否` 表示存在 partial-dirty victim 的旧字节覆盖不完整；即使数值通过，也不能升级为严格硬件验收。\n"
        "- 本报告不会把未进入矩阵的非因果或时序探索记作“缺失的零误差策略”。\n",
    ])
    report_path = args.output_root / "REPORT.md"
    report_path.write_text("\n".join(reports), encoding="utf-8")
    receipt = {
        "status": "COMPLETE_COMBINED_FULL_INFERENCE_POLICY_COMPARISON_NOT_HARDWARE_ACCEPTANCE",
        "workloads": list(workloads),
        "policy_count": len(combined),
        "joint_pass_count": sum(row["all_workloads_joint_below_10_pct"] for row in combined),
        "inputs": pins,
        "combined_summary_sha256": sha256(args.output_root / "combined-policy-summary.csv"),
        "report_sha256": sha256(report_path),
        "hardware_accuracy_accepted": False,
    }
    (args.output_root / "finish.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
