# 全模型估算(含显式建模类)— P32D2 / P128D2

快照日期:2026-09-24(Asia/Dubai)
状态:`FULL_MODEL_ESTIMATE_COMPLETE; MODELED_SHARE_DECLARED; HARDWARE_ACCURACY_NOT_ACCEPTED`

本文记录同一采样-扩展-回放链在**全模型覆盖**下的一次实测结果,以及其中被显式
建模的部分占了多大份额。它**不是**精度验收:本归档里 P32D2 没有同范围 NCU 参考,
因此没有任何一行可以写成"与硬件误差 x%"。精度口径仍以
`validation/p32d2_branch_status.csv` 与
[L2 缓存策略与 NCU 精度报告](L2_CACHE_STRATEGY_ACCURACY_REPORT.md) 为准。

- 判据与不可主张项:[扩张机制与拒绝原因](../EXPANSION_MECHANISM_AND_REFUSALS.md) 第 5 节
- 为何原来闭合不了:[P32D2 覆盖发现](../P32D2_COVERAGE_FINDING.md)
- 复现命令:[运行手册](../RUNBOOK.md) "Or complete the model"

## 1. 一次采集,两种口径

两个数据点都来自**已经存在的采集**,没有新增 GPU 时间。`--model-uncovered modeled`
只改变扩张阶段:未拟合类不再中止,而是变成带标签的 `numeric_modeled` launch。

| 点 | target | 精确 | 建模 | 建模原因 | 建模流量 | 流量占比 | 指令数 | 回放墙钟 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| P32D2 | 2060 | 1360 | 700(34.0%) | 350 缺模板 / 350 绑定歧义 | 1.23 GiB | 2.2% | 418,517,278 | 23.2 min |
| P128D2 | 2172 | 1162 | 1010(46.5%) | 686 缺模板 / 324 绑定歧义 | 3.76 GiB | 7.1% | 见回放回执 | 见回放回执 |

"流量占比"按 profile 的 opcode 宽度 × 活跃 lane × grid 求和的字节口径计算。

## 2. P32D2 全模型计数器

内存指令 418,517,278 条,2060 个 kernel 全部回放,状态
`PASS_COMPLETE_SAMPLED_MODEL_CACHE`。

| 口径 | L1 请求 / 命中 | L1 命中率 | L2 请求 / 命中 | L2 命中率 | DRAM read | DRAM write |
|---|---:|---:|---:|---:|---:|---:|
| 全模型(2060) | 916,034,810 / 158,351,348 | 17.287% | 765,334,094 / 212,121,924 | 27.716% | 17,587,980,864 B | 136,111,232 B |
| 其中建模的 700 | 39,527,392 | — | 33,101,568 | 99.56%(自身) | 2,611,200 B | — |
| 仅覆盖部分(1360) | 876,507,418 / 150,343,188 | 17.153% | 732,232,526 / 179,068,404 | 24.455% | 17,588,568,640 B | 135,537,920 B |

**建模部分是全模型里很小、且几乎全被 L2 吸收的一段。** 700 个建模 launch 贡献了
全 run 4.32% 的 L1 请求和 4.33% 的 L2 请求,但只贡献 **0.0148%** 的 DRAM read;
它们自身的 L2 命中率为 **99.56%**(全 run 27.72%)。这与建模规则一致:每个 CTA
只在对象自己的地址范围内仿射走位,对象小、可复用,于是请求在 L2 层就被吸收。

**这正是 `--partial` 会误导的地方。** 两个口径的 DRAM read 只差
**−587,776 B(−0.0033%)**,DRAM write 差 **+573,312 B(+0.42%)**。也就是说这个点上
"只覆盖 66% launch 的部分回放"恰好给出了几乎相同的 DRAM read,而它本来是下界、
不是全模型数字。差别不在总量,而在**能不能这么写**:现在它是全模型,建模份额、原因
与校准都随行记录在 `manifest.json` 与回放回执里,命中率也从 24.455% 升到 27.716%
(L2 层被建模流量"喂"得更满,这一项更接近真实,但同样没有硬件分母)。

## 3. 建模流量由谁贡献

按类聚合后,建模流量集中在真正搬数据的算子上,而不是零碎小核:

| 点 | 首位类 | 占建模流量 | 体积依据 |
|---|---|---:|---|
| P32D2 | `cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...>` | 71.0% | 拟合 census(实测) |
| P32D2 | `internal::gemvx` | 21.2% | 拟合 census(实测) |
| P128D2 | `ampere_bf16_s1688gemm_..._sliced1x2` | 63.0% | 类自身采样记录数 × 本 run 中位数 |
| P128D2 | `internal::gemvx` | 14.0% | 拟合 census(实测) |

体积依据只有两种,都记在每条 profile 的 `modeling.volume_basis` 里:
`template_census_whole_grid_divided_by_grid`(实测,该类有拟合模板)和
`run_calibrated_records_per_cta`(估计,该类从未拟合,用自身的采样记录数 × 本 run
的 phase 中位字节/指令)。后者是本方法唯一的估计环节,其跨类实测区间为 **4–512 B/指令**,
已写入 `modeled_calibration.bytes_per_record_range`:真实访问是 16 B/lane 的类最多被
低估 4 倍,字节宽的类会被高估。

## 4. 复现

```bash
# 扩张(CPU,秒级):把未拟合类补成带标签的 numeric_modeled
python3 -B integrations/sglang/memgen-adapter/expand_profiles.py \
  --sample-output  out/collect-20260923T165342Z/runs/qwen25_1p5b-p32-d2-collect/followthrough/sample \
  --layer-bindings out/collect-20260923T165342Z/runs/qwen25_1p5b-p32-d2-collect/followthrough/plan/layer-bindings.json \
  --output out/modeled-p32d2-test --model-uncovered modeled

# 全模型回放(CPU,约 23 分钟)
python3 -B integrations/sglang/memgen-adapter/run_memgen.py \
  --expanded out/modeled-p32d2-test \
  --binary out/collect-20260923T165342Z/engine/hbserve \
  --output out/modeled-p32d2-cache
```

一步到位的等价写法(需要重新采集时):
`./memgen collect --model-uncovered modeled --work <新目录> --model qwen25_1p5b --prefill-length 32 --decode-steps 2 --gpu-index 1`。
用 `--resume` 可以复用已通过的 census,只重跑 job 2。

计数器在 `<cache>/model/kernel_summary.csv`,建模份额在
`<expanded>/manifest.json` 与 `<cache>/finish.json`。
