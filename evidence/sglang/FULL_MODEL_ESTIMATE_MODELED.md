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

回放回执里与口径有关的那几个字段,两个点都是:

| 字段 | P32D2 | P128D2 |
|---|---|---|
| `status` | `PASS_COMPLETE_SAMPLED_MODEL_CACHE` | `PASS_COMPLETE_SAMPLED_MODEL_CACHE` |
| `complete_full_model` / `fully_exact` | `true` / `false` | `true` / `false` |
| `unsupported_launches` / `modeled_launches` | 0 / 700 | 0 / 1010 |
| `hardware_accuracy_accepted` | `false` | `false` |
| `allow_full_NCU_accuracy_comparison` | `false` | `false` |

## 1. 一次采集,两种口径

两个数据点都来自**已经存在的采集**,没有新增 GPU 时间。`--model-uncovered modeled`
只改变扩张阶段:未拟合类不再中止,而是变成带标签的 `numeric_modeled` launch。

| 点 | target | 精确 | 建模 | 建模原因 | 建模流量 | 流量占比 | 指令数 | 回放墙钟 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| P32D2 | 2060 | 1360 | 700(34.0%) | 350 缺模板 / 350 绑定歧义 | 1.23 GiB | 2.2% | 418,517,278 | 23.2 min |
| P128D2 | 2172 | 1162 | 1010(46.5%) | 686 缺模板 / 324 绑定歧义 | 10.47 GiB | 17.5% | 408,925,214 | 28.9 min |

"流量占比"按 profile 的 opcode 宽度 × 活跃 lane × grid 求和的字节口径计算。P128D2
的两列是第 4 节宽度校正之后的值(校正前为 3.76 GiB / 7.1%);P32D2 不受校正影响,
它的指令数逐位不变。两个点的墙钟不是同一条件:23.2 min 是单独跑,28.9 min 是与
P32D2 的校正后回放并发跑(同一台机,48 核)。

## 2. 两个点的全模型计数器

两者都是 `PASS_COMPLETE_SAMPLED_MODEL_CACHE`,6 个阶段(3 个 warmup + 3 个测量),
2060 / 2172 个 kernel 全部回放。

| 点 | L1 请求 / 命中 | L1 命中率 | L2 请求 / 命中 | L2 命中率 | DRAM read | DRAM write |
|---|---:|---:|---:|---:|---:|---:|
| P32D2 | 916,034,810 / 158,351,348 | 17.287% | 765,334,094 / 212,121,924 | 27.716% | 17,587,980,864 B | 136,111,232 B |
| P128D2 | 1,097,061,642 / 171,211,324 | 15.606% | 938,538,126 / 512,645,706 | 54.622% | 13,559,014,016 B | 77,980,544 B |

### 2.1 建模份额很小,而且几乎全被 L2 吸收

| 点 | 建模 launch | 占 L1 请求 | 占 L2 请求 | 占 DRAM read | 自身 L2 命中率 |
|---|---:|---:|---:|---:|---:|
| P32D2 | 700 | 4.32% | 4.33% | 0.0148%(2,611,200 B) | 99.56% |
| P128D2 | 1010 | 31.71% | 35.59% | 0.1192%(16,159,744 B) | 99.81% |

这与建模规则一致:每个 CTA 只在对象自己的地址范围内仿射走位,对象小、可复用,于是请求在
L2 层就被吸收。P128D2 的建模份额更大(launch 占 46.5%、L2 请求占 35.59%),但它在
DRAM 层的贡献仍然只有 0.12%。

P128D2 的 L1/L2 请求口径在宽度校正后变化很大而 DRAM 几乎不动,原因在命中数的分解里:
L1 命中在校正前后**完全相同**(171,211,324),新增的 225,290,240 条 L1 请求全是 miss;
它们在 L2 里命中 225,020,368 条。也就是说被校正放大的那段 GEMM 流量是流式权重读取——
不过 L1、被 L2 吸收——所以 L1 命中率从 19.64% 降到 15.61%、L2 命中率从 40.31% 升到
54.62%,而 DRAM read 只动 +0.0119%、write −0.54%。

### 2.2 逐阶段读数:两条物理上自洽的观察

`kernel_summary.csv` 按阶段拆分(该表本身没有 phase 列:用同目录 `semantic_cache_state.csv`
的 `phase`,或用 cache 的 `app.config` 里 `-kernel_N_llama_phase`),两个点的差别集中在 Prefill:

| 阶段 | 点 | L1 请求 | L1 命中率 | L2 命中率 | DRAM read (MiB) | DRAM write (MiB) |
|---|---|---:|---:|---:|---:|---:|
| Prefill | P32D2 | 152,326,954 | 5.05% | 37.83% | 2,742.9 | 60.7 |
| Prefill | P128D2 | 264,928,658 | 3.24% | 84.32% | 1,221.2 | 34.4 |
| 测量 Decode1/2(各) | P32D2 | 154,129,833 | 23.21% | 21.11% | 2,860.9 | 2.1 |
| 测量 Decode1/2(各) | P128D2 | 131,916,057 | 31.06% | 11.40% | 2,472.4 | 2.0 |
| warmup/Decode1/2(各) | P32D2 | 151,560,617 | 23.56% | 21.68% | 2,782.8 | 2.1 |
| warmup/Decode1/2(各) | P128D2 | 151,686,105 | 23.54% | 21.78% | 2,782.8 | 2.1 |

第 5 节的宽度校正只动了 P128D2 的 Prefill 一行(152,283,538 / 5.64% / 72.49% /
1,219.9 / 34.6),它的四个 Decode 行逐位不变——被校正的那个 GEMM 类只出现在 Prefill。

1. **更长 Prefill 的权重复用把 DRAM 读降了 2.25 倍。** token 数从 32 增到 128(4 倍),
   而 Prefill 的 DRAM read 从 2,742.9 MiB 降到 1,221.2 MiB,因为同一相位内每条权重
   行被更多 CTA 复用、在被驱逐前就命中(该相位 L2 命中率 37.83% → 84.32%)。这是缓存
   模型应有的行为,也是两个点总量差异的主因(P128D2 总 DRAM read 反而比 P32D2 低)。
2. **Decode 的 DRAM 读与 Prefill 长度无关。** 两个点的 4 个 warmup Decode 阶段读数
   一致到 0.1 MiB(均 2,782.8 MiB),符合"KV 常驻 L2"的预期:该配置下 KV 只有 MB 量级,
   而 L2 是 40 MiB,Prefill 写入的 KV 在 Decode 读回时不落到 DRAM。测量 Decode 两个点
   不同(2,860.9 vs 2,472.4 MiB),差的是**前面测量 Prefill 留在 L2 里的状态**,不是
   Decode 本身。

第 2 条同时也说明了本模型里 Decode 读数的敏感性来源:它由相位顺序与 L2 残留状态决定。
这正是 [精度报告](L2_CACHE_STRATEGY_ACCURACY_REPORT.md) 第 1 节把 Decode 写量与
L1/L2 命中率列为"未校准"的同一个机制。

### 2.3 这正是 `--partial` 会误导的地方

P32D2 有旧的仅覆盖部分回放(1360 个 launch),可与全模型直接对照:

| 口径 | L1 请求 | L2 请求 | L2 命中率 | DRAM read | DRAM write |
|---|---:|---:|---:|---:|---:|
| 全模型(2060) | 916,034,810 | 765,334,094 | 27.716% | 17,587,980,864 B | 136,111,232 B |
| 仅覆盖部分(1360) | 876,507,418 | 732,232,526 | 24.455% | 17,588,568,640 B | 135,537,920 B |
| 差值 | +39,527,392 | +33,101,568 | +3.26 pp | **−587,776 B(−0.0033%)** | +573,312 B(+0.42%) |

两个口径的 DRAM read 只差 **0.0033%**。也就是说在这个点上,"只覆盖 66% launch 的部分
回放"恰好给出了几乎相同的 DRAM read,而它本来是下界、不是全模型数字。差别不在于总量,
而在于**能不能这么写**:现在是全模型,建模份额、原因与校准随行记录在 `manifest.json`
与回放回执里;L2 命中率也从 24.455% 升到 27.716%(被建模流量喂得更满),这一项更接近
真实,但同样没有硬件分母。把一个 0.0033% 的巧合当成证据,就是这份文档要防的事。

## 3. 建模流量由谁贡献

按类聚合后,建模流量集中在真正搬数据的算子上,而不是零碎小核:

| 点 | 首位类 | 占建模流量 | 体积依据 |
|---|---|---:|---|
| P32D2 | `cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...>` | 71.0% | 拟合 census(实测) |
| P32D2 | `internal::gemvx` | 21.2% | 拟合 census(实测) |
| P128D2 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` | 86.7% | 拒绝记录实测宽度(16 B/lane) |
| P128D2 | `internal::gemvx` | 5.0% | 拟合 census(实测) |
| P128D2 | `ampere_bf16_s16816gemm_..._64x64_...` / `cutlass::Kernel2<...s16816gemm...>` | 2.5% / 2.4% | run 中位数(估计) |

体积依据有三种,都记在每条 profile 的 `modeling.volume_basis` 里,按证据强度排序:

1. `template_census_whole_grid_divided_by_grid`(实测):该类在本 run 里被拟合过,
   用其整网格 census 除以测量时的 grid;
2. `observed_refusal_width_records_per_cta`(实测):该类**自己**被拒的记录在
   projection 里按 opcode 宽度(B/lane)与 lane 数留了档,因此是实测宽度 × 实测 lane,
   再按该类记录数摊到每个 CTA。只有覆盖该类 ≥50% 记录时才采用(见第 4 节);
3. `run_calibrated_records_per_cta`(估计):该类从未拟合、也没有拒绝档,只能用自身
   采样记录数 × 本 run 的 phase 中位字节/指令。这是本方法**剩下**的估计环节,其跨类
   实测区间为 **4–512 B/指令**,已写入 `modeled_calibration.bytes_per_record_range`。

第 4 节记录第 2 种依据被补上之后,两个点的估计份额从多少降到多少。

## 4. 宽度校正:实测宽度取代 run 中位数

第 3 节第 3 条留下的那条不确定性——"真实访问是 16 B/lane 的类最多被低估 4 倍"——在
第二次扩张里被**实测**数据替代了一部分。证据本来就在采集回执里,没有新增 GPU 时间。

### 4.1 证据在哪

projection 在**拒绝**一条记录时仍按 opcode 给它分桶,桶里留着该 opcode 的 issue 宽度
(bytes per lane)和它在采样 CTA 上看到的 lane 数
(`projection_classification.rejection_classes[]`)。对一个从未拟合出模板的类,这是归档里
唯一留下它真实访存宽度的地方。`model_uncovered.observed_lane_census()` 读它,
`per_cta_volume()` 用它给出

```
该类 bytes/记录 = (width × lanes) / 被拒记录数
```

取代原来那个 run 中位数 128。方向也随实测:只 load 的类按 source mask 真正读出的 lane
计,只 store 的类按 active lane 计;`is_load` 与 `is_store` 同时置位的 atomic 计入写侧,
并在 profile 的 `directionless_opcodes` 里点名——它没有属于自己的方向可归。

### 4.2 一条覆盖门限,以及为什么它是两个实测值之间的分界

拒绝桶并不覆盖整类。P128D2 首位 GEMM 的桶覆盖 12000/12640 = **94.9%** 的记录;
`at::native::reduce_kernel` 只有 10/644 = **1.6%**——它剩下的 634 条记录是被 projection
**接受**的(该核因为那 10 条 atomic 而整体 UNLOWERED;已接受的记录只留计数、不留宽度)。

所以只有覆盖率 ≥ `OBSERVED_REFUSAL_COVERAGE_MIN`(0.5)时才采用实测宽度,否则保留 run
中位数:低于门限时,桶描述的是一段与被拒原因同源的残渣,把它的宽度外推到整类会是"用
一个错误宽度换另一个"。这条门限不是调参,它在两个点上都有实测支撑——把 GEMM 收进来、
把 atomic reduce 挡在外面。第一次实现没有它,`reduce_kernel` 的 6 个 launch 直接降级为
unsupported(2054/2060),这就是它拦住的错。

### 4.3 P32D2 是零变化对照,P128D2 只动一个类

| 点 | 走实测宽度的 launch | 类 | 校正前 | 校正后 | 倍数 |
|---|---:|---|---:|---:|---:|
| P32D2 | 0 | 没有类达到门限 | — | — | — |
| P128D2 | 56 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` | 161,792 B/CTA | 621,281 B/CTA | **3.84×** |

P32D2:全部 2060 条 profile 的 `template` / `kernel` / `model` / `source` / `status`
逐字段相同(只有 700 条建模行的 `modeling` 证据块多了新字段),回放后 6 组计数逐个一致。
P128D2:2172 条里只有那 56 条的 `template` 变了,`kernel` / `model` / `source` /
`status` 一个都没变。

那个 3.84× 不是"更大所以更准",它可以独立核对:P128D2 首位 GEMM 的 tile 是 128×64、
K=1536、bf16,一个 CTA 要读完整 K 的 A 瓦片与 B 瓦片,128×1536×2 + 64×1536×2 =
**589,824 B**,与该类自己采样的记录算出的 368,640 lane × 16 B ÷ 10 CTA =
**589,824 B/CTA** 完全相同;该 launch 自己的读对象范围是 393,216 + 393,216 + 1,024 +
1,024 B,足以容纳(所以没有触发 `issue_budget_truncated`)。命令见第 5 节。

每个依据各占多少(记在 manifest 的 `modeled_volume_basis`,单位是每 CTA 请求字节之和):

| basis | P32D2 launch / 字节 | P128D2 launch / 字节 |
|---|---:|---:|
| `template_census_whole_grid_divided_by_grid`(实测) | 350 / 22,358,304 | 324 / 9,862,272 |
| `observed_refusal_width_records_per_cta`(实测) | 0 / 0 | 56 / 34,791,751 |
| `run_calibrated_records_per_cta`(估计) | 350 / 2,929,555 | 630 / 19,919,405 |
| **其中估计份额** | **11.58%** | **74.61% → 30.85%** |

按第 1 节的发射字节口径,P128D2 的建模流量从 3.76 GiB 涨到 **10.47 GiB**(占全 run
7.06% → 17.45%),P32D2 不变(1.23 GiB / 2.20%)。P128D2 首位类因此从"占建模流量 63.0%、
依据是 run 中位数"变成"占建模流量 86.7%、依据是它自己实测的 16 B/lane"。

### 4.4 校正后的 P128D2

P32D2 与第 2 节逐位相同(它就是零变化对照)。P128D2 的变化全部落在 Prefill:

| 口径 | 校正前 | 校正后 | 差 |
|---|---:|---:|---:|
| 内存指令 | 394,860,254 | 408,925,214 | +3.6% |
| L1 请求 / 命中 | 871,771,402 / 171,211,324 | 1,097,061,642 / 171,211,324 | 请求 +25.8%,命中**不变** |
| L2 请求 / 命中 | 713,498,766 / 287,625,338 | 938,538,126 / 512,645,706 | 请求 +31.5% |
| DRAM read | 13,557,406,848 B | 13,559,014,016 B | **+1,607,168 B(+0.0119%)** |
| DRAM write | 78,403,968 B | 77,980,544 B | −423,424 B(−0.54%) |
| Prefill L2 命中率 | 72.49% | 84.32% | +11.83 pp |
| 回放墙钟 | 21.2 min(单独) | 28.9 min(与 P32D2 并发) | 不同条件,不作差 |

L1 命中数在校正前后**完全相同**,是这张表最该看的一行:新增的 225,290,240 条 L1 请求
全是 miss,其中 225,020,368 条在 L2 命中。也就是说被放大的那段 GEMM 流量是流式权重
读取——不过 L1、被 L2 吸收——所以 L1/L2 命中率口径动得很大(19.64% → 15.61%、
40.31% → 54.62%),而 DRAM read 只动 **+0.0119%**、write 掉 0.54%。谁把这份估算当硬件
对照,看到的是 L1 命中率变了近 4 个百分点;而这一项本来就没有 NCU 分母,它变的是
"建模流量占多少",不是"与硬件差多少"。

### 4.5 剩下的估计

P128D2 仍有 **30.85%** 的建模体积靠 run 中位数,主要是两个 GEMM(各 2.5%)与 flashinfer
attention 家族。它们在回执里没有宽度证据,而且不是被 projection 拒的:projection 接受了
它们的记录,拒绝来自拟合级联(`sample ordered lane mismatch`、
`categorical-y training or independent holdout missing`),而回执只给**被拒**记录留宽度档
(`template_adapter_r4/diagnostics.py` 的 `RejectionCensus.observe()` 在 `error is None`
时只累加计数就返回)。

要消掉这部分,需要采样器同时给**被接受**的记录留宽度档案。那是一处采样器改动
(`diagnostics.py`),而该文件被采样回执逐文件 pin 在 `source_pins` 里,所以它意味着
"改采样器 + 重新采样",扩张侧补不出来。在那之前,`modeled_estimated_share` 就是本归档
对这部分不确定性的公开记账。

## 5. 复现

```bash
# 扩张(CPU,秒级):把未拟合类补成带标签的 numeric_modeled
python3 -B integrations/sglang/memgen-adapter/expand_profiles.py \
  --sample-output  out/collect-20260923T165342Z/runs/qwen25_1p5b-p32-d2-collect/followthrough/sample \
  --layer-bindings out/collect-20260923T165342Z/runs/qwen25_1p5b-p32-d2-collect/followthrough/plan/layer-bindings.json \
  --output out/modeled-p32d2-width --model-uncovered modeled

# 全模型回放(CPU,约 25 / 29 分钟)
python3 -B integrations/sglang/memgen-adapter/run_memgen.py \
  --expanded out/modeled-p32d2-width \
  --binary out/collect-20260923T165342Z/engine/hbserve \
  --output out/modeled-p32d2-width-cache
```

P128D2 把两处 `20260923T165342Z` 换成 `20260923T185728Z`、`p32-d2` 换成 `p128-d2`、
`-p32d2-` 换成 `-p128d2-` 即可。一步到位的等价写法(需要重新采集时):
`./memgen collect --model-uncovered modeled --work <新目录> --model qwen25_1p5b --prefill-length 32 --decode-steps 2 --gpu-index 1`。
用 `--resume` 可以复用已通过的 census,只重跑 job 2。

计数器在 `<cache>/model/kernel_summary.csv`,建模份额在
`<expanded>/manifest.json` 与 `<cache>/finish.json`;第 4 节每个数字都从这三处读出。
按阶段拆分要借 `<cache>/model/semantic_cache_state.csv` 的 `phase` 列(`kernel_summary.csv`
本身没有 phase 列)。

第 4.3 节那条几何核对(实测 589,824 B/CTA = A 瓦片 + B 瓦片):

```bash
python3 -B -c "
import json,sys,glob; sys.path.insert(0,'integrations/sglang/memgen-adapter')
import expand_profiles as ep, model_uncovered as mu
s=glob.glob('out/collect-20260923T185728Z/runs/*/followthrough/sample')[0]
h=glob.glob(s+'/host/process-*/')[0]
b=ep.Binder(json.loads(open(h+'module_calls.json').read()),json.loads(open(h+'tensor_metadata.json').read()))
L=ep.launch_rows(glob.glob(s+'/observer/process-*/launch-journal.jsonl')[0])
t=dict(L[(1,24)],epoch_launch_ordinal=24)
print([hi-lo for lo,hi in mu.object_spans(b.contexts(t['call_id']),'inputs')])
print('A+B tile', 128*1536*2+64*1536*2)
"
```
