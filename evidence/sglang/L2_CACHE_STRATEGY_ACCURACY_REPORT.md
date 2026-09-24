# HBServe + Memgen L2 缓存策略与 NCU traffic 精度综合报告

快照日期：2026-09-21（Asia/Dubai）  
状态：`CURRENT_EVIDENCE_COMPLETE; REQUESTED_INDEPENDENT_MATRIX_INCOMPLETE_2_OF_5; HARDWARE_ACCURACY_NOT_ACCEPTED`

## 1. 执行结论

本报告回答两个问题：已经探索过哪些 L2（Level 2 Cache，二级缓存）策略；在统一的 1.5B 模型条件下，HBServe 动态访存生成加 Memgen 缓存回放与 NCU（NVIDIA Nsight Compute）真实硬件 DRAM traffic 的误差是多少。

结论先行：

1. **读流量目前稳定，写流量仍不稳定。** 在两个已经独立闭合的目标 workload 中，普通 LRU（Least Recently Used，最近最少使用）策略的完整请求 DRAM read 误差为 P128D2 `+0.54%`、P128D16 `+0.20%`；DRAM write 则从 P128D2 的 `+0.91%` 扩大到 P128D16 的 `+14.55%`。
2. **P128D2 的完整请求写误差小不是稳健性证据。** 同一 P128D2 中，Prefill write 为 `+0.40%`，连续两步 Decode write 已为 `+39.17%`；D1 为 `-27.82%`，D2 为 `+1148.67%`。完整请求由大体量 Prefill 主导，掩盖了 Decode 写服务错误。
3. **长 Decode 把缺口暴露出来。** P128D16 普通 LRU 的连续 Decode 硬件写量为 `6.29 MiB`，模型为 `47.09 MiB`，绝对多写 `40.80 MiB`，相对误差 `+648.63%`；完整请求 write 因而变为 `+14.55%`。
4. **phase11 主动清脏没有解决问题。** 它在 P128D2/P128D16 的连续 Decode write 误差分别为 `+39.20%`/`+648.62%`，与普通 LRU 实质相同；不能把它称为更准确的默认策略。
5. **q16 clean-first 保留策略被明确否决。** 它虽然保持 read 在约 1% 内，却使 P128D2/P128D16 完整请求 write 分别少 `92.37%`/`90.51%`，说明替换与 dirty 保留规则可以强烈改变写量，但该候选方向错误。
6. **用户指定的五个独立矩阵点只完成了 2/5。** P128D2 与 P128D16 有独立 profile、完整模型和三次 NCU 参考；P32D2 尚无同模型、同框架证据；P128D4/P128D8 只有从一次 P128D16 的逐步范围组合出的前缀诊断，不能冒充独立 workload 验收。
7. **当前目标矩阵没有同范围 NCU L1/L2 hit-rate oracle。** 因而本报告只给出可同分母比较的 DRAM read/write。Memgen 自身能输出命中率，但不能在缺少硬件分母时把它写成“L1/L2 accuracy 已通过”。

当前建议：把普通 LRU `disabled` 保留为最简单的工程控制策略；继续使用 HBServe + Memgen 估算已测范围的聚合 DRAM read；在新的独立证据出现前，不把 DRAM write、逐步 Decode write、L1/L2 命中率或未测 context 当成已校准结果。

## 2. 术语、误差和实验合同

### 2.1 术语

- **HBServe**：从稀疏动态采样及 packed profile（打包配置文件）重建完整推理的地址级 memory-SASS 请求流。本流程不持久化完整逐地址 raw trace。
- **Memgen**：消费 HBServe 请求流并模拟 L1/L2 状态、命中、脏数据和 DRAM 服务字节的缓存后端。
- **NCU**：NVIDIA Nsight Compute，本报告中的硬件 DRAM read/write 计数来源。
- **P\(n\)D\(m\)**：一次 batch 1 推理包含 \(n\) 个 Prefill（提示处理）token，随后连续执行 \(m\) 个 Decode（逐 token 解码）step。
- **whole**：完整测量范围；**Prefill**：只含提示处理；**Decode**：连续全部解码步；**D1/D2/...**：单独的逐步 NCU 范围。
- **MiB/GiB**：二进制单位，分别为 \(2^{20}\) B 和 \(2^{30}\) B。表中保留两位小数。

带符号相对误差定义为：

`100 × (model_bytes - NCU_bytes) / NCU_bytes`

正数表示模型多估，负数表示模型少估；绝对误差是两者字节差的绝对值。沿用当前项目的 DRAM traffic 数值门：每个受验范围的 read 和 write 相对误差都必须严格小于 10%。数值门通过、模型语义支持域完整、以及真实 NVIDIA 物理机制被识别，是三个不同结论。

### 2.2 当前统一目标 workload

| 项目 | 固定条件 |
|---|---|
| 模型 | Qwen2.5-1.5B-Instruct，BF16（Brain Floating Point 16 位），28 层 |
| 框架 | SGLang 0.4.10 + FlashInfer |
| 硬件 | NVIDIA RTX 4000 Ada Generation，48 个 SM（Streaming Multiprocessor，流式多处理器） |
| 执行 | batch 1、TP1（Tensor Parallel，张量并行度 1）、eager、关闭 CUDA Graph、KV 容量 1024 token |
| P128 输入 | 固定 prompt token IDs `1000..1127`；D16 使用固定交替解码 token `944/291` |
| 硬件统计 | 每个正式范围运行 3 次，表中使用中位数；独立范围的中位数不互相相加来冒充另一个独立范围 |

P128D2 的源人口为 2,173 个有序 kernel，其中测量段 1,086 个；P128D16 的源人口为 11,945 个，其中测量段 5,972 个。两者均包含一次同条件 warmup（预热）再进入 measurement（测量），缓存状态跨 kernel 保留。

D16 的三组参数没有使用 D16 计数重新选择，最终回执也记录 `parameters_selected_using_D16=false`；但研究过程中 D16 的硬件结果并非对开发者隐藏，所以它应称为“未在 D16 重选参数的迁移诊断”，不能追称严格 blind holdout（盲留出验收）。

### 2.3 当前统一 Memgen 缓存合同

- L2 总容量 40 MiB，128 B cache line，每行 4 个独立 32 B sector，16-way。
- 工程组织为 20 个逻辑 slice，每 slice 1,024 个 set，总计 20,480 个 set；使用现有 XOR 组索引。
- sector 分别保存 valid/dirty 状态；采用 write-back、write-allocate；脏 victim 只写实际 dirty sector。
- 缓存跨 kernel、跨 Prefill/Decode 持续存在；不在 kernel 边界或请求末尾强制 dirty drain。
- HBServe 请求按已生成的有序流进入后端；当前不是具备真实 GPU 到达并发、队列和完成时序的 cycle-accurate 模型。
- 策略名 `disabled` 只表示**禁用额外主动清脏/保留启发式**，并不表示关闭缓存；其替换控制是普通 LRU。

策略标签中，`clean-first` 表示读分配优先选择干净 victim；`qN` 表示每个 set 的 dirty-line 预算为 N；`ageN` 表示同一 set 的新行读分配年龄阈值为 N；`clean-retained` 表示脏 sector 已向下层发出但 tag 与有效数据继续留在缓存。`phase11` 只是这组 q4/age16 规则的版本名，不是第 11 个推理 phase。XOR（exclusive OR，按位异或）是当前 set/slice 组索引的一部分。`holdout` 是未参与对应规则拟合的留出 case。

其中 40 MiB、128 B/32 B、sector valid/dirty、脏 sector 逐出、跨 kernel 状态和无强制末尾排脏可以继续作为功能合同；20 个逻辑 slice、XOR、LRU 以及所有 dirty 释放规则仍是工程模型，不等于已经恢复出 NVIDIA 的真实物理实现。

### 2.4 三类证据不能混表

1. **独立完整 workload**：自身完成 profile、HBServe 展开、Memgen 回放以及同条件 NCU；可用于本 workload 的正式误差表。
2. **前缀组合诊断**：把同一次 P128D16 的 Prefill 和前 \(n\) 个逐步范围按字节相加；用于看误差趋势，但不是独立 P128D\(n\) 运行，也不包含短请求可能不同的结束行为。
3. **历史异构证据**：不同框架、模型精度、模型尺寸或缓存合同的数据；只用于解释策略探索，不能补当前目标矩阵的空格。

## 3. 用户指定矩阵的真实完成度

| 请求点 | 同一 1.5B/SGLang/BF16 条件下的独立证据 | 可给出的结论 |
|---|---:|---|
| P32D2 basic | **部分** | 现已有同栈的完整模型估算(1360 精确 + 700 建模,建模占流量 2.2%,见 [全模型估算](FULL_MODEL_ESTIMATE_MODELED.md)),但**仍无 NCU oracle**,不能作精度验收 |
| P128D2 | **完整** | 独立 profile、HBServe、Memgen、三次 NCU；可正式报告 |
| P128D4 | **缺失** | 只有 P128D16 前 4 步的组合诊断，不能验收 |
| P128D8 | **缺失** | 只有 P128D16 前 8 步的组合诊断，不能验收 |
| P128D16 | **完整** | 独立 profile、HBServe、三策略 Memgen、三次 NCU；可正式报告 |

因此，严格意义上的 requested matrix 是 **2/5 independently complete**。旧 `m1Bp32d2` 对象是 Llama-3.2-1B BF16，并且当时没有可用 NCU hardware oracle；它既不是 Qwen2.5-1.5B，也不能作为本报告的 P32D2 basic。P32D2 的完整模型估算填补的是"同栈、同框架的完整模型数字"这一格,不填补 NCU 那一格:第 1 节的精度结论与本节验收口径都不因它改变。

## 4. 独立完整 workload 的 NCU 对比

### 4.1 P128D2：四种当前可复核策略

下表中的 read 以 GiB 展示，绝对差以 MiB 展示；write 全部以 MiB 展示。每个单元格都是同一列定义，不混用模型内部 hit-rate。

| L2 策略 | 范围 | DRAM read NCU | DRAM read 模拟 | 绝对差 | 相对误差 | DRAM write NCU | DRAM write 模拟 | 绝对差 | 相对误差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 普通 LRU `disabled` | whole | 8.64 GiB | 8.68 GiB | 47.89 MiB | +0.54% | 268.54 MiB | 270.99 MiB | 2.44 MiB | +0.91% |
| 普通 LRU `disabled` | Prefill | 2.90 GiB | 2.92 GiB | 20.84 MiB | +0.70% | 264.04 MiB | 265.10 MiB | 1.06 MiB | +0.40% |
| 普通 LRU `disabled` | Decode 2 | 5.73 GiB | 5.76 GiB | 25.72 MiB | +0.44% | 4.23 MiB | 5.89 MiB | 1.66 MiB | +39.17% |
| clean-first + q4/age16 `phase11` | whole | 8.64 GiB | 8.68 GiB | 44.37 MiB | +0.50% | 268.54 MiB | 271.42 MiB | 2.87 MiB | +1.07% |
| clean-first + q4/age16 `phase11` | Prefill | 2.90 GiB | 2.92 GiB | 17.41 MiB | +0.59% | 264.04 MiB | 265.53 MiB | 1.49 MiB | +0.56% |
| clean-first + q4/age16 `phase11` | Decode 2 | 5.73 GiB | 5.76 GiB | 25.63 MiB | +0.44% | 4.23 MiB | 5.89 MiB | 1.66 MiB | +39.20% |
| clean-first q16、无主动清脏 | whole | 8.64 GiB | 8.64 GiB | 1.12 MiB | -0.01% | 268.54 MiB | 20.50 MiB | 248.05 MiB | -92.37% |
| clean-first q16、无主动清脏 | Prefill | 2.90 GiB | 2.88 GiB | 25.94 MiB | -0.87% | 264.04 MiB | 19.80 MiB | 244.24 MiB | -92.50% |
| clean-first q16、无主动清脏 | Decode 2 | 5.73 GiB | 5.76 GiB | 23.49 MiB | +0.40% | 4.23 MiB | 0.70 MiB | 3.53 MiB | -83.54% |
| 普通 LRU + q4/age16 cleaner | whole | 8.64 GiB | 8.68 GiB | 47.89 MiB | +0.54% | 268.54 MiB | 271.42 MiB | 2.87 MiB | +1.07% |
| 普通 LRU + q4/age16 cleaner | Prefill | 2.90 GiB | 2.92 GiB | 20.84 MiB | +0.70% | 264.04 MiB | 265.53 MiB | 1.49 MiB | +0.56% |
| 普通 LRU + q4/age16 cleaner | Decode 2 | 5.73 GiB | 5.76 GiB | 25.72 MiB | +0.44% | 4.23 MiB | 5.89 MiB | 1.66 MiB | +39.20% |

逐步 write 揭示完整请求中的抵消：普通 LRU 的 D1 是 NCU `4.08 MiB`、模型 `2.94 MiB`、误差 `-27.82%`；D2 是 NCU `0.24 MiB`、模型 `2.94 MiB`、误差 `+1148.67%`。phase11 对应为 `-27.81%` 和 `+1148.86%`。这不是 1.66 MiB 的 Decode 绝对差“突然变成 1148%”，而是 D2 硬件分母只有约 0.24 MiB。

尽管普通 LRU 与 phase11 的 whole read/write 数值均小于 10%，两者仍标记为 `MODEL_DOMAIN_INCOMPLETE`：普通 LRU 全流存在 9 个 partial dirty victim sector、208 B 未覆盖旧字节；phase11 为 7 个 sector、168 B。缺失字节很小，不足以解释 MiB 级 traffic 差额，但它使严格语义支持域不能声明闭合。

### 4.2 P128D16：三种策略的独立迁移结果

| L2 策略 | 范围 | DRAM read NCU | DRAM read 模拟 | 绝对差 | 相对误差 | DRAM write NCU | DRAM write 模拟 | 绝对差 | 相对误差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 普通 LRU `disabled` | whole | 48.90 GiB | 49.00 GiB | 98.02 MiB | +0.20% | 270.84 MiB | 310.25 MiB | 39.41 MiB | +14.55% |
| 普通 LRU `disabled` | Prefill | 2.90 GiB | 2.92 GiB | 20.39 MiB | +0.69% | 265.27 MiB | 263.16 MiB | 2.11 MiB | -0.80% |
| 普通 LRU `disabled` | Decode 16 | 46.00 GiB | 46.07 GiB | 77.00 MiB | +0.16% | 6.29 MiB | 47.09 MiB | 40.80 MiB | +648.63% |
| clean-first + q4/age16 `phase11` | whole | 48.90 GiB | 48.99 GiB | 93.93 MiB | +0.19% | 270.84 MiB | 312.82 MiB | 41.98 MiB | +15.50% |
| clean-first + q4/age16 `phase11` | Prefill | 2.90 GiB | 2.92 GiB | 17.06 MiB | +0.57% | 265.27 MiB | 265.73 MiB | 0.46 MiB | +0.17% |
| clean-first + q4/age16 `phase11` | Decode 16 | 46.00 GiB | 46.07 GiB | 76.25 MiB | +0.16% | 6.29 MiB | 47.09 MiB | 40.80 MiB | +648.62% |
| clean-first q16、无主动清脏 | whole | 48.90 GiB | 48.93 GiB | 30.89 MiB | +0.06% | 270.84 MiB | 25.71 MiB | 245.13 MiB | -90.51% |
| clean-first q16、无主动清脏 | Prefill | 2.90 GiB | 2.88 GiB | 27.14 MiB | -0.91% | 265.27 MiB | 24.64 MiB | 240.63 MiB | -90.71% |
| clean-first q16、无主动清脏 | Decode 16 | 46.00 GiB | 46.05 GiB | 57.41 MiB | +0.12% | 6.29 MiB | 1.06 MiB | 5.23 MiB | -83.08% |

普通 LRU 的三次 whole write 是 `270.75/270.88/270.84 MiB`，跨次极差约 `0.13 MiB`，远小于模型 `39.41 MiB` 的绝对差；NCU whole 噪声不能解释 `+14.55%`。连续 Decode write 的三次硬件值波动更大，约 `5.82/7.38/6.29 MiB`，但模型与中位数仍相差 `40.80 MiB`，也不能由重复波动解释。

D16 最终回执的 source-sector residual 和 dirty-version residual 均为 0；但仍有 32 个 partial dirty victim sector、760 B 未覆盖旧字节（所有三策略合计），因此 `semantic_admission=false`。这同样不是 40.80 MiB gap 的数值解释，而是严格支持域边界。

## 5. P128D2/D4/D8/D16 前缀组合诊断

本节专门满足“看 D2、D4、D8、D16 趋势”的需求，但必须降级解释：下表把一次 P128D16 的独立逐步 NCU 中位数和相同回放的 phase 行相加。它没有重新执行短请求、没有为每个 D 长度重新 profile，也没有重新测量独立 whole/Decode range。表中的 `Decode write 误差` 只对前 \(n\) 步逐步字节求和。

| 策略 | 组合点 | DRAM read NCU / 模拟 / 绝对差 / 误差 | DRAM write NCU / 模拟 / 绝对差 / 误差 | Decode write 误差 | 证据等级 |
|---|---|---:|---:|---:|---|
| 普通 LRU | P128D2 | 8.64 / 8.68 GiB / 46.60 MiB / +0.53% | 268.63 / 269.05 MiB / 0.41 MiB / +0.15% | +42.53% | 前缀诊断 |
| 普通 LRU | P128D4 | 14.39 / 14.44 GiB / 53.22 MiB / +0.36% | 269.01 / 274.93 MiB / 5.92 MiB / +2.20% | +161.21% | 前缀诊断 |
| 普通 LRU | P128D8 | 25.89 / 25.96 GiB / 67.21 MiB / +0.25% | 269.78 / 286.70 MiB / 16.93 MiB / +6.27% | +346.13% | 前缀诊断 |
| 普通 LRU | P128D16 | 48.90 / 49.00 GiB / 96.18 MiB / +0.19% | 271.29 / 310.25 MiB / 38.96 MiB / +14.36% | +593.70% | 前缀诊断 |
| phase11 | P128D2 | 8.64 / 8.68 GiB / 43.18 MiB / +0.49% | 268.63 / 271.62 MiB / 2.99 MiB / +1.11% | +42.53% | 前缀诊断 |
| phase11 | P128D4 | 14.39 / 14.44 GiB / 49.70 MiB / +0.34% | 269.01 / 277.50 MiB / 8.49 MiB / +3.16% | +161.20% | 前缀诊断 |
| phase11 | P128D8 | 25.89 / 25.96 GiB / 63.50 MiB / +0.24% | 269.78 / 289.27 MiB / 19.50 MiB / +7.23% | +346.13% | 前缀诊断 |
| phase11 | P128D16 | 48.90 / 48.99 GiB / 92.10 MiB / +0.18% | 271.29 / 312.82 MiB / 41.53 MiB / +15.31% | +593.69% | 前缀诊断 |
| q16 clean-first | P128D2 | 8.64 / 8.63 GiB / 3.00 MiB / -0.03% | 268.63 / 25.32 MiB / 243.31 MiB / -90.58% | -83.64% | 前缀诊断 |
| q16 clean-first | P128D4 | 14.39 / 14.39 GiB / 1.44 MiB / +0.01% | 269.01 / 25.37 MiB / 243.64 MiB / -90.57% | -83.81% | 前缀诊断 |
| q16 clean-first | P128D8 | 25.89 / 25.91 GiB / 10.75 MiB / +0.04% | 269.78 / 25.48 MiB / 244.30 MiB / -90.56% | -84.13% | 前缀诊断 |
| q16 clean-first | P128D16 | 48.90 / 48.93 GiB / 29.06 MiB / +0.06% | 271.29 / 25.71 MiB / 245.58 MiB / -90.52% | -84.32% | 前缀诊断 |

这个诊断显示：普通 LRU/phase11 的 read 一直在 0.6% 内；总 write 从 D2 到 D8 仍可能因大体量 Prefill 而落在 10% 门内，但 Decode-only write 从一开始就失败，并随步数累积。到 D16，完整请求 write 也越过 10%。q16 的 read 同样好，却把 write 压低约 90%，再次证明“总 read 接近”不能验证 dirty/writeback 机制。

前缀组合 P128D16 的 write `+14.36%` 与独立 whole 范围的 `+14.55%` 不完全相同；这是逐步中位数之和与独立 whole 中位数口径不同的正常结果，也是本节不能升级为独立验收的直接例子。

## 6. 已探索的 L2 策略全集与处置

下面的“面板”指由多个微基准和代表 kernel 组成的固定测试集合；它不是本报告的 SGLang 完整 workload。`通过数/总数` 只表示该面板数值门，不能跨证据域当作 full-inference accuracy。

### 6.1 历史完整推理基线

下表来自另一套历史栈：Qwen2.5-1.5B-Instruct Q8_0、llama.cpp、每点独立 profile、固定 D32。它可以说明 naïve 单层缓存与旧两级 Memgen 的差异，但不能补当前 SGLang/BF16 的 P32/P128 矩阵。

| 历史 workload | naïve DRAM read/write 误差 | 两级 Memgen L1/L2 命中率误差 | 两级 Memgen DRAM read/write 误差 |
|---|---:|---:|---:|
| P64D32 | 0.48% / 4.55% | 1.02% / 22.71% | 0.06% / 4.55% |
| P128D32 | 0.90% / 3.64% | 1.09% / 12.53% | 0.05% / 3.63% |
| P256D32 | 1.85% / 2.66% | 1.48% / 8.31% | 0.05% / 2.65% |
| P512D32 | 3.85% / 25.13% | 2.32% / 5.62% | 0.21% / 25.13% |
| P1024D32 | 7.81% / 23.07% | 4.10% / 1.14% | 1.08% / 23.07% |

两级 Memgen 显著改善了历史 read，但 naïve 与 Memgen 的 write 几乎相同，且都在 P512/P1024 失败。这是早期证据：增加 L1/两级读 locality 本身不能修复 writeback/service 口径。

### 6.2 策略目录

| 策略或策略族 | 主要变化 | 已观察表现 | 处置 |
|---|---|---|---|
| HBServe `ReferenceLRU` naïve cache | 40 MiB 单层 named cache、线性组索引，无独立 L1/L2 | 历史 llama.cpp 1.5B P64–P1024D32：read 误差 0.48%–7.81%；P512/P1024 write 误差 25.13%/23.07% | 保留为单层控制；不能称 NCU L2 模型 |
| 历史两级 linear baseline | 32 KiB/SM L1 + 40 MiB L2、线性组索引、store bypass、line-miss-only 写服务 | 历史 P64–P1024D32 read 0.05%–1.08%；write 在 P512/P1024 为 25.13%/23.07% | 仅历史已校准点可用；写路径拒绝泛化 |
| 历史两级 XOR | per-SM L1 容量按统一 L1/shared-memory 预算、XOR L2、partial-byte 状态 | P1024D32：L1 0.71%、L2 11.49%、read 1.17%、write 24.74% | 非当前固定合同；写仍失败 |
| 10 与 20 逻辑 partition、总 set 不变 | 改变逻辑拓扑，不改总容量/组数 | traffic 无可分辨变化 | 无法用现有流量识别物理 partition |
| linear 与 XOR 组索引 | 改变地址到 set 的映射 | 不同 case 有好有坏，无一致优势 | XOR 作为工程基线，不宣称已识别物理 hash |
| 普通 LRU `disabled` | 无额外 cleaner 或 clean-retained 规则 | P128D2 whole R/W `+0.54%/+0.91%`；P128D16 `+0.20%/+14.55%` | **当前最简单控制；read 可用，write 未验收** |
| MSHR/HIT_RESERVED | 加入 miss 并发保留状态；MSHR 是 Miss Status Holding Register（未完成 miss 状态表） | 1,099,796 次 lookup 中仅 8 次命中该状态，约 0.00073%；traffic 不变 | 只作时序诊断，不解决流量 |
| kernel-boundary high-water drain 75→50 | 每个 kernel 边界按 dirty 高水位清理 | full write 更接近，但逐 kernel WAPE（加权绝对百分比误差）更差 | 拒绝；人为边界补偿 |
| producer-stream drain | 按生产者请求流主动释放 dirty | write 可变近，但 read 约增加 333% | 拒绝 |
| aggressive high-water 2→1 | dirty 行到 2 即降至 1 | Prefill 多写、Decode 少写 | 拒绝 |
| dirty-age 0 对 302 | 按脏行年龄触发释放 | 两端 traffic 几乎相同 | 无识别力，不采用 |
| timestamp sort | 用采样 timestamp 重排请求 | 单 dominant kernel write 可由约 +252% 降至 <1% | 证明顺序敏感；timestamp 不是真实到达，不能作为 accuracy 策略 |
| fixed clean-retained budget 8 | 固定每组 dirty/clean-retained 预算 | 56-case 面板：iteration 40/56、warp 38/56，最大误差约 72.8% | 拒绝 |
| legacy quota 7 | 每组 quota 7 | 6/12 通过，普通 LRU 为 8/12；最大误差约 111% | 拒绝 |
| store8 / store8-read7 | 读写使用不同 quota | 初始面板最高 30/32，但快速 holdout 出现约 +50.6%/+68.7% | 过拟合，拒绝 |
| store8→7 / read7 | store path 从 8 收缩到 7 | 初始 24/32；扩展仅 8/24，最大约 23% | 拒绝 |
| pending-next-distinct-line | 延后到下一个不同 line 再释放 | iteration 29/32、warp 25/32；holdout 某 case 约 +194% | 拒绝 |
| overflow-armed pending8 | quota overflow 后武装 pending release | read 56/56；write iteration 48/56、warp 42/56，最大约 72.7% | 拒绝 |
| cancel-on-victim-store | victim line 随后被 store 时取消候选释放 | 两个 pilot 分别约 -9.30%/-0.62%，未跨 case 稳定 | 拒绝泛化 |
| dirty eviction 后配对 clean victim | 每次脏逐出再释放一个干净 victim | 实际正写 case 8/8 失败，多数更差 | 拒绝 |
| read clean-first LRU | read allocation 优先逐出 clean line | 17 个正写 case 0/17；26 个零写 case 26/26 | 拒绝 |
| clean-first + quota q1..q8 | 扫描每组 dirty quota | q2 在一个 gate 约 -3.64%，另一 down case 为 -100% | 拒绝 |
| q2 + read-pressure H1/H4/H8/H16 | quota 加读压力门限 | H16 在 gate 约 +4.6%，down 仍 -100% | 拒绝 |
| read-allocation age8、整行释放 | 读分配年龄达到 8 时释放整行 | 正写仅 1/17；代表 gate/down 为 +36.49%/+137.17% | 拒绝 |
| read-allocation age8、单 sector | 上述规则改为逐 sector | 正写 0/17；gate/down 为 +12.41%/+55.47% | 拒绝 |
| phase-release variant 0..15 | 多种 release 分支组合 | 没有一个在全部目标上联合通过 | 拒绝作为默认 |
| W64 pressure arms | 64-entry 压力窗口及多种 arm | 正写仅 1–3/10；零写 14/14 | 拒绝 |
| sector-independent age | 每个 sector 独立计龄 | fixture 行为改变，但 LLM 面板 traffic 不变 | 功能可实现，无 accuracy 增益 |
| disable-aged / actual-clean-victim clock | 改 aged 开关或按真实 clean victim 计时 | 无跨 case 一致改善 | 拒绝 |
| `phase11`：clean-first + q4 + age16 | 当前主要 cleaner 诊断候选 | P128D2 whole `+0.50%/+1.07%`，Decode write `+39.20%`；P128D16 whole `+0.19%/+15.50%`，Decode write `+648.62%` | 保留诊断；不优于 LRU，不作为已验收默认 |
| q16 clean-first、无主动清脏 | clean-first，dirty budget 16 | P128D2/P128D16 whole write `-92.37%/-90.51%` | 明确拒绝 |
| 普通 LRU + phase11 cleaner | 只把 phase11 victim 改回普通 LRU | P128D2 whole `+0.54%/+1.07%`，Decode write `+39.20%` | 与基线/phase11实质同样失败，拒绝 |
| byte-disjoint WTB capacity 1 | WTB 是 Write Transaction Buffer（写事务缓冲），仅相邻字节不重叠时合并 | P128D2 ingress whole/Prefill/Decode write 误差约 -1.36%/-0.15%/-20.60% | 改善一个入口口径但 Decode 未过门，且不是最终 DRAM 服务；不采用 |

策略探索的总括结论不是“还缺一个最佳 quota”，而是：简单 LRU、clean-first、quota、年龄、压力、pending、paired victim、按 kernel drain 和写事务缓冲都无法同时保持读准确、写准确、跨 workload 稳定与在线因果性。部分策略能在一个校准 case 上接近 NCU，但会在 holdout 或另一阶段出现数量级反例。

## 7. 为什么 DRAM write 始终不收敛

### 7.1 已被数据直接支持的判断

- **不是单纯 traffic 体量放大。** P128D2 到 P128D16，read 始终在约 0.2%–0.5%，而 Decode write 从 +39% 变为 +649%；同一请求生成/读取路径并没有同比失稳。
- **不是末尾少排一次 dirty。** 当前没有 terminal drain，且 partial dirty victim 的未覆盖旧字节仅百字节量级；它们使语义门失败，但无法解释 40.80 MiB 的 D16 Decode 绝对差。
- **不是 NCU whole 重复噪声。** P128D16 whole write 的三次极差约 0.13 MiB，远小于 39.41 MiB gap。
- **不是 phase11 与普通 LRU 之间的细小参数问题。** 两者在 Decode 产生几乎相同的约 47.09 MiB 模型写量，D16 硬件只有 6.29 MiB。
- **P128D2 whole 的“通过”主要由 Prefill 权重和阶段抵消造成。** 不能用 whole `+0.91%` 覆盖 Decode `+39.17%` 及单步反号误差。

### 7.2 离线 memory-SASS + cache replay 的天然边界

1. **生成的是内存指令地址，不是芯片物理 DRAM 完成事件。** 从 store 指令到 L1/L2 准入、合并、压缩、逐出、后台 service、控制器接收和最终完成之间仍有硬件状态。
2. **请求顺序是重建顺序，不是真实到达前沿。** CTA（Cooperative Thread Array，线程块）映射、warp 依赖、并发 miss、队列阻塞和完成反馈会改变相邻访问及 victim；timestamp 重排实验已经证明 order 是一阶敏感变量，但现有 timestamp 不能冒充真实周期。
3. **真实 L2 组织仍未唯一识别。** 20 个逻辑 slice、XOR、普通 LRU 是可运行的工程结构；真实 partition/hash、插入、读写非对称替换、store cache operator 和后台 dirty service 仍可能不同。
4. **非 kernel traffic 观察不完整。** memory-SASS profiler 以 kernel 指令为主体，`cuMemcpy*`、`cuMemset*`、初始化 DMA、metadata 或其他 copy-engine 事务不一定进入同一源流，而 NCU DRAM counter 可覆盖更宽的芯片活动。当前回执也明确列出 133 个 pre-epoch 调用及初始 DMA 未建模。
5. **物理写口径尚未闭合。** 模型输出 emitted dirty sector；硬件 counter 可能更接近 admitted/completed transaction。写合并、重复覆盖和保留会让 source store、L2 request 与 DRAM service 使用不同分母。
6. **当前 L1 store 策略和硬件并未证明等价。** 工程模型允许 store bypass 或代理规则，真实 Ada 的 store lookup/hit/writeback 行为不能仅从 read 准确反推。

因此，现有证据能证明“聚合 read 在两个已测 P128 点上很准、write 模型在 Decode 和长度迁移上失败”，但不能唯一证明失败只来自时序、只来自地址映射，或只来自某个 dirty quota。要区分这些原因，需要同一源的 arrival-aware replay、写漏斗硬件指标和非 kernel memop census，而不是继续只拟合 whole write 总数。

## 8. 可以继续采用与不能采用的部分

### 8.1 可以继续采用

- HBServe 对已建立 context-specific profile 的完整请求流式展开，不持久化大型 raw memory trace。
- 40 MiB、128 B line、4×32 B sector、sector valid/dirty、write-back/write-allocate、跨 kernel 状态和无强制 drain 的功能实现。
- 普通 LRU 作为最少假设的控制后端。
- P128D2/P128D16 已测条件下的聚合 DRAM read，误差分别约 0.54%/0.20%；使用时必须附带 workload 与证据范围。
- source/版本守恒、输入 SHA、三次 NCU 中位数、分阶段/逐步同时报告的验证协议。

### 8.2 当前不能采用为硬件准确结论

- 任意 context/model 的 DRAM write 预测；尤其不能把 P128D2 whole 通过外推到长 Decode。
- 逐步 Decode write、物理 writeback 时机或真实 Ada dirty-release 机制。
- 目标 SGLang 矩阵的 L1/L2 hit-rate accuracy；当前缺硬件同分母指标。
- q16、phase11 或其他 quota/age/pressure/pending 规则作为“已校准 NVIDIA 策略”。
- P32D2、独立 P128D4、独立 P128D8 的数值；当前没有对应闭合实验。
- 把内部 residual 为 0 解释为 hardware accuracy。内部守恒只证明账算完整，不证明输入域或物理机制完整。

## 9. 补齐严格矩阵的最小实验协议

后续应只补三个缺口：P32D2、P128D4、P128D8。每个点必须独立完成以下步骤，不能把 D16 前缀直接改名：

1. 固定同一 Qwen2.5-1.5B BF16、SGLang 0.4.10、prompt/token、GPU UUID、eager/TP1/KV 条件。
2. 为该 exact workload 独立执行 sparse capture、CTA placement、profile fit 和 held-out CTA 校验；可复用代码与已接受的相同 kernel 规则，但不得复制地址或把另一 context 的 profile 当作已验证。
3. 冻结 HBServe/Memgen 源码、配置、输入 pins 和普通 LRU 策略后，再做三次 NCU。至少保留 whole、Prefill、连续 Decode 和每个 Decode step 的 DRAM read/write。
4. 同时收集可严格定义分母的 NCU L1/L2 read hit 指标、global store、L2 write request 和 DRAM write，形成 source→L2→DRAM 写漏斗。
5. 所有参数选择在迁移/holdout 之前冻结；新 context 或模型必须保留至少一个真正未参与选参的 blind holdout。
6. 报告每个阶段，而不是只报告 whole；受验门仍为每个范围 read/write 都严格小于 10%。
7. 继续直接流式送入 Memgen，不保存 full raw trace；只保存 profile、配置、聚合账本、NCU 报告和 SHA 回执。

在这三个独立点闭合前，系统结论应保持为：`READ_TRAFFIC_CALIBRATED_ON_P128D2_AND_P128D16; WRITE_TRAFFIC_NOT_STABLE; CONTEXT_GENERALIZATION_UNPROVEN`。

## 10. 证据索引与完整性

本报告的本地 canonical 路径是 `C:\Users\82412\Documents\Codex\2026-09-03\llm-footprint-memgen\L2_CACHE_STRATEGY_ACCURACY_REPORT.md`；xmu 可直接阅读副本位于 `/home/xmu/nvidiagds/simulators/hyfiss/docs/L2_CACHE_STRATEGY_ACCURACY_REPORT.md`。远端是便于续接的同字节副本，本地文件仍是本次报告的编辑源。

### 10.1 远端 canonical 根

- P128D2 硬件：`/home/xmu/nvidiagds/codex-runs/l2-sglang-full-decode-20260919-01a08d87-r1`
- P128D2 普通 LRU/phase11：`/home/xmu/nvidiagds/codex-runs/l2-sglang-full-diagnostic-comparison-20260919-01a08d87-r1`
- P128D2 q16：`/home/xmu/nvidiagds/codex-runs/l2-sglang-full-retention-ablation-20260920-01a08d87-r1`
- P128D2 普通 LRU + cleaner：`/home/xmu/nvidiagds/codex-runs/l2-sglang-full-lru-cleaner-ablation-20260920-01a08d87-r1`
- P128D16 profile/source：`/home/xmu/nvidiagds/codex-runs/l2-sglang-full-qualified-d16-20260920-01a08d87-r1`
- P128D16 三策略：`/home/xmu/nvidiagds/codex-runs/l2-sglang-full-d16-model-20260920-01a08d87-r2`

### 10.2 本地 copy-only 回执

大数据未复制到 session 工作区；最终小回执保存在 `D:\codexdataspace\remote-sync\xmu`。以下本地 SHA-256 已与 xmu 原文件逐项比对一致：

| 对象 | 文件 | SHA-256 |
|---|---|---|
| P128D2 普通 LRU/phase11 | `l2-sglang-full-diagnostic-comparison-20260919-01a08d87-r1/summary.json` | `6765fb6b577e52251cce2c060bf3cbbb1b6fc7128a7c4f81c4522bd496f907d7` |
| P128D2 普通 LRU/phase11 | `.../finish.json` | `67734f614dfc24a62739d239f8c3c941734771729fba1a96f1759e2ceccad4eb` |
| P128D2 q16 | `l2-sglang-full-retention-ablation-20260920-01a08d87-r1/comparison.json` | `604cc3c13cd442d89f8334495c41f150f7ff6395446a789e0b70addedc81eabb` |
| P128D2 q16 | `.../finish.json` | `fa764bcd23e0e7b7e81efd8caa5ca76604d25acc3e4da558d2e74ae49e00e1e8` |
| P128D2 LRU + cleaner | `l2-sglang-full-lru-cleaner-ablation-20260920-01a08d87-r1/final-20260921/comparison.json` | `8caa684987a4d356114daf46cc436ec7834119035c0753f7cd29b53486f1447d` |
| P128D2 LRU + cleaner | `.../finish.json` | `9cca9f1a2992b4d56dc6e77a699608cfaaf19f62c5b338f4fc30c37277c83061` |
| P128D16 三策略 | `l2-sglang-full-d16-model-20260920-01a08d87-r2/final-20260921/comparison.json` | `58b3865fcf1c011cfa2cd5d2eb79db989e108f1a20fff785ae271560757c8323` |
| P128D16 三策略 | `.../finish.json` | `a83132fa22188380fdb0382d700624043995f91158e5bc02fc44e39b799d2ef0` |

P128D16 最终状态为 `COMPLETE_FULL_P128D16_COMPARISON_NOT_ACCEPTANCE`，运行产生 60 个对比行；source population、source-sector 与 dirty-version 账本闭合，但 `semantic_admission=false`、`hardware_accuracy_accepted=false`。普通 LRU + cleaner 的最终状态为 `COMPLETE_FULL_P128D2_LRU_CLEANER_ABLATION_NOT_ACCEPTANCE`。

## 11. 最终判定

| 目标 | 判定 |
|---|---|
| HBServe 是否能为当前 exact workload 动态生成完整请求 | **是**，P128D2/P128D16 已闭合，且无需持久化 full raw trace |
| Memgen 聚合 DRAM read 是否准确 | **在已测 P128D2/P128D16 上是**，whole 误差约 0.54%/0.20%；不能直接外推未测点 |
| Memgen DRAM write 是否准确且随 Decode 稳定 | **否**；D2 whole 接近但 Decode 已失败，D16 whole 14.55%、Decode 648.63% |
| phase11 是否比普通 LRU 更好 | **否**；读相近，写略差或等价 |
| q16 是否可作为替代 | **否**；write 少约 90% |
| 当前是否完成 P32D2、P128D2/D4/D8/D16 严格矩阵 | **否，2/5 独立完成**；D4/D8 仅有前缀诊断，P32D2 有同栈完整模型估算（建模占流量 2.2%）但无 NCU 参考，故不计入独立完成 |
| 当前是否可以证明任意规模误差上限 | **否**；现有是经验校准与迁移反例，不是可证明的普适上界 |

最准确的一句话是：**HBServe + Memgen 已经是一个在两个 P128 点上聚合读流量很准、但 Decode 写回服务不可靠且尚未证明跨 context 泛化的离线 traffic 模型。**
