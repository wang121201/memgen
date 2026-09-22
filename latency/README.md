# Simple-latency diagnostic

`simple_latency.py` consumes Memgen's `kernel_summary.csv` and integrates four
exclusive modeled outcomes: L1 resolved hits, L2 resolved hits, 32-byte DRAM
read sectors and 32-byte DRAM write sectors. Pending hits are charged at their
resolving cache level. The configured values are end-to-end costs for these
outcomes, so lookup latency is not added again at every cache level.

The tool validates each row's L1, L2 and DRAM request conservation before it
computes serial work. An optional `kernel_id,phase` CSV provides Prefill and
per-decode summaries. Both inputs and their SHA-256 digests are recorded.

The output `serial_memory_work_ns` is a deterministic sensitivity diagnostic.
It is **not** GPU execution time or achievable bandwidth: this branch contains
no resource scheduler, dependency scoreboard, queue replay, backpressure,
stall propagation or compute-memory overlap. Real acceptance also requires
calibrated latency constants; the fixture constants are deliberately synthetic.

Example:

```bash
python3 latency/simple_latency.py \
  --kernel-summary /path/to/model/kernel_summary.csv \
  --latency-config /path/to/calibrated-latencies.json \
  --phase-map /path/to/kernel-phases.csv \
  --output-json /tmp/simple-latency/result.json \
  --output-csv /tmp/simple-latency/by-phase.csv
```

Run the bounded implementation smoke with:

```bash
scripts/run_simple_latency_smoke.sh /tmp/memgen-simple-latency-smoke
```

## r4 同步与固定权重影响

本分支已合并主分支 b14d046 的 r4（第四轮缓存候选）核心。一级/二级缓存
（Level-1/Level-2 Cache，L1/L2）配置由 memgen 读取；本工具继续消费具名统计列，
不把缓存的零填充时序直接当成硬件延迟。ns 为纳秒，每项参数是对应最终服务
结果的总成本，不是各层附加周期；有限缺失状态保持寄存器
（Miss Status Holding Register，MSHR）及并发重叠仍未建模。

已用 Qwen2.5-1.5B 的 P32D2 配对结果测试：P32 为32个输入词元，D2为提示处理
后两次解码前向。原连续缓存回放包含预热；本测试仅从已生成账本筛选1030个
测量内核，排除1032个预热内核，未重新冷启动。合并前后工具消费同一份输入的
各阶段事件数及加权结果完全一致。

固定合成权重为 L1 命中1 ns、L2命中10 ns、显存 DRAM（动态随机存取存储器）
读扇区100 ns、写扇区120 ns；读写扇区均32字节。r4相对旧缓存的串行访存
工作量变化为：整体 -1.313847%，第一次解码 -2.004171%，第二次解码 -2.004151%，
提示处理 +0.001056%。主要变化来自L2服务转为L1服务，显存读取基本不变。
这些权重未经硬件校准，结果不是GPU执行加速或预测执行时间。

复现入口为 `scripts/test_simple_r4_impact.py --replay CLOSED_PAIRED_REPLAY
--prior-tool PRIOR_SIMPLE_LATENCY_PY --output FRESH_DIRECTORY`。
旧工具取自 a4fd8def464ab9b23e0915d48cea18731ec03ebd。
`scripts/test_latency_validation.py --output FRESH_DIRECTORY` 检查非有限成本、
数值溢出、不守恒、重复内核及缺失阶段映射的拒绝；原806 ns合成 smoke 保持。
提供阶段映射时必须覆盖每个输入内核；映射负责分组，不自动排除预热。
