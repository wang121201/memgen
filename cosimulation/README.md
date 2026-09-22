# HBFSim causal cosimulation

This branch lowers a compute/issue sideband and Memgen post-cache transactions
into one High Bandwidth Flash Simulator (HBFSim) dependency graph. Each JSONL
event contains a nominal ready time, compute duration, a cooperative thread
array (CTA) identifier and its streaming-multiprocessor/warp placement,
explicit prior events whose memory
completion must be observed, and its 32-byte-sector-aligned DRAM transactions.
Every input row is a memory-issue event and must contain at least one memory
transaction; compute-only events belong in the upstream scheduler model.

The lowering is causal:

1. an event's compute barrier cannot start before `ready_ns` or before every
   event named by `wait_for` completes;
2. its memory transactions depend on completion of that compute barrier;
3. HBFSim schedules the resulting channel, bank, row and queue work;
4. those physical completions delay later dependent compute barriers, changing
   their actual issue timestamps while independent streams may overlap.

`wait_for` must come from a real scheduler/scoreboard sideband. The runner does
not assume that every prior access in one warp is a true data dependency,
because doing so would over-serialize GPU execution. Queue-capacity feedback,
CTA residency and issue-port contention not represented by the input DAG are
also unavailable.

The bundled RTX 4000 Ada configuration uses HBFSim's HBM engine as a
GDDR6-targeted surrogate. Its capacity/interface arithmetic is pinned, but its
bank/row mapping, timings, refresh and queue policy are uncalibrated. Therefore
the resulting latency and bandwidth are deterministic mechanism diagnostics,
not hardware-accuracy results. The external HBFSim executable is not stored in
Git; `config/backend-pin.json` records its required digest and source revision.

The currently pinned xmu build is:

- executable:
  `/home/xmu/nvidiagds/codex-runs/llm-footprint-v1/hbfsim/memgen-cosim-backend-20260921-r2/build/hbfsim`;
- HBFSim commit: `d7a2ca64614a6d9ce8d7a69beb77ce78b66df1a8`;
- executable SHA-256:
  `349be108468f584f5e4ae0acf74b2c72789d4d26cb0f9f3747879cb203f69d2d`;
- build mode: clean detached worktree, CMake Release, Ninja, tests disabled;
- build evidence: `configure.log`, `build.log`, `version.txt`,
  `hbfsim.sha256` and `source-status.txt` in the parent run directory.

Run the bounded causal smoke on xmu with:

```bash
scripts/run_hbfsim_cosim_smoke.sh \
  /path/to/pinned/hbfsim \
  /tmp/memgen-hbfsim-cosim-smoke
```

For a real replay, invoke `cosimulation/coupled_replay.py` with a JSONL event
stream produced from the same independently profiled SGLang workload as the
Memgen traffic and NCU reference. A traffic-only trace is insufficient: it
cannot reconstruct true dependencies, compute duration, CTA residency, or
nominal issue time.

The current runner materializes one dependency graph and submits it as one
batch. Before a large full-inference run, perform an input census and confirm
that the graph fits host memory. A future chunked path must use HBFSim's
`retain` contract for cross-batch dependencies; splitting into timing-barrier
batches would change overlap and is not an acceptable optimization.


## r4 同步与接入边界

本分支已合并主分支 b14d046 的 r4（第四轮缓存候选）核心与硬件配置。
旧缓存回归、新r4显式回放、原有因果 smoke 均通过；合并前后相同三事件输入的
完成时间、依赖及流量完全一致：读取64字节、写入32字节。一级/二级缓存
（Level-1/Level-2 Cache，L1/L2）配置不会自动改写HBFSim存储系统的配置。

当前联合仿真入口仍消费外部事件文件，没有调用 memgen 生成这些事件。
Qwen2.5-1.5B 的P32D2（32输入词元、提示处理后两次解码前向）已存在闭合的
缓存汇总，但缺少实际显存事务出口和计算/依赖旁路数据，不能从汇总表重建地址、
线程束、依赖或时间。HBServe的timestamp仅是排序坐标，不能解释成纳秒。

后续接入应在真实DRAM（动态随机存取存储器）读填充与脏扇区写回出口发送事务，
保留被驱逐地址与32字节粒度；还须处理全部缓存命中、零显存事务的依赖事件。
线程束身份、计算时长及依赖必须由明确来源提供，不能为凑齐接口而假造。
将r4固定输出接到有向无环图（Directed Acyclic Graph，DAG）服务诊断，也不等于
完成时间反过来改变缓存请求顺序的闭环模拟。有限缺失状态保持寄存器
（Miss Status Holding Register，MSHR）仍未建模。

本次未修改HBFSim的存储体/队列/服务时序参数；其GDDR6替代模型仍未校准。
所以这里验证的是合并兼容性，完整LLM（大语言模型）的r4→HBFSim影响仍待接入。
