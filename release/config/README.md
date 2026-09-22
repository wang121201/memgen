# HBServe＋memgen 硬件配置

此目录的 `RTX4000Ada.r4.config` 是统一硬件配置 schema（格式版本）1。HBServe 是按采样 profile（地址规则文件）展开访存指令的前端；memgen 是消费展开结果、模拟缓存状态并统计访存流量的后端。本文件的“硬件配置”表示**有效行为模型**，不表示已经识别出真实芯片内部实现。

当前基线为 NVIDIA RTX 4000 Ada Generation 的 `r4-small-shared-20260922`：r4 指第四轮缓存候选，额外包含实际共享划分为 8/16 KiB 的扩展。此配置复现之前受测候选，**未升级为硬件准确度验收通过**。旧 `RTX4000Ada.paper-v1.config` 及既有实验冻结副本继续保留。

## 读取和使用

```bash
# 只解析、校验并显示配置，不读取模型或运行 GPU。
./hbserve --describe-hardware-config release/config/RTX4000Ada.r4.config

# compact profile、启动和 CTA 布局均沿用现有 workflow。
./hbserve --mode memgen \
  --profile-index profiles.index.jsonl \
  --app-config app.config --issue-config issue.config \
  --hw-config release/config/RTX4000Ada.r4.config \
  --r4-context r4-context.json \
  --stats source-stats.json --output-dir fresh-model-output
```

`hbserve` 指由 `release/source/tools/hbserve_profile_stream_cache_semantic_r17.cpp` 构建的本版可执行文件；历史冻结二进制不认识新格式。GPU 是图形处理器（Graphics Processing Unit），上述回放只使用中央处理器（CPU，Central Processing Unit）。输出目录和统计文件必须尚不存在；不保存展开的逐条 SASS（NVIDIA 机器指令汇编）访存 trace。

流程是：`--hw-config` → 注册表和严格解析 → 不可变 `HardwareProfile` → 前端核验动态上下文与 CTA 布局 → 后端生成生效参数 → 缓存构造。CTA（Cooperative Thread Array，协作线程数组）对应 CUDA 线程块；SM（Streaming Multiprocessor，流式多处理器）是其调度单元。解析发生在回放开始前，不在逐访存热路径内。

`memgen_hardware_config.h` 是唯一新格式解析器。调用者使用本版可执行文件的 `--describe-hardware-config` 获取生效值和配置身份，不复制另一套新格式解析规则。使用 r4 时必须提供匹配的原生地址上下文和轮转 CTA 布局；历史单层跨层重绑定的合成地址不能冒充原始分配信息。

本设计参考服务器 Accel-Sim 的组件选项注册、统一读取、交叉约束与生效值打印；未复制其整份参数或吞吐/时序模型。服务器 tuner（微基准参数生成工具）配置位于 `~/nvidiagds/simulators/accelsim2.0/util/tuner/NVIDIA_RTX_4000_Ada_Generation`。其 10×2 分区与受测 memgen 的 5×4 分区不能因总容量相同而互换；本配置保留后者。

## 字段、单位与边界

每行是一个 `-memgen_字段 值`；`#` 后为注释。文件不超过 64 KiB，所有注册字段必须出现且只能出现一次。数值是无符号十进制整数，未知字段、错误枚举、额外 token、缺项、重复项及溢出均报错。B 为字节（byte），KiB=1024 B，MiB=1024 KiB；容量全部由几何计算，不接受与几何矛盾的第二个容量值。

**读取请求、缺失填充和读流量统计的基本单位是32 B扇区；128 B是缓存行的标签和替换单位。** 当前配置 `sector_bytes=32`，而 `l1_line_bytes=l2_line_bytes=128`；一行包含4个扇区，分别记录有效状态。读取一个缺失扇区只填入并统计该32 B，不因首次分配128 B行标签就自动读取整行。行标签已存在而目标扇区无效时，仍为扇区未命中。

例如，两级缓存初始为空且中途没有逐出：先读某行第一个扇区中的4 B，模型显存读流量为32 B；再次读同一扇区命中，不新增下游读流量；随后读该行第二个扇区中的4 B，再产生32 B。累计64 B，不是两次各128 B。若一个线程束的访问覆盖完整、对齐的128 B，则合并为4个32 B请求，合计128 B；不能把4个扇区解释成4次128 B读取。

SASS指令的每线程操作数宽度仍可为4/8/16 B等，HBServe保留地址、宽度和活动字节；memgen在访存合并后按32 B扇区处理。统计模式下的模型显存读字节数按其读扇区数×32计算；与硬件计数器比较仍须匹配请求来源和测量范围，不能将某个来源的L2缺失扇区直接当作全部显存流量。这里明确的是当前模型与流量统计口径，不对未建模的DRAM物理突发长度作额外推断。现有r4代码已经按此处理，本次是术语澄清，不改变参数或算法。

| 字段组（省略共同前缀 `-memgen_`） | 定义、当前值和可接受范围 |
|---|---|
| `schema`、`profile_id`、`calibration_status` | 格式版本固定1；配置名称不超过128字符，限字母、数字、点、下划线、连字符；校准状态固定 `experimental`。配置名称不是内容身份，改参数后仍须以内容散列区分。 |
| `num_sms`、`warp_size` | 当前48个SM；支持1–256，受前端8位SM编号限制。warp（线程束）固定32个线程。 |
| `memory_channels`、`subpartitions_per_channel` | 当前5个逻辑通道、每通道4个逻辑分区；范围分别1–128、1–32，后者须为2的幂。乘积20不声称物理连线。 |
| `dram_banks`、`partition_index_bit`、`address_mapping` | 当前16个模型bank（存储体），8为地址分区起始位；范围为1–128的2次幂、7–31。映射固定 `fallback_quotient_v2`，沿用现有分区与局部分区地址计算。 |
| `l1_model`、`context_model_id` | 一级缓存（Level-1 Cache，L1）模型固定 `allocation_clock_v1`，运行上下文标识须等于 `r4-small-shared-20260922`。 |
| `l1_sets`、`l1_line_bytes`、`sector_bytes` | 当前16个集合，支持1–4096且为2的幂；缓存行固定128 B，sector（扇区）固定32 B。每行4个独立有效性扇区。 |
| `l1_shared_kib_to_ways` | 格式 `共享划分KiB:路数,...`。共享划分键0–1024且唯一，路数1–1024；当前 `8:64,16:56,32:50,64:33,100:14`。每SM有效容量为集合数×路数×128 B，即128/112/100/66/28 KiB。这是经验容量，不能解释成“128 KiB减共享内存”。 |
| `l1_replacement`、`l1_index`、`l1_tag` | 固定 CLOCK（二次机会时钟替换）、`allocation_relative_hash2_u32`（以原始分配起点为零的32位hash2索引）、`absolute`（绝对地址tag）。hash2为现有两次乘法混合，乘数0x7feb352d和0x846ca68b，移位16/15/16，最后取集合位；不冒充已知硬件hash。 |
| `l1_read_policy`、`l1_store_policy` | 固定 `ldg_strong_gpu_bypass_v1` 和 `bypass`：已分类的 `.STRONG.GPU` 读取绕过L1，写与原子也绕过L1。普通受支持读取走L1。 |
| `l1_preserve_across_kernels`、`l1_fill_latency` | 固定0：每kernel清空L1，同步填入读取扇区；不模拟等待周期。kernel指一次GPU内核启动。 |
| `l2_sets_per_partition`、`l2_ways`、`l2_line_bytes` | 二级缓存（Level-2 Cache，L2）每分区当前1024集合、16路、128 B行；集合范围1–1048576且为2的幂，路数1–1024，行固定128 B。总字节数须不超过32位无符号表示范围；当前20×1024×16×128=40 MiB。 |
| `l2_index`、`l2_replacement`、`l2_clean_first_k` | 当前 `X`（现有异或索引），也支持 `L`（线性索引）；替换固定 LRU（Least Recently Used，最近最少使用），清洁优先候选数K固定0，即未启用。 |
| `l2_data_validity`、`write_sector_policy` | 固定 `known_bytes_union_v1`，部分写按已知字节并集更新有效性；写扇区策略当前 `line-miss-only`，另支持已有 `all` 分支。前者保留“标签命中但扇区不完整”的状态，后者在该写分支分配缺失扇区；修改后需重新验收。 |
| `dram_store_policy` | 显存 DRAM（Dynamic Random-Access Memory，动态随机存取存储器）写统计当前 `writeback`，按淘汰的有效脏扇区计入流量；另支持既有 `request` 请求计数策略。两者不能混为同一硬件写口径。 |
| `l2_preserve_across_kernels` | 0或1，当前1：同一次回放内跨kernel保留L2状态。新进程仍从空状态开始。 |
| `l2_dirty_drain`、`l2_streaming_fill`、`l2_fill_latency` | 本版本固定0：禁用现有后台排脏启发式和流式填充分支，读取同步填入；不进行终端强制排脏。 |
| `mshr_model`、`timing_model` | 固定 `disabled`。MSHR（Miss Status Holding Register，缺失状态保持寄存器）的有限并发、合并和排队，以及互连/DRAM周期时序尚未建模。写192或其他值不会静默启用。 |
| `cta_placement`、`instruction_order`、`monotonic_sm` | 固定 `round_robin`、`timestamp`、1：前端轮转放置CTA，按已有模型时间戳合并指令，保持单SM时间单调。不是硬件实测到达顺序。HBServe逐CTA验证；直接后端API（Application Programming Interface，应用程序接口）仍由调用者提供布局，配置报告明确该范围。 |
| `issue_interval`、`kernel_gap` | 当前1和5000，是已有排序坐标的间隔，范围分别1–10^9、0–10^9；不是预测GPU周期或墙钟超时。HBServe已有时间戳的输入保持其值，仅在原后端需要合成/单调修正时使用。 |

改变支持范围内的参数确实会改变模型，但不会自动获得新硬件上的校准资格。若需要新的替换算法、有限MSHR、地址映射或写策略，应先实现组件、注册明确枚举并验证，不能只在文件中增加一个看似有效的数值。

敏感性须按字段作用解释：L1集合/路数与L2集合会改变缓存预测；`dram_banks` 只影响请求输出中的bank/row地址注释，在统计模式中不产生银行时序效应。`issue_interval`、`kernel_gap` 只影响已有模型的排序坐标；当前零填充延迟下不应把它们当成性能预测。当前fallback映射主要使用通道数×每通道分区数的乘积，不能从流量拟合独立识别这两个因子。

## 静态配置与动态上下文

配置保存模型参数。`r4-context.json` 继续保存每kernel实测的共享划分、原始分配代际、起点和长度，以及profile索引/启动文件的内容身份。要求分配起点128 B对齐、分配长度大于0且不超过4 GiB、范围不重叠、代际标识唯一；尾部扇区只检查实际访问的字节。缺context、模型标识冲突、配置没有对应共享档位、布局不符或实际访问越界均拒绝。

schema 1 文件对硬件策略具有唯一优先级；旧 `BackendOptions` 中硬件默认值只服务旧配置。输出格式、回调、地址来源、是否包含local（线程私有地址空间）等运行选项仍归调用者。旧配置没有 `-memgen_` 字段时继续使用旧reader和旧策略优先级。独立trace文件CLI（Command-Line Interface，命令行接口）没有所需动态context，因此明确拒绝新r4格式；应使用HBServe入口或有完整context的有序后端API。未扩展r4 checkpoint（缓存状态检查点）恢复。

## 生效记录与验证范围

新格式回放写出 `hardware.config`（实际解析的原文快照）、`hardware.resolved.json`（实际参数和容量）、`r4_l1_profiles.csv`（逐kernel共享档位/路数/容量）。HBServe另写 `hardware.identity.json`，包含配置与context的 SHA-256（256位安全散列）以及布局验证状态。`describe`与这两个JSON文件用Boost property_tree写出，**标量均为字符串**；数值需显式转换，硬件验收使用状态枚举 `hardware_accuracy_status=not_accepted`，不写易被误读为真值的字符串布尔字段。

配置读取本身不产生逐条trace，也不设任务时长截止。当前运行中的完整LLM（Large Language Model，大语言模型）矩阵保持原冻结实现和配置；新入口需在独立回放中使用。等价测试只证明重构保持软件行为，不证明L2写策略已经与真实计数对齐。尤其逐decode（逐词元解码）写流量仍是待校准项。

通用验证入口为 `python3 scripts/test_cache_core.py --output FRESH_OUTPUT_DIRECTORY`。它执行旧配置回归、独立 CLOCK 参考对照、五档显式配置等价、参数敏感性、非法输入拒绝、地址边界和失败取消测试。仅使用合成输入，不运行 GPU，也不赋予硬件准确度验收资格。
