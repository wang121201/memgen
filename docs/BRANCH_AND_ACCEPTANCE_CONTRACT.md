# Branch roles and unified acceptance contract

This document defines the common experiment contract for the Memgen repository.
It is normative for branch scope and result naming; it does not turn a
diagnostic result into hardware validation.

## 1. Terms

- **HBServe** is the sampled memory-SASS profile and full-inference address
  generation layer. Memory-SASS means the memory-accessing instructions in
  NVIDIA machine assembly plus their active-lane addresses.
- **Memgen** is the functional GPU cache-filter layer. It consumes generated
  addresses, models L1 and L2 cache state, and emits off-chip DRAM traffic.
- **NCU** means NVIDIA Nsight Compute. A same-workload NCU range is the hardware
  reference for cache counters, DRAM traffic and diagnostic kernel duration.
- **P<n>D<m>** means a batch-one inference with `n` prefill tokens and `m`
  independently executed decode steps. A prefix extracted from a longer run is
  diagnostic and is not an independent P/D workload.
- **Simple latency** means constant-cost integration over cache-hit or
  memory-miss events. It has no resource scheduler, dependency scoreboard,
  computation overlap, queue replay or backpressure.
- **HBFSim cosimulation** means dependency-aware memory-system replay in which
  memory completion can stall a producer and therefore change later memory
  issue timestamps and ordering. HBFSim is the High Bandwidth Flash Simulator;
  for RTX 4000 Ada its HBM engine is currently only a GDDR6-targeted surrogate.

## 2. Branch contract

| Branch | Responsibility | May claim | Must not claim |
| --- | --- | --- | --- |
| `main` | SGLang sampling, packed profile, HBServe full-inference address generation, stable L1/L2 cache filter and NCU comparison | Functional trace generation, cache/traffic counters, accuracy only at independently closed workloads | General latency, scheduling, compute overlap or unmeasured-context accuracy |
| `research/l2-writeback-dirty-management` | L2 indexing/replacement, write policy, sector-valid/dirty ownership, victim writeback and dirty release | Mechanism sensitivity and independently validated traffic accuracy | Promotion to the stable default from a fitted workload alone |
| `research/simple-latency` | Add configurable constant latency to exclusive L1-hit, L2-hit and DRAM read/write events | Deterministic serial memory-work and parameter sensitivity | GPU execution time, bandwidth saturation, queueing, stall propagation or compute-memory overlap |
| `research/hbfsim-cosimulation` | Lower post-cache transactions plus compute/issue sideband into an HBFSim dependency DAG; feed completion stalls into later issue | Causal memory replay, queue/bank/channel scheduling and compute-memory overlap when all required sideband is present | Hardware-accurate Ada timing while the GDDR profile or compute sideband remains uncalibrated/incomplete |

Research branches must remain separate. L2 policy changes affect which DRAM
transactions exist; latency and cosimulation branches evaluate when those
transactions complete. A latency model must never silently change traffic.

## 3. Common workload matrix

The inference framework is SGLang 0.4.10 with FlashInfer, batch size 1, tensor
parallelism 1, eager execution and CUDA Graph disabled. Every matrix point
requires its own sample, packed profile and NCU reference.

Models in scope are:

1. Qwen2.5-1.5B-Instruct BF16;
2. Meta-Llama-3-8B-Instruct BF16;
3. Meta-Llama-3-70B-Instruct BF16;
4. Qwen2-35B BF16. The shorthand `qwen235B` is ambiguous; a run must bind an
   exact model ID and must not reinterpret it as a 235B-parameter model.

The basic admission workload is P32D2. The scale series is the Cartesian
product of prefill lengths `128, 256, 512, 1024` and decode lengths
`2, 4, 8, 16, 32`. The immediate four-branch gate covers the 1.5B and 8B
models at P32D2. The 70B and 35B models remain future scale targets and may
require multi-GPU execution; they are not implied by a single-card result.

## 4. Hardware and cache identity

The hardware reference is an NVIDIA RTX 4000 Ada Generation GPU with 48
Streaming Multiprocessors (SMs).

- L1 cache: 32 KiB per SM, 4 sets, 64 ways, 128-byte line and four 32-byte
  sectors per line. The identity check is `4 * 64 * 128 = 32768` bytes.
- L2 cache: 40 MiB total, 20 slices, 1024 sets per slice, 16 ways, 128-byte
  line and four 32-byte sectors per line. The identity check is
  `20 * 1024 * 16 * 128 = 41943040` bytes.

Every receipt must preserve GPU UUID, model revision, SGLang/package versions,
prompt/decode token IDs, sampler/profile/source/config SHA-256 values, phase
boundaries and the exact cache configuration.

## 5. Metrics and denominators

For `whole`, `Prefill`, the continuous `Decode` range and each `Decode<i>`
step, report:

- L1 read hit rate and L2 read hit rate, but only compute hardware-relative
  error when NCU and the model use the same request denominator;
- DRAM read bytes and DRAM write bytes, absolute difference and signed/absolute
  relative error against the median of three NCU repeats;
- generated request, sector and byte conservation;
- modeled bandwidth only when traffic and elapsed time cover the same range;
- NCU kernel-duration-derived bandwidth as a hardware diagnostic, not as pure
  memory-service bandwidth;
- simple-latency serial memory-work or HBFSim completion/stall/bandwidth under
  their branch-specific names.

Relative traffic error is
`100 * (modeled_bytes - NCU_bytes) / NCU_bytes`. Zero hardware traffic is
reported with absolute bytes and an undefined relative error, never as 0%.

## 6. Admission gate

The mandatory current accuracy gate is DRAM write traffic: absolute relative
error must be at most 20% for the complete request **and for every individual
decode step**. Passing only the whole request is insufficient because prefill
traffic can hide decode error. DRAM read and cache hit-rate error remain
mandatory report columns; a stricter branch may add gates but may not weaken
the write rule.

All of the following are also mandatory:

1. independent profiling for every P/D point;
2. three NCU repeats and a declared aggregation rule;
3. exact phase/range identity and traffic conservation;
4. no unresolved partial-byte or semantic-domain loss promoted as acceptance;
5. latency claims separated from traffic claims;
6. HBFSim timing classified as diagnostic until the same-range GDDR timing,
   address mapping and compute sideband are calibrated.

## 7. Current P32D2 evidence boundary

- Meta-Llama-3-8B P32D2 has three NCU range repeats for whole, Prefill,
  Decode1 and Decode2, but the stable repository does not yet contain a closed
  full-inference Memgen replay joined to those exact ranges.
- Qwen2.5-1.5B P32D2 lacks an independent same-stack packed profile and NCU
  reference. Existing Qwen P128D2 evidence cannot be renamed P32D2.
- Therefore none of the four branches is currently admitted for both requested
  P32D2 models. Smoke tests validate implementation mechanics only.
