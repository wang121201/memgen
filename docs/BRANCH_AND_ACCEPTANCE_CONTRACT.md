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
4. `qwen235B`, an unresolved shorthand for a Qwen-family large model. It may
   denote Qwen2 35B or a 235B-parameter Qwen model; no run is admissible until
   the exact provider/model ID, immutable revision, precision and parallelism
   are recorded.

The basic admission workload is P32D2. The scale series is the Cartesian
product of prefill lengths `128, 256, 512, 1024` and decode lengths
`2, 4, 8, 16, 32`. The immediate four-branch gate covers the 1.5B and 8B
models at P32D2. The 70B and 35B models remain future scale targets and may
require multi-GPU execution; they are not implied by a single-card result.

### 3.1 Declared cases versus target matrices

The admission point and the scale series above are the *target* matrices. The
adapter declares four matrices of its own in
`integrations/sglang/memgen-adapter/contract.json`:

| Declared matrix | Prefills | Decodes | Models | Cases |
| --- | --- | --- | --- | --- |
| scale series (`prefills` / `decodes`) | 128, 256, 512, 1024 | 32, 64, 128 | 2 | 24 |
| basic admission (`basic_admission`) | 32 | 2 | 2 | 2 |
| evidence points (`evidence_points`) | 128 | 2, 16 | 1 | 2 |
| local acceptance (`local_acceptance_matrix`) | 32, 64, 128, 256, 512 | 2, 4, 8, 16 | 1 | 8 |

The first three blocks state prefill and decode axes, which are read as their
Cartesian product. The fourth states an explicit list of `[prefill, decode]`
points, because it is not a product: it is `PnD2` for `n` in `32, 64, 128, 256,
512` plus `P128Dm` for `m` in `2, 4, 8, 16`, with `P128D2` counted once. It is
the eight unique conditions of the local RTX 4000 Ada full-model traffic
calibration, carried here so the acceptance set those runs were measured on can
be selected and reproduced from this snapshot. Five of its points - `P64D2`,
`P256D2`, `P512D2`, `P128D4` and `P128D8` - are in no other block, so before it
existed they were undeclared and the driver refused them.

The scale series keeps the identity it had before the basic admission point was
declared, so the 24-case results are unaffected. The basic admission point is
declared in a separate block precisely so that producing `P32D2` can never be
read as extending the scale series. `evidence_points` exists for the two points
that have independent three-repeat NCU references, `P128D2` and `P128D16`, which
the declared axes could not express: reproducing accepted evidence is what the
declaration is for. Declaring a point makes it selectable and reproducible; it
admits nothing, and `hardware_accuracy_accepted` stays false until the gates in
this document are met.

Consequences:

- `P32D2` for Qwen2.5-1.5B and Meta-Llama-3-8B is producible from this
  snapshot: `memgen-adapter/matrix_workload.py` derives the permitted axes and
  accepted `(model, prefill, decode)` triples from `contract.json` instead of a
  hard-coded tuple, and `sample_pipeline.py` reuses that derivation. Pairs that
  no declared matrix contains, such as `P32D128` or `P512D8`, are rejected.
- **Producing a point is not an admission decision and not an accuracy result.**
  Each point still needs its own independent sample, packed profile and
  three-run NCU reference. The P32D2 rows remain `BLOCKED` in
  `validation/p32d2_branch_status.csv` until that evidence exists, and
  `P128D2` evidence may not be renamed.
- Six of the eight local acceptance conditions are shared with a block declared
  earlier and keep that block's identity: `P32D2` is `basic_admission`, and
  `P128D2` and `P128D16` are `evidence_points`. Only `P64D2`, `P256D2`,
  `P512D2`, `P128D4` and `P128D8` carry `local_acceptance_matrix`.
- Decode steps `4` and `8` are declared only by `local_acceptance_matrix`, and
  `D16` by both it and `evidence_points`. Every other decode value a block lists
  remains part of that block, so no result may be relabelled from one block to
  another.
- The three files that carry the declaration and the two vendored copies are
  hash-pinned by `integrations/sglang/memgen-adapter/deployment-files.json` and
  by `integrations/sglang/package.json`. Any further change to them must update
  those pins and the record in `integrations/sglang/revisions.json`, which is
  what `bootstrap_vendor.py` and `preflight.py` verify.

Run `python3 integrations/sglang/preflight.py` to print the declared and
documented matrices side by side.

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

### 6.1 Unreconciled write gate

This section states 20%. `evidence/sglang/L2_CACHE_STRATEGY_ACCURACY_REPORT.md`
states strictly below 10% for every measured range, and
[accuracy baselines](ACCURACY_BASELINES.md) carried the same 10% rule before it
was reduced to a pointer here. The two have not been reconciled and both are
still in the archive, so every result must be quoted together with the gate that
produced it. The difference decides rows: the P128D16 whole-request write error
is `+14.55%`, which passes 20% and fails 10%. This note records the conflict; it
does not choose a number.

## 7. Current P32D2 evidence boundary

- Meta-Llama-3-8B P32D2 has three NCU range repeats for whole, Prefill,
  Decode1 and Decode2, but the stable repository does not yet contain a closed
  full-inference Memgen replay joined to those exact ranges.
- Qwen2.5-1.5B P32D2 lacks an independent same-stack packed profile and NCU
  reference. Existing Qwen P128D2 evidence cannot be renamed P32D2.
- Therefore none of the four branches is currently admitted for both requested
  P32D2 models. Smoke tests validate implementation mechanics only.

## 8. Branch implementation-smoke status

The following bounded CPU checks were rerun on 2026-09-21. They prove that
each branch's archived mechanism executes deterministically; they do not use a
Qwen or Llama workload and do not satisfy the P32D2 hardware-accuracy gate.
The machine-readable source of this table is
`validation/branch_smoke_status.csv`.

| Branch | Checkout-local command | Result | Evidence identity | Claim boundary |
| --- | --- | --- | --- | --- |
| `main` | `scripts/run_cpu_smoke.sh FRESH_OUTPUT_DIRECTORY` | `PASS_FROZEN_CPU_SMOKE` | `kernel_summary.csv` SHA-256 `9e3b2ee1b69fce3650b9ae2e5a26583ff786d86008f2bb698fc1463915f2d9f2` | Frozen address-generation/cache-filter mechanics only |
| `research/l2-writeback-dirty-management` | `scripts/run_l2_r4_fixture.sh FRESH_OUTPUT_DIRECTORY` | `PASS_L2_R4_CAUSAL_FIXTURE` | policy registry SHA-256 `07e97eb46bdec863513113c261dd01253df2f66a8e282e2bb5c27344a05302cf` | Synthetic L2 causality, partial-write and dirty-policy fixture only |
| `research/simple-latency` | `scripts/run_simple_latency_smoke.sh FRESH_OUTPUT_DIRECTORY` | `PASS_SIMPLE_LATENCY_SMOKE` | observed result SHA-256 `0bb052963fd8580ebca4fead4ab56fa657e70083bbcf126a0b97f516218c612d` | Synthetic serial memory-work integration only; expected total is 806.00 ns |
| `research/hbfsim-cosimulation` | `scripts/run_hbfsim_cosim_smoke.sh HBFSim_BINARY FRESH_OUTPUT_DIRECTORY` | `PASS_HBFSIM_CAUSAL_COSIM_SMOKE` | observed result SHA-256 `38c7141d0bcf61112a1ad5a42aff5d9ea89c728377377181e4e1accb154af76a`; HBFSim binary SHA-256 `349be108468f584f5e4ae0acf74b2c72789d4d26cb0f9f3747879cb203f69d2d` | Synthetic dependency/stall propagation only; GDDR timing remains uncalibrated |

The P32D2 model rows remain `BLOCKED` in
`validation/p32d2_branch_status.csv`. A smoke `PASS` must never be reported as
Qwen2.5-1.5B or Meta-Llama-3-8B traffic, latency or bandwidth acceptance.

## 9. Explicit r4 software on main

The L1 geometry in section 4 describes the legacy configuration. The optional r4
(fourth cache candidate) configuration uses a shared-memory capacity table and
can be selected on `main` without accepting its hardware calibration; see
[release configuration](../release/config/README.md) and
[environment](ENVIRONMENT.md) section 5 for the revision record. The legacy
default policy is unchanged, and experimental L2 writes, finite MSHRs, private
sampling certificates and matrix controllers are not promoted by this core
extraction. The current source identity is recorded in
`release/current-core-manifest.json`, separately from the frozen import.
