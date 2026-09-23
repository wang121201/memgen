# Memgen: HBServe memory generation and GPU cache filtering

This repository archives the source needed to turn sampled GPU memory-SASS
(assembly memory instructions) into a full-inference address stream and pass
that stream directly through a functional GPU cache model. HBServe is the
address-generation layer; Memgen is the L1/L2 cache and DRAM-traffic layer.
NCU means NVIDIA Nsight Compute and is the hardware-counter reference used by
the acceptance reports.

The stable path is:

```text
SGLang workload
  -> sparse NVBit memory-SASS sample and CTA placement
  -> packed, workload-specific profile
  -> HBServe full-inference address generation
  -> naïve cache or two-level Memgen L1/L2 filter
  -> aggregate cache-hit and DRAM read/write counters
  -> same-workload NCU comparison
```

The pipeline streams generated requests into the cache model. Full raw memory
traces, model weights, NCU report databases and address-bearing workload
sidecars are deliberately excluded from Git.

## Collect a Qwen2.5-1.5B P32D2 run

This is the target: one batch-one inference with 32 prefill tokens and 2 decode
steps, Qwen2.5-1.5B-Instruct BF16 under SGLang 0.4.10, on an RTX 4000 Ada
Generation GPU. `P32D2` is the basic admission point declared in
`integrations/sglang/memgen-adapter/contract.json`.

**1. Check the host once.** Details and meaning are in the next section.

```bash
python3 integrations/sglang/preflight.py          # dependency paths, packages, GPU pool
python3 integrations/sglang/bootstrap_vendor.py   # materialize pinned files if absent
```

**2. See what can be collected, then review the plan** without spending GPU
time:

```bash
python3 integrations/sglang/collect_case.py --list-cases

python3 integrations/sglang/collect_case.py \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 \
  --gpu-index 1 --work /tmp/qwen15b-p32d2-r1 --dry-run
```

**3. Collect it:**

```bash
python3 integrations/sglang/collect_case.py \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 \
  --gpu-index 1 --work /tmp/qwen15b-p32d2-r1
```

### Parameters

| Parameter | Meaning |
| --- | --- |
| `--model`, `--prefill-length`, `--decode-steps` | the case to collect, as three independent values. Any declared combination works; `--list-cases` prints all 26 |
| `--case` | shorthand for the three, e.g. `qwen25_1p5b-p32-d2` |
| `--list-cases` | print every declared case with its matrix name, then exit |
| `--work DIR` | fresh output directory. It must not exist; a retry needs a new one |
| `--gpu-index N` | which admitted GPU, using the index `preflight.py` prints. Default 0 |
| `--gpu UUID` | the same choice by UUID, for scripted callers |
| `--cpu N` | one CPU id from the shared `0..15` pool. Default 8 |
| `--python PATH` | interpreter that carries the SGLang stack |
| `--census-seconds`, `--sample-seconds` | budgets for the two GPU stages |
| `--job-seconds` | job 2 wall-clock ceiling. `0`, the default, runs to completion |
| `--observer PATH` | reuse a built observer; otherwise one is built into `--work` |
| `--dry-run` | write the two job specs and print the plan, execute nothing |

The triple is validated against the declaration in
`integrations/sglang/memgen-adapter/contract.json` before anything runs, so a
combination that no declared matrix contains fails at once and tells you to run
`--list-cases`. Changing the workload needs no other edit: those three values
flow into the census host, the sample plan and the replay command. To rehearse
the toolchain first, use `--prefill-length 128 --decode-steps 32`, the point the
archived deployment actually drove through this chain.

**The cache replay is not given a wall-clock deadline.** `run_memgen.py`
documents that it has none, and `followthrough.py`, `profile_cache.py` and this
driver no longer add one. Only the two GPU stages carry budgets, because they
hold a leased device.

It runs two jobs under the lease controller,
which owns the CPU and GPU locks, the CPU affinity, the memory guard and
`CUDA_VISIBLE_DEVICES`:

| Job | Stage | Device | Budget |
| --- | --- | --- | --- |
| 1 | census: the SGLang run under the NVBit metadata observer | 1 GPU | 1800 s |
| 2 | plan: select one decoder layer and the sample set | CPU | 300 s |
| 2 | build: compile the sparse sampler from the real plan | CPU | 900 s |
| 2 | sample: sparse memory-SASS sampling and profile fitting | 1 GPU | 7200 s |
| 2 | expand and replay: address generation and the L1/L2 cache filter | CPU | to completion |

What you get, all under `--work`:

| Artifact | Meaning |
| --- | --- |
| `observers/<case>-census/process-*/launch-journal.jsonl` | every kernel launch, ordered |
| `observers/<case>-census/process-*/finish.json` | `PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE` |
| `runs/<case>-census/host/process-*/finish.json` | measured phases and tensor metadata |
| `runs/<case>-collect/followthrough/plan/sample-plan.json` | which layer and CTAs are sampled |
| `.../sample/profiles/profiles.index.jsonl` | the packed, workload-specific profile |
| `.../expanded/manifest.json` | `complete_full_model` and unsupported-launch counts |
| `.../cache/model/kernel_summary.csv` | per-kernel cache, hit and DRAM counters |
| `.../cache/model/cache_observation.json` | occupancy, writeback, residual checks |
| `collect-receipt.json` | case identity, stage receipts and the artifact map |

The full-inference address stream is generated and streamed straight into the
cache model, so **no per-address trace reaches the disk**: `materialized_raw_sass_bytes`
stays 0 and the receipt records `raw_trace_persisted: false`. What persists is
the packed profile and the aggregate counters.

Collected counters are **not** an admission. A P32D2 accuracy claim additionally
needs an independent three-repeat NCU reference over the same ranges and
denominators, which this archive does not contain for the SGLang/BF16 stack.
`validation/p32d2_branch_status.csv` therefore stays `BLOCKED`.

[`docs/RUNBOOK.md`](docs/RUNBOOK.md) gives the same stages one command at a
time, with the expected receipt status of each and the failure-triage order.

## Branch contract

- `main` is the stable integration and acceptance branch. It contains the
  frozen functional cache release, the SGLang sampling/HBServe adapter, bounded
  fixtures, machine-readable baselines and reproduction instructions.
- `research/l2-writeback-dirty-management` is the mechanism-research branch.
  It adds L2 indexing/replacement variants, write-back policies, dirty-sector
  ownership and release experiments. A research result must not become the
  `main` default merely because one workload improves.
- `research/simple-latency` integrates configurable constant latency over
  cache and DRAM events. It is a serial memory-work diagnostic without
  scheduling, backpressure or compute overlap.
- `research/hbfsim-cosimulation` is the dependency-aware path where HBFSim
  completion can delay future memory issue. It requires compute/issue sideband
  and calibrated memory timing before hardware timing claims are allowed.

The common SGLang workload matrix, RTX 4000 Ada cache geometry, metric
denominators and the per-decode DRAM-write gate are defined in
[the branch and acceptance contract](docs/BRANCH_AND_ACCEPTANCE_CONTRACT.md).

## Evidence boundary

The five-point P64D32 through P1024D32 table is preserved as a historical
regression baseline. It used Qwen2.5-1.5B-Instruct Q8_0 with llama.cpp, not the
current SGLang/BF16 stack. Current SGLang acceptance uses Qwen2.5-1.5B-Instruct
BF16, SGLang 0.4.10 and an RTX 4000 Ada Generation GPU. These two experiment
families share an output schema but their numerical values are not
interchangeable. See [accuracy baselines](docs/ACCURACY_BASELINES.md).

At the current frozen SGLang boundary, P128D2 and P128D16 are independently
closed against NCU. Aggregate DRAM read errors are 0.54% and 0.20%; aggregate
DRAM write errors are 0.91% and 14.55%. Decode-only write remains inaccurate,
so this repository does not claim generally accurate NVIDIA write-back timing
or dirty-release behavior.

## Check the host and the archive (no GPU)

Two different questions, two different command groups. Neither one collects
workload data; that is the section above.

**Is this machine ready to run the target?** `preflight.py` verifies the
dependency paths, the pinned package versions and the admitted GPU pool, and
prints the `--gpu-index` numbering that `collect_case.py` accepts.
`bootstrap_vendor.py` materializes the six pinned controller files the archive
does not carry. Both are read-only until told to write.

```bash
python3 integrations/sglang/preflight.py                  # add --gpus none on a CPU-only host
python3 integrations/sglang/bootstrap_vendor.py --check    # drop --check to materialize
```

**Is the frozen archive still intact after a change?** These need no GPU, no
model and no NVBit, so any reviewer can run them anywhere.

```bash
python3 -B scripts/verify_archive.py                     # manifests, pins, forbidden artifacts
bash scripts/run_cpu_smoke.sh /tmp/memgen-cpu-smoke-r1   # frozen engine, twice, equal output
python3 -B tests/sglang/test_declared_cases.py           # declared cases and rejection rules
python3 -B tests/sampling/test_profile_census.py         # sector-census differential tests
```

They need `mpic++`, a C++17 toolchain and the zstd/boost/OpenSSL libraries.
`verify_archive.py` rejects bytecode caches, model files, NCU databases and any
file over 10 MiB, so invoke the Python entry points with `-B` as shown.

A built NVBit tool has two distinct identities: the artifact hash of one build,
which includes the build-id and nvcc's temporary file names, and a content hash
that is stable across rebuilds of the same source. Compare them with:

```bash
python3 integrations/sglang/tool_identity.py /absolute/fresh/observer-build/observer.so
```

The bounded commands for the three research branches and their latest
deterministic evidence identities are recorded in
[`validation/branch_smoke_status.csv`](validation/branch_smoke_status.csv).
These checks establish implementation health only. The separate
[`validation/p32d2_branch_status.csv`](validation/p32d2_branch_status.csv)
remains the authority for Qwen2.5-1.5B and Meta-Llama-3-8B hardware-accuracy
admission.

Full SGLang/NCU reproduction requires the external runtime, model and GPU listed
in [reproduction](docs/REPRODUCTION.md). It does not require committing or
retaining a full raw memory trace.

## Repository layout

- `release/`: frozen cache engine source, configuration, small fixtures and
  its original validation receipts; prebuilt binaries are excluded.
- `integrations/sglang/`: SGLang sampling, packed-profile and HBServe adapter
  sources used by the current full-inference workflow, plus the host preflight
  and pinned-file materialization entry points.
- `validation/`: machine-readable historical and current SGLang comparison
  tables.
- `evidence/`: small, address-free canonical reports and receipts.
- `docs/`: provenance, metric definitions, environment prerequisites, the
  reproduction boundaries and the operator runbook.

No repository-wide license is assigned by this archival change. Source-level
origin and licensing observations are recorded in
[provenance](docs/PROVENANCE.md); downstream users must verify third-party
rights before redistribution.

## Optional r4 cache core

The r4 (fourth cache candidate) software is available on main with explicit
configuration and context. The legacy configuration remains unchanged.
See [configuration and limitations](release/config/README.md).
Build and run the current CPU (central processing unit) regression with:

```bash
python3 scripts/test_cache_core.py --output /absolute/fresh/test-directory
```

This includes the historical cache smoke test, an independent CLOCK reference,
configuration rejection and source/context failure checks. It stores summaries
and compact synthetic fixtures, not expanded address traces. Passing it does
not establish NVIDIA hardware-traffic accuracy. The L2 (level-two cache) write
policy and finite MSHR (miss-status holding register) modeling remain research.
