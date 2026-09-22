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

## Quick verification

The archive-level check is read-only:

```bash
python3 scripts/verify_archive.py
```

The bounded CPU smoke builds the frozen engine and writes output only to a
fresh caller-selected directory:

```bash
scripts/run_cpu_smoke.sh /tmp/memgen-cpu-smoke-r1
```

The corresponding bounded commands for the three research branches and their
latest deterministic evidence identities are recorded in
[`validation/branch_smoke_status.csv`](validation/branch_smoke_status.csv).
These checks establish implementation health only. The separate
[`validation/p32d2_branch_status.csv`](validation/p32d2_branch_status.csv)
remains the authority for Qwen2.5-1.5B and Meta-Llama-3-8B hardware-accuracy
admission.

Full SGLang/NCU reproduction requires the external runtime, model and GPU
listed in [reproduction](docs/REPRODUCTION.md). It does not require committing
or retaining a full raw memory trace.

## Repository layout

- `release/`: frozen cache engine source, configuration, small fixtures and
  its original validation receipts; prebuilt binaries are excluded.
- `integrations/sglang/`: SGLang sampling, packed-profile and HBServe adapter
  sources used by the current full-inference workflow.
- `validation/`: machine-readable historical and current SGLang comparison
  tables.
- `evidence/`: small, address-free canonical reports and receipts.
- `docs/`: provenance, metric definitions and reproduction boundaries.

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
