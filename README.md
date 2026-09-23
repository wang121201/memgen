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

The pipeline streams generated requests into the cache model; full raw traces,
model weights, NCU report databases and address-bearing workload sidecars are
deliberately excluded from Git.

## Documentation map

| Document | Owns |
| --- | --- |
| This file | What the repository is, the one-command target run, the claim boundary |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Commands: host checks, prerequisites, every stage, parameters, artifacts, failure handling |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | Host dependencies, pinned-file materialization, how to build the tools, tool build reproducibility |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | The acceptance checklist: what a claim must preserve |
| [`docs/BRANCH_AND_ACCEPTANCE_CONTRACT.md`](docs/BRANCH_AND_ACCEPTANCE_CONTRACT.md) | Normative: branch roles, workload matrices, cache geometry, metrics, gates |
| [`docs/ACCURACY_BASELINES.md`](docs/ACCURACY_BASELINES.md) | The measured numbers, and which experiment family each belongs to |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | Where the sources came from, and licensing observations |

## Collect a Qwen2.5-1.5B P32D2 run

This is the target: one batch-one inference with 32 prefill tokens and 2 decode
steps, Qwen2.5-1.5B-Instruct BF16 under SGLang 0.4.10, on an RTX 4000 Ada
Generation GPU. `P32D2` is the basic admission point declared in
`integrations/sglang/memgen-adapter/contract.json`.

**1. Check the host once.** What each check means is in
[verify without a GPU](#verify-without-a-gpu).

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

It runs two jobs under the lease controller, which owns the CPU and GPU locks,
the CPU affinity, the memory guard and `CUDA_VISIBLE_DEVICES`: job 1 is the
census, the SGLang run under the NVBit metadata observer; job 2 is the plan, the
sampler build, the sparse sampling and the expansion and cache replay, which
runs to completion with **no wall-clock deadline**.

Everything lands under `--work`: the ordered launch journal, the census and host
receipts, the sample plan, the packed profile, the expansion manifest,
`kernel_summary.csv` with the per-kernel cache, hit and DRAM counters,
`cache_observation.json`, and `collect-receipt.json` with the artifact map.

Because the address stream is generated and streamed straight into the cache
model, **no per-address trace reaches the disk**: `materialized_raw_sass_bytes`
stays 0 and the receipt records `raw_trace_persisted: false`. What persists is
the packed profile and the aggregate counters.

Collected counters are **not** an admission: `validation/p32d2_branch_status.csv`
stays `BLOCKED` until P32D2 has its own profile and NCU reference.

Parameters, the full artifact table, the stage-by-stage form and the
expected receipt of every stage are in [the runbook](docs/RUNBOOK.md). To
rehearse the toolchain first, use `--prefill-length 128 --decode-steps 32`, the
point the archived deployment actually drove through this chain.

## Claim boundary

`main` is the stable branch; three research branches stay separate. A research
result must not become the `main` default merely because one workload improves.
Branch roles, workload matrices, cache geometry, metric denominators and the
admission gates are normative in
[the branch and acceptance contract](docs/BRANCH_AND_ACCEPTANCE_CONTRACT.md).

The five-point P64D32 to P1024D32 table is a historical llama.cpp/Q8_0
baseline, not current SGLang evidence. At the frozen SGLang boundary only
P128D2 and P128D16 are independently closed against NCU, with aggregate DRAM
read errors 0.54% and 0.20% and write errors 0.91% and 14.55%. Decode-only write
stays inaccurate, so no generally accurate NVIDIA write-back or dirty-release
behaviour is claimed. Numbers: [accuracy baselines](docs/ACCURACY_BASELINES.md).

## Verify without a GPU

`preflight.py` answers "is this machine ready": dependency paths, pinned package
versions, the admitted GPU pool and the `--gpu-index` numbering.
`bootstrap_vendor.py` materializes the six pinned controller files the archive
does not carry. The next four answer "is the frozen archive intact" and need no
GPU, model or NVBit, so any reviewer can run them anywhere.

```bash
python3 integrations/sglang/preflight.py                  # add --gpus none on a CPU-only host
python3 integrations/sglang/bootstrap_vendor.py --check    # drop --check to materialize
python3 -B scripts/verify_archive.py                     # manifests, pins, forbidden artifacts
bash scripts/run_cpu_smoke.sh /tmp/memgen-cpu-smoke-r1   # frozen engine, twice, equal output
python3 -B tests/sglang/test_declared_cases.py           # declared cases, driver, pin structure
python3 -B tests/sampling/test_profile_census.py         # sector-census differential tests
```

These need `mpic++`, a C++17 toolchain and the zstd/boost/OpenSSL libraries.
Passing them is implementation health only;
[`validation/p32d2_branch_status.csv`](validation/p32d2_branch_status.csv)
remains the authority for hardware-accuracy admission. Host details, tool builds
and the pinned-file records are in [environment](docs/ENVIRONMENT.md).

## Repository layout

- `release/`: frozen cache engine source, configuration, fixtures and its
  original validation receipts. Prebuilt binaries are excluded.
- `integrations/sglang/`: the SGLang sampling, packed-profile and HBServe
  adapter sources, plus `preflight.py`, `bootstrap_vendor.py`,
  `collect_case.py` and `tool_identity.py`.
- `validation/`: the comparison and admission tables. `evidence/`: address-free
  canonical reports and receipts. `docs/`: the documents above.

No repository-wide licence is assigned. Source origins and licensing
observations are in [provenance](docs/PROVENANCE.md); downstream users must
verify third-party rights before redistribution.

## Optional r4 cache core

r4 is the fourth cache candidate, available on `main` with an explicit
configuration and a required context file; the legacy default is unchanged. See
[configuration and limitations](release/config/README.md) and run
`python3 -B scripts/test_cache_core.py --output /absolute/fresh/test-directory`.
It uses synthetic fixtures only, so passing it does not establish
traffic accuracy; the L2 write policy and finite MSHR modelling remain research.
