# memgen

Memgen turns sampled GPU memory-SASS (assembly memory instructions) into a
full-inference address stream and passes that stream through a functional L1/L2
cache model, so the generated DRAM traffic can be compared against NVIDIA
Nsight Compute (NCU). HBServe is the address-generation layer; Memgen is the
cache and DRAM-traffic layer; NCU is the hardware counter reference used by the
acceptance reports.

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

## What it does

- Declares a closed set of workload points and refuses anything else, so a
  result always names the exact case that produced it.
- Samples one decoder layer sparsely instead of recording a full trace, then
  rebuilds the whole inference address stream from that sample.
- Models 32 KiB L1 per SM and 40 MiB L2 as 128 B lines with four independently
  tracked 32 B sectors, and emits per-kernel cache, hit and DRAM counters.
- Pins every input, source and configuration by SHA-256, and labels each receipt
  with the claims it is allowed to support.

## What it is not

- Not a timing model: no DRAM bank timing, no MSHR queueing, no hardware
  write-back or dirty-release behaviour, no completion scheduling.
- Not an accuracy authority: `validation/p32d2_branch_status.csv` decides
  admission, and it stays `BLOCKED` until a case has its own profile and a
  three-repeat NCU reference. This archive ships no NCU harness for the
  SGLang/BF16 stack.
- Not a trace recorder: the address stream is generated in memory and consumed
  by the cache model. No per-address trace is written, and `memgen
  capabilities` says why.

## Requirements

| Tier | Needs |
| --- | --- |
| Checks and CPU replay | `mpic++`, a C++17 toolchain, `libzstd`, `boost_mpi`, `libcrypto`, Python 3.10+ |
| Real collection | An RTX 4000 Ada Generation GPU, CUDA 12.8 (`nvcc`, `ncu`), NVBit, the SGLang/PyTorch stack, the pinned model checkpoints |

```bash
./memgen check        # host dependencies, pinned package versions, archive pins
./memgen capabilities # the machine-readable claim boundary
```

## Quick start

Collect one Qwen2.5-1.5B P32D2 run: 32 prefill tokens, 2 decode steps, batch 1,
BF16 under SGLang 0.4.10. `P32D2` is the basic admission point declared in
`integrations/sglang/memgen-adapter/contract.json`.

```bash
./memgen cases        # every declared point; P32D2 is one of them
./memgen gpus         # admitted GPUs and the --gpu-index numbering

./memgen plan \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 \
  --gpu-index 1       # writes the job specs, runs nothing

./memgen collect \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 \
  --gpu-index 1       # census, sample, expand, replay
```

Every run writes a fresh timestamped directory under `out/` unless `--work`
nominates one, and existing results are never overwritten. The CLI prints the
underlying command before running it, so the entry point teaches the pipeline
instead of hiding it.

Parameters live in [the runbook](docs/RUNBOOK.md) section 3; the stages and
their budgets are in `./memgen capabilities` and in the same section.

It runs two jobs under the lease controller, which owns the CPU and GPU locks,
the CPU affinity, the memory guard and `CUDA_VISIBLE_DEVICES`: job 1 is the
census, the SGLang run under the NVBit metadata observer; job 2 is the plan, the
sampler build, the sparse sampling, the expansion and the cache replay, which
runs to completion with **no wall-clock deadline**. The run's directory holds
the launch journal and receipts, the sample plan, the packed profile, the
expansion manifest, `kernel_summary.csv` with the per-kernel cache, hit and DRAM
counters, `cache_observation.json` and `collect-receipt.json`.

Because the address stream is generated and streamed straight into the cache
model, **no per-address trace reaches the disk**: `materialized_raw_sass_bytes`
stays 0 and the receipt records `raw_trace_persisted: false`. What persists is
the packed profile and the aggregate counters. Collected counters are **not** an
admission: `validation/p32d2_branch_status.csv` stays `BLOCKED` until P32D2 has
its own profile and an NCU reference. To rehearse the toolchain first, collect
`--prefill-length 128 --decode-steps 32`, the point the archived deployment
drove through this chain.

## Commands

| Command | What it does | GPU |
| --- | --- | --- |
| `memgen check` | host dependencies, pinned package versions, archive pins, optional `--r4` | no |
| `memgen test` | portable regression: entry point, declared cases, driver, pins, census | no |
| `memgen smoke` | frozen engine replayed twice on a synthetic fixture | no |
| `memgen cases` | the declared workload points, with their matrix and model | no |
| `memgen gpus` | admitted GPUs and the `--gpu-index` numbering | no |
| `memgen capabilities` | machine-readable claim boundary, stages and budgets | no |
| `memgen plan` | write the job specs for one case and print the plan | no |
| `memgen collect` | census, sampler build, sparse sampling, expand and replay | **yes** |
| `memgen replay` | an admitted profile stream through the cache model | no |

Run `memgen <command> --help` for options, and `docs/RUNBOOK.md` for the
stage-by-stage form, the expected receipt of each stage and failure handling.

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

## Verify the installation

The first three commands need no GPU, no model and no NVBit, so any reviewer can
run them anywhere. They establish implementation health, not accuracy.

```bash
./memgen check        # add --gpus none on a CPU-only host
./memgen test
./memgen smoke
```

`check` fails if a pinned file drifts from its record. A file that this archive
deliberately revised reports `PRESENT_REVISED` and is explained in
`integrations/sglang/revisions.json`; anything unexplained reports
`PRESENT_DRIFT` and is a failure. Host details, tool builds and the pinned-file
records are in [environment](docs/ENVIRONMENT.md).

A built NVBit tool has two distinct identities: the artifact hash of one build,
which includes the build-id and nvcc's temporary file names, and a content hash
that is stable across rebuilds of the same source. Compare them with
`python3 integrations/sglang/tool_identity.py <so>`.

## Documentation map

| Document | Owns |
| --- | --- |
| This file | What it is, what it is not, requirements, quick start, the command set |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Every command, parameter, artifact, receipt and failure path |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | Host dependencies, pinned-file materialization, how to build the tools, tool build reproducibility |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | The acceptance checklist: what a claim must preserve |
| [`docs/BRANCH_AND_ACCEPTANCE_CONTRACT.md`](docs/BRANCH_AND_ACCEPTANCE_CONTRACT.md) | Normative: branch roles, workload matrices, cache geometry, metrics, gates |
| [`docs/ACCURACY_BASELINES.md`](docs/ACCURACY_BASELINES.md) | The measured numbers, and which experiment family each belongs to |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | Where the sources came from, and licensing observations |

## Repository layout

- `memgen`: the entry point; `memgen_cli/`: its verb implementations.
- `release/`: frozen cache engine source, configuration, fixtures and its
  original validation receipts. Prebuilt binaries are excluded.
- `integrations/sglang/`: the SGLang sampling, packed-profile and HBServe
  adapter sources, plus `preflight.py`, `bootstrap_vendor.py`,
  `collect_case.py` and `tool_identity.py`.
- `tests/`, `scripts/`, `workflow/`: the tests, the verification scripts and the
  pipeline scripts these verbs drive.
- `validation/`: the comparison and admission tables. `evidence/`: address-free
  canonical reports and receipts. `docs/`: the documents above.
- `out/`: generated run directories, gitignored.

No repository-wide licence is assigned. Source origins and licensing
observations are in [provenance](docs/PROVENANCE.md); downstream users must
verify third-party rights before redistribution.

## Optional r4 cache core

r4 is the fourth cache candidate, available on `main` with an explicit
configuration and a required context file; the legacy default is unchanged. See
[configuration and limitations](release/config/README.md) and run
`./memgen check --r4`. It uses synthetic fixtures only, so passing it does not
establish traffic accuracy; the L2 write policy and finite MSHR modelling remain
research.
