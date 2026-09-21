# Source and evidence provenance

This document defines the imported snapshots. SHA-256 means Secure Hash
Algorithm 256-bit content digest. CTA means Cooperative Thread Array (CUDA
thread block). NVBit is NVIDIA's dynamic binary instrumentation framework. NCU
means NVIDIA Nsight Compute. No raw address trace or model weight is included.

## Frozen cache release

Imported from the copy-only XMU snapshot:

`/home/xmu/nvidiagds/codex-runs/memgen-paper-ada-v1-20260916-01a08d87-r1`

The source release status is `PASS_FROZEN_PAPER_WORKFLOW`; it passed build,
geometry, synthetic smoke and real two-kernel prefix checks. Its original
`release-manifest.json` SHA-256 is
`a97b93c93247acf42c542453f3b936cf726ef71db5212dd04f82f8cba678eeda`.
The frozen engine is `hbserve_profile_stream_cache_semantic_r17.cpp`; its
published build did not materialize raw traces. The original receipt explicitly
states `hardware_acceptance=false`, so the functional release alone is not a
hardware-accuracy claim.

Prebuilt binaries, `evidence-export.tar.gz`, Python bytecode and compressed
profile payloads are excluded. Small source, configuration, fixtures and
address-free validation receipts are retained.

## SGLang integration snapshot

Imported from the copy-only source subset under:

`/home/xmu/nvidiagds/simulators/hyfiss/analysis/full-inference-matrix-20260919-r1/sampled-workflow-r1`

The retained subset is `compact-sources-r1`, `memgen-adapter-r1` and the small
top-level orchestration sources. It contains the NVBit sampler, SGLang-to-packed
conversion, HBServe template adapter, profile expansion and direct Memgen
launch path. Runtime caches, generated Triton/CUDA products, model files and
experiment outputs are excluded.

## Historical five-point evidence

The remote experiment root was:

`/home/xmu/nvidiagds/codex-runs/llm-footprint-v1/memgen/hyfiss-prefill-matrix-20260913-01a06837-r1`

The canonical address-free report was copied from the verified local mirror
`D:\codexdataspace\reports\hyfiss-prefill-matrix-20260913-01a06837-r3`:

- `finish.json`: 659,858 bytes, SHA-256
  `996e26bb170d6469bb06fb2ff79a9b138594e260f63cae5275f9b558501960aa`.
- `tables.md`: 25,092 bytes, SHA-256
  `45c5686bdf8b1eb50e2f6238a54cca69f9ac6d93db17dbeac3671b744de79d9b`.

The model was Qwen2.5-1.5B-Instruct Q8_0 under llama.cpp. This evidence is a
historical regression and must not be relabeled as SGLang/BF16 acceptance.

## Current SGLang evidence

The self-contained report in `evidence/sglang/L2_CACHE_STRATEGY_ACCURACY_REPORT.md`
freezes the current Qwen2.5-1.5B-Instruct BF16 + SGLang 0.4.10 evidence. Only
P128D2 and P128D16 are independently complete. Its policy experiments and
write-back conclusions belong on the research branch; `main` keeps the report
to define the acceptance boundary, not to promote a research policy.

## Licensing observation

The archived HBServe material observed in its source bundle is MIT-licensed.
The inspected HyFiSS checkout points to
`https://github.com/ConvolutedDog/HyFiSS.git` but did not contain a root license
file at the inspected revision; only `parda/LICENSE` was present. This archive
therefore does not assert a repository-wide license or grant rights beyond
those already held by the source owners. Attribution and license review are
required before third-party redistribution.

