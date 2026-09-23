# The accepted point does not close either

Status: `MEASURED_2026-09-24_ON_THIS_HOST`. Measured, not inferred. This is a
negative result about reproducibility, and it outranks the earlier P32D2 finding:
the gap is not a property of the small workload.

## 1. What was run

After declaring `evidence_points` (prefill 128, decode 2 and 16), the point with
independent NCU references was collected through the same single entry point:

```bash
./memgen collect --model qwen25_1p5b --prefill-length 128 --decode-steps 2 --gpu-index 1 --cpu 9
```

Work directory `out/collect-20260923T185728Z`. Census closed in 5.33 min with
2306 launches. Job 2 ran every stage to returncode 0 and then stopped at its own
gate.

## 2. Both declared small points stop the same way

| | P32D2 (`…T165342Z`) | P128D2 (`…T185728Z`) |
| --- | --- | --- |
| census launches | 2194 | 2306 |
| target launches (plan) | 2060 | 2172 |
| sampled classes | 176 | 180 |
| profiles accepted / rejected | 150 / 26 | **142 / 38** |
| packed launches | 1360 | **1162** |
| unsupported launches | 700 | **1010** |
| `complete_full_model` | false | **false** |
| followthrough status | `STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC` | same |

A wider prompt did not help: P128D2 rejects *more* classes (38 of 180) and covers
*fewer* launches (1162 of 2172) than the 32-token point. The earlier hypothesis —
that a 32-token workload cannot supply enough CTAs, so a longer prompt would
close — is **wrong**, and this table is why.

## 3. The rejections are the same refusals in both runs

The packer's own messages, counted per run:

| Message | P32D2 | P128D2 |
| --- | --- | --- |
| `sparse heterogeneous CTA structure has no categorical-y witness` | 6 | 12 |
| `categorical-y training or independent holdout missing` | 6 | 6 |
| `sample ordered lane mismatch` | 2 | 6 |
| `source mask differs from effective guard; not silently lowered` | 4 | 4 |
| `non-affine training addresses in CTA xyz coordinates; … at least five training CTAs; … at least seven training CTAs` | 2 | 2 |
| `RMW/atomic or directionless memory op not admitted` | 6 | 6 |

Same families, similar counts, different workloads. That is the signature of a
limit in the pinned sampler rather than in the workload: these classes are
refused because the address-rule fitter cannot fit them, at either size.

## 4. What this means for acceptance

The two points with independent NCU references are P128D2 (DRAM read +0.54%,
write +0.91%) and P128D16 (read +0.20%, write +14.55%), per
[the accuracy report](../evidence/sglang/L2_CACHE_STRATEGY_ACCURACY_REPORT.md).
Those numbers require a full-model expansion. With the sampler as pinned in this
repository, no full-model expansion is reachable at either point:

- P32D2: 700 of 2060 launches unsupported;
- P128D2: 1010 of 2172 launches unsupported.

So the accepted numbers cannot currently be reproduced by this repository, and
nothing in this archive should be read as confirming or contradicting them.
Three explanations were considered:

1. **The sampler revision differs — ruled out for the code that refuses.** The
   refusals come from the address-rule fitter. Its two inputs are pinned by the
   sampler itself, in
   `compact-sources/upstream/template_adapter_r4/template_census.py`: the codec
   (`CODEC_SHA c4b90ea4…`) and `shape_aware_cta_rules.py`
   (`4dcd235e…`). Both files as loaded by this run hash to exactly those values.
   The fit driver `sglang_sample_to_packed.py` does differ from the copy under
   `sampled-workflow-r1/`, but the 22-line difference replaces an inline census
   loop with `profile_census.count_profile` and states that it changes only
   counting cost. So the code that refuses is the revision the archive pins.
2. **The accepted numbers came from a different flow.** Still open, and the
   archive contains a candidate: the historical family
   (`evidence/historical-prefill-d32`, 33 phases / 19,258 kernels per point) did
   reach full inference, and its receipt names the engine and mode it used —
   `hbserve_profile_stream_cache_profile_rules_r7 --mode memgen` from the
   `hyfiss-prefill-matrix-20260913` work root, not the SGLang sampled chain and
   not the frozen `semantic_r17` core. Its binaries are still on this host.
3. **The accepted run used a different plan or a hand-built expansion.** Not
   excluded; it would need the 2026-09-21 work directory to settle.

Until one of 2 or 3 is answered, the honest statement is: this chain, as pinned,
cannot close the points it is compared against, while a different full-inference
flow in this archive demonstrably did cover whole models.

## 5. What can still be measured

`--partial` measures the covered part at either point, labelled:
`out/collect-20260923T165342Z/partial-cache-<UTC>/model/kernel_summary.csv` for
P32D2 (1360 of 2060 launches). P128D2's covered part (1162 of 2172) can be
produced the same way:

```bash
./memgen collect --resume --partial --work out/collect-20260923T185728Z \
  --model qwen25_1p5b --prefill-length 128 --decode-steps 2 --gpu-index 1 --cpu 9
```

Those counters are a lower bound over a reduced access set, exactly as in
[the P32D2 finding](P32D2_COVERAGE_FINDING.md) section 5, and are not comparable
with the NCU reference.

## 6. Two defects this run exposed, both fixed

- `run_job`'s busy retry read the whole accumulated controller log, so a previous
  attempt's `ResourceBusy` made a later attempt's *different* failure look like
  another busy device. It now reads only the current attempt's output.
- A job 2 that stops at its own gate returned 1 with no collection receipt, while
  the runbook documents exit 2 plus coverage. `stopped_at_the_gate()` now
  distinguishes "the chain decided to stop" from "a stage failed", so the receipt
  is written and the exit code matches the documentation.
