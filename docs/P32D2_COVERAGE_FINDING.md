# Why P32D2 does not close, measured

Status: `MEASURED_2026-09-24_ON_THIS_HOST`. This is a finding about one workload
point, not an accuracy result. Nothing here changes an admission, and
`hardware_accuracy_accepted` stays `false` everywhere it appears.

The question was whether one Qwen2.5-1.5B P32D2 run (32 prefill tokens, 2 decode
steps, batch 1, BF16 under SGLang 0.4.10) produces L1, L2 and DRAM traffic. It
runs, and it does not close. This is what stops it.

## 1. What was run

```bash
./memgen collect --model qwen25_1p5b --prefill-length 32 --decode-steps 2 --gpu-index 1
./memgen collect --resume --partial --work out/collect-20260923T165342Z \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 --gpu-index 1
```

Both under the lease controller, on `GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13`
(RTX 4000 Ada, 48 SM), with `out/collect-20260923T165342Z` as the run directory.
The census ran once; the second command re-verified it and continued from it.

## 2. What closed

| Stage | Status | Wall | Note |
| --- | --- | --- | --- |
| census | `PASS_PROCESS_ONLY` | 5.11 min | observer pid 946923, 2194 launches, 10.3 MB of 256 MiB metadata |
| plan | rc 0 | 0.01 min | 2060 launches, 176 selected classes, primary layer 0 |
| sampler build | rc 0 | 0.16 min | `sampler.so` for the planned classes |
| sparse sample (GPU) | `PASS_SINGLE_LAYER_SAMPLES_AND_PROFILE_FITTING` | 5.69 min | 611672 selected records, 150 of 176 classes packed |
| expand | rc 0 | 0.45 min | `PARTIAL_PROFILE_EXPANSION_UNSUPPORTED_RETAINED` |
| cache replay | **not run** | — | the gate below stops the chain first |

Every stage that ran returned 0. The chain stopped because it refuses to emit a
full-model number from an incomplete stream, which is the documented behaviour
of `followthrough.py`:

```python
if not manifest['complete_full_model']:
    result.update(status='STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC', ...)
```

`expanded/manifest.json` reports `target_launches: 2060`, `packed_launches:
1360`, `unsupported_launches: 700`, `complete_full_model: false`,
`complete_declared_profile_stream: false`.

## 3. The 700 are two different problems

First, what is **not** the problem: the expansion does reach the whole model.
`plan/layer-bindings.json` names, for every target launch, the sampled launch
that supplies its profile. In this run 1884 of the 2060 target launches (91%)
reuse a primary-layer profile, and those targets are **layers 1 through 27**, 64
to 70 launches per layer. One layer's packed profile is expanded onto every layer
of the model, which is exactly the full-model expansion. It worked for 1360 of
the launches; where it did not, the limit is in the *source* profile set, not in
the expansion.

Cross-referencing `plan/layer-bindings.json` (2060 bindings),
`sample/profiles/profiles.index.jsonl` (150 packed profiles) and
`expanded/manifest.json` gives an exact split. Both halves are 350:

| Count | Reason as recorded | Mechanism |
| --- | --- | --- |
| 350 | `Source profile missing/unsupported` | the template the binding names has no packed profile: 324 same-signature expansions, 12 primary-layer, 14 phase-global |
| 350 | `Ambiguous observed tensor binding: conflicting target deltas` | the template is packed, but two candidate source intervals contain the rule with different deltas, so `expand_profiles.py:211` refuses to pick one |

Neither is a crash. The first says a sampled class never produced a profile; the
second says the layer-to-layer rebinding could not be resolved without guessing.

## 4. Root cause of the missing profiles: the sample is too small to fit rules

`sample/profiles/receipt.json` accounts for all 176 sampled classes: 170
`SOURCE_DECODED_FOR_PACKED_CALLBACK`, 6 `UNLOWERED`. Only 150 reached
`profiles.index.jsonl`. The packer records why for each of the other 26 — these
are the message strings, counted:

| Count | Message |
| --- | --- |
| 6 | `sparse heterogeneous CTA structure has no categorical-y witness` |
| 6 | `categorical-y training or independent holdout missing` |
| 6 | `RMW/atomic or directionless memory op not admitted` (`at::native::reduce_kernel<512, 1, ReduceOp<float, ...>>`, SGLang's sampling path) |
| 4 | `source mask differs from effective guard; not silently lowered` |
| 2 | `non-affine training addresses in CTA xyz coordinates; ... x-floor rule requires at least five training CTAs; axis-permutation rule requires at least seven training CTAs` |
| 2 | `sample ordered lane mismatch` |

Every one of these is a statement that the sample does not contain enough
structure to fit an address rule: not enough training CTAs, no categorical-y
witness, no independent holdout, a mask that differs from the guard. The
thresholds are visible in the text — *at least five training CTAs*, *at least
seven training CTAs*, *y or z extent > 1*.

A 32-token prefill with 2 decode steps gives exactly those kernels very few
CTAs. The sampler refuses to lower them, and the expansion then cannot cover the
launches that would reuse them. The two points that do have independently
accepted hardware evidence, P128D2 and P128D16, required a complete expansion to
produce their numbers, and their prompt is four times longer. So this is a
property of the workload point, not a defect in this run:

- the refusals are deliberate, and each names the condition it needed;
- the blocking classes are the small-grid and phase-global kernels, which a
  32-token workload cannot supply in quantity;
- no configuration in the current chain can turn refusals into profiles.

What was **not** established here: whether a longer prompt is sufficient on its
own, since P128D32 was only driven as far as the sample in the archived
deployment and its expansion was not available to inspect. Section 6 keeps that
as an experiment, not a conclusion.

## 5. What the covered part measures

`--partial` replays the 1360 covered launches with
`run_memgen.py --allow-partial-diagnostic`. The counters are real cache-model
output over 66% of the launches and **are not the case traffic**: the missing
700 are not modelled and are not zero.

Measured, 2026-09-24, over the 1360 covered launches
(`partial-cache-<UTC>/model/kernel_summary.csv`, 1360 rows, summed; hit rates are
ratios of sums, not means of per-kernel rates):

| Quantity | Value |
| --- | --- |
| L1 requests / hits / hit rate | 876,507,418 / 150,343,188 / 0.171525 |
| L2 requests / hits / hit rate | 732,232,526 / 179,068,404 / 0.244551 |
| DRAM read | 17,588,568,640 B (16.38 GiB) |
| DRAM write | 135,537,920 B (129.3 MiB) |
| L2 writeback dirty sectors | 4,235,560 |
| generated memory instructions / lane addresses | 415,947,152 / 13,256,768,256 |

How to read this table:

- the DRAM totals are **lower bounds**, because 700 launches of the run are not
  modelled at all;
- the hit rates are ratios over the launches that were modelled, so they are not
  the point's hit rate and must not be compared with a full-model number;
- the replay receipt says `PASS_PARTIAL_MODEL_CACHE_DIAGNOSTIC` with
  `complete_full_model: false`, `unsupported_launches: 700` and
  `hardware_accuracy_accepted: false`.

The receipt labels this: the collect receipt carries `partial_diagnostic` with
`target_launches`, `packed_launches`, `unsupported_launches` and the counter
values it read, and the exit code stays 2 because the model is still not covered.

Counter values for this run are recorded in section 8 of the run receipt at
`out/collect-20260923T165342Z/collect-receipt.json` under `partial_diagnostic`.

## 6. What to do about it

Ordered by cost, with what each one buys:

1. **Accept P32D2 as partial.** Zero cost. Quote nothing as the P32D2 traffic;
   use the partial counters only to see shape, not totals.
2. **Measure a point whose grids are wide enough.** `P128D32` is declared in the
   `scale_series` matrix and is the point the archived deployment drove through
   this chain, so it is the cheapest declared point that can plausibly close. It
   costs a full census and a larger sample.
3. **Improve the fitter for small grids.** This is where a fix would live:
   the fallback chain in the sampler's address-rule fitting, and the
   `Ambiguous observed tensor binding` resolution in `expand_profiles.py`. Both
   are upstream sampler work, not driver work, and both need the refusal
   conditions to stay honest.
4. **Do not** lower the refusals to make the numbers appear. A zero-filled or
   guessed region is indistinguishable from measured traffic in a summary table.

## 7. A separate gap this run exposed

The two points with independently accepted hardware evidence — P128D2 and
P128D16, per
[the accuracy report](../evidence/sglang/L2_CACHE_STRATEGY_ACCURACY_REPORT.md) —
are not in the declared matrix. `contract.json` declares `scale_series` as
prefills {128, 256, 512, 1024} × decodes {32, 64, 128} plus the P32D2 basic
admission points, so `./memgen collect --prefill-length 128 --decode-steps 2` is
refused as undeclared. The chain can therefore reproduce the accepted points
only after they are declared, and the declared matrix cannot currently express
the evidence it is being compared against. That is a contract decision, not a
code defect.

## 8. Proposed update to the admission table

`validation/p32d2_branch_status.csv` currently records the Qwen P32D2 row as
`traffic_input=missing`, reason `P128D2 cannot be renamed P32D2`. That was true
before this run; now a partial profile exists. Suggested row, keeping the status
blocked and the admission false:

```csv
main,Qwen2.5-1.5B-Instruct BF16,P32D2,partial_profiles_available,missing,not_applicable,BLOCKED_PARTIAL_PROFILE_EXPANSION,false,1360 of 2060 launches covered; 26 of 176 sampled classes have no rule fit for a 32-token workload
```

This file is the admission authority, so it was not edited here.

## 9. Reproduce

```bash
cd <repository root>

# implementation health, no GPU
./memgen check && ./memgen test && ./memgen smoke

# the collection itself, 32 prefill tokens and 2 decode steps
./memgen collect --model qwen25_1p5b --prefill-length 32 --decode-steps 2 --gpu-index 1

# counters for the covered part, labelled partial; exit stays 2
./memgen collect --resume --partial --work <the run directory> \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 --gpu-index 1
```

Read coverage from `<work>/runs/qwen25_1p5b-p32-d2-collect/followthrough/expanded/manifest.json`
and the per-stage reasons from
`<work>/runs/qwen25_1p5b-p32-d2-collect/followthrough/sample/profiles.stdout`.

## 10. Evidence

| Artifact | Path under `out/collect-20260923T165342Z/` |
| --- | --- |
| census journal and observer receipt | `observers/qwen25_1p5b-p32-d2-census/process-946923-918204006/` |
| measured phases and tensor metadata | `runs/qwen25_1p5b-p32-d2-census/host/process-946923/` |
| plan, bindings and the 176 selected classes | `runs/qwen25_1p5b-p32-d2-collect/followthrough/plan/` |
| packed profiles and the 26 refusals | `runs/qwen25_1p5b-p32-d2-collect/followthrough/sample/profiles/` and `profiles.stdout` |
| coverage, accepted and unsupported | `runs/qwen25_1p5b-p32-d2-collect/followthrough/expanded/manifest.json` |
| collection receipt with `partial_diagnostic` | `collect-receipt.json` |

Counters, if the partial replay has been run, are in
`partial-cache-<UTC>/model/kernel_summary.csv`; its columns are described in the
README under "Where the traffic numbers are".
