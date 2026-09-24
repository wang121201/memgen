# What the sampler refuses, and how far one layer carries

Status: `MEASURED_2026-09-24_ON_THIS_HOST`. Two questions, answered from the runs
and from the code that does the refusing. No admission changes.

## 1. Why a class is refused

Every unsupported launch traces back to one of two decisions:

1. **The class never produced a profile.** `p.fit()` in
   `compact-sources/upstream/sglang_sample_to_packed.py` fits a compact
   *address-rule template* for a kernel class from a handful of sampled CTAs,
   split into training and holdout. If no rule in the cascade can reproduce the
   observed addresses, the class is refused. The cascade lives in
   `release/workflow/hyfiss_sampled_sass_trace_profile_rules_r12.py`, and its
   inputs are pinned by the sampler itself in `template_census.py` (codec
   `c4b90ea4…`, `shape_aware_cta_rules.py` `4dcd235e…`); both files as loaded
   hash to those values.
2. **The template exists but cannot be rebound** to the target launch's tensors,
   because two candidate source intervals contain the rule with different deltas
   (`expand_profiles.py`, "Ambiguous observed tensor binding").

The cascade, in order, with the condition each branch needs:

| Branch | Needs | Refusal when it fails |
| --- | --- | --- |
| affine in CTA x | addresses lie on a line in CTA coordinates | `non-affine training addresses in CTA xyz coordinates` |
| x-linear / y-categorical | a categorical y axis **and** an independent holdout CTA | `categorical-y training or independent holdout missing`, `sparse heterogeneous CTA structure has no categorical-y witness` |
| geometry-fixed z-half | `grid.z >= 4` and a `z = 0` anchor | `z-half partition requires z extent >= 4` |
| tiled x/y/z | y or z extent > 1 | `tiled multidimensional rule requires y or z extent > 1` |
| unique x-floor | a one-dimensional grid, **>= 5 training CTAs**, an `x = 0` anchor, >= 3 observed address levels, a within-group plateau, a uniquely identifiable divisor | `x-floor rule requires at least five training CTAs` |
| unique axis permutation | a one-dimensional grid, **>= 7 points**, >= 4 address levels | `axis-permutation rule requires at least seven training CTAs` |

Two further refusals are correctness gates rather than fitting failures:

- `source mask differs from effective guard; not silently lowered` — the
  observed load mask is not the effective guard, so admitting it would drop
  bytes silently;
- `RMW/atomic or directionless memory op not admitted` — a read-modify-write has
  no direction to attribute bytes to, so it is left `UNLOWERED`.

The thresholds are about **identifiability**, not resources: below five or seven
points, several different rules reproduce the same observations, and the fitter
refuses instead of picking one. A guessed rule would emit plausible addresses
that no observation supports, which is exactly what the archive's provenance
rules exist to prevent.

The classes that hit this, in these two runs:

| Family | Class | P32D2 | P128D2 | Refusal |
| --- | --- | --- | --- | --- |
| attention post-processing | `flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheHeadPara…` | 6 | 6 | categorical-y / holdout missing |
| paged / ragged attention | `flashinfer::BatchPrefillWithPagedKVCacheKernel`, `…WithRaggedKVCacheKernel` | 4 | 6 | source mask differs from effective guard, sparse heterogeneous CTA structure |
| state merge | `flashinfer::PersistentVariableLengthMergeStatesKernel` | — | 3 | sparse heterogeneous CTA structure |
| elementwise / copy | `at::native::elementwise_kernel`, `unrolled_elementwise_kernel` | 8 | 8 | non-affine in CTA xyz, sparse heterogeneous CTA structure |
| gather | `at::native::indexSelectLargeIndex…` | 4 | — | sample ordered lane mismatch |
| GEMM | `cutlass::Kernel2`, `ampere_bf16_s16816gemm_…` | — | 6 | sample ordered lane mismatch |
| reduction | `at::native::reduce_kernel<…, ReduceOp<float, …>>` | 6 | 8 | RMW/atomic not admitted |

Counts per run: P32D2 refuses 20 fitted classes plus 6 atomic ones (26 of 176);
P128D2 refuses 30 plus 8 (38 of 180). The extra classes at the longer prompt are
the GEMMs and the ragged-attention variant, which is why the wider point gets
*worse* rather than better.

## 2. How far one layer carries

The chain expands **layers only**. Measured over every binding:

| | P32D2 | P128D2 |
| --- | --- | --- |
| bindings | 2060 | 2172 |
| own-sampled (no reuse) | 176 | 180 |
| reuse another launch's template | 1884 | 1992 |
| **cross-phase bindings** | **0** | **0** |
| **cross-epoch bindings** | **0** | **0** |
| coverage of layer 0 (the template layer) | 0.83 | 0.68 |
| coverage of layers 1..27 | 0.63–0.66 | 0.49–0.53 |

Two conclusions, both directly answering the question:

- **Carrying to other layers is uniform.** Every layer from 1 to 27 lands inside
  the same 3-point band, because a class either has a usable template — and then
  all 27 siblings receive it — or it has none, and then all 27 miss it. The cost
  of the rebinding itself is the gap between the template layer and its
  siblings: 0.83 → 0.63–0.66 (P32D2) and 0.68 → 0.49–0.53 (P128D2), i.e. the
  ambiguous-delta refusals.
- **Nothing is carried across decode steps or across warmup/measurement.** Every
  binding stays inside one `(epoch, phase)`; the template is the same phase of a
  different layer. Each decode step is planned and sampled in the same run, so
  there is no decode-step expansion to evaluate — the question does not apply by
  construction.

Per phase, coverage still differs because the *classes* differ:

| Phase | P32D2 | P128D2 |
| --- | --- | --- |
| Prefill | 498/776 = 0.642 | 344/776 = 0.443 |
| Decode1 | 431/642 = 0.671 | 409/698 = 0.586 |
| Decode2 | 431/642 = 0.671 | 409/698 = 0.586 |

Decode1 and Decode2 match exactly, which is what independently planned,
identically structured phases should look like.

## 3. Re-sampling per point is already the design

Changing model, prompt length or decode steps produces a different declared point,
and each point gets its own census and its own sample: `collect` requires a fresh
`--work`, and the planner refuses a census that is not complete for the point
(`Complete warmup/measurement phase population required`,
`Original layer census incomplete`). No profile, rule or template is reused
across points. The only reuse inside a point is the layer dimension measured
above.

## 4. What closing would require

For a point to reach `complete_full_model` under the default posture, both must
hold:

1. every sampled class fits, i.e. 0 of 176 (P32D2) or 0 of 180 (P128D2) refusals.
   That means extending the cascade for the families in section 1 — attention
   post-processing, GEMMs, gathers, elementwise, and admitting atomics with a
   defined byte attribution — each of which is a research change, not a switch;
2. every target rebinds without ambiguity, i.e. the 350 (P32D2) or 324 (P128D2)
   ambiguous-delta refusals resolved in `expand_profiles.py`.

Until then, `--partial` measures the covered part with its coverage recorded, and
the two findings
([P32D2](P32D2_COVERAGE_FINDING.md),
[accepted point](EVIDENCE_POINT_REPRODUCIBILITY_FINDING.md)) say what that number
is and is not.

## 5. Modeled completion: the HBServe posture, opt-in

`--model-uncovered modeled` completes the model without pretending the missing
classes were fitted. It is the same posture HBServe takes in
`hbserve/traces/_reference/full_model_trace_plan.py`, where a class with no
numeric template is either `class_only` (symbolic, excluded from the numeric
replay) or modeled with `mode: numeric_modeled`, a named evidence string, and a
`not_claimed` list. Measured on the two existing runs:

| point | targets | exact | modeled | modeled by cause | modeled traffic | share of traffic |
| --- | --- | --- | --- | --- | --- | --- |
| qwen25_1p5b P32D2 | 2060 | 1360 | 700 (34.0 %) | 350 missing template, 350 ambiguous delta | 1.23 GiB | 2.2 % |
| qwen25_1p5b P128D2 | 2172 | 1162 | 1010 (46.5 %) | 686 missing template, 324 ambiguous delta | 3.76 GiB | 7.1 % |

The estimate is dominated by the classes that carry real traffic: on P32D2,
71 % of the modeled volume is one cutlass WMMA GEMM class and 21 % a `gemvx`
class; on P128D2 the top four are ampere/cutlass GEMMs. The largest of them uses
the *fitted census* basis, i.e. measured per-CTA bytes, not an estimate.

A modeled launch is built in `integrations/sglang/memgen-adapter/model_uncovered.py`
from evidence the exact path cannot use, in this order:

1. **addresses** come from the launch's *own* allocation context — the same
   `module_calls.json`/`tensor_metadata.json` ledger `Binder.mapping()` walks for
   an exact rebind. A launch that binds no tensor object at all (the phase-global
   reducers) falls back to the sampled host's largest storage root and says so
   (`address_basis: representative_persistent_arena`);
2. **volume per CTA** comes from the class's own sample, not from a guess:
   * when the class has a fitted census (the ambiguous-rebind case, and any
     missing class that another launch of the same class got fitted), the census
     is used. The census is a **whole-grid** total (`mem_insts == entries ×
     grid_size`, verified on multi-CTA classes), so it is divided by the grid it
     was measured on;
   * otherwise `consumer.json`'s per-class `selected_records` is divided by the
     CTA count the sampler *actually* selected for that class — the whole grid
     when `selected_all_grid_ctas` is true, else `|fit_ctas| + |holdout_ctas|`.
     That divisor is per class and measured: 104 of 176 P32D2 classes selected
     their whole grid, the rest 1–10 CTAs;
   * the per-instruction byte figure is the run's phase median
     (`bytes_per_record_by_phase`, 128 B on both points), and its measured range
     (4–512 B per instruction) is recorded as `bytes_per_record_range`. That range
     is the honest uncertainty of this estimator: a class whose real accesses are
     16 B per lane is under-modeled by up to 4×, one that is byte-wide is
     over-modeled;
3. **the walk** is affine over the object and never leaves it: a private per-CTA
   tile when the object can hold the whole grid's tiles, otherwise HBServe's
   `shared_template_arena`, where every CTA re-reads the same representative
   lines. Access width narrows (.128 → .8) and the lane mask shrinks so that no
   modeled issue reads past its own allocation, which is why a 20-byte object can
   still be modeled.

What a modeled row may not claim is enforced, not documented: the status is
`PASS_MODELED_UNCOVERED_CLASS`, `modeling.mode` is always `numeric_modeled`,
`exact_cross_layer_identity_claimed` is false, `model.target_addresses_hardware_observed`
is false so the r4 original-allocation guard refuses such a row without a release
re-pin, and the manifest carries `fully_exact: false`, `modeled_launches`,
`modeled_fraction`, `modeled_by_cause`, `modeled_policies`, the per-phase
calibration and HBServe's `not_claimed` list. `complete_full_model` then means
"every target launch has a profile", which is why the receipt and the replay carry
the modeled share beside every counter.

Two bugs this measurement caught, both of which had inflated the modeled share to
34 % / 49 % of traffic before being fixed: a bare `LDG.E` opcode is inferred as
four bytes per lane by the engine, which silently quartered every 16-byte modeled
issue; and the census/record figures were treated as per-CTA when they are
per-launch totals.
