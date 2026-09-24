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

For a point to reach `complete_full_model`, both must hold:

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
