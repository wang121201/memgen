# Where the DRAM traffic error lives, and why the write path dominates it

Status: `MEASURED_2026-09-27_ON_THIS_HOST`. This is a finding about the model, not
an admission. `hardware_accuracy_accepted` stays `false` everywhere and
`validation/p32d2_branch_status.csv` remains the authority.

Everything below is read from the archived comparison
(`evidence/sglang/p32d2-p64d2-traffic-comparison-20260923/`) and from local
replays of this repository's own expansion of one phase, so no new GPU time was
spent.

## 1. The error is layered, and the layers are separable

Per-metric WAPE over the ten archived ranges, r4 candidate
(`comparison.json`, 160 rows):

| Layer | Metric | legacy | r4 | Hardware repeat spread |
| --- | --- | ---: | ---: | ---: |
| request generation | L1 global read sectors | 0.0000% | 0.0000% | 0.00% |
| request generation | L1 global write sectors | 0.0001% | 0.0001% | 0.00% |
| L1 hit decision | L1 read lookup hit | 38.22% | 2.75% | — |
| L2 read | L2 read sectors | 16.20% | 1.16% | 0.01% |
| L2 read | L2 read lookup miss | 0.4851% | 0.4853% | — |
| **L2 write** | **L2 write sectors** | **8.7781%** | **8.7781%** | 0.03% |
| DRAM read | DRAM read bytes | 0.4205% | 0.4206% | 0.01% |
| **DRAM write** | **DRAM write bytes** | **3.9879%** | **3.8326%** | up to 232% |

Two properties of that table decide everything else:

1. **The request stream is already right.** L1 read and write *sector* counts
   match hardware to 1e-5 relative error, so no error below this line is caused
   by the address stream, the lane coalescer, or the sector geometry.
2. **The L2 write row does not move with the cache candidate.** legacy and r4
   produce the identical 8.7781%, and on the Decode/steps ranges both produce the
   identical sector count, so this error is not a capacity, associativity or
   replacement question.

## 2. The read residual is downstream of one number

Every read metric is arithmetically tied to the L1 read hit count, because
`L2 read requests = L1 read sectors - L1 read hits` and the L1 sector count is
exact:

| Range | L1 read sectors | L1 read hits error | L2 read requests error | L2 read miss error | DRAM read error |
| --- | ---: | ---: | ---: | ---: | ---: |
| P32D2 whole | 477,471,001 (=hardware) | −3.98% | +1.48% | +0.49% | +0.40% |
| P64D2 whole | 477,471,001 (=hardware) | −3.98% | +1.48% | +0.49% | +0.40% |

The model keeps the same number of read sectors but hits on ~4% fewer of them, so
every downstream counter is high by exactly the propagated amount. The declared
cause is in the model's own limits: `cross-kernel L1 retention ... not modeled`.
Because stores bypass L1 entirely (section 3), allocated store lines cannot
either absorb or displace anything there, so this residual is *not* a write-path
effect.

**Granularity is not the problem.** DRAM bytes per L2 read miss is `32.00` in the
model and `32.01`–`32.07` in hardware on every archived range, so an L2 read miss
already fetches one 32 B sector, never a 128 B line, and the small excess is
hardware's own non-sector-aligned edges.

## 3. The write gap is one structural omission plus one timing effect

Measured on this repository's expansion, measured-Decode phases only (642
kernels), replayed locally, compared with the archived hardware rows for the same
phase:

| Quantity | Hardware | This model | Archived model |
| --- | ---: | ---: | ---: |
| L1 store sectors | 947,560 | 1,112,888 | 947,560 |
| L2 write requests | 657,937 | 1,112,888 | 947,558 |
| L1 store → L2 write ratio | **1.4402** | **1.0000** | **1.0000** |
| store sectors that are partial (not full 32 B) | — | **72.9%** | — |
| mean bytes covered per store sector | — | **11.4 / 32** | — |
| DRAM store bytes | 3,066,880 | 4,365,920 | 5,675,872 |

Read that as three statements:

1. **Hardware merges 30.6% of Decode store sectors before L2; the model merges
   none.** `l1_store_policy` is `bypass` in both candidates, so every store sector
   becomes exactly one L2 write request. The store stream is dominated by narrow
   partial-sector writes (11.4 of 32 bytes on average), which is precisely the
   pattern a hardware store buffer or write-combining path merges and the model
   cannot.
2. **The model's writeback volume per L2 write sector is too high.** Inside the
   same range the model writes back one 32 B sector per 5.36 store sectors it sent
   to L2 (177,371 of 947,558 in the archived model) where hardware writes back one
   per 6.86 (95,840 of 657,937): dirty lines leave L2 sooner than they do on
   hardware. Combined, the two factors reproduce the observed Decode error:
   `1.44 x 1.28 = 1.85`, i.e. the `+85.07%` reported for P32D2 Decode in the
   archive.
3. **Prefill does not show either effect**, which is the control: its stores are
   wide and contiguous, hardware merges only 0.9% of its store sectors, and the
   model's DRAM store error there is `−1.18%`(+0.36% on the second workload).
   A candidate that fixes merging must therefore move Decode and leave Prefill
   alone.

The model's DRAM store accounting is otherwise sound: with
`dram_store_policy=writeback` a store miss inserts the line dirty and charges DRAM
only when it is evicted, and stores never issue a DRAM read fill
(`needs_dram_read = inst.op != 'W'`), so no read-for-ownership is invented.

## 4. What is *not* attributable to the write path

- the L1 read hit residual (−3.98%), hence the whole read chain in section 2;
- the L1 read hit improvement the r4 candidate bought (38.22% → 2.75%), which is a
  capacity/parameter effect;
- the hardware denominator instability on the tail rows: P64D2 Decode2 hardware
  repeats span 130,560–434,432 bytes, a 232.5% spread, so its `+2074.71%` cannot
  be read as a model-error magnitude;
- the two metrics the comparison excludes by design because the model has no
  validated semantics for them (`l1tex_global_store_lookup_hit`,
  `lts_tex_write_lookup_miss`).

What *is* attributable to the write path is exactly the two mechanisms in
section 3, and they are now measurable separately because the store policy is
configurable (`release/source/tools/memgen_hardware_config.h`,
`..._r4_semantic.cc`): `-memgen_l1_store_policy allocate` selects an allocating
store path, `-memgen_write_sector_policy` and `-memgen_dram_store_policy` select
the write-sector emission and the DRAM store attribution, and an unset key leaves
the frozen behaviour byte-identical.

## 5. The allocating store path does not restore the merge (measured)

Measured on the same Decode subset, same profile stream, only the store policy
changed:

| Quantity | `bypass` | `allocate` | Change |
| --- | ---: | ---: | ---: |
| L1 store sectors, full/partial, covered bytes | 1,112,888 / 301,480 / 811,408 / 12,721,012 | identical | 0.00% |
| `l1_requests` | 308,259,667 | 309,372,555 | **+1,112,888** (stores now touch L1) |
| `l1_hits` | 71,544,843 | 71,779,437 | +234,594 (21% of store sectors hit a resident line) |
| **`l2_write_requests`** | **1,112,888** | **1,112,888** | **0.00%** |
| DRAM store bytes, writeback events, dirty sectors | 4,339,936 / 33,914 / 135,623 | identical | 0.00% |
| DRAM read bytes | 5,999,704,704 | 5,999,704,704 | 0.00% |

So the merge factor stays `1.0000` against hardware's `1.4402`, and the negative
result has a name in the code rather than an explanation: the L2 lookup is

```cpp
const bool l2_lookup = write_like || bypass_l1 || !l1_hit;
```

and `write_like` is true for every store, so a store always performs its L2 access
whatever the L1 says. This engine's allocating store path is a **write-through L1
with store-allocate**: it counts store hits — which is why the excluded metric
`l1tex_global_store_lookup_hit` has no NCU counterpart, the candidate's own
documentation says it bypasses L1 stores — and absorbs nothing. The knob is
therefore harmless (reads are unchanged, +448 L2 read requests out of 236.7M) but
it cannot be the fix.

Two things follow. First, the merge axis needs a mechanism that does not exist
yet: a bounded write-combining buffer before the L2 access, which merges partial
writes to one sector within a window and issues one L2 write per merged sector.
Second, the opportunity is real and the model already measures its size: 21% of
Decode store sectors hit a line that is already L1-resident, against the 30.6% of
store sectors hardware merges — the same order, which is what a bounded window
would recover.

## 6. The write residual is per range, and each range has two factors

The hardware protocol records `Decode1`, `Decode2` and the whole `Decode` range as
separate measurements, so a whole-range residual can hide two opposite errors.
Splitting the 642-kernel Decode subset by `-kernel_<n>_llama_phase`
(`scripts/phase_write_split.py`, `scripts/write_acceptance.py`) shows exactly that,
with this replay's frozen default:

| Range | Model write | Hardware write | Signed error | Store sectors | Dirty fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Decode1` | 2,156,064 | 2,288,768 | −5.80% | ×1.1745 | **×0.8021** |
| `Decode2` | 2,183,872 | 828,160 | **+163.70%** | ×1.1745 | **×2.2453** |
| `Decode` | 4,339,936 | 3,066,880 | +41.50% | ×1.1745 | ×1.2049 |

The last two columns are an identity, not an approximation: write traffic is the
store sectors the model sent times the fraction of them it evicted dirty, and
`1.1745 × 1.2049 = 1.4152` reproduces the +41.5% of the whole range. Every range
shares the same store-sector factor, because the store stream is upstream of the
cache model: `write_sector_requests` is the expansion's output, and the archived
comparison's own model column sent **947,558** store sectors where hardware sent
947,560 — a match in every range, including `Prefill` at 2,753,475 exactly. This
replay sends 1,112,888 for the same 642 launches (+17.45%), so the stream factor
belongs to the expansion, not to the cache decision, and no cache candidate can
remove it.

The dirty fraction is the cache decision, and it is not one error but a split:
`Decode1` retains *more* than hardware (×0.8021) while `Decode2` retains *less*
(×2.2453). The model evicts 12.19% of the store sectors it sent, in both steps
almost identically (12.11% and 12.26%), because a steady-state replay with a cold
start makes the two steps symmetric. Hardware is not symmetric: its `Decode1`
rate is 15.10% and its `Decode2` rate 5.46% of 473,780 store sectors per step,
which is the signature of a transient — the first decode step also drains whatever
the prefill left dirty, and by the second step the dirty footprint fits. A
per-range match therefore cannot be reached by a single steady-state rate; the
combined range (`Decode`) is the range a retention policy can be tuned against,
and the per-step split is where the residue goes.

`-memgen_l2_clean_first_k` is the lever for the dirty fraction: it prefers a clean
victim inside a window of the `k` least recently used entries of a set, so dirty
lines survive longer. Its floor is set by the stream factor — with the dirty
fraction at hardware's 1.0000 the combined range would be 1.1745 × 3,066,880 =
3,602,000 bytes, still +17.45% — which is the boundary of what any cache-model
change can claim on this replay.

## 7. The retention lever is degenerate, so the knob cannot land on hardware

Measured on the same Decode subset, same profile stream, one window per run, with
everything else at the frozen default:

| Window `k` | `l2_write_requests` | L2 write hits | L2 write misses | Dirty sectors evicted | DRAM store bytes (Decode) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (frozen) | 1,112,888 | 988,321 | 124,567 | 135,623 | 4,339,936 (+41.50%) |
| 2 | 1,112,888 | 1,089,755 | 23,133 | 2,767 | 88,544 (−97.11%) |
| 4 | 1,112,888 | 1,089,787 | 23,101 | **0** | **0** (−100.00%) |
| 8 | 1,112,888 | 1,089,787 | 23,101 | **0** | **0** (−100.00%) |
| hardware | 657,937 | — | — | 95,840 | 3,066,880 |

The window saturates at `k=4`: `k=4` and `k=8` are identical, so the reachable
values are exactly three, and hardware's is in none of them.

The store stream and the L2 write requests are identical in all three runs, so the
knob changes only the victim choice: DRAM read moves by 0.02%. What it does not do
is move gradually. A two-entry window already changes the *composition* of the set
rather than its recency order — dirty lines stay resident, later stores to them
become hits, allocating misses fall 81.4%, and with less allocation pressure there
is less eviction, which keeps the next dirty line resident in turn. The self
reinforcing loop is real (it is why any write-retention policy helps), but its
response to this knob is a step: the achievable values on this replay are
{4,339,936 (k≤1), 88,544 (k=2), 0 (k≥4)}, and the hardware value 3,066,880 sits
inside the gap. **No value of `k` reaches hardware's number.**

That negative result is informative. Reads outnumber store sectors in a Decode
step by roughly 168 to 1 (3.0 GB of read bytes against 17.8 MB of stores), so the
least recently used end of every set is nearly always clean; a two-entry clean
preference is therefore already enough to make dirty lines almost never leave, and
any larger window pins them completely. Hardware does *not* behave that way — it
evicts 95,840 dirty sectors in the same read-dominated regime — so its victim
choice is close to recency based, which is what the frozen default already models.
On the rate axis the default is within 2.6% of hardware after the stream factor is
taken out (`1.2049 / 1.1745`), and in `Prefill`, where the store stream is 4.87%
below hardware, the rate factor is 0.993 — the evicted dirty count tracks the store
stream rather than fighting it.

Three consequences, and only the third is actionable here:

1. The per-step asymmetry is not reachable by a steady-state retention rule. The
   model evicts 12.11% and 12.26% of its store sectors in the two steps; hardware's
   15.10% and 5.46% are a transient — the first step also drains what the prefill
   left dirty — and a cold-start continuous replay has no boundary to carry that
   residue across. The documented limit ("scope selection does not recreate
   physical range synchronization") is the right place to record this.
2. `Decode2` alone cannot be matched by any cache-model change, because its
   hardware value is smaller than a proportional share of a steady-state rate.
3. The combined `Decode` range is the range that can be matched in principle, and
   the gap there is dominated by the store stream, which is upstream of the cache
   model: the archived comparison's own model column matched hardware's store
   sectors in every range (947,558 against 947,560 in `Decode`, 2,753,475 exactly
   in `Prefill`), while this replay sends 1,112,888 for the same 642 launches. An
   expansion whose store stream matches hardware's is the prerequisite for the
   per-range write acceptance; until then the write residual has a floor of
   +17.45% that no cache candidate can claim to remove. The cache model is not the
   layer that inflates it: the observation layer's own source-side counter
   (`source_write_sectors` in `cache_observation.csv`) totals 1,112,888 for this
   subset, equal to `write_sector_requests` and to `l2_write_requests`, so the
   extra store sectors arrive in the generated address stream and are passed
   through unchanged.

## 8. The sector angle belongs to the expansion, and a fifth of it is modeled

Section 7 named the layer; this section measures it, because "the input does not
match the hardware counter" is only useful once it is split by *which launches*
the input came from. The expansion manifest declares that split: of 2,060 launches,
1,360 are `exact` and 700 are `numeric_modeled` (34.0%), the latter from
`missing_template_profile` (350) and `ambiguous_address_binding` (350), calibrated
as `run_calibrated_records_per_cta` at 128 bytes per record for both decode steps.
Its `not_claimed` list already says what a modeled class does not promise:
"exact access width, lane mapping, or predicate mask of a modeled class" — the
three quantities that decide a store instruction's sector count.

Joining that split with the full-model replay gives, for the measurement ranges:

| Range | Mode | Launches | Store sectors | Read sectors | DRAM store bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| `Decode1` | exact | 222 | 429,404 | 152,681,657 | 2,182,048 |
| `Decode1` | `numeric_modeled` | 99 | 127,040 | 1,448,176 | **0** |
| `Decode2` | exact | 222 | 429,404 | 152,681,658 | 2,182,464 |
| `Decode2` | `numeric_modeled` | 99 | 127,040 | 1,448,176 | 1,408 |
| `Prefill` | exact | 249 | 2,188,652 | 136,737,562 | 63,699,136 |
| `Prefill` | `numeric_modeled` | 139 | 430,592 | 15,589,392 | 1,408 |

The modeled classes are 22.8% of the `Decode` store stream and 0.9% of its read
stream, which is the whole reason the read side looks healthy and the write side
does not: load sectors are dominated by launches with native addresses, store
sectors are not. Two consequences follow.

First, the modeled classes are a *pressure* source rather than a traffic source.
They contribute 127,040 store sectors per step and are charged 0 bytes of DRAM
write traffic, because their synthetic sectors land on lines that the exact
launches already made resident and their own dirty data never leaves the cache
inside the range. What they do instead is allocate: 22.8% more store sectors means
22.8% more allocations, and the evictions they cause are charged to the exact
launches' dirty data. That is why the DRAM write residual (×1.4152) is larger than
the store-stream residual (×1.1745) even though the cache model itself only adds
the dirty fraction ×1.2049.

Second, no cache-side change can absorb this, and no sector-level calibration is
available for those launches either: the archive records hardware counters per
range, not per launch, and only 42 of the 642 subset profiles carry an
`independent_source_census` (those 42 match the engine exactly, so the census is
self-consistent, not an independent hardware reference). The fix is therefore where
the manifest points: replace `missing_template_profile` and
`ambiguous_address_binding` with native address coverage for those 700 launches, or
calibrate the modeled classes against per-launch NCU sector counters once such a
reference exists. Until one of those happens, the honest acceptance scope is the
exact launches for the sector angle, with the modeled share reported next to it —
which is what the manifest already encodes as
`allow_full_NCU_accuracy_comparison: false` with
`allow_declared_estimate_NCU_comparison: true`.

## 9. The three write factors are not independent, and Accel-Sim does not close them

The write residual decomposes into three factors, but they are **arithmetically
coupled, not three independent knobs**. Write traffic is the store sector count
times the fraction that survives to L2 times the fraction that is evicted dirty,
so a change in the store stream moves the denominator of the merge ratio and the
dirty fraction, not only its own term. Concretely:

- **Store-sector count (expansion, +17.45% on this replay).** The 700 modeled
  launches contribute 127,040 store sectors per Decode step that hardware does not
  have. Removing them (native coverage) shrinks the stream, but it also changes
  which lines are resident and therefore which exact launches' dirty data is
  evicted, so the dirty-fraction residual is re-measured, not preserved.
- **Store merge (cache, ×1.44).** Hardware merges ~30.6% of Decode store sectors
  before L2; this model merges none. This factor is real hardware behaviour, not a
  fitting artefact: NVIDIA write-combining / store buffers coalesce narrow
  partial-sector stores to the same 32 B sector, and the archived comparison's
  `L1 store → L2 write ratio = 1.4402` is measured from NCU, not invented. It is
  workload-dependent, not a constant: Prefill merges only 0.9% because its stores
  are wide and contiguous, so a merge window calibrated on Decode does not
  transfer to Prefill or to another model's store pattern.
- **Dirty fraction / early writeback (cache, ×1.28).** The model evicts dirty
  lines sooner than hardware. This is the least solid of the three: the
  `-memgen_l2_clean_first_k` sweep saturates at `k=4` with exactly three reachable
  values, none of which is hardware's, and the per-step split (Decode1 ×0.8021,
  Decode2 ×2.2453) is a cold-start transient, not a steady-state rate, so no
  single retention parameter can land on it.

The three factors multiply into the observed error (`1.1745 × 1.44 × 1.28 ≈ 2.16`
on the combined Decode range), but they are **not orthogonal**: the merge ratio is
computed over the store stream that the expansion produced, and the dirty fraction
is computed over the post-merge L2 write stream, so any fix re-defines the other
two factors' baselines rather than leaving them untouched. Eliminating one factor
does not remove the others; it only re-baselines them.

**Accel-Sim 2.0 does not close the write gap either.** Its Ada configuration's L2
coalescer is read-only: `l2cache.cc` documents and enforces
`// For write request, it is not coalesced with LRC` with
`if (m_lrc && !(req->is_write()))`, so Accel-Sim's L2 issue path passes each store
sector through un-merged exactly like this model. Its L1 is write-through
(`dl1 ... T ...`), and its store path has no write-combining buffer before L2.
Accel-Sim models the same two write mechanisms this model already models
(write-back L2, sector-level dirty byte masks) plus full DRAM/MSHR timing, but the
write *traffic* mismatch is a store-merge and dirty-release question that neither
the functional model nor Accel-Sim's current Ada config answers. Matching the
address decomposition (section on `RTX4000Ada.accelsim.config`) removes a
partition/set-distribution term, but it does not introduce the missing merge
mechanism, so it cannot by itself move the +39% / +648% Decode write figures.

## 10. Store merge is a real, workload-independent mechanism, and it is portable to Accel-Sim

The store merge is not a fitting artefact invented to close one workload's write
gap. It is NVIDIA write-combining: narrow partial-sector stores to the same 32 B
sector are merged in a bounded store buffer before the L2 access, so the number of
L2 write requests is lower than the number of store sectors. The mechanism is
workload-independent — the *merge count* varies with the store pattern (Prefill
merges 0.9% because its stores are wide, Decode merges 30.6% because its stores
are narrow), but the *mechanism* (merge same-sector partial byte masks within a
window, flush on full or on window overflow) is one fixed component.

The correct implementation is therefore a per-SM bounded write-combining buffer
keyed by 32 B sector address, not a per-workload calibration:

- a full-sector store (byte_mask == 0xFFFFFFFF) passes through immediately;
- a partial-sector store merges its byte_mask into the buffer entry for that
  sector; when the entry becomes full, or when the buffer reaches its capacity
  (the window), the entry is flushed as one L2 write request;
- the buffer flushes at every kernel boundary.

The window is a mechanism capacity (buffer size), not a fitted parameter: the
measured merge ratio emerges from the store pattern, so the same component serves
Prefill, Decode, and any other model without per-point tuning.

The same component is portable to Accel-Sim 2.0. Accel-Sim's L2 path already breaks
every request into 32 B sector requests in
`memory_sub_partition::breakdown_request_to_sector_requests`, and its
`L2RequestCoalescer` already merges by `sector_addr` via `equal_range` — it is
deliberately gated off for writes by `if (m_lrc && !(req->is_write()))`. Enabling a
write coalescer there (or a sibling write-combining buffer beside the LRC) is the
same mechanism this model needs, so a store-merge component is a shared, portable
answer rather than a memgen-only patch.

## 11. Per-step write comparison is too strict, but the whole-Decode gap is real

The per-decode-step write split (`Decode1 ×0.8021`, `Decode2 ×2.2453`) reflects a
hardware cold-start transient — the first decode step also drains the dirty data
the prefill left resident, so hardware's own Decode1 and Decode2 dirty rates differ
(15.10% vs 5.46%). Demanding a per-step match therefore treats the model's
cold-start semantics as an error even when the combined range is correct. The
right comparison scope is the combined `Decode` range, with the per-step split kept
as a diagnostic for the transient, not as an admission gate.

However, the combined Decode gap (+41.5%) is not a scope artefact: it is the store
merge (×1.44) and modeled store stream (+17.45%) multiplying, both of which survive
range pooling. Widening the comparison removes the transient term (×1.28) but not
the other two, so the honest target is to close the merge and the modeled store
stream first, and treat the dirty transient as a separately reported, currently
uncalibrated residual rather than a parameter to fit.

## 12. Store-merge sweep: the mechanism works, but must be measured on native sectors

`-memgen_store_merge_window` was swept over the 642-kernel Decode subset
(`out/subset-decode`, the same stream section 3 measured). `write_sector_requests`
stays 1,112,888 at every window (the store stream is upstream of the merge); the
L2 write count moves:

| window | l2_write_requests | merge factor (l2w/ws) | dram_store_bytes |
| ---: | ---: | ---: | ---: |
| 0 (frozen) | 1,112,888 | 1.0000 | 4,339,936 |
| 4 | 1,068,248 | 0.9599 | 4,336,736 |
| 16 | 970,776 | 0.8723 | 4,328,800 |
| 64 | 837,144 | 0.7522 | 4,307,552 |
| 256 | 737,176 | 0.6624 | 4,233,824 |
| hardware | 657,937 | 0.6943 (1.44:1) | 3,066,880 |

The merge grows with the window and does not saturate by 256, which is expected:
the mechanism merges same-32B-sector stores however far apart they are, and the
window is a buffer capacity, not a fitted constant. The sweep confirms the
mechanism is real and monotone, but **the numbers cannot be compared with
hardware's 1.44:1 yet**, for two reasons:

1. **The denominator is wrong.** This stream carries the +17.45% modeled store
   surplus (1,112,888 vs hardware's 947,560 native sectors), and modeled sectors
   repeat object-walk addresses far more than native ones, so they merge more
   readily. The merge factor must be measured on a native store stream (native-full
   capture) before it can be read against hardware's 1.44:1.
2. **The merge granularity is unverified.** The sweep merges same-32B-sector byte
   masks (sector cache byte-level write coalescing). Hardware's 30.6% store merge
   could be the same sector-level merge, or a 128B-line-level merge across sectors;
   the NCU `l1tex` sector count versus the L2 write-request count does not by itself
   decide which. Until the granularity is pinned, the mechanism is a candidate, not
   a validated hardware match.

The correct order is therefore: close the native store stream first (priority 1,
native-full), then re-sweep the merge on native sectors and compare against the
1.44:1 hardware ratio (priority 2). The two are independent factors and must not be
tuned together, or a window that compensates the modeled surplus looks calibrated
on one point and is wrong everywhere else.
