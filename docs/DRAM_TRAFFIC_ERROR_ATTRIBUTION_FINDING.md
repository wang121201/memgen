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
| hardware | 657,937 | — | — | 95,840 | 3,066,880 |

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
   +17.45% that no cache candidate can claim to remove.
