# The closed P32D2 and P64D2 traffic comparison, and what its rows show

Status: `MEASURED_2026-09-25_ON_THIS_HOST`. This is a finding about archived
evidence, not about a traffic model. It archives the traffic comparison that was
produced on 2026-09-23 by an external calibration run root, verifies it, and
records what its rows show. It changes no admission:
`hardware_accuracy_accepted` stays `false`, the archived comparison states that
itself, and `validation/p32d2_branch_status.csv` remains the authority.

## 1. Why this was archived

Before this, `validation/sglang_current.csv` and `docs/ACCURACY_BASELINES.md`
carried P128D2 and P128D16 with absolute GiB and MiB, and no P32D2 row existed
anywhere in the tree. The rows that decide whether the write path is acceptable
were only in external calibration logs, so the repository could show the rows that
pass and not the rows that fail.

## 2. What was archived, and how it was checked

`evidence/sglang/p32d2-p64d2-traffic-comparison-20260923/` holds the copy, with
`archive-provenance.json` recording for each file the absolute source path, its
byte count, its SHA-256 and whether it was archived:

| Archived | Bytes |
| --- | ---: |
| `traffic-comparison/comparison.json` | 143,173 |
| `traffic-comparison/assessment.md` | 1,853 |
| `evaluation/finish.json` | 2,916 |
| `evaluation/progress.json` | 2,916 |
| `p32d2-replay/finish.json` | 3,913 |
| `p64d2-replay/finish.json` | 3,936 |
| `p64d2-replay/allocation-audit.json` | 318 |

The comparison depends on inputs that cannot be archived. Two are larger than the
archive's 10 MiB limit — `r4-context.json` at 22,528,413 bytes for each workload,
and `issue.config` at 23,232,100 and 23,691,192 bytes — and the raw NCU reports are
a forbidden suffix. Those are recorded by digest in the same provenance file
rather than omitted, so a reader can tie the rows to specific inputs without the
repository shipping them. The two `r4-context.json` copies have different digests,
so the context is per-workload, not per-config.

Both replay receipts are named for the boundary they keep:
`PASS_FULL_SOURCE_CACHE_REPLAY_NOT_HARDWARE_ACCEPTANCE`. The comparison itself is
`PARTIAL_COMPARISON` with `hardware_accuracy_accepted: false`.

## 3. The rows

160 rows cover two workloads, two cache candidates and three independent range
protocols. Errors are `100 x (simulated - hardware median) / hardware median`, over
three hardware repeats. This is the DRAM ledger:

| Workload | Candidate | Scope / phase | R/W | Simulated | Hardware median | Error | Hardware min–max |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| P32D2 | r4 | whole | read | 9,279,987,552 | 9,240,374,144 | **+0.4287%** | 9,240,184,320 – 9,240,436,864 |
| P32D2 | r4 | whole | write | 75,722,784 | 72,884,480 | **+3.8943%** | 72,834,560 – 73,577,984 |
| P32D2 | r4 | steps / Prefill | write | 70,046,912 | 70,882,944 | −1.1795% | 70,810,624 – 71,260,160 |
| P32D2 | r4 | Decode | write | 5,675,872 | 3,066,880 | **+85.0699%** | 3,066,496 – 3,151,360 |
| P32D2 | r4 | steps / Decode1 | write | 2,837,632 | 2,288,768 | +23.9808% | 2,204,544 – 2,322,304 |
| P32D2 | r4 | steps / Decode2 | write | 2,838,240 | 828,160 | **+242.7164%** | 516,096 – 834,304 |
| P64D2 | r4 | whole | read | 9,288,289,280 | 9,251,509,632 | **+0.3976%** | 9,250,289,152 – 9,251,771,264 |
| P64D2 | r4 | whole | write | 155,439,776 | 152,238,720 | **+2.1027%** | 152,124,800 – 152,248,704 |
| P64D2 | r4 | Decode | write | 5,684,320 | 4,158,080 | **+36.7054%** | 3,540,992 – 4,293,248 |
| P64D2 | r4 | steps / Decode2 | write | 2,842,080 | 130,688 | **+2074.7062%** | 130,560 – 434,432 |

The legacy candidate is in the same file for the same ranges, which is what makes
the read improvement checkable rather than asserted: P32D2 whole read is
`+0.4285%` under legacy and `+0.4287%` under r4, so r4 does not change the read
side; P32D2 whole write is `+4.0941%` under legacy and `+3.8943%` under r4; and
P32D2 Decode write is `+89.8195%` under legacy against `+85.0699%` under r4. The
r4 candidate improves the decode write error by about five points, and it does not
make that range acceptable.

## 4. What the rows establish

Reading is settled at these two points. Every DRAM read error in the file is
between `0.027%` and `0.493%`, and the hardware repeats for a read are within
about `0.02%` of their median, so the read verdict does not turn on which gate is
applied.

Writing is not. The whole-request numbers look small — `+3.8943%` and `+2.1027%` —
because a prefill writes tens of megabytes while a single decode step writes
hundreds of kilobytes, and the comparison's own `summary` block makes that
concrete: over the two Decode ranges the legacy write WAPE is `61.16%` while the
read WAPE is `0.47%`. The `Decode2` rows show why the small denominators are
unstable: P32D2 hardware repeats are `828,160 / 834,304 / 516,096` bytes, so the
minimum is 62% of the median, and P64D2's are `130,688 / 130,560 / 434,432`.

One property holds across the whole file and is worth stating on its own: **every
DRAM row is `within_observed_range: false`**. The read errors are small in
percentage but still outside the spread of the three hardware repeats, because
those repeats are tight. Passing a percentage gate and being inside the observed
range are therefore different claims, and no row here satisfies the second one.

## 5. Scope, so the numbers are not read as wider than they are

| Field | Value |
| --- | --- |
| `status` | `PARTIAL_COMPARISON` |
| `expected_workloads` | 8 declared conditions |
| `hardware_ready_workloads` | 8 of 8 — the NCU reference exists for all of them |
| completed model comparisons | 2 of 8, P32D2 and P64D2 |
| `missing_workloads` | P128D2, P256D2, P512D2, P128D4, P128D8, P128D16 |
| `hardware_accuracy_accepted` | `false` |
| `isolated_speedup_qualified` | `false` |

Its `limits` list travels with the rows and bounds them: no MSHR concurrency, no
real arrival order, CTA order modeled rather than measured, and hardware ranges
are independent protocols that must not be combined into one accuracy denominator.
The same file's `execution_schedule` records that the CPU replay limit was raised
without any model change, which is why it is a scheduling fact and not a
performance result.

## 6. What is still missing

1. The other six conditions' model sides. Their hardware reference is ready; the
   replays are not closed.
2. The gate decision. `docs/ACCURACY_BASELINES.md` records two DRAM write gates —
   at most 20% in the branch contract, strictly below 10% in the L2 strategy
   report — and `+242.7164%` and `+2074.7062%` fail either of them, so this file
   does not depend on resolving that. `+3.8943%` and `+2.1027%` pass both, which is
   exactly the case where the gate choice matters.
3. The r4 context producer. The `r4-context.json` that a unified r4 replay needs is
   **not** produced by anything on `main`; the producing code is
   `research/r4_l1_filter/` on the `research/llm-traffic-calibration` branch
   (commit `b9a9022`), together with `collect_shared.py`, `validate_context.py` and
   `sparse_profile_adapter.py`. That branch is not published to `origin`, so the
   step that makes this comparison reproducible is currently one host's git object.

## 7. Reproduce

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
D = Path('evidence/sglang/p32d2-p64d2-traffic-comparison-20260923')
prov = json.loads((D / 'archive-provenance.json').read_text())
bad = []
for rel, row in prov.items():
    if not isinstance(row, dict) or not row.get('archived'):
        continue
    if hashlib.sha256((D / rel).read_bytes()).hexdigest() != row['sha256']:
        bad.append(rel)
print('archived entries verified:', sum(1 for r in prov.values()
      if isinstance(r, dict) and r.get('archived')) - len(bad))
print('mismatched:', bad)
print('rows:', len(json.loads((D / 'traffic-comparison/comparison.json')
                              .read_text())['rows']))
PY
```

See also `docs/P32D2_CALIBRATION_EVIDENCE_FINDING.md`, which covers the other
archived copy (the r4 parameter selection against 360 fitted conditions and 12
historical anchors) and not the traffic rows.
