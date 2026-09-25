# What the archived P32D2 calibration evidence does and does not substantiate

Status: `MEASURED_2026-09-25_ON_THIS_HOST`. This is a finding about evidence,
not about a traffic model. It archives a copy that was produced on 2026-09-22 by
an external calibration run root, verifies that copy against its own manifest,
and records what it does and does not establish. It changes no admission:
`hardware_accuracy_accepted` stays `false` everywhere, and
`validation/p32d2_branch_status.csv` remains the authority for hardware-accuracy
admission.

## 1. Why this was archived

The repository carried no P32D2 traffic comparison at all while the calibration
logs that produced it lived outside the archive. The asymmetry was measurable:
`git grep` over tracked files finds the favourable P128D2 and P128D16 rows —
`validation/sglang_current.csv` and `docs/ACCURACY_BASELINES.md` carry them with
absolute GiB and MiB — and finds none of the P32D2 numbers, nor the numbers that
limit the claim elsewhere (the continuous-Decode DRAM write errors, which are the
ones that fail). A reader of this repository could therefore see the rows that
pass and not the rows that fail.

## 2. What was archived, and how the copy was checked

`evidence/sglang/p32d2-traffic-comparison-20260922/` holds the copy. It brings
its own `copy-manifest.json`, which records for each file the absolute path it
came from, its byte count and its SHA-256. That manifest is what makes this an
evidence copy rather than a pile of files, so the copy was verified against it
before anything was added to the tree:

| Check | Result |
| --- | --- |
| files listed in the manifest | 13 |
| files whose bytes and digest match the manifest | 13 of 13 |
| bytes covered by the manifest | 1,939,899 |
| `copy-manifest.json` SHA-256 | `973e2f0c5d3ef00df7cf1e90856bd4fa424771cb5bd548bc48a53e7615549978` |
| the same identity as recorded by the calibration log that produced it | matches, and the file count and byte total match too |

Four further files travel with the copy and are **not** covered by that manifest:
`assessment.json` (the calibration run's own assessment), `copy-manifest.json`
itself, `delivery.json` (a patch-delivery record), and
`previous-deliverable.patch`. Everything else in this document is read from the
manifest-verified set.

## 3. What the archived copy substantiates

Its own assessment (`assessment.json`, observed `2026-09-22T12:44:18Z`, code
commit `ace70256950ff436ed9457a9eb6b8bcb9b1290d0`) scores the three cache
candidates over weighted absolute percentage error:

| Family | Conditions | r2 | r3 | r4 | r4 within threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fresh_ca` | 360 | 21.5276% | 18.1411% | **12.0130%** | 98 of 360 |
| `anchors` | 12 | 29.8570% | 25.1537% | **37.0510%** | 0 of 12 |

This is worth stating plainly because it is unfavourable and it is now in the
archive: r4 is the best candidate on the 360 fitted conditions and **the worst on
the 12 historical anchors**, where it is outside the threshold on every one of
them. `r4/summary_r4.json` records `COMPLETED_SERIAL_ONLY_NOT_DEPLOYED`, and the
hardware aggregates behind the table are 384 conditions, all of family
`serial_chain_r4`, each with a median, five repeats and a stability flag.

The copy also substantiates the P32D2 profile closure, which is the half of the
P32D2 gap this archive can now answer. `assessment.json.profile_summary` records
2062 source launches, 40,956,470 selected records, 2062 of 2062 packed models
accepted, 0 missing, 0 unlowered, `replay_closed: true`,
`full_launch_population_complete: true`, 5950.23 s and 330,976,111 profile bytes,
sourced from `qwen1p5b-P32D2/profiles/receipt.json` at digest
`14de1fb75722611646e7005530be785aca3b17264637178131223fabcd309c72`.

One identity ties this copy to the rest of the repository. The archived
`configuration/paper-v1.config` has SHA-256
`75aecac4fa003b71fec12a9d4c42e604969aa9aa932742093b1bcad875e96974`, which is the
`config_sha256` this repository's own replay reports for the one-command
collection path. The archived copy and the replay agree about which hardware
config was in force.

## 4. What it does not substantiate

The copy states its own limits, and they are the reason this document exists:

- `snapshot/comparison.json` is `PARTIAL_COMPARISON` with `rows: []` and
  `matrix_status: RUNNING`, and it lists `qwen1p5b-P32D2` among the workloads it
  does not have. The snapshot was taken while the six-condition matrix was still
  running.
- `assessment.json.profile_summary` records `complete_inference_model_comparison:
  false` and `hardware_timing_qualified: false`.

So the P32D2 traffic errors quoted in the calibration log — DRAM read `+0.4287%`,
whole-request DRAM write `+3.8943%`, continuous-Decode DRAM write `+85.0699%` —
are **not** verifiable from this repository, and this document does not repeat
them as if they were. The closed comparison rows live in an external execution
root (`evaluation/traffic-comparison/comparison.json`) that is not archived, and
the three-range NCU reports behind them are not archived either.

`validation/p32d2_branch_status.csv` therefore keeps its `main` row for
`Qwen2.5-1.5B-Instruct BF16 / P32D2` at `BLOCKED_MISSING_INDEPENDENT_PROFILE_AND_NCU`.
Splitting that status: the independent profile half is now present in the archive
and manifest-verified; the NCU-comparison half is not in *this* copy, and the copy
says so itself. The closed traffic comparison was archived separately, from its own
run root, in
[the traffic comparison finding](P32D2_P64D2_TRAFFIC_COMPARISON_FINDING.md); this
document is about the calibration copy and does not restate its rows.

## 5. What is still missing, named

1. The other six declared conditions' model sides. Their NCU reference exists; the
   replays that would pair a model value with it are not closed.
2. A gate decision. `docs/ACCURACY_BASELINES.md` records that this repository
   states two different DRAM write gates — at most 20% in the branch contract and
   strictly below 10% in the L2 strategy report — and that they have not been
   reconciled. No number archived here resolves that.

The copy does carry one of the comparison's own rules, which belongs with these
missing pieces: its `limits` list says hardware ranges are independent protocols
and that `whole`, `Decode` and `steps` must not be combined into one accuracy
denominator. That is the same rule `docs/ACCURACY_BASELINES.md` applies when it
refuses to let a whole-request number stand in for per-decode accuracy.

## 6. What this does not change

`validation/sglang_current.csv` is unchanged. The archive contract
(`scripts/verify_archive.py`) pins it at exactly two workloads and three ranges,
and P32D2 is not one of them; adding a P32D2 row there would fail
`PASS_ARCHIVE_CONTRACT`. The archived copy is evidence for what was measured, not
an admission, and not a replacement for that table.

## 7. Reproduce

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
D = Path('evidence/sglang/p32d2-traffic-comparison-20260922')
manifest = json.loads((D / 'copy-manifest.json').read_text())
bad = [rel for rel, row in manifest.items()
       if hashlib.sha256((D / rel).read_bytes()).hexdigest() != row['sha256']]
print('manifest entries verified:', len(manifest) - len(bad), 'of', len(manifest))
print('mismatched:', bad)
PY
```

See also `docs/P32D2_COVERAGE_FINDING.md`, which is about the expansion coverage
of the same declared point and not about traffic.
