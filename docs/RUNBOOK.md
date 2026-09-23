# Runbook: manual launch and full verification

This is the operator document. It gives the exact commands, the expected
receipts and the claim boundary for each step. The document map is in the
[README](../README.md); read [environment](ENVIRONMENT.md) first for the
dependency inventory and [branch contract](BRANCH_AND_ACCEPTANCE_CONTRACT.md)
for what a result is allowed to claim.

Nothing here is an accuracy claim. Producing an artifact is not an admission
decision; see section 5.

---

## 1. Implementation health, no GPU

No GPU, no model, no NVBit. Run this after any change to the repository. Each
command writes only to a fresh directory you name. A pass here says the frozen
engine, the pinned deployment and the declared-case contract are intact; it says
nothing about a workload. Section 3 is what collects the data, and it is not a
substitute for these pin-integrity checks.

```bash
cd <repository root>
./memgen check      # host dependencies, pinned packages, archive pins
./memgen test       # entry point, declared cases, driver, pin structure, census counting
./memgen smoke      # frozen engine replayed twice, output must be identical
```

Expected: `PASS host and archive checks`, `PASS portable regression tests` and
`PASS_FROZEN_CPU_SMOKE`, each exiting 0. `./memgen check --gpus none` skips the
GPU checks on a CPU-only host, and `--quiet` prints only the final statuses.

Under the hood these run `integrations/sglang/preflight.py`,
`integrations/sglang/bootstrap_vendor.py --check`, `scripts/verify_archive.py`,
`tests/cli/test_cli.py`, `tests/sglang/test_declared_cases.py` and
`tests/sampling/test_profile_census.py`, plus `scripts/run_cpu_smoke.sh` with
`mpic++`. The CLI prints each command it runs, so any of them can be invoked
directly when a failure needs detail.

`verify_archive.py` fails if a bytecode cache, a model file, an NCU database or
any file over 10 MiB is present, so invoke the Python entry points with `-B`.
Every tool here sets `sys.dont_write_bytecode` itself.

If `check` reports `PRESENT_REVISED`, that is a recorded, reviewed revision; the
reason is in `integrations/sglang/revisions.json`. `PRESENT_DRIFT` or `DRIFT` is
a failure and must be investigated, never silenced.

Optional, and only for the r4 core:

```bash
./memgen check --r4     # cache-core regression into a fresh out/cache-core-<UTC>
```

---

## 2. Prerequisites for the target run

`./memgen check` covers the host and the pinned files. It is the last step that
is not part of a collection, so run it before anything else:

```bash
./memgen check
./memgen gpus                                      # admitted GPUs by index
```

The observer is built automatically by `memgen collect`. The frozen CPU engine
is only needed for `memgen replay`; build it with:

```bash
rm -rf /tmp/mg-engine && mkdir -p /tmp/mg-engine
mpic++ -std=c++17 -O2 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  release/source/tools/hbserve_profile_stream_cache_semantic_r17.cpp \
  -l:libzstd.so.1 -lz -lboost_mpi -lboost_serialization -lcrypto -pthread \
  -o /tmp/mg-engine/hbserve
```
# build the metadata observer (no GPU needed, about 5 s)
rm -rf /tmp/mg-observer
python3 integrations/sglang/compact-sources/observer/build.py --output /tmp/mg-observer
#    expect: {"status": "PASS_BUILD_ONLY_NO_GPU", ...}
python3 integrations/sglang/tool_identity.py /tmp/mg-observer/observer.so
#    record artifact_sha256 and content_sha256 in your notebook; see ENVIRONMENT.md 4.4
```

Resource model, taken from the recorded deployment:

| Resource | Value | Where it is enforced |
| --- | --- | --- |
| CPU ids | `0..15` only | `run_job.py`, `CPU pool is 0..15` |
| GPU ids | the three UUIDs in `run_job.py` `GPU_POOL` | `run_job.py`, `GPU2 and unknown GPU are excluded` |
| Guarded RSS | 64 GiB by default | `run_job.py` `rss_limit_bytes` |
| Stage budgets | census 1800 s, sample 7200 s; the replay runs to completion | per job `seconds` (`0` means no deadline) |
| Fresh admission | a GPU must be idle: no compute apps, under 512 MiB used, `utilization.gpu == 0` | `parent_controller.py` `gpu_admission`; the driver waits for it |

`run_job.py` takes an exclusive lock per CPU id and per GPU id, sets CPU
affinity and single-threaded library variables, and kills its child if the
controller dies. Do not launch two jobs with the same `cpu`/`gpu`.

Admission is checked once, and the utilization figure is an average the driver
decays, so the second after another job releases the device can still read as
busy on an idle GPU. `collect_case.py` therefore retries that one refusal for up
to `--gpu-wait-seconds`, printing each wait; this is a wait, not a kill. Every
controller run writes its output to `runs/<job>.controller.stdout` and
`.stderr`, so the driver prints one line per stage instead of the controller's
whole receipt, which stays in `runs/<job>/job-finish.json`.

---

## 3. The target run

### 3.1 One command

`integrations/sglang/collect_case.py` drives the whole chain for one declared
case as two jobs under the lease controller. It builds the observer, writes
both job specs with their sources pinned by hash, verifies every stage receipt
and writes a collection receipt with the artifact map.

```bash
# review the plan: writes both specs, runs nothing, uses no GPU
./memgen plan \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 \
  --gpu-index 1 --work /absolute/fresh/qwen15b-p32d2-r1

# collect
./memgen collect \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 \
  --gpu-index 1 --work /absolute/fresh/qwen15b-p32d2-r1
```

Without `--work` the CLI creates a fresh timestamped directory under `out/`.
Both verbs forward to `integrations/sglang/collect_case.py` and print the
command they run.

`--gpu-index` is the numbering `preflight.py` prints. `--case qwen25_1p5b-p32-d2`
is accepted as a shorthand for the three case values, and `--list-cases` prints
every declared case.

| Parameter | Meaning |
| --- | --- |
| `--model`, `--prefill-length`, `--decode-steps` | the case to collect, as three independent values. Any declared combination works; `--list-cases` prints all 26 |
| `--case` | shorthand for the three |
| `--list-cases` | print every declared case with its matrix name, then exit |
| `--work DIR` | fresh output directory. It must not exist; a retry needs a new one |
| `--gpu-index N` | which admitted GPU, by the index `preflight.py` prints. Default 0 |
| `--gpu UUID` | the same choice by UUID, for scripted callers |
| `--cpu N` | one CPU id from the shared `0..15` pool. Default 8 |
| `--python PATH` | interpreter that carries the SGLang stack |
| `--census-seconds`, `--sample-seconds` | budgets for the two GPU stages |
| `--job-seconds` | job 2 wall-clock ceiling. `0`, the default, runs to completion |
| `--gpu-wait-seconds` | how long to wait for a device that is busy at fresh admission. Default 600, `0` fails at once |
| `--observer PATH` | reuse a built observer; otherwise one is built into `--work` |
| `--dry-run` | write the two job specs and print the plan, execute nothing |

The triple is checked against `memgen-adapter/contract.json` before anything
runs, so an undeclared combination fails at once and points at `--list-cases`.
Changing model, prefill length or decode steps needs no other edit. To rehearse
the toolchain first, use `--prefill-length 128 --decode-steps 32`, the point the
archived deployment actually drove through this chain.

**No stage imposes a wall-clock deadline on the cache replay.** `run_memgen.py`
documents that it has none, and `followthrough.py`, `profile_cache.py` and this
driver no longer add one. Only the two GPU stages carry budgets, because they
hold a leased device.

Artifacts, all under `--work`:

| Artifact | Meaning |
| --- | --- |
| `observers/<case>-census/process-*/launch-journal.jsonl` | every kernel launch, ordered |
| `observers/<case>-census/process-*/finish.json` | `PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE` |
| `runs/<case>-census/host/process-*/finish.json` | measured phases and tensor metadata |
| `runs/<case>-collect/followthrough/plan/sample-plan.json` | which layer and CTAs are sampled |
| `.../sample/profiles/profiles.index.jsonl` | the packed, workload-specific profile |
| `.../expanded/manifest.json` | `complete_full_model` and unsupported-launch counts |
| `.../cache/model/kernel_summary.csv` | per-kernel cache, hit and DRAM counters |
| `.../cache/model/cache_observation.json` | occupancy, writeback, residual checks |
| `collect-receipt.json` | case identity, stage receipts and the artifact map |
| `runs/<job>.controller.stdout`, `.stderr` | the controller's own output, kept out of the terminal |

It refuses a case outside the declared matrix, a GPU outside the admitted pool,
an existing `--work` and a missing interpreter, so a typo fails before any GPU
time is spent. Exit 0 means every stage receipt closed; exit 2 means the profile
stream did not cover the full model, which the receipt reports as
`STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC` rather than hiding.

`--dry-run` writes `census-spec.json` and `collect-spec.json`. The second names
`process-<pid>` as a placeholder because job 1 resolves the real census process
directory before job 2 is written.

**Budgets are hard limits, not estimates.** A stage that exceeds its budget is
killed: the census job and the whole of job 2 are bounded by `run_job.py`
through `parent_controller`, and the sampling stage additionally enforces its
own budget inside `sample_pipeline.py`, which is why `--sample-seconds` must
stay within `60..21600` and cannot be set to zero. A kill releases the lease and
exits non-zero. `--census-seconds 0` removes the census ceiling, and the replay
has no deadline at any layer.

**This is not the smoke test.** `scripts/run_cpu_smoke.sh` replays a synthetic
two-kernel fixture on the CPU with no GPU, no model and no NVBit, and only
checks that the frozen engine is deterministic; it belongs to section 1.
`collect_case.py` runs the real Qwen workload through SGLang, NVBit and the
cache model. Collect nothing until section 1 passes, and never read a passing
smoke as evidence about a workload.

### 3.2 Stage by stage

Use this when a stage fails and you want to rerun one of them, or when you want
to drive `wait_then_sample.py` instead. Set the working paths once; the case is
the P32D2 basic admission point.

```bash
REPO=$PWD                                   # repository root
CASE=qwen25_1p5b-p32-d2                     # or qwen25_1p5b-p128-d32 to rehearse
MODEL=qwen25_1p5b; P=32; D=2                # keep consistent with CASE
WORK=/absolute/fresh/work-$CASE             # must not exist yet, and must be canonical
OBSERVER=/tmp/mg-observer/observer.so
ENGINE=/tmp/mg-engine/hbserve
PY=/home/xmu/sgl/bin/python                 # interpreter that carries SGLang
GPU=GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13   # any UUID from the preflight.py GPU table
mkdir -p "$WORK"
```

`$WORK` must be absolute and canonical, with no symlink in any component: the
observer compares `SG_NVBIT_OUTPUT_ROOT` with its own `realpath` and refuses a
mismatch. `collect_case.py` canonicalises the root and creates it for you; in
this hand-driven form both are your job.

> Rehearse first on `CASE=qwen25_1p5b-p128-d32` with `P=128 D=32`. That point
> was the one actually driven through this chain in the archived deployment, so
> it exercises the same code path as the recorded receipts.

### Stage 1, census (GPU): run the host under the metadata observer

The census is `host.py` preloaded with `observer.so`. It records the launch
journal and static metadata; it captures no addresses and stores no raw trace.

Generate the job spec with correct pins, then run it under the lease
controller:

```bash
python3 - "$REPO" "$WORK" "$CASE" "$MODEL" "$P" "$D" "$OBSERVER" "$PY" "$GPU" <<'PY'
import hashlib, json, sys
from pathlib import Path
repo, work, case, model, p, d, observer, python, gpu = sys.argv[1:10]
work = Path(work).resolve()
# The observer initialises before the child interpreter runs and refuses to start
# unless this root exists and is byte-identical to its own realpath.
(work / 'observers' / (case + '-census')).mkdir(parents=True, exist_ok=True)
def pin(path):
    path = Path(path).resolve(); b = path.read_bytes()
    return dict(path=str(path), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
sources = [pin(repo + '/integrations/sglang/run_job.py'),
           pin(repo + '/integrations/sglang/memgen-adapter/host.py'),
           pin(repo + '/integrations/sglang/memgen-adapter/contract.json'),
           pin(repo + '/integrations/sglang/memgen-adapter/matrix_workload.py'),
           pin(observer)]
spec = dict(case_id=case, tool='memgen', input_kind='single_layer_profile',
            cpu=8, gpu=gpu, seconds=1800, cache_directory=str(work / 'cache'),
            argv=[python, '-B', repo + '/integrations/sglang/memgen-adapter/host.py',
                  '--model', model, '--prefill-length', p, '--decode-steps', d,
                  '--output', str(work / 'runs' / (case + '-census') / 'host')],
            environment={'LD_PRELOAD': observer, 'SG_NVBIT_SCOPE_ABI': '1',
                         'SG_NVBIT_OUTPUT_ROOT': str(work / 'observers' / (case + '-census')),
                         'SG_NVBIT_MAX_BYTES': str(256 << 20),
                         'ACK_CTX_INIT_LIMITATION': '1'},
            sources=sources)
(work / 'census-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
print('wrote', work / 'census-spec.json', 'with', len(sources), 'pinned sources')
PY

python3 -B integrations/sglang/run_job.py \
  --spec "$WORK/census-spec.json" \
  --output "$WORK/runs/$CASE-census" \
  --execute
```

Running without `--execute` prints the prepared receipt and writes nothing;
use that to check the spec before spending GPU time.

Expected, all three must hold before continuing:

| Artifact | Field | Value |
| --- | --- | --- |
| `$WORK/observers/$CASE-census/process-<pid>/finish.json` | `status` | `PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE` |
| `$WORK/observers/$CASE-census/process-<pid>/launch-journal.jsonl` | — | present, non-empty |
| `$WORK/runs/$CASE-census/host/process-<pid>/finish.json` | `status` | `PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE` |
| `$WORK/runs/$CASE-census/job-finish.json` | `status` | `PASS_PROCESS_ONLY` |

The observer and the host must share one `pid`. Their directories are named
differently on purpose — the observer writes `process-<pid>-<ticks>` because it
does not know the controller's name, the controller writes `process-<pid>` — so
compare pids, never directory names. `collect_case.py` applies the same gate as
`wait_then_sample.py`: both statuses, one pid, `epoch_begin_count ==
epoch_end_count` with `active_epoch == 0`, metadata below the quota, and
receipts that name the requested case, CPU and GPU. A census that hits the
metadata cap is a failure, not a partial success. The census is also the stage
that fails first if the observer root is missing or non-canonical: the observer
reports `fatal: canonical existing output root` and the job exits before any
CUDA call.

### Stage 2, plan (no GPU): pick one decoder layer and the sample set

```bash
JOURNAL=$(ls -d "$WORK"/observers/$CASE-census/process-*)
HOST_FINISH=$(ls -d "$WORK"/runs/$CASE-census/host/process-*)/finish.json

python3 -B integrations/sglang/memgen-adapter/make_sample_plan.py \
  --journal "$JOURNAL" --host-finish "$HOST_FINISH" \
  --output "$WORK/plan" --layer 0
#    expect: PASS_PLAN_ONLY_NOT_SAMPLED, launches and selected_launches counts
```

Outputs `$WORK/plan/sample-plan.json`, `layer-bindings.json`, `census.json`.

### Stage 3, build the sampler (no GPU)

```bash
python3 -B integrations/sglang/compact-sources/upstream/nvbit_sampler_r4/build.py \
  --plan "$WORK/plan/sample-plan.json" --output "$WORK/sampler-build"
python3 integrations/sglang/tool_identity.py "$WORK/sampler-build/sampler.so"
```

The sampler build consumes the real plan, so it can only run after stage 2. If
`build.py` offers no `--nvbit`/`--cuda` override you need, pass them explicitly;
the defaults are the paths in `ENVIRONMENT.md` section 2.

### Stage 4, sparse sampling (GPU)

Stages 2 to 5 are also covered by one wrapper, `followthrough.py`, which runs
`plan -> build -> sample -> expand -> memgen` in order and stops where you ask:

```bash
python3 -B integrations/sglang/memgen-adapter/followthrough.py \
  --journal "$JOURNAL" --host-finish "$HOST_FINISH" \
  --sources integrations/sglang/compact-sources \
  --output "$WORK/followthrough" \
  --stop-after sample \
  --python "$PY" \
  --sample-seconds 7200
#    expect last stage receipt: PASS_THROUGH_SAMPLE
```

To run the sampling stage directly instead:

```bash
CUDA_VISIBLE_DEVICES=$GPU python3 -B integrations/sglang/memgen-adapter/sample_pipeline.py \
  --upstream integrations/sglang/compact-sources/upstream \
  --sampler-lib "$WORK/sampler-build/sampler.so" \
  --plan "$WORK/plan/sample-plan.json" \
  --model "$MODEL" --prefill-length "$P" --decode-steps "$D" \
  --output "$WORK/sample" --seconds 7200
#    expect: PASS_SINGLE_LAYER_SAMPLES_AND_PROFILE_FITTING
```

Two guards run before any GPU work: the case triple is re-validated against
`contract.json`, and `CUDA_VISIBLE_DEVICES` must be set by you, not inherited.
The pipeline requires a sparse plan: it rejects a plan that samples every
launch.

### Stage 5, expand and replay (no GPU after expansion)

Expand turns the packed profile into a full-inference address stream; replay
streams that stream through the functional L1/L2 cache filter and writes
aggregate counters. Neither stage is HBFSim cosimulation: there is no
dependency DAG, no queue, bank or channel scheduling and no completion stall, so
no timing claim can come from here. HBFSim lives on
`research/hbfsim-cosimulation` in a separate worktree and is not reachable from
`main`.

```bash
python3 -B integrations/sglang/memgen-adapter/followthrough.py \
  --journal "$JOURNAL" --host-finish "$HOST_FINISH" \
  --sources integrations/sglang/compact-sources \
  --output "$WORK/expand-memgen" \
  --stop-after memgen \
  --python "$PY"
#    expect last stage receipt: PASS_THROUGH_MEMGEN
```

Or explicitly, with the engine you built in section 2:

```bash
python3 integrations/sglang/memgen-adapter/profile_cache.py \
  --sample "$WORK/sample" \
  --bindings "$WORK/plan/layer-bindings.json" \
  --output "$WORK/profile-cache"
```

`profile_cache.py` runs `expand_profiles.py` then `run_memgen.py --expanded`.
Both set `hardware_accuracy_accepted=false` and
`allow_full_NCU_accuracy_comparison=false` in their receipts. To drive the
frozen engine yourself on a profile index you already admitted:

```bash
python3 integrations/sglang/memgen-adapter/run_memgen.py \
  --binary "$ENGINE" \
  --profile-index "$WORK/expanded/profiles.index.jsonl" \
  --app-config "$WORK/expanded/app.config" \
  --issue-config "$WORK/expanded/issue.config" \
  --hw-config release/config/RTX4000Ada.paper-v1.config \
  --output "$WORK/replay"
```

Read the result from `<replay>/model/kernel_summary.csv` and
`<replay>/model/cache_observation.json`. That JSON states
`hardware_acceptance = DIAGNOSTIC_NOT_HARDWARE_ACCEPTANCE`; keep that label. The
receipt says `raw_trace_materialized: false` and
`hardware_accuracy_accepted: false`, so the counters are summary traffic
counters, not an accepted hardware result.

### Optional: the one-shot waiter

`wait_then_sample.py` waits for a census job, then runs the sample and
cache stages under the lease controller. It needs the census to have closed:

```bash
python3 -B integrations/sglang/memgen-adapter/wait_then_sample.py \
  --census-job-finish "$WORK/runs/$CASE-census/job-finish.json" \
  --observer-root     "$WORK/observers/$CASE-census" \
  --host-root         "$WORK/runs/$CASE-census/host" \
  --sources           integrations/sglang/compact-sources \
  --observer-binary   "$OBSERVER" \
  --controller        integrations/sglang/run_job.py \
  --output            "$WORK/sample-chain" \
  --cache-directory   "$WORK/cache" \
  --wait-seconds 2400 --sample-seconds 7200
```

It verifies `memgen-adapter/deployment-files.json` before waiting, pins every
input, and refuses to continue if a pinned file changes while it waits. Its
final status is `PASS_DECLARED_PROFILE_CACHE_MODEL_NOT_NATIVE_ACCURACY`, which
names its own limit.

### 3.3 Explicit r4 replay

The r4 L1 model needs its own hardware configuration and a matching native
address context, which the wrapper does not invent; what must be supplied is in
[the acceptance procedure](REPRODUCTION.md) section 4.

```bash
python3 -B scripts/test_cache_core.py --output /absolute/fresh/core-tests
python3 -B scripts/test_replay_entry.py \
  --binary /absolute/fresh/core-tests/legacy/build/hbserve \
  --output /absolute/fresh/replay-tests

python3 integrations/sglang/memgen-adapter/run_memgen.py \
  --binary /absolute/fresh/core-tests/legacy/build/hbserve \
  --profile-index /absolute/admitted/profiles.index.jsonl \
  --app-config /absolute/admitted/app.config \
  --issue-config /absolute/admitted/issue.config \
  --hw-config release/config/RTX4000Ada.r4.config \
  --r4-context /absolute/admitted/r4-context.json \
  --output /absolute/fresh/replay
```

`PASS_PROFILE_STREAM_CACHE_REPLAY` means the supplied stream closed, not NCU
agreement.

---

## 4. Stopping, cleanup and failure handling

- Cancel with `SIGTERM`, `SIGINT` or `SIGHUP`. `run_job.py`,
  `sample_pipeline.py` and `replay` clean up their children and release their
  leases; `run_memgen.py` has no wall-clock deadline and relies on this.
- Every output directory must be fresh. All entry points refuse an existing
  path, so a retry needs a new `$WORK`, not a deleted one.
- `run_memgen.py --seconds` is accepted and ignored. Do not rely on it.
- Failure triage, in order: the job receipt
  (`runs/<case>-*/job-finish.json`), then the controller stdout/stderr next to
  it, then the stage logs written by `followthrough.py` and
  `sample_pipeline.py`.
- A `DRIFT` or `PRESENT_DRIFT` from `bootstrap_vendor.py`, or a non-zero exit
  from `verify_archive.py`, means stop: the archive no longer matches its own
  pins.

---

## 5. What each stage proves, and what it does not

| Artifact | Proves | Does not prove |
| --- | --- | --- |
| Step 0 suite | implementation health, determinism, pin integrity | anything about a real workload |
| `observer.so` build receipt | the frozen source and headers compiled, no GPU used | that the binary is byte-reproducible, see `ENVIRONMENT.md` 4.4 |
| census receipt | the launch journal closed for one case and no metadata cap was hit | that a sample is representative |
| sample plan | one decoder layer plus exceptions were selected from a real census | full-model coverage |
| packed profile | a bounded sparse fit was accepted | that every launch is supported |
| `kernel_summary.csv` | address generation and cache filtering ran on the declared stream | any hardware-accuracy claim |
| NCU comparison | traffic agreement for the same range and denominator | other ranges, other contexts |

The admission authority remains
`validation/p32d2_branch_status.csv`. Declaring and producing `P32D2` does not
change its rows: they stay `BLOCKED` until an independent profile and a
three-repeat NCU reference exist for that exact case. This repository contains
no NCU harness for the SGLang/BF16 stack, so a P32D2 accuracy admission cannot
be closed with in-repo tooling alone.
