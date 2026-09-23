# Runbook: manual launch and full verification

This is the operator document. It gives the exact commands, the expected
receipts and the claim boundary for each step. Read
[environment](ENVIRONMENT.md) first for the dependency inventory, and
[branch contract](BRANCH_AND_ACCEPTANCE_CONTRACT.md) for what a result is
allowed to claim.

| Document | Answers |
| --- | --- |
| [`README.md`](../README.md) | What the repository is, and the one-line checks |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | Which host dependencies exist, how to check them, how to build the tools |
| [`REPRODUCTION.md`](REPRODUCTION.md) | The acceptance procedure and its evidence rules |
| This runbook | How to actually launch each stage, by hand |

Nothing in this runbook is an accuracy claim. Producing an artifact is not an
admission decision; see section 5.

---

## 1. Step 0: implementation health, no GPU

No GPU, no model, no NVBit. Run this after any change to the repository. Each
command writes only to a fresh directory you name.

The split from section 3 exists because the two kinds of check cost different
things and answer different questions. This section is cheap, needs nothing
beyond a C++17 toolchain, and can therefore be run by anyone reviewing an
archive change to confirm that the frozen engine, the pinned deployment and the
declared-case contract are intact. Section 3 needs the GPU host, the SGLang
stack and NVBit, and is what actually collects the target data. Do not treat a
passing Step 0 as evidence about a workload, and do not treat section 3 as a
substitute for the pin-integrity checks.

```bash
cd <repository root>

# 1. archive contract, derived-core identity, forbidden-artifact scan
python3 -B scripts/verify_archive.py
#    expect: PASS_ARCHIVE_CONTRACT ... and PASS_CURRENT_CACHE_CORE_IDENTITY

# 2. frozen engine smoke: two runs, cache observation off and on, equal output
rm -rf /tmp/mg-smoke && bash scripts/run_cpu_smoke.sh /tmp/mg-smoke
#    expect: PASS_FROZEN_CPU_SMOKE, about 17 s

# 3. optional r4/cache-core regression, CLOCK reference and rejection cases
rm -rf /tmp/mg-core && python3 -B scripts/test_cache_core.py --output /tmp/mg-core
#    expect: /tmp/mg-core/validation.json status PASS_CACHE_CORE_SOFTWARE_REGRESSION

# 4. census counting differential tests
python3 -B tests/sampling/test_profile_census.py
#    expect: OK (5 tests)

# 5. declared-case contract, rejection rules, mirror identity, pin structure
python3 -B tests/sglang/test_declared_cases.py
#    expect: OK (13 tests)

# 6. the six files pinned in package.json, verified against their origins
python3 integrations/sglang/bootstrap_vendor.py --check
#    expect: PASS_VENDOR_PINS_SATISFIED

# 7. host prerequisites, pinned package versions, declared vs documented matrices
python3 integrations/sglang/preflight.py
#    expect: PASS_HOST_PREREQUISITES
#    add --gpus none to skip the GPU checks on a CPU-only host

# 8. nothing may leave bytecode behind
find . -path ./.git -prune -o -name '__pycache__' -print
#    expect: no output
```

`verify_archive.py` fails if a bytecode cache, a model file, an NCU database or
any file over 10 MiB is present, so always invoke the Python entry points with
`-B` as shown. `preflight.py` sets `sys.dont_write_bytecode` itself.

If step 6 reports `PRESENT_REVISED`, that is a recorded, reviewed revision; the
reason is in `integrations/sglang/revisions.json`. `PRESENT_DRIFT` or `DRIFT`
is a failure and must be investigated, never silenced.

---

## 2. Prerequisites for the target run

```bash
python3 integrations/sglang/preflight.py            # all required checks OK
python3 integrations/sglang/bootstrap_vendor.py     # materialize pinned files if absent

# build the frozen CPU engine (used by the replay stage)
rm -rf /tmp/mg-engine && mkdir -p /tmp/mg-engine
mpic++ -std=c++17 -O2 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  release/source/tools/hbserve_profile_stream_cache_semantic_r17.cpp \
  -l:libzstd.so.1 -lz -lboost_mpi -lboost_serialization -lcrypto -pthread \
  -o /tmp/mg-engine/hbserve

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

`run_job.py` takes an exclusive lock per CPU id and per GPU id, sets CPU
affinity and single-threaded library variables, and kills its child if the
controller dies. Do not launch two jobs with the same `cpu`/`gpu`.

---

## 3. The target run

### 3.1 One command

`integrations/sglang/collect_case.py` drives the whole chain for one declared
case as two jobs under the lease controller. It builds the observer, writes
both job specs with their sources pinned by hash, verifies every stage receipt
and writes a collection receipt with the artifact map.

```bash
# review the plan: writes both specs, runs nothing, uses no GPU
python3 integrations/sglang/collect_case.py \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 \
  --gpu-index 1 --work /absolute/fresh/qwen15b-p32d2-r1 --dry-run

# collect
python3 integrations/sglang/collect_case.py \
  --model qwen25_1p5b --prefill-length 32 --decode-steps 2 \
  --gpu-index 1 --work /absolute/fresh/qwen15b-p32d2-r1
```

`--gpu-index` is the numbering `preflight.py` prints. `--case qwen25_1p5b-p32-d2`
is accepted as a shorthand for the three case values, and `--list-cases` prints
every declared case.

It refuses a case outside the declared matrix, a GPU outside the admitted pool,
an existing `--work` and a missing interpreter, so a typo fails before any GPU
time is spent. Exit 0 means every stage receipt closed; exit 2 means the profile
stream did not cover the full model, which the receipt reports as
`STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC` rather than hiding.

`--dry-run` writes `census-spec.json` and `collect-spec.json`. The second names
`process-<pid>` as a placeholder because job 1 resolves the real census process
directory before job 2 is written.

### 3.2 Stage by stage

Use this when a stage fails and you want to rerun one of them, or when you want
to drive `wait_then_sample.py` instead. Set the working paths once; the case is
the P32D2 basic admission point.

```bash
REPO=$PWD                                   # repository root
CASE=qwen25_1p5b-p32-d2                     # or qwen25_1p5b-p128-d32 to rehearse
MODEL=qwen25_1p5b; P=32; D=2                # keep consistent with CASE
WORK=/absolute/fresh/work-$CASE             # must not exist yet
OBSERVER=/tmp/mg-observer/observer.so
ENGINE=/tmp/mg-engine/hbserve
PY=/home/xmu/sgl/bin/python                 # interpreter that carries SGLang
GPU=GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13   # any UUID from the preflight.py GPU table
mkdir -p "$WORK"
```

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
work = Path(work)
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

The observer and the host must share one `pid`; `wait_then_sample.py` also
checks `epoch_begin_count == epoch_end_count`, `active_epoch == 0`, and that
`max_metadata_bytes` was not exhausted. A census that hits the metadata cap is
a failure, not a partial success.

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
`hardware_acceptance = DIAGNOSTIC_NOT_HARDWARE_ACCEPTANCE`; keep that label.

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
