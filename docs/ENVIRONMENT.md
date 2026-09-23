# Environment and prerequisites

This document lists every machine-local dependency the archived workflow assumes,
the value verified on the reference host, and how to check it. HBServe is the
address generator; Memgen is the L1/L2 cache filter; NCU is NVIDIA Nsight
Compute; NVBit is the instrumentation framework used to collect the sparse
memory-SASS sample.

Two tiers must be separated:

| Tier | Needs | Commands |
| --- | --- | --- |
| CPU-only implementation health | `mpic++`, a C++17 toolchain, `libzstd`, `boost_mpi`, `libcrypto`, `python3` | `scripts/verify_archive.py`, `scripts/run_cpu_smoke.sh`, `scripts/test_cache_core.py` |
| SGLang sampling and NCU reference | NVIDIA GPU + driver, CUDA 12.8 (`nvcc`, `ncu`), NVBit, SGLang/PyTorch stack, the model checkpoints | `integrations/sglang/preflight.py` and the chain in [reproduction](REPRODUCTION.md) |

Nothing in either tier establishes hardware accuracy. The tiers only decide
whether a command can run at all.

## 1. One command to check the host

```bash
python3 integrations/sglang/preflight.py
```

Each line is `STATUS NAME DETAIL`. `OK` means present and matching the pin.
`MISSING` and `DRIFT` block the tier that needs them. `SKIPPED` marks an
optional dependency that is absent. The command also prints the three workload
matrices and which of them the current adapter can actually build; see
section 5.

Useful variants:

```bash
python3 integrations/sglang/preflight.py --gpus none          # CPU tier only
python3 integrations/sglang/preflight.py --binary /abs/path/to/hbserve
python3 integrations/sglang/preflight.py --receipt /tmp/preflight-r1.json
```

## 2. Verified values on the reference host

These are the values recorded by `preflight.py` on the host that produced the
archived receipts. They are defaults inside archived scripts, not preferences.

| Dependency | Verified value | Used as a default by |
| --- | --- | --- |
| Python | `/usr/bin/python3` | this repository's `scripts/` |
| SGLang interpreter | `/home/xmu/sgl/bin/python` (symlink to `python3`) | `followthrough.py`, `sample_pipeline.py`, `wait_then_sample.py` `--python` |
| MPI C++ compiler | `/usr/bin/mpic++` (GCC 11.4) | `scripts/run_cpu_smoke.sh` |
| CUDA toolkit | `/usr/local/cuda-12.8` (`nvcc`, `ncu`) | `compact-sources/observer/build.py`, `compact-sources/upstream/nvbit_sampler_r4/build.py` |
| NVBit | `/home/xmu/nvidiagds/simulators/hyfiss/tracing-tool/nvbit` (`libnvbit.a`, `nvbit.h`, `nvbit_tool.h`) | both NVBit `build.py` scripts `--nvbit` |
| GPU | RTX 4000 Ada Generation, 4 visible; 3 in `run_job.py` `GPU_POOL` | `run_memgen.py`, `run_job.py`, `wait_then_sample.py` |
| Frozen engine root | `/home/xmu/nvidiagds/codex-runs/memgen-paper-ada-v1-20260916-01a08d87-r1` | `run_memgen.py` `--binary` default, `wait_then_sample.py` `FROZEN` |
| Qwen checkpoint | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa798...` | `memgen-adapter/contract.json` |
| Llama checkpoint | `~/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct` | `memgen-adapter/contract.json` |

`preflight.py` also verifies the six pinned package versions in
`contract.json` (`sglang 0.4.10`, `torch 2.7.1+cu126`, `sgl-kernel 0.2.8`,
`triton 3.3.1`, `flashinfer-python 0.2.9rc2`, `transformers 4.54.1`) against the
selected interpreter, because `matrix_workload.check_packages()` rejects any
drift at sampling time.

## 3. Materialize the pinned controller deployment

`integrations/sglang/package.json` records the absolute origin and SHA-256 of
every file the frozen controller deployment needs. Six of them are imported at
runtime but are not carried in this archive, so without them `run_job.py`
cannot be imported and `wait_then_sample.py` cannot pin its controller:

```text
vendor/parent_controller.py
metadata-host/metadata_host.py
metadata-host/vendor/metadata_host.py
metadata-host/vendor/matrix_common.py
metadata-host/vendor/matrix_workload.py
metadata-host/vendor/contract.json
```

Materialize them, verifying both ends of the copy:

```bash
python3 integrations/sglang/bootstrap_vendor.py --check          # report only
python3 integrations/sglang/bootstrap_vendor.py                  # copy missing files
python3 integrations/sglang/bootstrap_vendor.py --receipt /tmp/vendor-r1.json
```

Statuses are `PRESENT_IDENTICAL`, `MATERIALIZED`, `PRESENT_DRIFT`,
`ORIGIN_DRIFT`, `MISSING_ORIGIN`. Only the first two are success. The script
never downloads anything, never overwrites a file whose content differs from
the pin, and never runs a GPU. A materialized file is byte-identical to the
deployment that produced the archived receipts; a mismatch is reported and
never repaired.

`metadata-host/vendor/matrix_common.py`, `matrix_workload.py` and
`contract.json` are byte-identical to their `memgen-adapter/` counterparts by
construction. The duplicate copies are what the frozen deployment used, so they
are materialized rather than replaced by symlinks.

Two files no longer match their historical pin, because this archive
deliberately revised them. `integrations/sglang/revisions.json` records each
such change, its before/after SHA-256 and its reason:

- `bootstrap_vendor.py --check` reports `PRESENT_REVISED` for a file that
  matches a revision record, and `PRESENT_DRIFT` for one that does not;
- `revisions_check` verifies every copy named by the record, so a half-applied
  revision (one mirror updated, the other not) fails instead of passing
  quietly;
- the statuses are distinct on purpose. `PRESENT_REVISED` is a reviewed change;
  `PRESENT_DRIFT` is unexplained and is always a failure.

## 4. Build the pieces the archive does not carry

Prebuilt shared objects, NVBit itself, model weights and NCU report databases
are deliberately excluded. The engine has no prebuilt binary either.

### 4.1 CPU engine (no GPU)

This is exactly what `scripts/run_cpu_smoke.sh` runs:

```bash
mpic++ -std=c++17 -O2 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  release/source/tools/hbserve_profile_stream_cache_semantic_r17.cpp \
  -l:libzstd.so.1 -lz -lboost_mpi -lboost_serialization -lcrypto -pthread \
  -o /absolute/fresh/build/hbserve
```

### 4.2 NVBit metadata observer (GPU)

```bash
python3 integrations/sglang/compact-sources/observer/build.py --output /absolute/fresh/observer-build
```

The three recorded steps, taken verbatim from
`integrations/sglang/transition-proof-r1.json` (`compact_build.steps`, all
`returncode 0`), are:

```bash
/usr/local/cuda-12.8/bin/nvcc -dc -c -std=c++11 \
  -I/home/xmu/nvidiagds/simulators/hyfiss/tracing-tool/nvbit \
  -Xptxas -cloning=no -Xcompiler -Wall \
  -gencode arch=compute_89,code=sm_89 -O3 -Xcompiler -fPIC \
  <sources>/observer/observer.cu -o <build>/observer.o

/usr/local/cuda-12.8/bin/nvcc -gencode arch=compute_89,code=sm_89 -O3 <build>/observer.o \
  -L/home/xmu/nvidiagds/simulators/hyfiss/tracing-tool/nvbit -lnvbit \
  -L/usr/local/cuda-12.8/lib64 -lcuda -lcudart_static -lcrypto -lpthread -ldl \
  -shared -o <build>/observer.so

/usr/bin/nm -D <build>/observer.so
```

### 4.3 NVBit sparse sampler (GPU)

The sampler build consumes a real sample plan, so it can only run after
`make_sample_plan.py`:

```bash
python3 integrations/sglang/compact-sources/upstream/nvbit_sampler_r4/build.py \
  --plan /absolute/admitted/sample-plan.json --output /absolute/fresh/sampler-build
```

Both `build.py` scripts default `--nvbit` and `--cuda` to the values in
section 2. Override them explicitly on any other host.

### 4.4 The NVBit tool build is not byte-reproducible

This matters because `transition-proof-r1.json` records `observer.so` SHA-256
`9e446ae7…c285` as an evidence identity. That hash identifies one artifact; it
is not a property of the recipe and cannot be re-derived. Measured on the
reference host with the frozen source:

| Observation | Value |
| --- | --- |
| archived `observer.so` | 2741496 bytes, artifact `9e446ae7…`, content `971de6c9…` |
| rebuild, twice, from the frozen source | 2741496 bytes, artifact `a9fbeb1c…` and `18f9d9c4…`, content `1cf0f7fa…` both times |
| rebuild versus archived | content differs; `.text` 1067 of 1292002 bytes (0.083%), plus `.dynsym` 7807, `.gnu.hash` 4837, `.rela.plt` 939, `.rela.dyn` 171, `__nv_module_id` 16 |
| two consecutive rebuilds | content identical, `.strtab` differs by 4 bytes only, stripped binaries byte-identical |

All declared inputs were verified byte-identical first, including `observer.cu`
(`a2ca02f7…`), `build.py`, `nvcc` (`3aadf006…`) and `libnvbit.a` (`bd5fd2f0…`).
The residual difference is inside nvcc/ptxas and host-link code generation.

Two consequences:

1. `observer/manifest.json` declares `build_inputs` and `nvbit_headers` only.
   `nvcc` and `libnvbit.a` are **recorded** in the build receipt but never
   **compared** to a declared expectation, so a changed NVBit or CUDA compiler
   would not fail the build. They match the archived recording today, but that
   was verified by hand, not by the build.
2. The reproducible identity of the sampling toolchain is the source and header
   SHA-256 values plus the recorded argv, which `build.py` does verify. Do not
   use the `.so` hash as a tamper check.

Report and compare the two identities with:

```bash
python3 integrations/sglang/tool_identity.py /absolute/fresh/observer-build/observer.so
python3 integrations/sglang/tool_identity.py <build-a>/observer.so --compare <build-b>/observer.so
```

`artifact_sha256` is the whole file; `content_sha256` covers every section
except `.symtab`, `.strtab`, `.comment` and `.note.gnu.build-id`, so it is
equal across rebuilds of the same source and differs when the code differs.
The command exits 1 when the two contents differ.

The same treatment applies to `sampler.so`, which uses the same nvcc flow. Its
identity has not been measured here because its build needs a real sample plan.

## 5. Declared workload matrices

The adapter declares two matrices in `memgen-adapter/contract.json`. They are
derived, not duplicated: `matrix_workload.add_arguments` and
`matrix_workload.contract` read the permitted axes and the accepted
`(model, prefill, decode)` triples from that file, and `sample_pipeline.py`
reuses the same derivation instead of repeating hard-coded choices.

| Declared matrix | Prefills | Decodes | Models | Cases | Status |
| --- | --- | --- | --- | --- | --- |
| scale series (`prefills` / `decodes`) | 128, 256, 512, 1024 | 32, 64, 128 | 2 | 24 | buildable |
| basic admission (`basic_admission`) | 32 | 2 | 2 | 2 | buildable |

The scale series keeps the identity it had before the admission point existed,
so the archived 24-case results are unaffected. The admission point is declared
in its own block so that producing `P32D2` can never be read as extending the
scale series. The contract `schema` string was bumped to
`SGLANG_FULL_INFERENCE_V2` so a receipt cannot silently mix the two.

The documentation names three matrices, which are not the same set:

| Documented matrix | Decodes | Divergence |
| --- | --- | --- |
| basic admission point `P32D2` | 2 | declared and buildable |
| scale series | 2, 4, 8, 16, 32 | `D=32` declared; `D=2` only via the admission point; `D=4`, `D=8`, `D=16` undeclared |

Undeclared pairs are rejected rather than silently accepted:

```text
ACCEPT ('qwen25_1p5b', 32, 2)    -> matrix=basic_admission
ACCEPT ('qwen25_1p5b', 128, 32)  -> matrix=scale_series
REJECT ('qwen25_1p5b', 32, 128)  -> Case outside the declared matrix
REJECT ('qwen25_1p5b', 128, 2)   -> Case outside the declared matrix
```

**Producing a declared point is not an admission decision and not an accuracy
result.** Every point still needs its own independent sample, packed profile and
three-run NCU reference. The P32D2 rows stay `BLOCKED` in
`validation/p32d2_branch_status.csv` until that evidence exists.

`python3 integrations/sglang/preflight.py` prints both sets and the exact
divergence, and flags the documented sentences it reads with
`[DOC TRACE NOT FOUND]` if the prose moves.

### 5.1 Revision record

`integrations/sglang/revisions.json` explains each deliberate difference from a
historical pin, naming every copy of a revised file, the before/after SHA-256
and the reason. The verification rules are:

- a file matching its historical pin reports `PRESENT_IDENTICAL`;
- a file matching a revision record reports `PRESENT_REVISED`;
- a file matching neither reports `PRESENT_DRIFT` and fails;
- every copy listed in a revision must carry the recorded current hash, so a
  half-applied revision fails.

A revision entry is a change record, never a way to silence drift. Adding one is
a reviewed change.

## 6. Behaviours that surprise callers

- `run_memgen.py --seconds` is accepted and ignored. There is no wall-clock
  deadline on replay; cancel with SIGTERM/SIGINT/SIGHUP, which cleans up the
  child.
- Every output directory must be fresh. `run_cpu_smoke.sh`, `test_cache_core.py`
  and `run_memgen.py` all refuse an existing path.
- `release/config/RTX4000Ada.r4.config` requires `--r4-context`; the legacy
  `release/config/RTX4000Ada.paper-v1.config` does not. Known modeled
  cross-layer binding markers are rejected by the C++ reader when r4 context is
  requested, including when a caller bypasses the Python wrapper.
- The r4 context fixture in `scripts/test_cache_core.py` is synthetic. Its
  `claim_boundary` says so; no test allocation is hardware evidence.
- `run_job.py` only admits CPUs `0..15` and the three GPU UUIDs in its
  `GPU_POOL`. A fourth visible GPU is deliberately excluded.
- r4 L1 state is cleared per kernel (`l1_preserve_across_kernels=0`) while L2 is
  preserved across kernels in one replay. A new process still starts from an
  empty L2.
- Importing the pinned controller writes `__pycache__/*.pyc` next to it, and
  `scripts/verify_archive.py` rejects bytecode as a forbidden artifact. Always
  invoke these entry points as `python3 -B`, which is what the recorded
  invocations in `transition-proof-r1.json` do. `preflight.py` additionally sets
  `sys.dont_write_bytecode` before its first local import.

## 7. What this archive does not carry

| Absent | Why | How to obtain |
| --- | --- | --- |
| Prebuilt `hbserve` binary | excluded from Git | build per section 4.1 |
| `observer.so`, `sampler.so` | excluded (`*.so` in `.gitignore`) | build per sections 4.2, 4.3 |
| NVBit itself | third-party, outside the repo | install NVBit; then point `--nvbit` at it |
| Model weights | large third-party artifacts | fetch the checkpoint; verify `config_sha256` |
| `.ncu-rep` databases | excluded (`*.ncu-rep`) | re-profile on the target GPU |
| Raw memory traces | excluded by design | regenerated by HBServe from the packed profile |
| `vendor/parent_controller.py`, `metadata-host/*` | pinned but not committed | materialize per section 3 |

Third-party rights are not assessed here. See
[provenance](PROVENANCE.md) before redistributing anything.

## 8. Change record

Three reviewable passes touched this archive. None produced an accuracy claim.

**Pass 1, environment and preflight.** Added `integrations/sglang/preflight.py`,
`integrations/sglang/bootstrap_vendor.py` and this document. The six files
pinned by `integrations/sglang/package.json` were materialized from their
recorded origins and verified byte-identical.

**Pass 2, declared basic admission point.** Added
`integrations/sglang/revisions.json` and revised exactly five files, so that the
`P32D2` admission point named in the branch contract can be selected:

| File | Change |
| --- | --- |
| `memgen-adapter/contract.json` | declare `basic_admission`; bump `schema` |
| `memgen-adapter/matrix_workload.py` | derive axes and accepted triples from the contract |
| `memgen-adapter/sample_pipeline.py` | reuse `matrix_workload.add_arguments`, and validate the case triple in the parent process |
| `metadata-host/vendor/contract.json` | mirror of the first |
| `metadata-host/vendor/matrix_workload.py` | mirror of the second |

`memgen-adapter/deployment-files.json` was updated with the new `bytes` and
`sha256` for the three adapter files, and `revisions.json` records the change
for the two vendored copies. Everything else is unchanged:

- `release/current-core-manifest.json` (6 hashed core files) is untouched;
- the other 10 hash-pinned adapter files are untouched;
- `release/config/*.config` and all `validation/*.csv` values are untouched;
- `integrations/sglang/package.json` is untouched: it remains the historical
  provenance record, and `revisions.json` is what explains the divergence.

The frozen runtime check in `wait_then_sample.py` was replayed against the
updated manifest and passes for all 13 rows. Both negative tests fail as
required: an unexplained tamper reports `PRESENT_DRIFT`, and a half-applied
revision reports `DRIFT`.

**Pass 3, tool build reproducibility.** Added
`integrations/sglang/tool_identity.py` and section 4.4. The NVBit metadata
observer was built from the frozen source on the reference host and compared
against the archived build. No file of the frozen sampling chain was changed.
The finding is that the `.so` hash recorded as an evidence identity is an
artifact identity, that `nvcc` and `libnvbit.a` are recorded but not verified
by `build.py`, and that the reproducible identity is the source and header
hashes plus the recorded argv.

Re-verified after all three passes: `scripts/verify_archive.py`,
`scripts/run_cpu_smoke.sh`, `scripts/test_cache_core.py`,
`tests/sampling/test_profile_census.py`, `tests/sglang/test_declared_cases.py`,
`bootstrap_vendor.py --check`, `preflight.py` and `tool_identity.py` all pass.
Nothing added here is a hardware-accuracy claim, and no evidence table was
changed.
