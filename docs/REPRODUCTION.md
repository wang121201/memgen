# Reproduction and acceptance procedure

HBServe is the full-inference address generator. Memgen is the functional L1
(Level-1) and L2 (Level-2) cache filter. NCU is NVIDIA Nsight Compute. DRAM is
off-chip graphics-memory traffic. `P<n>D<m>` denotes `n` prefill tokens and `m`
decode steps in a batch-one inference.

## 1. Verify the archive

Run the read-only manifest/schema check:

```bash
python3 scripts/verify_archive.py
```

Run the bounded CPU smoke in a new output directory:

```bash
scripts/run_cpu_smoke.sh /tmp/memgen-cpu-smoke-r1
```

The smoke compiles the current C++17 engine in legacy mode with `mpic++`, replays a synthetic
packed profile twice with cache observation off/on, and requires identical
`kernel_summary.csv` output with SHA-256
`9e3b2ee1b69fce3650b9ae2e5a26583ff786d86008f2bb698fc1463915f2d9f2`.
It is a functional check, not hardware accuracy.

## 2. Exact SGLang workload contract

For current acceptance use:

- Qwen2.5-1.5B-Instruct in BF16 (Brain Floating Point 16-bit);
- SGLang 0.4.10 + FlashInfer;
- batch 1, tensor parallelism 1, eager execution, CUDA Graph disabled;
- RTX 4000 Ada Generation, 48 streaming multiprocessors;
- a workload-specific sparse sample/profile for every independent P/D point;
- three NCU runs for each whole, prefill and continuous-decode range.

Do not reuse a P128D16 profile as proof for an independent P128D4 or P128D8
workload. A prefix sum is diagnostic only.

## 3. Generate without a full raw trace

The current integration source is under `integrations/sglang/`:

1. `compact-sources/upstream/nvbit_sampler_r4/` collects bounded sparse
   memory-SASS and CTA placement.
2. `compact-sources/upstream/sglang_sample_to_packed.py` converts qualified
   samples into packed inputs.
3. `compact-sources/upstream/template_adapter_r4/hbserve_adapter.py` and its
   supporting modules construct the HBServe profile/address rules.
4. `memgen-adapter/expand_profiles.py` expands the workload-specific profiles.
5. `memgen-adapter/run_memgen.py` streams the generated source into the cache
   backend and records aggregate counters.

These scripts retain their frozen path contracts and should first be exercised
with their included manifests. Portability cleanup must be a reviewed change,
not an unrecorded edit to the archived snapshot.

## 4. NCU comparison

For each independent workload, freeze model/runtime/token/GPU identity before
profiling. Compare only identical ranges and denominators. Preserve:

- source/profile/config SHA-256 values;
- number of phases, kernels, memory instructions, lane addresses and 32-byte
  sectors;
- NCU whole/prefill/decode counters for three repeats and the median;
- Memgen whole/prefill/decode counters;
- explicit missing or incompatible L1/L2 denominators.

Traffic signed relative error is
`100 * (model_bytes - NCU_bytes) / NCU_bytes`. The repository-wide admission
gate requires DRAM write absolute relative error at most 20% for the complete
request and every individual decode step. DRAM read and cache hit-rate error
remain mandatory report columns. Internal conservation is mandatory but does
not prove hardware accuracy. See
[the unified branch contract](BRANCH_AND_ACCEPTANCE_CONTRACT.md).

## 5. Promotion rule

`main` keeps ordinary LRU (least recently used replacement) as the legacy control. The optional r4 L1 software does not promote an L2 write policy. An L2 write-back or
dirty-management candidate may be promoted from
`research/l2-writeback-dirty-management` only after it passes deterministic
fixtures, preserves source/phase conservation, improves independent SGLang
workloads including decode-only write, and does not regress read traffic. A
single workload fit or aggregate whole-request cancellation is insufficient.

## 6. Explicit r4 cache replay

r4 is the fourth experimental cache candidate. CTA (cooperative thread array)
means a CUDA thread block; SM (streaming multiprocessor) is the block placement
unit. The L1 replacement/tag line is 128 bytes, while reads and fills are counted
in 32-byte sectors. The shared-memory capacity table and unchanged experimental
L2 write behavior are specified in `release/config/README.md`.

```bash
python3 scripts/test_cache_core.py --output /absolute/fresh/core-tests
python3 scripts/test_replay_entry.py \
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

Supply an admitted profile stream, the matching per-kernel allocation lifetimes
and measured shared-memory context, and round-robin CTA placement. The wrapper
does not invent this information. Known modeled cross-layer binding markers are
rejected by the C++ reader whenever r4 context is requested, including when a
caller bypasses the Python wrapper. Unmarked input is not independently certified
as hardware-native merely because it passes this rejection check.

The historical `--expanded DIRECTORY` mode remains available with legacy
configuration. It retains its synthetic-address boundary. Existing callers
without `--binary` retain the archived XMU executable path; new callers should
always supply the built binary explicitly. The `--seconds` argument is accepted
for compatibility but ignored. This replay entry has no wall-clock deadline;
SIGTERM/SIGHUP/interrupt cancellation cleans up its child. Historical sampling
and outer controller deadlines were not changed by this extraction.

The wrapper records input/configuration/binary hashes, actual configuration,
source closure and timing in a fresh output directory. It stores no expanded
address trace. `PASS_PROFILE_STREAM_CACHE_REPLAY` means the supplied stream
closed successfully, not full-model coverage or NCU agreement. Only separately
joined workload/range evidence can establish those claims.


## 7. Exact sample-profile census optimization

The census is the aggregate count of instructions, active lane addresses and
32-byte sectors generated by an already fitted profile. When an address rule
has the same base residue modulo 32 across its CTA domain, the new counter
evaluates its sector count once and multiplies by the domain size. Otherwise it
enumerates every CTA as before. Exact sampled-address validation and the six
published census fields are unchanged. The counted profiles are byte-equivalent
to the prior fitter on the tested synthetic complete, heterogeneous and sparse
inputs. This does not improve address prediction or hardware calibration.

Run `python3 -B tests/sampling/test_profile_census.py` for portable differential
tests against per-lane byte enumeration. Source preparation copies the helper
when present, and the sampler controller includes it in its source pins. The
direct single-pass collector, extended sparse fitting rules, specialized entry
certificates and research matrix controller remain outside main. No GPU sampling
or full inference was rerun to validate this counting-only change.
