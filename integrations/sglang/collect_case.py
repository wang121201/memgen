#!/usr/bin/env python3
"""Collect one declared case end to end: census, sample, expand and cache replay.

Target use is the basic admission point, `qwen25_1p5b-p32-d2`. Nothing here
invents evidence and nothing here claims accuracy.

Two jobs, both under `run_job.py`, which owns the CPU/GPU leases, the CPU
affinity, the RSS guard and `CUDA_VISIBLE_DEVICES`. This script acquires no
lease itself and must not be run outside that controller.

    job 1  host.py under LD_PRELOAD=observer.so   -> census receipts
    job 2  followthrough.py --stop-after memgen   -> packed profile, expansion,
                                                     cache counters

The GPU is chosen by index into the admitted pool, the same numbering
`preflight.py` prints, so no UUID has to be typed. The cache replay has no
wall-clock deadline: `run_memgen.py` documents that, and neither this driver nor
`followthrough.py` adds one. Only the GPU stages carry budgets, because they
hold a leased device.

Data flow, and what is deliberately never written to disk:

    SGLang run (GPU)  ->  sparse memory-SASS sample  ->  packed profile
      ->  HBServe full-inference address stream  ->  L1/L2 cache filter
      ->  aggregate kernel_summary.csv and cache_observation.json

The address stream is streamed straight into the cache model. No per-address
trace and no raw SASS are materialized; `materialized_raw_sass_bytes` stays 0 in
the source statistics.

`--dry-run` writes only the two job specs and prints every command, so the plan
can be reviewed before any GPU time is spent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

# A GPU that has just been released still reports a decayed utilization average,
# so the second after another job ends can read as busy on an idle device.
GPU_ADMISSION_WAIT_SECONDS = 600
GPU_ADMISSION_RETRY_SECONDS = 20

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADAPTER = HERE / 'memgen-adapter'
SOURCES = HERE / 'compact-sources'
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ADAPTER))
# One home for the projection-policy names: the adapter that applies them.
sys.path.insert(0, str(SOURCES / 'upstream/template_adapter_r4'))
import matrix_workload as workload  # noqa: E402
import tool_identity  # noqa: E402
from memory_projection import MODEL_POLICIES  # noqa: E402


def pin(path: Path) -> dict:
    path = Path(path).resolve()
    payload = path.read_bytes()
    return dict(path=str(path), bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest())


def gpu_pool() -> set[str]:
    """Read the admitted UUIDs from the controller rather than restating them."""
    return set(re.findall(r'GPU-[0-9a-f-]{36}', (HERE / 'run_job.py').read_text()))


def gpu_table() -> list[tuple[int, str, str]]:
    """The admitted pool by index, with the hardware name when it can be read.

    The numbering is the sorted pool, so index 0, 1 and 2 always name the same
    devices and `preflight.py` prints the same table.
    """
    names: dict[str, str] = {}
    try:
        import subprocess
        out = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout
        names = {uuid.strip(): name.strip()
                 for uuid, name in (line.rsplit(',', 1) for line in out.strip().splitlines())}
    except Exception:  # noqa: BLE001  (a missing nvidia-smi must not break listing)
        pass
    return [(index, uuid, names.get(uuid, 'name unavailable'))
            for index, uuid in enumerate(sorted(gpu_pool()))]


def declared_cases() -> list[tuple[str, int, int]]:
    """Every declared (model, prefill, decode), for --list-cases and error hints."""
    spec = workload.spec()
    return sorted(set().union(*workload.declared_cases(spec).values()))


def source_pins(files, compact: bool) -> list[dict]:
    """Pin the job implementation and its explicit inputs, without duplicates.

    `files` entries are either absolute paths or names inside memgen-adapter.
    The census job pins only what it runs; the collect job also pins every
    compact-source file its stages compile or import.
    """
    paths = [HERE / 'run_job.py']
    paths += [ADAPTER / name if isinstance(name, str) else Path(name) for name in files]
    if compact:
        paths += sorted(p for p in SOURCES.rglob('*')
                        if p.is_file() and p.suffix in ('.py', '.json', '.h', '.cu', '.inc'))
    seen, pins = set(), []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        pins.append(pin(resolved))
    return pins


def observer_root(work: Path, case: str) -> Path:
    """The census observer's output root, created because it must pre-exist.

    `observer.cu` refuses to initialize unless `SG_NVBIT_OUTPUT_ROOT` exists and
    is byte-identical to its own `realpath`, and it initializes before the child
    interpreter runs, so `host.py` cannot create it in time. Both properties are
    established here rather than left to the caller's `--work` spelling.
    """
    root = (work / 'observers' / f'{case}-census').resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def census_spec(case: str, work: Path, contract: dict, args) -> dict:
    pins = source_pins(['host.py', 'contract.json', 'matrix_workload.py'], compact=False)
    if Path(args.observer).is_file():
        pins.append(pin(Path(args.observer)))
    return dict(case_id=case, tool='memgen', input_kind='single_layer_profile',
                cpu=args.cpu, gpu=args.gpu, seconds=args.census_seconds,
                cache_directory=str(work / 'cache'),
                argv=[args.python, '-B', str(ADAPTER / 'host.py'),
                      '--model', contract['model_key'],
                      '--prefill-length', str(contract['prefill_length']),
                      '--decode-steps', str(contract['decode_steps']),
                      '--output', str(work / 'runs' / f'{case}-census' / 'host')],
                environment={'LD_PRELOAD': str(args.observer),
                             'SG_NVBIT_SCOPE_ABI': '1',
                             'SG_NVBIT_OUTPUT_ROOT': str(observer_root(work, case)),
                             'SG_NVBIT_MAX_BYTES': str(256 << 20),
                             'ACK_CTX_INIT_LIMITATION': '1'},
                sources=pins)


def census_process_names(observer: dict, observer_finish: Path) -> tuple[str, str]:
    """The two census process directories, which differ on purpose.

    The observer adds its start ticks to its own directory because it cannot
    know the controller's name for it, and the controller writes `process-<pid>`
    alone. Both are derived from the observer's receipt and checked against the
    directory that receipt was read from, so job 2 cannot be pointed at a path
    that does not exist.
    """
    journal = f"process-{observer['pid']}-{observer['start_ticks']}"
    if observer_finish.parent.name != journal:
        raise SystemExit('census receipt and its directory disagree: '
                         f"{journal} vs {observer_finish.parent.name}")
    return journal, f"process-{observer['pid']}"


def collect_spec(case: str, work: Path, args, journal: str, host: str,
                 engine: Path | None = None) -> dict:
    """Job 2's spec, given the two census process names it has to read.

    `engine` is the frozen CPU binary the cache stage replays through. It is named
    here rather than left to `run_memgen.py`'s own default so the replay uses the
    engine built from this repository's source, and so the spec records its hash.
    """
    follow = work / 'runs' / f'{case}-collect' / 'followthrough'
    argv = [args.python, '-B', str(ADAPTER / 'followthrough.py'),
            '--journal', str(observer_root(work, case) / journal),
            '--host-finish', str(work / 'runs' / f'{case}-census' / 'host' / host
                                 / 'finish.json'),
            '--sources', str(SOURCES),
            '--output', str(follow),
            '--stop-after', 'memgen',
            '--python', args.python,
            '--model-uncovered', args.model_uncovered,
            '--sample-seconds', str(args.sample_seconds)]
    sources = source_pins(['followthrough.py', 'make_sample_plan.py', 'sample_pipeline.py',
                           'expand_profiles.py', 'model_uncovered.py', 'run_memgen.py', 'profile_cache.py',
                           'contract.json', 'matrix_workload.py'], compact=True)
    if engine is not None:
        argv += ['--engine', str(engine)]
        if engine.is_file():
            sources.append(pin(engine))
    return dict(case_id=case, tool='memgen', input_kind='sample_and_cache',
                cpu=args.cpu, gpu=args.gpu, seconds=args.job_seconds,
                cache_directory=str(work / 'cache'),
                argv=argv,
                # The stage env is allow-listed, so the policy travels in the spec
                # rather than in the ambient environment, and the written spec is
                # then the record of which policy this run used.
                environment={'SG_TEMPLATE_MODEL_POLICY': args.model_policy},
                sources=sources)


def only(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise SystemExit(f'expected exactly one {pattern} under {root}, found {len(paths)}')
    return paths[0]


def status_of(path: Path, expected: str) -> dict:
    value = json.loads(path.read_text())
    if value.get('status') != expected:
        raise SystemExit(f'{path} status is {value.get("status")!r}, expected {expected!r}')
    return value


def gpu_admission_is_busy(text: str) -> bool:
    """True when the controller refused a fresh locked admission.

    The vendor criterion is `utilization_percent != 0` on an otherwise idle
    device, and that figure is an average the driver decays, so a busy reading
    right after another job ends is worth a bounded wait rather than throwing
    away a finished census.
    """
    return 'selected GPU not idle at fresh locked admission' in text


def reused_census(work: Path, case: str, cpu: int, gpu: str) -> tuple[Path, dict]:
    """Re-verify the census receipts an earlier run left in this `--work`.

    `--resume` exists because job 1 is the only stage whose result cannot be
    recomputed on the CPU: everything after it runs off the packed profile. Its
    receipts are the proof that it happened, so they are re-verified by the same
    gate rather than trusted because they are there.
    """
    observer_finish = only(observer_root(work, case), 'process-*/finish.json')
    host_finish = only(work / 'runs' / f'{case}-census' / 'host', 'process-*/finish.json')
    observer = verify_census(observer_finish, host_finish,
                             work / 'runs' / f'{case}-census' / 'job-finish.json',
                             case, cpu, gpu)
    return observer_finish, observer


def decided_job_two(follow: Path) -> dict | None:
    """The outcome an earlier job 2 decided here, or None if it never decided.

    A job that failed did not decide anything, so its directory is retried
    rather than reused: the stages hold leases that a failure can leave behind,
    and a status of FAIL names no coverage to report.
    """
    finish = follow / 'finish.json'
    if not finish.is_file():
        return None
    try:
        closed = json.loads(finish.read_text())
    except ValueError:
        return None
    if not isinstance(closed, dict) or str(closed.get('status', '')).startswith('FAIL'):
        return None
    return closed


def reused_policy(follow: Path) -> str:
    """The projection policy the sample under `follow` was taken with.

    The fitter hard-coded strict before `--model-policy` existed, so a receipt
    that does not name a policy records one taken with strict. That is a reading
    of the archive, not a default chosen here.
    """
    for receipt in sorted((follow / 'sample').rglob('receipt.json')):
        try:
            row = json.loads(receipt.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(row, dict) and 'model_policy' in row:
            return row['model_policy']
    return 'strict'


def reused_model_uncovered(follow: Path) -> str:
    """The `--model-uncovered` the expansion under `follow` was completed with.

    The expansion records the setting it ran under as `modeled_completion` in its
    own manifest, so the value is read back from the artifact rather than from a
    copy of the command line. `--model-uncovered` is not a property of the sample:
    the same packed profile is expanded either to refuse a class with no admitted
    template or to give it an explicit numeric_modeled profile, which changes what
    the expansion covers. A tree with no readable expansion reports the default,
    and the caller only asks after `decided_job_two` named a decided job.
    """
    manifest = follow / 'expanded' / 'manifest.json'
    try:
        row = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return 'refuse'
    return str(row.get('modeled_completion', 'refuse'))


def stash_job_output(work: Path, case: str) -> Path | None:
    """Move a previous job 2 directory aside instead of deleting its evidence."""
    stale = work / 'runs' / f'{case}-collect'
    if not stale.exists():
        return None
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    moved = stale.with_name(f'{stale.name}.attempt-{stamp}')
    attempt = 1
    while moved.exists():
        attempt += 1
        moved = stale.with_name(f'{stale.name}.attempt-{stamp}-{attempt}')
    stale.rename(moved)
    return moved


def controller_reason(stderr_text: str, stderr_path: Path, output: Path) -> str:
    """The one line worth printing when a controller run fails.

    A traceback on stderr is the clearest signal, but a child that exits nonzero
    leaves only the receipt, whose `process.error` names the failure.
    """
    lines = [line for line in stderr_text.splitlines()
             if line.strip() and not line.startswith('# attempt ')]
    if lines:
        return lines[-1]
    receipt = output / 'job-finish.json'
    if receipt.is_file():
        value = json.loads(receipt.read_text())
        process = value.get('process') or {}
        detail = process.get('error') or f"returncode {process.get('returncode')}"
        return f"{value.get('status', 'FAIL')}: {detail}"
    return f'controller failed; see {stderr_path}'


def stopped_at_the_gate(follow: Path) -> bool:
    """True when job 2 wrote a finish.json that names a stop of its own.

    `followthrough.py` exits 2 for a stop it decided, such as an expansion that
    does not cover the full model, and that is a result with a coverage to report
    rather than a failed stage.
    """
    finish = follow / 'finish.json'
    return finish.is_file() and json.loads(finish.read_text())['status'].startswith('STOP_')


def run_job(spec_path: Path, output: Path, dry: bool,
            wait_seconds: int = GPU_ADMISSION_WAIT_SECONDS) -> int:
    """Run one job under the lease controller, reporting it in one line.

    The controller writes the whole receipt, including the observed process
    identities, to `<output>/job-finish.json`, so echoing it here would only bury
    the terminal. Its output is kept beside that receipt instead.
    """
    argv = [sys.executable, '-B', str(HERE / 'run_job.py'),
            '--spec', str(spec_path), '--output', str(output), '--execute']
    if dry:
        print('  ' + ' '.join(argv[:-1]))
        return 0
    import subprocess
    log_dir = output.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = log_dir / f'{output.name}.controller.stdout'
    stderr = log_dir / f'{output.name}.controller.stderr'
    deadline = time.monotonic() + wait_seconds
    attempt = 0
    while True:
        attempt += 1
        with stdout.open('a') as out, stderr.open('a') as err:
            # Flush now: the child writes straight to these descriptors, so a
            # buffered header would land after the output it explains.
            handle_text = f"# attempt {attempt}: {' '.join(argv)}\n"
            for handle in (out, err):
                handle.write(handle_text)
                handle.flush()
            # Only this attempt's output decides the verdict: a previous attempt's
            # busy message stays in the file and must not be read as this one's.
            offset = stderr.stat().st_size
            code = subprocess.call(argv, stdout=out, stderr=err)
        with stderr.open() as handle:
            handle.seek(offset)
            message = handle.read()
        if code == 0:
            receipt = json.loads((output / 'job-finish.json').read_text())
            print(f"  {receipt['status']}  wall {receipt['wall_minutes']:.2f} min  "
                  f"cpu {receipt['CPU_minutes']:.2f} min  ->  {output / 'job-finish.json'}")
            return 0
        if not gpu_admission_is_busy(message):
            print('  ' + controller_reason(message, stderr, output))
            print('  controller output ' + str(stderr))
            return 1
        remaining = deadline - time.monotonic()
        if output.exists():
            print('  the device is busy and this job already wrote its output directory; '
                  'nothing was retried')
            return 1
        if wait_seconds == 0:
            print('  the device is busy at fresh admission and --gpu-wait-seconds is 0')
            print('  controller output ' + str(stderr))
            return 1
        if remaining <= 0:
            print(f'  the device stayed busy for the whole {wait_seconds} s wait budget; '
                  'raise --gpu-wait-seconds or free the GPU')
            print('  controller output ' + str(stderr))
            return 1
        pause = min(GPU_ADMISSION_RETRY_SECONDS, remaining)
        print(f'  device busy at fresh admission, so this stage is waiting: {pause:.0f} s '
              f'of a {remaining:.0f} s budget left (a wait, not a kill)')
        time.sleep(pause)


def verify_census(observer_finish: Path, host_finish: Path, job_finish: Path, case: str,
                  cpu: int, gpu: str) -> dict:
    """The census gate `wait_then_sample.py` applies, with its exact meanings.

    The two process directories are named differently on purpose: the observer
    writes `process-<pid>-<ticks>` because it does not know the controller's
    name, and the controller writes `process-<pid>`. They must therefore be
    compared by pid, not by directory name.
    """
    observer = status_of(observer_finish, 'PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE')
    host = status_of(host_finish, 'PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE')
    job = status_of(job_finish, 'PASS_PROCESS_ONLY')
    if host_finish.parent.name != f"process-{observer['pid']}":
        raise SystemExit(f"census observer pid {observer['pid']} and host process "
                         f"{host_finish.parent.name} are not one process")
    if observer['epoch_begin_count'] != observer['epoch_end_count'] or observer['active_epoch'] != 0:
        raise SystemExit('census epochs did not close: '
                         f"{observer['epoch_begin_count']} begin, "
                         f"{observer['epoch_end_count']} end, "
                         f"{observer['active_epoch']} still active")
    if observer['metadata_bytes_before_finish'] >= observer['max_total_bytes']:
        raise SystemExit('census exhausted the observer metadata quota: '
                         f"{observer['metadata_bytes_before_finish']} of "
                         f"{observer['max_total_bytes']} bytes")
    if (job['cpu'], job['gpu'], job['case_id']) != (cpu, gpu, case):
        raise SystemExit('census receipt names other resources: '
                         f"cpu {job['cpu']}, gpu {job['gpu']}, case {job['case_id']}")
    if host['input_contract']['case_id'] != case:
        raise SystemExit('census host receipt names another case: '
                         f"{host['input_contract']['case_id']}")
    return observer


def kernel_counters(path: Path) -> dict:
    """Case totals from the one-row-per-kernel summary.

    Summing is not optional: the table has one row per launch, and a hit rate is a
    ratio of sums, never a mean of ratios.
    """
    import csv
    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    def total(column: str) -> int:
        return sum(int(row[column]) for row in rows)

    result = {'kernels': len(rows),
              'dram_load_bytes': total('dram_load_bytes'),
              'dram_store_bytes': total('dram_store_bytes'),
              'l2_writeback_dirty_sectors': total('l2_writeback_dirty_sectors')}
    for level in ('l1', 'l2'):
        requests, hits = total(f'{level}_requests'), total(f'{level}_hits')
        result[f'{level}_requests'], result[f'{level}_hits'] = requests, hits
        result[f'{level}_hit_rate'] = (f'{hits / requests:.6f}' if requests else None)
    return result


def build_engine(work: Path) -> Path:
    """Build the frozen CPU engine from the archived source, per RUNBOOK 2.

    It is built into `--work` so a replay names a binary that this run produced
    and can hash, instead of depending on a machine-local build. The command
    itself lives in followthrough.py, which is also the stage that replays
    through the binary, so the two paths cannot compile the source differently.
    """
    sys.path.insert(0, str(ADAPTER))
    import followthrough
    return followthrough.build_engine(work / 'engine' / 'hbserve')


def run_partial_replay(case: str, args, follow: Path) -> dict:
    """Replay an expansion that does not cover the full model, labelled partial.

    `followthrough.py` stops at `STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC`
    rather than emit a full-model number from a partial stream, which is right.
    This step exists so the covered part can still be measured, and every number
    it produces is labelled with the coverage it came from.
    """
    expansion = follow / 'expanded'
    manifest = json.loads((expansion / 'manifest.json').read_text())
    engine = Path(args.engine).resolve() if args.engine else build_engine(args.work)
    if not engine.is_file():
        raise SystemExit('engine not found: ' + str(engine))
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    output = args.work / f'partial-cache-{stamp}'
    spec = args.work / 'partial-spec.json'
    sources = source_pins(['run_memgen.py', 'contract.json'], compact=False)
    sources.append(pin(engine))
    sources.append(pin(expansion / 'manifest.json'))
    spec.write_text(json.dumps(dict(
        case_id=case, tool='memgen', input_kind='partial_model_cache', cpu=args.cpu, gpu=None,
        seconds=args.job_seconds, cache_directory=str(args.work / 'cache'),
        argv=[args.python, '-B', str(ADAPTER / 'run_memgen.py'), '--expanded', str(expansion),
              '--allow-partial-diagnostic', '--binary', str(engine), '--output', str(output)],
        sources=sources), indent=2) + '\n')
    print()
    print('== partial replay: what the expansion does cover ==')
    print('  spec ' + str(spec))
    if run_job(spec, args.work / 'runs' / f'{case}-partial', False, 0):
        print('  the partial replay failed; the expansion is untouched', file=sys.stderr)
        return None
    counters = output / 'model' / 'kernel_summary.csv'
    result = dict(complete_full_model=False,
                  target_launches=manifest['target_launches'],
                  packed_launches=manifest['packed_launches'],
                  unsupported_launches=manifest['unsupported_launches'],
                  engine=str(engine), kernel_summary=str(counters),
                  counters=kernel_counters(counters) if counters.is_file() else None)
    print(f"  coverage {result['packed_launches']} of {result['target_launches']} launches; "
          f"{result['unsupported_launches']} have no profile")
    if result['counters']:
        for name, value in result['counters'].items():
            print(f'  {name:26} {value}')
    print('  PARTIAL_MODEL_DIAGNOSTIC: these counters are not the case traffic, because')
    print('  the missing launches are not modelled and are not zero.')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_argument_group('what to collect')
    what.add_argument('--model', help='declared model key, see --list-cases')
    what.add_argument('--prefill-length', type=int, help='declared prefill length')
    what.add_argument('--decode-steps', type=int, help='declared decode steps')
    what.add_argument('--case', help='shorthand for the three above, e.g. qwen25_1p5b-p32-d2')
    what.add_argument('--list-cases', action='store_true',
                      help='print every declared case with its matrix and exit')
    where = parser.add_argument_group('where to run')
    where.add_argument('--work', type=Path, help='fresh output directory; must not exist')
    where.add_argument('--gpu-index', type=int, default=0,
                       help='index into the admitted GPU pool as printed by preflight.py (default 0)')
    where.add_argument('--gpu', help='admitted GPU UUID; overrides --gpu-index')
    where.add_argument('--cpu', type=int, default=8, help='one CPU id from 0..15 (default 8)')
    where.add_argument('--python', default='/home/xmu/sgl/bin/python',
                       help='interpreter that carries the SGLang stack')
    budget = parser.add_argument_group('budgets in seconds; a stage that exceeds its budget is killed')
    budget.add_argument('--census-seconds', type=int, default=1800,
                        help='census job ceiling; 0 means no limit (default 1800)')
    budget.add_argument('--sample-seconds', type=int, default=7200,
                        help='sampling ceiling; sample_pipeline.py requires 60..21600 of its own')
    budget.add_argument('--job-seconds', type=int, default=0,
                        help='job 2 wall-clock ceiling; 0 means run to completion (default 0)')
    budget.add_argument('--gpu-wait-seconds', type=int, default=GPU_ADMISSION_WAIT_SECONDS,
                        help='wait at most this long for a device that is busy at fresh locked '
                             'admission; 0 fails immediately (default 600)')
    parser.add_argument('--observer', type=Path, help='prebuilt observer.so; built into --work if omitted')
    parser.add_argument('--resume', action='store_true',
                        help='reuse an existing --work whose census passed the gate and start '
                             'from job 2; job 1 is the only stage that costs GPU time twice')
    parser.add_argument('--partial', action='store_true',
                        help='when the expansion does not cover the full model, replay what it '
                             'does cover and label the counters partial')
    parser.add_argument('--engine', type=Path,
                        help='frozen CPU engine for --partial; built into --work by default. The '
                             'cache stage of the normal path always builds and passes its own, '
                             'because run_memgen.py would otherwise use a machine-local binary '
                             'that this repository cannot hash')
    parser.add_argument('--model-uncovered', choices=('refuse', 'modeled'), default='refuse',
                        help='refuse stops when a class has no admitted template (default); '
                             'modeled completes the full model with an explicit numeric_modeled label')
    parser.add_argument('--model-policy', choices=MODEL_POLICIES, default='strict',
                        help='which sampled memory records the projection admits; strict '
                             '(default) refuses predicated global reads, the two ldg_source_predicate '
                             'policies admit them instead of losing a weight-streaming class to a refusal')
    parser.add_argument('--dry-run', action='store_true', help='write the specs and print the plan')
    args = parser.parse_args()

    if args.list_cases:
        models = workload.spec()['models']
        print(f"{'case id':24} {'workload':26} {'matrix':16} model")
        for name in sorted(declared_cases()):
            value = workload.contract(*name)
            workload_label = f'prefill {name[1]}, decode {name[2]}'
            print(f"{value['case_id']:24} {workload_label:26} {value['declared_matrix']:16} "
                  f"{models[name[0]].get('display', name[0])}")
        print()
        print('The model key is the shipped identifier; the last column is what it means.')
        print('Example: qwen25_1p5b is Qwen2.5-1.5B, and qwen25_1p5b-p128-d32 is that')
        print('model with 128 prefill tokens and 32 decode steps.')
        return 0

    if args.case:
        if any(value is not None for value in (args.model, args.prefill_length, args.decode_steps)):
            raise SystemExit('give either --case, or --model with --prefill-length and --decode-steps')
        match = re.fullmatch(r'(.+)-p(\d+)-d(\d+)', args.case)
        if not match:
            raise SystemExit('--case must look like <model>-p<prefill>-d<decode>, got ' + args.case)
        model, prefill, decode = match.group(1), int(match.group(2)), int(match.group(3))
    else:
        if None in (args.model, args.prefill_length, args.decode_steps):
            raise SystemExit('give --model, --prefill-length and --decode-steps, or --case, '
                             'or run --list-cases')
        model, prefill, decode = args.model, args.prefill_length, args.decode_steps
    try:
        contract = workload.contract(model, prefill, decode)
    except (ValueError, KeyError) as error:
        raise SystemExit(f'{error}\nrun --list-cases to see every declared case')

    table = gpu_table()
    if args.gpu:
        if args.gpu not in {uuid for _, uuid, _ in table}:
            raise SystemExit('GPU is not in the admitted pool of run_job.py: ' + args.gpu)
        index, gpu = next((i, u) for i, u, _ in table if u == args.gpu)
    else:
        if not 0 <= args.gpu_index < len(table):
            raise SystemExit(f'--gpu-index must be 0..{len(table) - 1} for this pool')
        index, gpu = table[args.gpu_index][0], table[args.gpu_index][1]
    if not 0 <= args.cpu < 16:
        raise SystemExit('--cpu must be 0..15')
    if args.census_seconds != 0 and not 1 <= args.census_seconds <= 86400:
        raise SystemExit('--census-seconds must be 0 (no limit) or 1..86400')
    if not 60 <= args.sample_seconds <= 21600:
        raise SystemExit('--sample-seconds must be 60..21600: sample_pipeline.py enforces its own '
                         'bounded runtime and cannot be asked to run unbounded')
    if args.job_seconds != 0 and not 1 <= args.job_seconds <= 86400:
        raise SystemExit('--job-seconds must be 0 (run to completion) or 1..86400')
    if not 0 <= args.gpu_wait_seconds <= 86400:
        raise SystemExit('--gpu-wait-seconds must be 0..86400')
    if not args.work:
        raise SystemExit('--work is required')
    if args.work.exists() and not args.resume:
        raise SystemExit('refusing existing work directory: ' + str(args.work) +
                         '\npass --resume to continue one whose census closed')
    if args.resume and not args.work.exists():
        raise SystemExit('--resume needs an existing --work: ' + str(args.work))
    if args.resume and args.dry_run:
        raise SystemExit('--resume and --dry-run do not combine; the specs already exist')
    # Canonical from here on: `SG_NVBIT_OUTPUT_ROOT` must equal its own realpath,
    # so a `--work` reached through a symlink would otherwise fail at observer
    # init. The receipt then names the real directory.
    args.work = args.work.resolve()
    if not args.dry_run and not Path(args.python).exists():
        raise SystemExit('interpreter not found: ' + args.python)
    journal = host = None
    args.gpu = gpu
    case = contract['case_id']

    print(f"case       {contract['case_id']}  (matrix {contract['declared_matrix']})")
    print(f"model      {contract['model_key']}  "
          f"{workload.spec()['models'][contract['model_key']].get('display', '')}")
    print(f"workload   prefill {contract['prefill_length']} tokens, "
          f"decode {contract['decode_steps']} steps, ids {contract['decode_input_ids']}")
    print(f"gpu        index {index} of 0..{len(table) - 1} -> it is {table[index][1]} "
          f"({table[index][2]})")
    for i, uuid, name in table:
        print(f"             [{i}] {uuid}  {name}")
    print(f"cpu        {args.cpu} of 0..15")
    print(f"work       {args.work}")
    census = 'no limit' if args.census_seconds == 0 else f'{args.census_seconds} s hard limit'
    print(f"budgets    census {census}; sample {args.sample_seconds} s hard limit; "
          f"replay {'to completion' if args.job_seconds == 0 else str(args.job_seconds) + ' s'}")
    print('           a stage that exceeds its budget is killed and its lease released;')
    print('           the replay has no wall-clock deadline at any layer')
    print()

    args.work.mkdir(parents=True, exist_ok=True)
    observer_root(args.work, case)
    # Resolved here so the reuse path can name it, but the build below has to know
    # whether the path was asked for: the default still has to be built.
    build_observer = args.observer is None
    if build_observer:
        args.observer = args.work / 'observer-build' / 'observer.so'

    print()
    if args.resume:
        print('== job 1: census reused from --work (--resume) ==')
        observer_finish, observer = reused_census(args.work, case, args.cpu, args.gpu)
        journal, host = census_process_names(observer, observer_finish)
        print(f"  closed {journal}: observer pid {observer['pid']}, "
              f"{observer['launch_before_count']} launches, "
              f"{observer['metadata_bytes_before_finish']} of "
              f"{observer['max_total_bytes']} metadata bytes")
        print('  the census journal and host receipt are the inputs job 2 reads')
    else:
        if build_observer:
            print('== build the metadata observer (no GPU) ==')
            if args.dry_run:
                print(f'  would build {args.observer} and pin its identity before job 1')
            else:
                import subprocess
                build = subprocess.run([sys.executable, '-B', str(SOURCES / 'observer' / 'build.py'),
                                        '--output', str(args.observer.parent)],
                                       capture_output=True, text=True)
                if build.returncode:
                    raise SystemExit('observer build failed:\n' + build.stdout + build.stderr)
                identity = tool_identity.report(args.observer)
                print(f"  artifact {identity['artifact_sha256'][:16]}  "
                      f"content {identity['content_sha256'][:16]}")
                print('  identity is per build; see docs/ENVIRONMENT.md section 4.4')
        args.observer = Path(args.observer).resolve()
        if not args.dry_run and not args.observer.is_file():
            raise SystemExit('observer not found: ' + str(args.observer))

        print()
        print('== job 1: census under the metadata observer (GPU) ==')
        spec1 = args.work / 'census-spec.json'
        spec1.write_text(json.dumps(census_spec(case, args.work, contract, args), indent=2) + '\n')
        print('  spec ' + str(spec1))
        if run_job(spec1, args.work / 'runs' / f'{case}-census', args.dry_run,
                   args.gpu_wait_seconds):
            return 1
        if not args.dry_run:
            observer_finish = only(observer_root(args.work, case), 'process-*/finish.json')
            host_finish = only(args.work / 'runs' / f'{case}-census' / 'host',
                               'process-*/finish.json')
            observer = verify_census(observer_finish, host_finish,
                                     args.work / 'runs' / f'{case}-census' / 'job-finish.json',
                                     case, args.cpu, args.gpu)
            journal, host = census_process_names(observer, observer_finish)
            print(f"  closed {journal}: observer pid {observer['pid']}, "
                  f"{observer['launch_before_count']} launches, "
                  f"{observer['metadata_bytes_before_finish']} of "
                  f"{observer['max_total_bytes']} metadata bytes")

    print()
    print('== job 2: sample, expand and cache replay ==')
    follow = args.work / 'runs' / f'{case}-collect' / 'followthrough'
    closed = decided_job_two(follow) if args.resume else None
    reused = False
    if closed is not None:
        sampled = reused_policy(follow)
        if sampled != args.model_policy:
            raise SystemExit(
                f'reused job 2 was sampled with --model-policy {sampled}, not '
                f'{args.model_policy}: the policy decides which sampled records the '
                'projection admits, and therefore what the reused expansion covers. '
                f'Move {follow} aside to sample again, or ask for the policy it was '
                'sampled with')
        expanded = reused_model_uncovered(follow)
        if expanded != args.model_uncovered:
            # The sample is reusable, but the expansion beside it was completed under
            # the other --model-uncovered. That flag decides whether a launch whose
            # class has no admitted template is refused or receives an explicit
            # numeric_modeled profile, so the expansion on disk answers a different
            # question and its coverage is not this run's coverage. Keep it as
            # evidence and expand the same sample again rather than report it.
            print(f'  reused job 2 was expanded with --model-uncovered {expanded}, not '
                  f'{args.model_uncovered}: expanding the same sample again')
        else:
            reused = True
            print(f"  reused from --work (--resume): {closed['status']}"
                  f'  [--model-policy {sampled}, --model-uncovered {expanded}]')
            print(f"  expansion {follow / 'expanded' / 'manifest.json'}")
    if not reused:
        if args.resume:
            stashed = stash_job_output(args.work, case)
            if stashed is not None:
                print(f'  previous job 2 output kept as runs/{stashed.name}')
        spec2 = args.work / 'collect-spec.json'
        if journal is None:
            journal, host = 'process-<pid>-<ticks>', 'process-<pid>'
        # Name the engine the cache stage must replay through. followthrough.py
        # builds it from this repository's source when the stage actually runs, so
        # a run that never reaches the replay does not pay for the compile, and the
        # replay never falls back to run_memgen.py's machine-local default.
        spec2.write_text(json.dumps(collect_spec(case, args.work, args, journal, host,
                                                args.work / 'engine' / 'hbserve'),
                                   indent=2) + '\n')
        print('  spec ' + str(spec2))
        if run_job(spec2, args.work / 'runs' / f'{case}-collect', args.dry_run,
                   args.gpu_wait_seconds):
            # A stop the chain decided is reported with its coverage below; only a
            # job that left no finish.json is a hard failure here.
            if not stopped_at_the_gate(follow):
                return 1
            print('  job 2 stopped at its own gate; the receipt records the coverage')
        follow = args.work / 'runs' / f'{case}-collect' / 'followthrough'

    if args.dry_run:
        print()
        print('DRY_RUN_PLAN_ONLY: nothing was executed and no GPU time was used')
        print('  collect-spec.json names process-<pid>-<ticks> and process-<pid> as')
        print('  placeholders; job 1 resolves both from its own receipts before job 2')
        print('  is written.')
        return 0

    follow = args.work / 'runs' / f'{case}-collect' / 'followthrough'
    finish = json.loads((follow / 'finish.json').read_text())
    replay = follow / 'cache' / 'model' / 'kernel_summary.csv'
    expansion = json.loads((follow / 'expanded' / 'manifest.json').read_text())
    # Recorded, not gated: an expansion written before modeled completion existed
    # simply has no coverage fields, and status stays the authority below.
    coverage = {k: expansion.get(k) for k in ('target_launches', 'packed_launches', 'unsupported_launches',
                                             'exact_launches', 'modeled_launches', 'modeled_fraction',
                                             'modeled_by_cause', 'fully_exact', 'modeled_completion',
                                             'exact_cross_layer_identity_claimed')}
    partial = None
    if finish['status'] == 'STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC':
        manifest = expansion
        if not args.partial:
            print()
            print(f"  the expansion covers {manifest['packed_launches']} of "
                  f"{manifest['target_launches']} launches, so no counters were produced.")
            print('  `--partial` replays the covered part and labels the result partial.')
            print('  `--model-uncovered modeled` completes the model instead, with the '
                  'modeled share recorded in the receipt.')
            print('  A refusal caused by the projection policy is not a missing sample: '
                  '`--model-policy estimate_ldg_source_predicate` admits the predicated '
                  'global reads that strict refuses, which is how a weight-streaming '
                  'GEMM gets fitted instead of modeled. The `validated_` spelling of '
                  'that policy additionally re-checks every admitted record against an '
                  'independently audited source scope, and needs '
                  '`source-qualification.json` from that audit beside the upstream '
                  'sources; this tree does not carry it, so only `estimate_` runs here.')
        else:
            partial = run_partial_replay(case, args, follow)
            if partial is None:
                return 1
    # The replay's own hardware identity, propagated through the job-2 stage receipt.
    # It decides whether these counters may be compared with NCU at all, so the
    # receipt names it instead of leaving the reader to infer it from prose. A run
    # that never replayed has no identity and records none.
    hardware = {k: finish.get(k) for k in (
        'hardware_config', 'config_sha256', 'hardware_schema', 'hardware_accuracy_status',
        'cache_policy', 'allow_full_NCU_accuracy_comparison', 'input_scope')}
    if all(value is None for value in hardware.values()):
        hardware = None
    claim_boundary = ('Collected counters for one declared case. Not an accuracy '
                      'admission: that needs an independent three-repeat NCU reference '
                      'for the same ranges, recorded in validation/.')
    if hardware and hardware.get('allow_full_NCU_accuracy_comparison') is not True:
        claim_boundary += (f" This replay ran {hardware.get('hardware_config')}"
                           f" (schema {hardware.get('hardware_schema')}, calibration "
                           f"status {hardware.get('hardware_accuracy_status')}, policy "
                           f"{hardware.get('cache_policy')}), which does not permit a "
                           'full NCU accuracy comparison, so these counters are not '
                           'evidence of hardware agreement.')
    receipt = dict(schema='SG_CASE_COLLECTION_V1', case_id=case,
                   declared_matrix=contract['declared_matrix'],
                   status=finish['status'], stages=finish['stages'],
                   expansion_coverage=coverage,
                   hardware=hardware,
                   work=str(args.work), artifacts=dict(
                       census_observer_finish=str(observer_root(args.work, case)
                                                  / journal / 'finish.json'),
                       census_host_finish=str(args.work / 'runs' / f'{case}-census' / 'host'
                                              / host / 'finish.json'),
                       sample_plan=str(follow / 'plan' / 'sample-plan.json'),
                       packed_profiles=str(follow / 'sample' / 'profiles' / 'profiles.index.jsonl'),
                       expanded_manifest=str(follow / 'expanded' / 'manifest.json'),
                       kernel_summary=str(replay) if replay.is_file() else None,
                       cache_observation=str(follow / 'cache' / 'model' / 'cache_observation.json')
                       if (follow / 'cache' / 'model' / 'cache_observation.json').is_file() else None),
                   raw_trace_persisted=False, full_native_address_coverage=False,
                   hardware_accuracy_accepted=False,
                   partial_diagnostic=partial,
                   claim_boundary=claim_boundary)
    (args.work / 'collect-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print()
    print(json.dumps({k: receipt[k] for k in ('case_id', 'declared_matrix', 'status',
                                              'hardware_accuracy_accepted')}, indent=2))
    if hardware:
        print(f"  hardware                 {hardware.get('hardware_config')}"
              f" ({hardware.get('hardware_schema')}, {hardware.get('cache_policy')}, "
              f"NCU comparison permitted: "
              f"{hardware.get('allow_full_NCU_accuracy_comparison')})")
    print(f"  launches                 {coverage['exact_launches']} exact + "
          f"{coverage['modeled_launches']} modeled of {coverage['target_launches']} "
          f"({coverage['modeled_fraction']:.1%} modeled)" if coverage['modeled_fraction'] is not None else
          f"  launches                 {coverage['packed_launches']} packed of "
          f"{coverage['target_launches']} (this expansion declares no modeled completion)")
    for name, value in receipt['artifacts'].items():
        print(f"  {name:24} {value if value else 'not produced'}")
    print(f"  receipt                  {args.work / 'collect-receipt.json'}")
    # Exit 2 even when --partial produced counters: the documented meaning of 0 is
    # "every stage receipt closed", and a partial model never satisfies that.
    return 0 if receipt['status'].startswith('PASS_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
