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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADAPTER = HERE / 'memgen-adapter'
SOURCES = HERE / 'compact-sources'
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ADAPTER))
import matrix_workload as workload  # noqa: E402
import tool_identity  # noqa: E402


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


def collect_spec(case: str, work: Path, args) -> dict:
    follow = work / 'runs' / f'{case}-collect' / 'followthrough'
    return dict(case_id=case, tool='memgen', input_kind='sample_and_cache',
                cpu=args.cpu, gpu=args.gpu, seconds=args.job_seconds,
                cache_directory=str(work / 'cache'),
                argv=[args.python, '-B', str(ADAPTER / 'followthrough.py'),
                      '--journal', str(observer_root(work, case) / args.journal_process),
                      '--host-finish', str(work / 'runs' / f'{case}-census' / 'host'
                                           / args.journal_process / 'finish.json'),
                      '--sources', str(SOURCES),
                      '--output', str(follow),
                      '--stop-after', 'memgen',
                      '--python', args.python,
                      '--sample-seconds', str(args.sample_seconds)],
                sources=source_pins(['followthrough.py', 'make_sample_plan.py', 'sample_pipeline.py',
                                     'expand_profiles.py', 'run_memgen.py', 'profile_cache.py',
                                     'contract.json', 'matrix_workload.py'], compact=True))


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


def run_job(spec_path: Path, output: Path, dry: bool) -> int:
    argv = [sys.executable, '-B', str(HERE / 'run_job.py'),
            '--spec', str(spec_path), '--output', str(output)]
    print('  ' + ' '.join(argv))
    if dry:
        return 0
    import subprocess
    return subprocess.call(argv + ['--execute'])


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
    parser.add_argument('--observer', type=Path, help='prebuilt observer.so; built into --work if omitted')
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
    if not args.work:
        raise SystemExit('--work is required')
    if args.work.exists():
        raise SystemExit('refusing existing work directory: ' + str(args.work))
    # Canonical from here on: `SG_NVBIT_OUTPUT_ROOT` must equal its own realpath,
    # so a `--work` reached through a symlink would otherwise fail at observer
    # init. The receipt then names the real directory.
    args.work = args.work.resolve()
    if not args.dry_run and not Path(args.python).exists():
        raise SystemExit('interpreter not found: ' + args.python)
    args.journal_process = None
    args.gpu = gpu
    case = contract['case_id']

    print(f"case       {contract['case_id']}  (matrix {contract['declared_matrix']})")
    print(f"model      {contract['model_key']}  "
          f"{workload.spec()['models'][contract['model_key']].get('display', '')}")
    print(f"workload   prefill {contract['prefill_length']} tokens, "
          f"decode {contract['decode_steps']} steps, ids {contract['decode_input_ids']}")
    print(f"gpu        index {index} of {len(table) - 1} -> it is {table[index][1]} "
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

    args.work.mkdir(parents=True)
    observer_root(args.work, case)
    if args.observer is None:
        args.observer = args.work / 'observer-build' / 'observer.so'
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
            print(f"  artifact {identity['artifact_sha256'][:16]}  content {identity['content_sha256'][:16]}")
            print('  identity is per build; see docs/ENVIRONMENT.md section 4.4')
    args.observer = Path(args.observer).resolve()
    if not args.dry_run and not args.observer.is_file():
        raise SystemExit('observer not found: ' + str(args.observer))

    print()
    print('== job 1: census under the metadata observer (GPU) ==')
    spec1 = args.work / 'census-spec.json'
    spec1.write_text(json.dumps(census_spec(case, args.work, contract, args), indent=2) + '\n')
    print('  spec ' + str(spec1))
    if run_job(spec1, args.work / 'runs' / f'{case}-census', args.dry_run):
        return 1
    if not args.dry_run:
        observer_finish = only(observer_root(args.work, case), 'process-*/finish.json')
        host_finish = only(args.work / 'runs' / f'{case}-census' / 'host', 'process-*/finish.json')
        status_of(observer_finish, 'PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE')
        status_of(host_finish, 'PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE')
        status_of(args.work / 'runs' / f'{case}-census' / 'job-finish.json', 'PASS_PROCESS_ONLY')
        if observer_finish.parent.name != host_finish.parent.name:
            raise SystemExit('observer and host did not close in one process')
        args.journal_process = observer_finish.parent.name
        print(f"  closed {args.journal_process}: census receipts verified")

    print()
    print('== job 2: sample, expand and cache replay ==')
    spec2 = args.work / 'collect-spec.json'
    if args.journal_process is None:
        args.journal_process = 'process-<pid>'
    spec2.write_text(json.dumps(collect_spec(case, args.work, args), indent=2) + '\n')
    print('  spec ' + str(spec2))
    if run_job(spec2, args.work / 'runs' / f'{case}-collect', args.dry_run):
        return 1

    if args.dry_run:
        print()
        print('DRY_RUN_PLAN_ONLY: nothing was executed and no GPU time was used')
        print('  collect-spec.json names process-<pid> as a placeholder; job 1 resolves the')
        print('  real census process directory before job 2 is written.')
        return 0

    follow = args.work / 'runs' / f'{case}-collect' / 'followthrough'
    finish = json.loads((follow / 'finish.json').read_text())
    replay = follow / 'cache' / 'model' / 'kernel_summary.csv'
    receipt = dict(schema='SG_CASE_COLLECTION_V1', case_id=case,
                   declared_matrix=contract['declared_matrix'],
                   status=finish['status'], stages=finish['stages'],
                   work=str(args.work), artifacts=dict(
                       census_observer_finish=str(observer_root(args.work, case)
                                                  / args.journal_process / 'finish.json'),
                       census_host_finish=str(args.work / 'runs' / f'{case}-census' / 'host'
                                              / args.journal_process / 'finish.json'),
                       sample_plan=str(follow / 'plan' / 'sample-plan.json'),
                       packed_profiles=str(follow / 'sample' / 'profiles' / 'profiles.index.jsonl'),
                       expanded_manifest=str(follow / 'expanded' / 'manifest.json'),
                       kernel_summary=str(replay) if replay.is_file() else None,
                       cache_observation=str(follow / 'cache' / 'model' / 'cache_observation.json')
                       if (follow / 'cache' / 'model' / 'cache_observation.json').is_file() else None),
                   raw_trace_persisted=False, full_native_address_coverage=False,
                   hardware_accuracy_accepted=False,
                   claim_boundary='Collected counters for one declared case. Not an accuracy '
                                  'admission: that needs an independent three-repeat NCU reference '
                                  'for the same ranges, recorded in validation/.')
    (args.work / 'collect-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print()
    print(json.dumps({k: receipt[k] for k in ('case_id', 'declared_matrix', 'status',
                                              'hardware_accuracy_accepted')}, indent=2))
    for name, value in receipt['artifacts'].items():
        print(f"  {name:24} {value if value else 'not produced'}")
    print(f"  receipt                  {args.work / 'collect-receipt.json'}")
    return 0 if receipt['status'].startswith('PASS_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
