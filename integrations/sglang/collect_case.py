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
                             'SG_NVBIT_OUTPUT_ROOT': str(work / 'observers' / f'{case}-census'),
                             'SG_NVBIT_MAX_BYTES': str(256 << 20),
                             'ACK_CTX_INIT_LIMITATION': '1'},
                sources=pins)


def collect_spec(case: str, work: Path, args) -> dict:
    follow = work / 'runs' / f'{case}-collect' / 'followthrough'
    return dict(case_id=case, tool='memgen', input_kind='sample_and_cache',
                cpu=args.cpu, gpu=args.gpu, seconds=args.sample_seconds + args.memgen_seconds,
                cache_directory=str(work / 'cache'),
                argv=[args.python, '-B', str(ADAPTER / 'followthrough.py'),
                      '--journal', str(work / 'observers' / f'{case}-census' / args.journal_process),
                      '--host-finish', str(work / 'runs' / f'{case}-census' / 'host'
                                           / args.journal_process / 'finish.json'),
                      '--sources', str(SOURCES),
                      '--output', str(follow),
                      '--stop-after', 'memgen',
                      '--python', args.python,
                      '--sample-seconds', str(args.sample_seconds),
                      '--memgen-seconds', str(args.memgen_seconds)],
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
    parser.add_argument('--case', required=True, help='declared case id, e.g. qwen25_1p5b-p32-d2')
    parser.add_argument('--gpu', required=True, help='one UUID from the run_job.py pool')
    parser.add_argument('--work', type=Path, required=True, help='fresh output directory')
    parser.add_argument('--cpu', type=int, default=8, help='single CPU id from 0..15')
    parser.add_argument('--python', default='/home/xmu/sgl/bin/python',
                        help='interpreter that carries the SGLang stack')
    parser.add_argument('--observer', type=Path, help='prebuilt observer.so; built into --work if omitted')
    parser.add_argument('--census-seconds', type=int, default=1800)
    parser.add_argument('--sample-seconds', type=int, default=7200)
    parser.add_argument('--memgen-seconds', type=int, default=21600)
    parser.add_argument('--dry-run', action='store_true', help='write the specs and print the plan')
    args = parser.parse_args()

    match = re.fullmatch(r'(.+)-p(\d+)-d(\d+)', args.case)
    if not match:
        raise SystemExit('case must look like <model>-p<prefill>-d<decode>, got ' + args.case)
    model, prefill, decode = match.group(1), int(match.group(2)), int(match.group(3))
    contract = workload.contract(model, prefill, decode)
    if contract['case_id'] != args.case:
        raise SystemExit(f"case ids differ: {args.case} versus {contract['case_id']}")
    if args.gpu not in gpu_pool():
        raise SystemExit('GPU is not in the admitted pool of run_job.py: ' + args.gpu)
    if args.work.exists():
        raise SystemExit('refusing existing work directory: ' + str(args.work))
    if not args.dry_run and not Path(args.python).exists():
        raise SystemExit('interpreter not found: ' + args.python)
    args.journal_process = None

    print(f"case     {args.case}  ({contract['declared_matrix']})")
    print(f"gpu      {args.gpu}   cpu {args.cpu}")
    print(f"work     {args.work}")
    print(f"prompt   {contract['prefill_length']} tokens, decode {contract['decode_steps']} steps "
          f"ids {contract['decode_input_ids']}")
    print(f"model    {contract['model']}")
    print()

    args.work.mkdir(parents=True)
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
    spec1.write_text(json.dumps(census_spec(args.case, args.work, contract, args), indent=2) + '\n')
    print('  spec ' + str(spec1))
    if run_job(spec1, args.work / 'runs' / f'{args.case}-census', args.dry_run):
        return 1
    if not args.dry_run:
        observer_finish = only(args.work / 'observers' / f'{args.case}-census', 'process-*/finish.json')
        host_finish = only(args.work / 'runs' / f'{args.case}-census' / 'host', 'process-*/finish.json')
        status_of(observer_finish, 'PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE')
        status_of(host_finish, 'PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE')
        status_of(args.work / 'runs' / f'{args.case}-census' / 'job-finish.json', 'PASS_PROCESS_ONLY')
        if observer_finish.parent.name != host_finish.parent.name:
            raise SystemExit('observer and host did not close in one process')
        args.journal_process = observer_finish.parent.name
        print(f"  closed {args.journal_process}: census receipts verified")

    print()
    print('== job 2: sample, expand and cache replay ==')
    spec2 = args.work / 'collect-spec.json'
    if args.journal_process is None:
        args.journal_process = 'process-<pid>'
    spec2.write_text(json.dumps(collect_spec(args.case, args.work, args), indent=2) + '\n')
    print('  spec ' + str(spec2))
    if run_job(spec2, args.work / 'runs' / f'{args.case}-collect', args.dry_run):
        return 1

    if args.dry_run:
        print()
        print('DRY_RUN_PLAN_ONLY: nothing was executed and no GPU time was used')
        print('  collect-spec.json names process-<pid> as a placeholder; job 1 resolves the')
        print('  real census process directory before job 2 is written.')
        return 0

    follow = args.work / 'runs' / f'{args.case}-collect' / 'followthrough'
    finish = json.loads((follow / 'finish.json').read_text())
    replay = follow / 'cache' / 'model' / 'kernel_summary.csv'
    receipt = dict(schema='SG_CASE_COLLECTION_V1', case_id=args.case,
                   declared_matrix=contract['declared_matrix'],
                   status=finish['status'], stages=finish['stages'],
                   work=str(args.work), artifacts=dict(
                       census_observer_finish=str(args.work / 'observers' / f'{args.case}-census'
                                                  / args.journal_process / 'finish.json'),
                       census_host_finish=str(args.work / 'runs' / f'{args.case}-census' / 'host'
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
