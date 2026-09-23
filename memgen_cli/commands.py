"""Verb implementations.

Each verb is a thin, transparent dispatcher: it prints the underlying command
before running it, so the one entry point teaches the pipeline instead of hiding
it. Nothing is reimplemented here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ._repo import (ADAPTER, BOOTSTRAP, COLLECT, PREFLIGHT, REPLAY, ROOT, SCRIPTS, SGLANG,
                    TESTS, timestamped)


def _run(argv: list[str], echo: bool = True) -> int:
    if echo:
        print('+ ' + ' '.join(argv), flush=True)
    return subprocess.call(argv)


def _python() -> str:
    return sys.executable


def _forward(argv: list[str], script: Path) -> int:
    return _run([_python(), '-B', str(script), *argv])


def _segments(argv: list[str]) -> set[str]:
    return {item.split('=', 1)[0] for item in argv}


def _summarize(lines: list[str]) -> str:
    """One line for `--quiet`. JSON receipts collapse to their status fields."""
    if not lines:
        return ''
    text = '\n'.join(lines)
    try:
        payload = json.loads(text)
    except ValueError:
        return lines[-1]
    keys = ('status', 'hardware_accuracy_accepted')
    return json.dumps({key: payload[key] for key in keys if key in payload})


def check(argv: list[str]) -> int:
    """Host dependencies and archive pins. No GPU is touched."""
    parser = argparse.ArgumentParser(prog='memgen check',
                                     description='Verify the host and the frozen archive.')
    parser.add_argument('--gpus', choices=('pool', 'any', 'none'), default='pool',
                        help='"none" skips the GPU checks on a CPU-only host')
    parser.add_argument('--quiet', action='store_true', help='print only the final statuses')
    parser.add_argument('--r4', action='store_true',
                        help='additionally run the optional r4 cache-core regression')
    args = parser.parse_args(argv)

    steps = [
        ('host and dependencies', [_python(), '-B', str(PREFLIGHT), '--gpus', args.gpus]),
        ('pinned deployment files', [_python(), '-B', str(BOOTSTRAP), '--check']),
        ('archive manifests and pins', [_python(), '-B', str(SCRIPTS / 'verify_archive.py')]),
    ]
    if args.r4:
        steps.append(('r4 cache core regression',
                      [_python(), '-B', str(SCRIPTS / 'test_cache_core.py'),
                       '--output', str(timestamped('cache-core'))]))
    failures = []
    for label, command in steps:
        print(f'\n== {label} ==')
        capture = subprocess.run(command, capture_output=True, text=True)
        output = (capture.stdout + capture.stderr).strip().splitlines()
        if args.quiet:
            print(_summarize(output))
        else:
            print('\n'.join(output))
        if capture.returncode:
            failures.append(label)
    print()
    if failures:
        print('FAIL ' + ', '.join(failures), file=sys.stderr)
        return 1
    print('PASS host and archive checks. Run `memgen test` and `memgen smoke` next.')
    return 0


def cases(argv: list[str]) -> int:
    """List the declared workload points."""
    return _forward(['--list-cases', *argv], COLLECT)


def gpus(argv: list[str]) -> int:
    """The admitted GPUs and the index `--gpu-index` accepts."""
    parser = argparse.ArgumentParser(prog='memgen gpus',
                                     description='List the admitted GPUs by index.')
    parser.parse_args(argv)
    import collect_case
    table = collect_case.gpu_table()
    if not table:
        print('no admitted GPU found in run_job.py', file=sys.stderr)
        return 1
    print(f"{'index':>5}  {'uuid':40} name")
    for index, uuid, name in table:
        print(f'{index:>5}  {uuid:40} {name}')
    print('\nPass one with --gpu-index, or --gpu UUID for scripted callers.')
    return 0


def _with_default_work(argv: list[str], prefix: str) -> list[str]:
    if '--work' in _segments(argv):
        return argv
    work = timestamped(prefix)
    print(f'no --work given, using a fresh directory: {work}')
    return [*argv, '--work', str(work)]


def plan(argv: list[str]) -> int:
    """Write the job specs for one case and print the plan. Runs nothing."""
    forwarded = _with_default_work(argv, 'plan')
    if '--dry-run' not in forwarded:
        forwarded = [*forwarded, '--dry-run']
    return _forward(forwarded, COLLECT)


def collect(argv: list[str]) -> int:
    """Collect one declared case: census, sample, expand and replay."""
    forwarded = _with_default_work(argv, 'collect')
    if '--dry-run' in forwarded:
        print('--dry-run passed to `memgen collect`; use `memgen plan` for that', file=sys.stderr)
        return 2
    return _forward(forwarded, COLLECT)


def replay(argv: list[str]) -> int:
    """Run an admitted profile stream through the cache model."""
    return _forward(argv, REPLAY)


def smoke(argv: list[str]) -> int:
    """Replay the frozen engine's synthetic fixture on the CPU."""
    parser = argparse.ArgumentParser(prog='memgen smoke',
                                     description='Bounded CPU smoke of the frozen engine.')
    parser.add_argument('--out', type=Path, help='fresh output directory')
    args = parser.parse_args(argv)
    target = args.out or timestamped('smoke')
    return _run(['bash', str(SCRIPTS / 'run_cpu_smoke.sh'), str(target)])


def test(argv: list[str]) -> int:
    """Portable regression tests. No GPU, no model, no NVBit."""
    parser = argparse.ArgumentParser(prog='memgen test',
                                     description='Portable regression tests.')
    parser.parse_args(argv)
    modules = [TESTS / 'cli/test_cli.py',
               TESTS / 'sglang/test_declared_cases.py',
               TESTS / 'sampling/test_profile_census.py']
    failures = []
    for module in modules:
        print(f'\n== {module.relative_to(ROOT)} ==')
        if _run([_python(), '-B', str(module)]) != 0:
            failures.append(str(module.relative_to(ROOT)))
    print()
    if failures:
        print('FAIL ' + ', '.join(failures), file=sys.stderr)
        return 1
    print('PASS portable regression tests.')
    return 0
