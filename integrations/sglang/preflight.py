#!/usr/bin/env python3
"""Check the machine-local prerequisites the frozen integration silently assumes.

Nothing here proves accuracy. This script only reports whether the host can run
the archived workflow at all, which of the documented workload matrices the
current adapter contract can actually build, and which pinned files are absent.

Every path checked here is a value that some archived script uses as a default.
Running this on a different host is the fastest way to learn what has to be
supplied explicitly.

Exit status is 0 when every required host check passes. GPU, NVBit and CUDA are
required because the sampling and NCU steps cannot run without them; the frozen
CPU smoke and cache regression do not need any of them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Importing a sibling module would otherwise write __pycache__/*.pyc into the
# tree, which scripts/verify_archive.py correctly rejects as a forbidden
# artifact. Guard before the first local import so a plain `python3
# preflight.py` cannot dirty the archive.
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import bootstrap_vendor  # noqa: E402  (same directory, pinned-manifest checker)

NVCC = Path('/usr/local/cuda-12.8/bin/nvcc')
NCU = Path('/usr/local/cuda-12.8/bin/ncu')
NVBIT = Path('/home/xmu/nvidiagds/simulators/hyfiss/tracing-tool/nvbit')
FROZEN = Path('/home/xmu/nvidiagds/codex-runs/memgen-paper-ada-v1-20260916-01a08d87-r1')
CONTRACT = HERE / 'memgen-adapter/contract.json'
ADAPTER = HERE / 'memgen-adapter'
OK = ('OK', 'OPTIONAL', 'SKIPPED')


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section: str, name: str, status: str, detail: str = '', required: bool = True) -> None:
        if status != 'OK' and not required:
            status = 'SKIPPED' if status == 'MISSING' else status
        self.rows.append(dict(section=section, name=name, status=status, detail=detail,
                              required=required))

    def failures(self) -> list[dict]:
        return [row for row in self.rows if row['required'] and row['status'] not in OK]

    def render(self) -> str:
        width = max(len(row['name']) for row in self.rows) if self.rows else 0
        lines, section = [], None
        for row in self.rows:
            if row['section'] != section:
                section = row['section']
                lines.append('')
                lines.append(f'== {section} ==')
            lines.append(f"  {row['status']:<9} {row['name']:<{width}}  {row['detail']}".rstrip())
        return '\n'.join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_tools(report: Report) -> None:
    for name in ('mpic++', 'g++', 'nvidia-smi'):
        found = shutil.which(name)
        report.add('TOOLCHAIN', name, 'OK' if found else 'MISSING', found or 'not on PATH')
    report.add('TOOLCHAIN', 'python3', 'OK', sys.executable)
    for binary in (NVCC, NCU):
        report.add('TOOLCHAIN', binary.name, 'OK' if binary.is_file() else 'MISSING', str(binary))


def check_nvbit(report: Report) -> None:
    for name in ('libnvbit.a', 'nvbit.h', 'nvbit_tool.h'):
        path = NVBIT / name
        report.add('NVBIT', name, 'OK' if path.is_file() else 'MISSING', str(path))


def check_gpu(report: Report, gpus: str) -> None:
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001  (report, never raise)
        report.add('GPU', 'nvidia-smi query', 'MISSING', repr(error))
        return
    present = {uuid: name for uuid, name in
               (line.rsplit(',', 1) for line in out.strip().splitlines())}
    pool = sorted(set(re.findall(r'GPU-[0-9a-f-]{36}', (HERE / 'run_job.py').read_text())))
    if not pool:
        report.add('GPU', 'GPU_POOL', 'MISSING', 'no UUID found in run_job.py')
    for uuid in pool:
        name = present.get(uuid, 'absent from nvidia-smi')
        report.add('GPU', uuid, 'OK' if uuid in present else 'MISSING', name,
                   required=gpus == 'pool')
    report.add('GPU', 'admitted devices', 'OK', f'{len(present)} visible; pool of {len(pool)}',
               required=False)


def check_models(report: Report) -> None:
    try:
        models = json.loads(CONTRACT.read_text())['models']
    except Exception as error:  # noqa: BLE001
        report.add('MODELS', 'contract.json', 'MISSING', repr(error))
        return
    for key, row in models.items():
        path = Path(row['path'])
        report.add('MODELS', f'{key} path', 'OK' if path.is_dir() else 'MISSING', str(path))
        config = path / 'config.json'
        if config.is_file():
            actual = sha256(config)
            report.add('MODELS', f'{key} config.json',
                       'OK' if actual == row['config_sha256'] else 'DRIFT', actual)
        for name in row.get('weight_sha256', {}):
            weight = path / name
            report.add('MODELS', f'{key}/{name}', 'OK' if weight.is_file() else 'MISSING',
                       f'{weight.stat().st_size} bytes' if weight.is_file() else str(weight),
                       required=False)


def check_pins(report: Report) -> None:
    manifest = ROOT / 'integrations/sglang/package.json'
    accepted = ('PRESENT_IDENTICAL', 'PRESENT_REVISED')
    covered = tuple(row['relative'] for row in json.loads(manifest.read_text())['files'])
    for row in bootstrap_vendor.classify(manifest, HERE):
        if row['status'] not in accepted:
            detail = f"{row['status']}; run integrations/sglang/bootstrap_vendor.py"
        elif row['status'] == 'PRESENT_REVISED':
            detail = f"reviewed revision: {row.get('revision')}"
        else:
            detail = 'byte-identical to package.json pin'
        report.add('VENDOR_PINS', row['relative'], 'OK' if row['status'] in accepted else 'MISSING',
                   detail)
    for row in bootstrap_vendor.revisions_check(HERE, covered):
        report.add('REVISIONS', row['relative'], 'OK' if row['status'] == 'OK' else 'DRIFT',
                   'matches the recorded current pin, all copies identical' if row['status'] == 'OK'
                   else row['actual'])
    deployment = json.loads((ADAPTER / 'deployment-files.json').read_text())['files']
    for row in deployment:
        path = ADAPTER / row['name']
        if not path.is_file():
            report.add('ADAPTER_PINS', row['name'], 'MISSING', str(path))
        else:
            actual = sha256(path)
            report.add('ADAPTER_PINS', row['name'],
                       'OK' if actual == row['sha256'] else 'DRIFT', actual)


def check_frozen_roots(report: Report, binary: Path | None) -> None:
    report.add('ENGINE', 'frozen root', 'OK' if FROZEN.is_dir() else 'MISSING', str(FROZEN),
               required=False)
    if binary is not None:
        report.add('ENGINE', 'requested binary', 'OK' if binary.is_file() else 'MISSING',
                   str(binary))
    else:
        source = ROOT / 'release/source/tools/hbserve_profile_stream_cache_semantic_r17.cpp'
        report.add('ENGINE', 'engine source', 'OK' if source.is_file() else 'MISSING',
                   str(source.relative_to(ROOT)))
        report.add('ENGINE', 'no prebuilt binary', 'OK',
                   'supply --binary or build per docs/ENVIRONMENT.md section 4.1',
                   required=False)


def check_runtime(report: Report, python: Path) -> None:
    if not python.exists():
        report.add('RUNTIME', str(python), 'MISSING', '--sglang-python', required=False)
        return
    try:
        packages = json.loads(CONTRACT.read_text())['packages']
    except Exception as error:  # noqa: BLE001
        report.add('RUNTIME', 'contract packages', 'MISSING', repr(error), required=False)
        return
    code = ('import importlib.metadata, json, sys;'
            'print(json.dumps({n: importlib.metadata.version(n) for n in json.loads(sys.argv[1])}))')
    got = subprocess.run([str(python), '-c', code, json.dumps(sorted(packages))],
                         capture_output=True, text=True)
    if got.returncode:
        report.add('RUNTIME', 'package identity', 'MISSING', got.stderr.strip()[:120],
                   required=False)
        return
    actual = json.loads(got.stdout)
    for name, version in packages.items():
        report.add('RUNTIME', name, 'OK' if actual.get(name) == version else 'DRIFT',
                   f"expected {version}, found {actual.get(name)}", required=False)


def declared_cases() -> tuple[dict, dict]:
    """Reuse the adapter's own derivation so preflight cannot disagree with it."""
    sys.path.insert(0, str(ADAPTER))
    import matrix_workload  # noqa: PLC0415  (only importable from the adapter directory)
    contract = matrix_workload.spec()
    return contract, matrix_workload.declared_cases(contract)


def matrices() -> list[dict]:
    contract, declared = declared_cases()
    models = sorted(contract['models'])
    basic = contract.get('basic_admission') or {}
    # The documented matrices are prose, so verify the sentences that carry them
    # instead of trusting a copied constant. Drift here means the doc moved.
    doc = ROOT / 'docs/BRANCH_AND_ACCEPTANCE_CONTRACT.md'
    text = doc.read_text() if doc.is_file() else ''
    admission_trace = 'The basic admission workload is P32D2.'
    scale_trace = '`2, 4, 8, 16, 32`'
    rows = [
        dict(matrix='implemented scale series', source=f'{CONTRACT.name} prefills/decodes',
             prefills=contract['prefills'], decodes=contract['decodes'], models=len(models),
             cases=len(declared['scale_series']),
             note='every point needs its own sample, packed profile and NCU reference'),
        dict(matrix='implemented basic admission point',
             source=f"{CONTRACT.name} basic_admission + docs section 3",
             prefills=basic.get('prefills', []), decodes=basic.get('decodes', []),
             models=len(basic.get('models', [])), cases=len(declared['basic_admission']),
             note=('PRODUCIBLE from this snapshot. Producing it is not an accuracy result and '
                   'not an admission decision; P32D2 stays BLOCKED in '
                   'validation/p32d2_branch_status.csv until its own profile and NCU reference exist')
                  + ('' if admission_trace in text else ' [DOC TRACE NOT FOUND]')),
        dict(matrix='documented scale series',
             source='docs/BRANCH_AND_ACCEPTANCE_CONTRACT.md section 3',
             prefills=contract['prefills'], decodes=[2, 4, 8, 16, 32], models=len(models), cases=0,
             note=('partially producible: D=32 comes from the scale series and D=2 only from the '
                   'admission point; D=4, D=8 and D=16 remain undeclared')
                  + ('' if scale_trace in text else ' [DOC TRACE NOT FOUND]')),
    ]
    return rows


def render_matrices(rows: list[dict]) -> str:
    lines = ['', '== MATRICES ==']
    for row in rows:
        lines.append(f"  {row['matrix']}")
        lines.append(f"    source    {row['source']}")
        lines.append(f"    prefills  {row['prefills']}")
        lines.append(f"    decodes   {row['decodes']}")
        lines.append(f"    cases     {row['cases']}   {row['note']}")
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sglang-python', type=Path, default=Path('/home/xmu/sgl/bin/python'),
                        help='interpreter that carries the SGLang/PyTorch stack')
    parser.add_argument('--binary', type=Path, help='hbserve binary to require')
    parser.add_argument('--gpus', choices=('pool', 'any', 'none'), default='pool',
                        help='pool: require every GPU_POOL device; none: skip GPU entirely')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()

    report = Report()
    check_tools(report)
    check_nvbit(report)
    if args.gpus != 'none':
        check_gpu(report, args.gpus)
    check_models(report)
    check_pins(report)
    check_frozen_roots(report, args.binary)
    check_runtime(report, args.sglang_python)
    try:
        rows = matrices()
    except Exception as error:  # noqa: BLE001  (report, never raise)
        report.add('MATRICES', 'contract derivation', 'MISSING', repr(error))
        rows = []
    failures = report.failures()

    if args.json:
        print(json.dumps(dict(schema='SG_PREFLIGHT_V1', status=report_status(failures),
                              checks=report.rows, matrices=rows,
                              hardware_accuracy_accepted=False), indent=2))
    else:
        print(report.render())
        print(render_matrices(rows))
        print()
        print(report_status(failures))
        if failures:
            print('blocked by: ' + ', '.join(f"{row['section']}/{row['name']}" for row in failures),
                  file=sys.stderr)
    if args.receipt:
        if args.receipt.exists():
            raise SystemExit('refusing existing receipt path: ' + str(args.receipt))
        args.receipt.write_text(json.dumps(
            dict(schema='SG_PREFLIGHT_V1', status=report_status(failures), checks=report.rows,
                 matrices=rows, hardware_accuracy_accepted=False), indent=2) + '\n')
    return 1 if failures else 0


def report_status(failures: list[dict]) -> str:
    return 'PASS_HOST_PREREQUISITES' if not failures else 'FAIL_HOST_PREREQUISITES'


if __name__ == '__main__':
    raise SystemExit(main())
