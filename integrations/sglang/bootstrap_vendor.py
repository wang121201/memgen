#!/usr/bin/env python3
"""Materialize hash-pinned deployment files that this archive does not carry.

`package.json` records, for every file the frozen controller deployment needs,
both its machine-local origin and its SHA-256. Six of those files are imported
at runtime but are deliberately not committed here, so without them
`run_job.py` cannot even be imported and `wait_then_sample.py` cannot pin its
controller.

This script copies each missing file from its recorded origin and verifies
bytes and SHA-256 before and after the copy. It never downloads anything, never
overwrites a file whose content differs from the pin, and never runs a GPU or
loads a model. A materialized file is byte-identical to the deployment that
produced the archived receipts; a mismatch is reported, never repaired.

`revisions.json` explains each deliberate difference from a historical pin, so a
reviewed change reports `PRESENT_REVISED` while an unexplained difference still
reports `PRESENT_DRIFT`.

Exit status is 0 only when every pinned file ends in an accepted status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / 'package.json'
REVISIONS = 'revisions.json'
OK_STATUSES = ('PRESENT_IDENTICAL', 'MATERIALIZED', 'PRESENT_REVISED')


def pin(path: Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(path=str(Path(path).resolve()), bytes=Path(path).stat().st_size,
                sha256=digest.hexdigest())


def matches(got: dict, row: dict) -> bool:
    return got['bytes'] == row['bytes'] and got['sha256'] == row['sha256']


def revision_index(base: Path) -> dict[str, dict]:
    """Map a base-relative path to the reviewed revision explaining its content."""
    path = base / REVISIONS
    if not path.is_file():
        return {}
    index: dict[str, dict] = {}
    for entry in json.loads(path.read_text())['revisions']:
        for relative in entry['paths']:
            index[relative] = entry
    return index


def revisions_check(base: Path, covered: tuple[str, ...] = ()) -> list[dict]:
    """Verify every copy named by revisions.json carries its recorded current pin.

    This catches a half-applied revision: one mirror updated and the other not.
    """
    path = base / REVISIONS
    if not path.is_file():
        return []
    rows = []
    for entry in json.loads(path.read_text())['revisions']:
        for relative in entry['paths']:
            if relative in covered:
                continue
            target = base / relative
            if not target.is_file():
                status, detail = 'MISSING', str(target)
            else:
                actual = pin(target)['sha256']
                status = 'OK' if actual == entry['current_sha256'] else 'DRIFT'
                detail = actual
            rows.append(dict(relative=relative, status=status, actual=detail,
                             expected_sha256=entry['current_sha256'], name=entry['name'],
                             record=entry.get('record', '')))
    return rows


def classify(manifest: Path, base: Path) -> list[dict]:
    """Report one status per pinned file without modifying anything."""
    rows = json.loads(manifest.read_text())['files']
    index = revision_index(base)
    report = []
    for row in rows:
        target = base / row['relative']
        record = dict(relative=row['relative'], origin=row['path'],
                      expected_sha256=row['sha256'], expected_bytes=row['bytes'])
        if target.is_file():
            got = pin(target)
            if matches(got, row):
                record.update(status='PRESENT_IDENTICAL', actual_sha256=got['sha256'],
                              actual_bytes=got['bytes'])
            else:
                entry = index.get(row['relative'])
                if entry and got['sha256'] == entry['current_sha256']:
                    record.update(status='PRESENT_REVISED', actual_sha256=got['sha256'],
                                  actual_bytes=got['bytes'], revision=entry['name'],
                                  record=entry.get('record', ''))
                else:
                    record.update(status='PRESENT_DRIFT', actual_sha256=got['sha256'],
                                  actual_bytes=got['bytes'])
        elif not Path(row['path']).is_file():
            record.update(status='MISSING_ORIGIN')
        else:
            got = pin(Path(row['path']))
            record.update(status='ORIGIN_DRIFT' if not matches(got, row) else 'MATERIALIZABLE',
                          actual_sha256=got['sha256'])
        report.append(record)
    return report


def materialize(manifest: Path, base: Path) -> list[dict]:
    """Copy missing files from their pinned origin, verifying both ends."""
    report = classify(manifest, base)
    for row, record in zip(json.loads(manifest.read_text())['files'], report):
        if record['status'] != 'MATERIALIZABLE':
            continue
        target = base / row['relative']
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(target.name + '.staged')
        if staged.exists():
            raise SystemExit('refusing to clobber staging file: ' + str(staged))
        shutil.copyfile(row['path'], staged)
        got = pin(staged)
        if not matches(got, row):
            staged.unlink()
            raise SystemExit('origin changed during copy: ' + row['path'])
        os.replace(staged, target)
        record.update(status='MATERIALIZED', actual_sha256=got['sha256'], actual_bytes=got['bytes'])
    return report


def render(report: list[dict]) -> str:
    lines = []
    for row in report:
        lines.append(f"{row['status']:<18} {row['relative']}")
        if row['status'] != 'PRESENT_IDENTICAL':
            lines.append(f"{'':<18}   origin: {row['origin']}")
            if 'actual_sha256' in row:
                lines.append(f"{'':<18}   expected {row['expected_sha256']}")
                lines.append(f"{'':<18}   actual   {row['actual_sha256']}")
            if row.get('revision'):
                lines.append(f"{'':<18}   revision {row['revision']} -> {row.get('record', '')}")
    return '\n'.join(lines)


def render_revisions(rows: list[dict]) -> str:
    lines = ['', '== REVISIONS ==']
    if not rows:
        lines.append('  none declared')
    for row in rows:
        lines.append(f"  {row['status']:<9} {row['relative']}")
        if row['status'] != 'OK':
            lines.append(f"{'':<12} expected {row['expected_sha256']}")
            lines.append(f"{'':<12} actual   {row['actual']}")
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--base', type=Path, default=HERE,
                        help='directory the relative paths resolve against')
    parser.add_argument('--check', action='store_true',
                        help='report status only; write nothing')
    parser.add_argument('--json', action='store_true', help='machine-readable report')
    parser.add_argument('--receipt', type=Path,
                        help='also write the report to this fresh path')
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit('manifest not found: ' + str(args.manifest))
    base = args.base.resolve()
    manifest = args.manifest.resolve()
    report = (classify if args.check else materialize)(manifest, base)
    covered = tuple(row['relative'] for row in json.loads(manifest.read_text())['files'])
    revisions = revisions_check(base, covered)
    failed = ([row for row in report if row['status'] not in OK_STATUSES]
              + [row for row in revisions if row['status'] != 'OK'])
    payload = dict(schema='SG_VENDOR_MATERIALIZE_V1', mode='check' if args.check else 'materialize',
                   manifest=str(manifest), files=report, revisions=revisions,
                   status='PASS_VENDOR_PINS_SATISFIED' if not failed else 'FAIL_VENDOR_PINS_UNSATISFIED',
                   hardware_accuracy_accepted=False)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(report))
        print(render_revisions(revisions))
        print(payload['status'])
    if args.receipt:
        if args.receipt.exists():
            raise SystemExit('refusing existing receipt path: ' + str(args.receipt))
        args.receipt.write_text(json.dumps(payload, indent=2) + '\n')
    if failed:
        print('unresolved: ' + ', '.join(row['relative'] for row in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
