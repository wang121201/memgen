#!/usr/bin/env python3
"""Build a phase subset of a packed expansion, for fast policy ablations.

Not a workflow stage: it exists so a single cache-policy question (here: what the
store path does to L2 write requests and DRAM stores) can be answered in minutes
on one phase instead of re-replaying the whole model.

The engine requires dense kernel ids 1..N, so the subset renumbers them and
rewrites only the ``kernel.id`` field of each profile payload. Nothing else in a
profile changes, and every index row keeps ``source_kernel_id`` plus the SHA-256
of the rewritten payload, so a subset profile can always be traced back to the
expansion profile it came from. A subset is never a full-model stream and its
receipt says so.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def need(ok, message):
    if not ok:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--phases', required=True, help='comma separated phase labels, e.g. Decode1,Decode2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0,
                        help='keep only the first N selected kernels; for fast control runs')
    parser.add_argument('--replay-config', type=Path, required=True,
                        help='a replay config whose input pins are copied into the subset receipt')
    args = parser.parse_args()

    wanted = {phase.strip() for phase in args.phases.split(',') if phase.strip()}
    need(wanted, 'no phase requested')
    expanded = args.expanded.resolve()
    app_rows = {}
    phases = {}
    for line in (expanded / 'app.config').read_text().splitlines():
        match = re.match(r'-kernel_(\d+)_(\w+) (.*)$', line)
        need(match is not None, 'unexpected app.config row: ' + line[:60])
        kid, field, value = int(match.group(1)), match.group(2), match.group(3)
        app_rows.setdefault(kid, {})[field] = value
    for kid, fields in app_rows.items():
        phases[kid] = fields['llama_phase']
    selected = {kid for kid, phase in phases.items() if phase in wanted}
    need(selected, 'no kernel carries the requested phases')
    if args.limit:
        need(args.limit > 0, '--limit must be positive')
        selected = set(sorted(selected)[:args.limit])

    index = [json.loads(line) for line in (expanded / 'profiles.index.jsonl').read_text().splitlines()]
    by_id = {row['kernel_id']: row for row in index}
    need(set(by_id) == set(app_rows), 'index and app.config disagree about which kernels exist')

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pack = output / 'profiles.pack'
    offset = 0
    kept = []
    renumber = {kid: index for index, kid in enumerate(sorted(selected), start=1)}
    with (expanded / 'profiles.pack').open('rb') as source, pack.open('wb') as destination:
        for kid in sorted(selected):
            row = by_id[kid]
            source.seek(row['offset'])
            raw = source.read(row['bytes'])
            need(hashlib.sha256(raw).hexdigest() == row['sha256'],
                 'profile %d does not match the expansion index' % kid)
            value = json.loads(raw)
            need(int(value['kernel']['id']) == kid, 'expansion profile carries another kernel id')
            value['kernel']['id'] = renumber[kid]
            rewritten = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
            destination.write(rewritten)
            kept.append(dict(kernel_id=renumber[kid], source_kernel_id=kid, path=str(pack),
                             offset=offset, bytes=len(rewritten),
                             sha256=hashlib.sha256(rewritten).hexdigest(), status=row['status'],
                             rewritten_fields=['kernel.id']))
            offset += len(rewritten)
    (output / 'profiles.index.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in kept))

    fields = ('kernel_name', 'llama_phase', 'grid_dim_x', 'grid_dim_y', 'grid_dim_z',
              'grid_size', 'block_size')
    (output / 'app.config').write_text(''.join(
        '-kernel_%d_%s %s\n' % (renumber[kid], field, app_rows[kid][field])
        for kid in sorted(selected) for field in fields))

    issued = {}
    pattern = re.compile(r'-trace_issued_sm_id_(\d+) (.*)$')
    for line in (expanded / 'issue.config').read_text().splitlines():
        match = pattern.match(line)
        need(match is not None, 'unexpected issue.config row: ' + line[:60])
        sm, entries = int(match.group(1)), match.group(2).split()
        keep = []
        for entry in entries:
            kernel_id, cta, timestamp = entry.strip('()').split(',')
            if int(kernel_id) in selected:
                keep.append('(%d,%s,%s)' % (renumber[int(kernel_id)], cta, timestamp))
        if keep:
            issued[sm] = keep
    need(issued, 'no CTA was issued for the selected phases')
    (output / 'issue.config').write_text(''.join(
        '-trace_issued_sm_id_%d %s\n' % (sm, ' '.join(entries)) for sm, entries in sorted(issued.items())))

    semantic = (expanded / 'semantic.ranges').read_bytes()
    (output / 'semantic.ranges').write_bytes(semantic)

    phases_kept = sorted({phases[kid] for kid in selected})
    receipt = dict(schema='SG_PHASE_SUBSET_EXPANSION_V1', source_expansion=str(expanded),
                   phases=phases_kept, kernels=len(selected), ctas=sum(
                       int(app_rows[kid]['grid_size']) for kid in selected),
                   pack_bytes=offset, renumbered_kernel_ids=True,
                   recorded_source_kernel_ids=True, limited_to=args.limit or None,
                   replay_config=str(args.replay_config.resolve()),
                   note='filtered copy of a packed expansion; only kernel.id differs from the '
                        'source profile, the source id is recorded per row, and this is not a '
                        'full-model stream')
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('phases', 'kernels', 'ctas', 'pack_bytes')}))


if __name__ == '__main__':
    main()
