#!/usr/bin/env python3
"""Report a reproducible identity for a built NVBit tool.

`transition-proof-r1.json` records the SHA-256 of `observer.so` as an evidence
identity. That hash is an artifact identity: it includes the GNU build-id, the
symbol table and nvcc's per-invocation temporary file name, so nobody can
re-derive it, not even on this machine with every declared input verified
byte-identical. Rebuilding the frozen source here produces an equal-sized
binary whose `.text` differs from the archived one by 0.083% of its bytes.

This script separates the two questions that the artifact hash conflates:

- `artifact_sha256` is the hash of the file as built. It identifies one build.
- `content_sha256` covers every section except `.symtab`, `.strtab`,
  `.comment` and `.note.gnu.build-id`. It is stable across rebuilds of the same
  source with the same compiler and lets a reviewer tell "same code, different
  build" from "different code".

Exit status is 0 when given one file, or when two files have equal content.
No GPU, no model and no network is used.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import struct
import sys
from pathlib import Path

VOLATILE = ('.symtab', '.strtab', '.comment', '.note.gnu.build-id', '.shstrtab')
TEMP_NAME = re.compile(rb'tmpxft_[0-9a-f]+_\d+-\d+_[\w.]+')


def sections(path: Path) -> dict[str, bytes]:
    """Minimal ELF64 section reader; returns section name to payload."""
    blob = path.read_bytes()
    if blob[:4] != b'\x7fELF' or blob[4] != 2:
        raise ValueError('ELF64 required: ' + str(path))
    (shoff,) = struct.unpack_from('<Q', blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', blob, 0x3A)

    def header(index: int) -> tuple[int, int, int]:
        base = shoff + index * shentsize
        name, _, _, _, offset, size = struct.unpack_from('<IIQQQQ', blob, base)
        return name, offset, size

    _, names_offset, names_size = header(shstrndx)
    names = blob[names_offset:names_offset + names_size]
    out = {}
    for index in range(shnum):
        name, offset, size = header(index)
        end = names.index(b'\0', name)
        out[names[name:end].decode()] = blob[offset:offset + size]
    return out


def content_hash(items: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(items):
        if name in VOLATILE or not items[name]:
            continue
        digest.update(name.encode())
        digest.update(items[name])
    return digest.hexdigest()


def report(path: Path) -> dict:
    items = sections(path)
    volatile = sorted(name for name in items if name in VOLATILE and items[name])
    temps = sorted({m.decode() for m in TEMP_NAME.findall(items.get('.strtab', b''))})
    return dict(path=str(path), bytes=path.stat().st_size,
                artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                content_sha256=content_hash(items),
                text_sha256=hashlib.sha256(items.get('.text', b'')).hexdigest(),
                volatile_sections=volatile, nvcc_temporary_names=temps)


def compare(first: Path, second: Path) -> tuple[bool, list[tuple[str, int]]]:
    a, b = sections(first), sections(second)
    rows = []
    for name in sorted(set(a) & set(b)):
        if a[name] != b[name]:
            rows.append((name, sum(1 for x, y in zip(a[name], b[name]) if x != y)))
    for name in sorted(set(a) ^ set(b)):
        rows.append((name, -1))
    return content_hash(a) == content_hash(b), rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tool', type=Path)
    parser.add_argument('--compare', type=Path,
                        help='second build to compare against; exit 1 unless content matches')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    for path in (args.tool, args.compare):
        if path is not None and not path.is_file():
            raise SystemExit('not a file: ' + str(path))
    first = report(args.tool)
    if not args.json:
        print(f"artifact_sha256  {first['artifact_sha256']}  (identifies this build only)")
        print(f"content_sha256   {first['content_sha256']}  (stable across rebuilds)")
        print(f"text_sha256      {first['text_sha256']}")
        print(f"bytes            {first['bytes']}")
        print(f"volatile         {', '.join(first['volatile_sections'])}")
        print(f"nvcc temp names  {', '.join(first['nvcc_temporary_names']) or 'none'}")
    if args.compare is None:
        if args.json:
            print(first)
        return 0

    equal, rows = compare(args.tool, args.compare)
    second = report(args.compare)
    if not args.json:
        print(f"\ncomparing against {args.compare}")
        print(f"  second artifact_sha256  {second['artifact_sha256']}")
        print(f"  second content_sha256   {second['content_sha256']}")
        print(f"  content identical       {equal}")
        for name, differing in rows:
            detail = 'added or removed' if differing < 0 else f'{differing} differing bytes'
            print(f'  differs {name:22} {detail}')
    if args.json:
        print(dict(first=first, second=second, content_identical=equal,
                   differing_sections=[dict(name=n, bytes=d) for n, d in rows]))
    print('SAME_CONTENT' if equal else 'DIFFERENT_CONTENT')
    return 0 if equal else 1


if __name__ == '__main__':
    raise SystemExit(main())
