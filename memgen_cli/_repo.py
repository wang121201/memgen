"""Repository paths and import bridges shared by the commands.

The pipeline lives under `integrations/sglang/`, and its modules are written to
be run from their own directory. This module puts those directories on the
import path once, disables bytecode so `scripts/verify_archive.py` never sees a
`__pycache__`, and exposes the paths the commands need.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SGLANG = ROOT / 'integrations/sglang'
ADAPTER = SGLANG / 'memgen-adapter'
SCRIPTS = ROOT / 'scripts'
TESTS = ROOT / 'tests'
COLLECT = SGLANG / 'collect_case.py'
PREFLIGHT = SGLANG / 'preflight.py'
BOOTSTRAP = SGLANG / 'bootstrap_vendor.py'
REPLAY = ADAPTER / 'run_memgen.py'

sys.dont_write_bytecode = True
for path in (str(SGLANG), str(ADAPTER)):
    if path not in sys.path:
        sys.path.insert(0, path)


def interpreter() -> str:
    """The interpreter the pipeline expects to carry the SGLang stack."""
    return '/home/xmu/sgl/bin/python'


def timestamped(prefix: str) -> Path:
    """A fresh output directory, the way every stage here insists on one."""
    import time
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    return ROOT / 'out' / f'{prefix}-{stamp}'
