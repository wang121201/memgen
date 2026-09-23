#!/usr/bin/env python3
"""Command-line entry point for memgen."""

from __future__ import annotations

import sys
from typing import Sequence

USAGE = """usage: memgen <command> [options]

commands:
  check         verify the host and the frozen archive before collecting, no GPU
  cases         list the declared workload points
  gpus          list the admitted GPUs and their --gpu-index numbering
  plan          write the job specs for one case and print the plan, run nothing
  collect       collect one declared case: census, sample, expand, replay
  replay        run an admitted profile stream through the cache model
  smoke         replay the frozen engine's synthetic fixture on the CPU
  test          portable regression tests
  capabilities  print the machine-readable claim boundary

Run `memgen <command> --help` for the command's options.
"""


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {'-h', '--help'}:
        print(USAGE, end='')
        return 0
    if arguments[0] == '--version':
        from memgen_cli import __version__

        print(__version__)
        return 0

    command, rest = arguments[0], arguments[1:]
    if command == 'capabilities':
        from memgen_cli.capabilities import main as capabilities_main

        return capabilities_main(rest)
    if command in {'check', 'cases', 'gpus', 'plan', 'collect', 'replay', 'smoke', 'test'}:
        from memgen_cli import commands

        return getattr(commands, command)(rest)

    print(f'unknown command: {command}', file=sys.stderr)
    print(USAGE, end='', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
