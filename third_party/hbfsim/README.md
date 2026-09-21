# Vendored HBFSim Python client

This directory contains the minimal Python client snapshot from the same clean
HBFSim source commit as the external executable pinned by the cosimulation
branch:

`d7a2ca64614a6d9ce8d7a69beb77ce78b66df1a8`

The snapshot came from the clean detached worktree at
`/home/xmu/nvidiagds/codex-runs/llm-footprint-v1/hbfsim/memgen-cosim-backend-20260921-r2/source`.
File identities are:

| File | SHA-256 |
| --- | --- |
| `hbfsim_client/__init__.py` | `ec639c973a6dca34def2bdaf673c78e82be70a5675408cce10d825e456647415` |
| `hbfsim_client/simulation_session.py` | `dca7cd6647d785704dca7c02d34a31725c313310de782e0dc2edff95c96cc555` |
| `hbfsim_client/transaction_protocol.py` | `4c13b075047b16f86dcae6b6b57033d159bb14a842cb584a592a05fa2b9f471b` |
| `hbfsim_client/provenance.py` | `82d6d4acddeb17929edba45ab1a45040317581ca085a0bee871f246f47866a00` |
| `LICENSE` | `ef719c401c789166de786951fa9805d2505d8144840f633a06e2aaca0a9715e2` |

The snapshot is distributed under the included MIT license. Keeping client and
backend on one commit is mandatory because transaction protocol, resolved
configuration formulas and ready/completion receipt schemas evolve together.
