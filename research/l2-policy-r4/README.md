# L2 write-back and dirty-management research snapshot r4

L2 means Level-2 cache. DRAM means off-chip graphics-memory traffic. LRU means
Least Recently Used replacement. A dirty sector contains data modified in L2
that has not yet been emitted as a lower-level write. RMW means read-modify-write
and CTA means CUDA Cooperative Thread Array (thread block).

This directory is the frozen r4 mechanism-research snapshot imported from:

`/home/xmu/nvidiagds/codex-runs/l2-sglang-full-policy-matrix-20260921-01a08d87-r1`

It is intentionally isolated on branch
`research/l2-writeback-dirty-management`. It is not the `main` cache default
and does not claim that NVIDIA's physical L2 policy has been recovered.

## Frozen identity

- `full-policy-r4` binary SHA-256:
  `abf48f900070c78dd58c0539aa1dc5c03f810f6404b023fd92b3601fd2c25d40`.
- `policies-r4.csv` SHA-256:
  `07e97eb46bdec863513113c261dd01253df2f66a8e282e2bb5c27344a05302cf`.
- Registry: 45 causal online policies, with ordinary LRU `disabled` always
  first as the reference.
- Stable fixture output: exact boundaries, partial dirty sectors, RMW,
  producer/version conservation, clean-victim protection, age/spacing gates,
  pending selection/cancellation and cross-kernel state all pass.
- Regression anchor: 24 keys, 648 compared fields, no differences and equal
  semantic digests.

The later r5 source and performance work were still active when this branch was
created and are deliberately excluded.

## Strategy families

The registry includes ordinary LRU; fixed dirty quotas; clean-first quotas;
read-pressure release; age- and sector-age variants; pending selection,
overflow and cancellation; phase variants 0 through 15; LRU-victim and
behavior-arm ablations. `phase11` is a policy version name (clean-first, write
quota 4, age 16), not inference phase number 11.

The current hardware result is negative but useful: ordinary LRU keeps SGLang
P128D2/P128D16 aggregate DRAM read error at 0.54%/0.20%, while whole-request
write changes from +0.91% to +14.55% and decode-only write reaches
+39.17%/+648.63%. `phase11` does not improve decode write; clean-first q16
under-writes whole requests by roughly 90%. See
`../../evidence/sglang/L2_CACHE_STRATEGY_ACCURACY_REPORT.md` for the complete
scope and evidence classes.

## Contents

- `source/`: frozen r4 cache/policy implementation and deterministic fixtures.
- `runner/`: immutable shard launcher and aggregation utilities.
- `policy-registry.csv`: the exact 45-policy registry.
- `evidence/`: small fixture, regression and runner-smoke receipts only; no
  full raw trace or unfinished matrix output.

## Verification

From the repository root:

```bash
scripts/run_l2_r4_fixture.sh /tmp/memgen-l2-r4-fixture-r1
```

The script compiles from source, runs the deterministic fixture, regenerates
the policy registry and requires a byte-identical registry SHA-256. It is a
functional/causal validation, not NCU hardware acceptance.

Promotion to `main` requires independent SGLang workloads, separate prefill and
decode traffic, complete source/dirty conservation, improved decode write and
no read regression. A single aggregate fit is insufficient.

