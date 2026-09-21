#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 FRESH_OUTPUT_DIRECTORY" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$1"
if [[ -e "$out" ]]; then
  echo "refusing existing output path: $out" >&2
  exit 2
fi
mkdir -p "$out"

mpic++ -std=c++17 -O2 \
  "$root/research/l2-policy-r4/source/tools/llm_full_policy_matrix.cpp" \
  -l:libzstd.so.1 -lz -lboost_mpi -lboost_serialization -lcrypto -pthread \
  -o "$out/full-policy-r4"

"$out/full-policy-r4" --fixture "$root/release/config/RTX4000Ada.paper-v1.config" \
  >"$out/fixture.stdout" 2>"$out/fixture.stderr"
FULL_POLICY_MATRIX=all "$out/full-policy-r4" --list-policies >"$out/policies-r4.csv"

actual="$(sha256sum "$out/policies-r4.csv" | awk '{print $1}')"
expected="07e97eb46bdec863513113c261dd01253df2f66a8e282e2bb5c27344a05302cf"
if [[ "$actual" != "$expected" ]]; then
  echo "policy registry SHA-256 mismatch: $actual" >&2
  exit 1
fi
cmp "$out/policies-r4.csv" "$root/research/l2-policy-r4/policy-registry.csv"
grep -q '^PASS_PENDING_SELECT_OVERFLOW_CANCEL_BOUNDARY_PARTIAL_ATOMIC_AND_VERSION_CONSERVATION$' "$out/fixture.stdout"
echo "PASS_L2_R4_CAUSAL_FIXTURE output=$out"

