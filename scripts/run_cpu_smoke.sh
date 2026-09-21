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

mkdir -p "$out/build" "$out/off" "$out/on"
mpic++ -std=c++17 -O2 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  "$root/release/source/tools/hbserve_profile_stream_cache_semantic_r17.cpp" \
  -l:libzstd.so.1 -lz -lboost_mpi -lboost_serialization -lcrypto -pthread \
  -o "$out/build/hbserve"

common=(
  --mode memgen
  --profile-index "$root/release/fixtures/smoke/profiles.index.jsonl"
  --app-config "$root/release/fixtures/smoke/configs/app.config"
  --issue-config "$root/release/fixtures/smoke/configs/issue.config"
  --hw-config "$root/release/config/RTX4000Ada.paper-v1.config"
  --include-local false
)

"$out/build/hbserve" "${common[@]}" --stats "$out/off/source-stats.json" --output-dir "$out/off/model" --observe-cache false
"$out/build/hbserve" "${common[@]}" --stats "$out/on/source-stats.json" --output-dir "$out/on/model" --observe-cache true

cmp "$out/off/model/kernel_summary.csv" "$out/on/model/kernel_summary.csv"
actual="$(sha256sum "$out/on/model/kernel_summary.csv" | awk '{print $1}')"
expected="9e3b2ee1b69fce3650b9ae2e5a26583ff786d86008f2bb698fc1463915f2d9f2"
if [[ "$actual" != "$expected" ]]; then
  echo "kernel summary SHA-256 mismatch: $actual" >&2
  exit 1
fi
python3 "$root/scripts/verify_archive.py"
echo "PASS_FROZEN_CPU_SMOKE output=$out"
