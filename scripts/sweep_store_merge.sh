#!/usr/bin/env bash
# Sweep -memgen_store_merge_window over the 642-kernel Decode subset and report
# the L1-store -> L2-write merge ratio and DRAM store bytes per window.
#
# usage: bash scripts/sweep_store_merge.sh <engine> <profile-index> <app-config> \
#        <issue-config> <hw-config-base> <output-root>
set -euo pipefail

engine="$1"
profile_index="$2"
app_config="$3"
issue_config="$4"
hw_base="$5"
out_root="$6"

mkdir -p "$out_root"

echo "window,write_sector_requests,l2_write_requests,merge_ratio,dram_store_bytes"

for window in 0 2 4 8 16 32 64; do
  cfg="$out_root/w${window}.config"
  cp "$hw_base" "$cfg"
  printf -- '-memgen_store_merge_window %d\n' "$window" >> "$cfg"
  run="$out_root/w${window}"
  rm -rf "$run"
  "$engine" --mode memgen \
    --profile-index "$profile_index" \
    --app-config "$app_config" \
    --issue-config "$issue_config" \
    --hw-config "$cfg" \
    --include-local false \
    --stats "$out_root/w${window}-stats.json" \
    --output-dir "$run" \
    --observe-cache false >/dev/null 2>&1
  python3 - "$run" "$window" <<'PY'
import csv, sys
run, window = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(run + '/model/kernel_summary.csv')))
ws = sum(int(r['write_sector_requests']) for r in rows)
l2w = sum(int(r['l2_write_requests']) for r in rows)
dsb = sum(int(r['dram_store_bytes']) for r in rows)
ratio = l2w / ws if ws else 0.0
print(f"{window},{ws},{l2w},{ratio:.4f},{dsb}")
PY
done
