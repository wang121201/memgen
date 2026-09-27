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

# Relocate index references without changing packed profile bytes.
python3 - "$root" "$out" <<'PY'
import json, pathlib, sys
root, out = map(pathlib.Path, sys.argv[1:])
source = root / 'release/fixtures/smoke'
rows = [json.loads(line) for line in (source/'profiles.index.jsonl').read_text().splitlines()]
for row in rows:
    row['path'] = str(source/'profiles.pack.jsonl')
(out/'profiles.index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
PY

common=(
  --mode memgen
  --profile-index "$out/profiles.index.jsonl"
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

# The write-path policies are configurable, and that knob has three properties
# worth pinning: it is inert unless a config asks for it (the hash above), a
# policy change is visible as a different store path rather than a different
# request stream, and an invalid policy is refused instead of silently defaulted.
mkdir -p "$out/wc" "$out/bad"
cp "$root/release/config/RTX4000Ada.paper-v1.config" "$out/wc.config"
printf -- '-memgen_l1_store_policy allocate\n' >> "$out/wc.config"
"$out/build/hbserve" "${common[@]}" --hw-config "$out/wc.config" \
  --stats "$out/wc/source-stats.json" --output-dir "$out/wc/model" --observe-cache false

cp "$root/release/config/RTX4000Ada.paper-v1.config" "$out/bad.config"
printf -- '-memgen_l1_store_policy write-through\n' >> "$out/bad.config"
if "$out/build/hbserve" "${common[@]}" --hw-config "$out/bad.config" \
     --stats "$out/bad/source-stats.json" --output-dir "$out/bad/model" --observe-cache false \
     >/dev/null 2>&1; then
  echo "an invalid write-path policy was accepted" >&2
  exit 1
fi

python3 - "$out" <<'PY'
import csv, pathlib, sys
out = pathlib.Path(sys.argv[1])

def totals(path):
    rows = list(csv.DictReader(path.open()))
    return {name: sum(int(float(row[name])) for row in rows if row.get(name))
            for name in ('l1_requests', 'write_sector_requests', 'l2_write_requests', 'dram_load_bytes')}

default = totals(out / 'off' / 'model' / 'kernel_summary.csv')
configured = totals(out / 'wc' / 'model' / 'kernel_summary.csv')
failures = []
# The LSU-level store stream must not move: only its path through the caches may.
if configured['write_sector_requests'] != default['write_sector_requests']:
    failures.append('store sector count moved with the store policy')
# Stores must never fetch: reads are not allowed to move either.
if configured['dram_load_bytes'] != default['dram_load_bytes']:
    failures.append('DRAM read moved with the store policy')
# Bypass skips the L1 access for stores; allocating must show up there.
if configured['l1_requests'] <= default['l1_requests']:
    failures.append('allocating stores did not access L1')
if configured['l2_write_requests'] > default['l2_write_requests']:
    failures.append('allocating stores increased L2 write requests')
if failures:
    raise SystemExit('write-path policy check failed: ' + '; '.join(failures))
print('PASS_WRITE_PATH_POLICY: l1_requests {} -> {}, l2_write_requests {} -> {}, '
      'dram_load_bytes unchanged'.format(default['l1_requests'], configured['l1_requests'],
                                         default['l2_write_requests'], configured['l2_write_requests']))
PY

# The victim preference is configurable too. The smoke fixture is too small to
# show a retention difference, so the gate pins the contract instead of a number:
# a valid window is accepted, and a value outside the documented range is refused
# rather than silently clamped.
mkdir -p "$out/cfw" "$out/cfbad"
cp "$root/release/config/RTX4000Ada.paper-v1.config" "$out/cf.config"
printf -- '-memgen_l2_clean_first_k 4\n' >> "$out/cf.config"
"$out/build/hbserve" "${common[@]}" --hw-config "$out/cf.config" \
  --stats "$out/cfw/source-stats.json" --output-dir "$out/cfw/model" --observe-cache false

cp "$root/release/config/RTX4000Ada.paper-v1.config" "$out/cfbad.config"
printf -- '-memgen_l2_clean_first_k 65\n' >> "$out/cfbad.config"
if "$out/build/hbserve" "${common[@]}" --hw-config "$out/cfbad.config" \
     --stats "$out/cfbad/source-stats.json" --output-dir "$out/cfbad/model" --observe-cache false \
     >/dev/null 2>&1; then
  echo "an out-of-range clean-first window was accepted" >&2
  exit 1
fi
echo "PASS_CLEAN_FIRST_WINDOW: window 4 accepted, window 65 refused"

python3 "$root/scripts/verify_archive.py"
echo "PASS_FROZEN_CPU_SMOKE output=$out"
