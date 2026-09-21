#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${1:-/tmp/memgen-simple-latency-smoke}

if [[ -e "$output_root" ]]; then
  echo "refusing to overwrite existing output: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root"

python3 "$repo_root/latency/simple_latency.py" \
  --kernel-summary "$repo_root/latency/fixtures/kernel_summary.csv" \
  --latency-config "$repo_root/latency/fixtures/synthetic_latencies.json" \
  --phase-map "$repo_root/latency/fixtures/kernel_phases.csv" \
  --output-json "$output_root/result.json" \
  --output-csv "$output_root/by-phase.csv"

python3 - "$output_root/result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "PASS_SIMPLE_LATENCY_SERIAL_MEMORY_WORK"
assert result["kernel_count"] == 2
assert result["whole"]["event_counts"] == {
    "l1_hit": 6,
    "l2_hit": 4,
    "dram_read_sector": 4,
    "dram_write_sector": 3,
}
assert result["whole"]["serial_memory_work_ns"] == 806.0
assert result["by_phase"]["Prefill"]["serial_memory_work_ns"] == 355.0
assert result["by_phase"]["Decode1"]["serial_memory_work_ns"] == 451.0
print("PASS_SIMPLE_LATENCY_SMOKE")
PY
