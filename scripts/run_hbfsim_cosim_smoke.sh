#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
simulator=${1:?usage: run_hbfsim_cosim_smoke.sh HBFSim-binary [new-output-dir]}
output_root=${2:-/tmp/memgen-hbfsim-cosim-smoke}

if [[ -e "$output_root" ]]; then
  echo "refusing to overwrite existing output: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root"

python3 "$repo_root/cosimulation/coupled_replay.py" \
  --events "$repo_root/cosimulation/fixtures/two_warp_causal.jsonl" \
  --simulator "$simulator" \
  --system-config "$repo_root/cosimulation/config/eight-stack-baseline.cfg" \
  --system-config "$repo_root/cosimulation/config/rtx4000-ada-uncalibrated.cfg" \
  --output "$output_root/result.json"

python3 - "$output_root/result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "PASS_HBFSIM_CAUSAL_COSIM_DIAGNOSTIC"
assert result["hardware_accuracy_accepted"] is False
assert result["event_count"] == 3
assert result["memory_transaction_count"] == 3
assert result["traffic"]["read_bytes"] == 64
assert result["traffic"]["write_bytes"] == 32
events = {row["event_id"]: row for row in result["events"]}
assert events["w0_e1"]["compute_start_ns"] >= events["w0_e0"]["event_finish_ns"]
assert events["w0_e1"]["causal_stall_ns"] > 0
assert events["w1_e0"]["compute_start_ns"] == 0
assert events["w1_e0"]["compute_start_ns"] < events["w0_e0"]["event_finish_ns"]
print("PASS_HBFSIM_CAUSAL_COSIM_SMOKE")
PY
