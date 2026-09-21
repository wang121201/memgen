# Simple-latency diagnostic

`simple_latency.py` consumes Memgen's `kernel_summary.csv` and integrates four
exclusive modeled outcomes: L1 resolved hits, L2 resolved hits, 32-byte DRAM
read sectors and 32-byte DRAM write sectors. Pending hits are charged at their
resolving cache level. The configured values are end-to-end costs for these
outcomes, so lookup latency is not added again at every cache level.

The tool validates each row's L1, L2 and DRAM request conservation before it
computes serial work. An optional `kernel_id,phase` CSV provides Prefill and
per-decode summaries. Both inputs and their SHA-256 digests are recorded.

The output `serial_memory_work_ns` is a deterministic sensitivity diagnostic.
It is **not** GPU execution time or achievable bandwidth: this branch contains
no resource scheduler, dependency scoreboard, queue replay, backpressure,
stall propagation or compute-memory overlap. Real acceptance also requires
calibrated latency constants; the fixture constants are deliberately synthetic.

Example:

```bash
python3 latency/simple_latency.py \
  --kernel-summary /path/to/model/kernel_summary.csv \
  --latency-config /path/to/calibrated-latencies.json \
  --phase-map /path/to/kernel-phases.csv \
  --output-json /tmp/simple-latency/result.json \
  --output-csv /tmp/simple-latency/by-phase.csv
```

Run the bounded implementation smoke with:

```bash
scripts/run_simple_latency_smoke.sh /tmp/memgen-simple-latency-smoke
```
