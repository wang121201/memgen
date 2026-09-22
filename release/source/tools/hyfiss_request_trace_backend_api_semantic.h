#ifndef HYFISS_REQUEST_TRACE_BACKEND_API_H
#define HYFISS_REQUEST_TRACE_BACKEND_API_H

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "../trace-parser/trace-parser.h"
#include "r4_l1_read_filter.h"

namespace hyfiss_request_trace {

struct OrderedMemoryInst {
  std::uint64_t pc = 0;
  std::vector<unsigned long long> addr;
  std::uint64_t timestamp = 0;
  std::uint32_t mask = 0;
  std::string opcode;
  unsigned block_id = 0;
  unsigned sm_id = 0;
  std::uint64_t sequence = 0;
};

struct L2AccessObservation {
  int kernel_id=0; unsigned partition=0;
  uint64_t addr=0,index_addr=0; char operation='R';uint32_t byte_mask=0;
  int outcome=0; // 0 hit, 1 hit reserved, 2 line miss, 3 sector miss
  uint32_t valid_before=0,dirty_before=0,valid_after=0,dirty_after=0;
  bool victim=false;uint64_t victim_addr=0;uint32_t victim_valid=0,victim_dirty=0;
};
struct BackendOptions {
  // Experimental serial-read candidate; legacy remains the default.
  bool r4_l1_read_filter = false;
  std::string r4_model_id = "CLOCK_u128_s16_h2_c1062";
  std::function<void(const L2AccessObservation&)> observe_l2_access;
  // Same state payload; partition is SM id. Bypassed L1 operations are excluded.
  std::function<void(const L2AccessObservation&)> observe_l1_access;
  // A validated immutable profile may be shared by frontend and backend.
  // In schema 1 the file owns all hardware policies; legacy API policy members
  // below apply only to legacy files. Workload/output callbacks stay caller-owned.
  std::shared_ptr<const HardwareProfile> hardware_profile;
  std::string hw_config;
  std::string output_dir;
  // Optional llama.cpp CUDA tensor RANGE sidecar.  When set, the summary-only
  // backend writes a small semantic_traffic.csv ledger; it never materializes
  // request or footprint traces for this purpose.
  std::string semantic_file;
  bool semantic_summary = false;
  bool observe_cache = false;
  std::string output_format = "summary";
  std::string emit_level = "DRAM";
  std::string order = "hyfiss-sm";
  std::string l1_store_policy = "bypass";
  std::string write_sector_policy = "line-miss-only";
  std::string dram_store_policy = "writeback";
  bool l2_dirty_drain = true;
  bool l2_streaming_fill = false;
  bool l2_dirty_drain_latency_set = false;
  uint64_t l2_dirty_drain_latency = 0;
  uint64_t l2_dirty_drain_max_sectors_per_kernel = 0;
  uint64_t l2_dirty_drain_high_watermark_sectors = 0;
  uint64_t l2_dirty_drain_target_sectors = 0;
  bool preserve_l2 = true;
  bool preserve_l1 = false;
  bool monotonic_sm = true;
  bool include_local = true;
  bool l1_fill_latency_set = false;
  bool l2_fill_latency_set = false;
  uint64_t l1_fill_latency = 0;
  uint64_t l2_fill_latency = 0;
  unsigned sector_size = 32;
  unsigned issue_interval = 1;
  unsigned kernel_gap = 5000;
};

struct KernelTraceRef {
  unsigned r4_shared_kib = 0;
  std::vector<R4Allocation> r4_allocations;
  int kernel_id = 0;
  std::string kernel_name;
  std::string llm_phase = "unknown";
  const std::map<int, std::vector<mem_instn>> *sm_traces = nullptr;
  // Optional already globally ordered, one-instruction-at-a-time source.  It
  // avoids materializing or duplicating a complete kernel trace.  Exactly one
  // of sm_traces and next_ordered_inst may be set for a yielded kernel.
  std::function<bool(OrderedMemoryInst &)> next_ordered_inst;
};

int run_from_sm_trace_source(
    const BackendOptions &backend_opt,
    const std::function<bool(KernelTraceRef &)> &next_kernel);
int run_from_sm_traces(const std::vector<KernelTraceRef> &kernels,
                       const BackendOptions &backend_opt);
int run_cli(int argc, char **argv);

} // namespace hyfiss_request_trace

#endif
