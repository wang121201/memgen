#include <algorithm>
#include <array>
#include <bitset>
#include <iterator>
#include <cctype>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/crc.hpp>

#include "hyfiss_request_trace_backend_api_semantic.h"
#include "memc_reader.h"

namespace fs = std::filesystem;

namespace {

constexpr unsigned kWarpSize = 32;

enum class SetIndexFunction {
  Linear,
  Ipoly,
  BitwiseXor,
  Fermi,
};

enum class FootprintFormat {
  FullCsv,
  MinimalCsv,
  Binary,
};

const char *set_index_name(SetIndexFunction fn) {
  switch (fn) {
  case SetIndexFunction::Linear:
    return "L";
  case SetIndexFunction::Ipoly:
    return "P";
  case SetIndexFunction::BitwiseXor:
    return "X";
  case SetIndexFunction::Fermi:
    return "H";
  }
  return "L";
}

SetIndexFunction parse_set_index_function(const std::string &s) {
  if (s == "P" || s == "p" || s == "HASH_IPOLY")
    return SetIndexFunction::Ipoly;
  if (s == "X" || s == "x" || s == "BITWISE_XORING")
    return SetIndexFunction::BitwiseXor;
  if (s == "H" || s == "h" || s == "FERMI_HASH")
    return SetIndexFunction::Fermi;
  return SetIndexFunction::Linear;
}

FootprintFormat parse_footprint_format(const std::string &s) {
  if (s == "minimal" || s == "minimal-csv" || s == "csv-minimal")
    return FootprintFormat::MinimalCsv;
  if (s == "binary" || s == "bin" || s == "fbin")
    return FootprintFormat::Binary;
  return FootprintFormat::FullCsv;
}

const char *footprint_format_name(FootprintFormat format) {
  switch (format) {
  case FootprintFormat::FullCsv:
    return "csv";
  case FootprintFormat::MinimalCsv:
    return "minimal";
  case FootprintFormat::Binary:
    return "binary";
  }
  return "csv";
}

template <typename T> void write_pod(std::ostream &os, const T &value) {
  os.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

template <typename T> T read_pod(std::istream &is, const char *what) {
  T value{};
  is.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!is)
    throw std::runtime_error(std::string("truncated checkpoint while reading ") +
                             what);
  return value;
}

struct Options {
  fs::path trace_root;
  fs::path configs_dir;
  fs::path memory_dir;
  fs::path hw_config;
  fs::path output_dir = "request_trace_out";
  std::string kernels = "all";
  std::string emit_kernels = "all";
  bool emit_all_kernels = true;
  std::set<int> emit_kernel_ids;
  std::string input_format = "auto";
  std::string emit_level = "all";
  std::string output_phase = "all";
  std::string output_format = "both";
  std::string footprint_format = "csv";
  fs::path checkpoint_dir;
  fs::path restore_checkpoint;
  uint64_t checkpoint_interval = 0;
  bool checkpoint_only = false;
  uint64_t output_rotate_bytes = uint64_t{1} << 40;
  std::string semantic_output = "none";
  fs::path semantic_file;
  bool semantic_summary = false;
  std::string l1_store_policy = "bypass";
  std::string write_sector_policy = "line-miss-only";
  std::string dram_store_policy = "writeback";
  bool l2_dirty_drain = true;
  bool l2_streaming_fill = false;
  uint64_t l2_dirty_drain_latency = 0;
  bool l2_dirty_drain_latency_set = false;
  uint64_t l2_dirty_drain_max_sectors_per_kernel = 0;
  uint64_t l2_dirty_drain_high_watermark_sectors = 0;
  uint64_t l2_dirty_drain_target_sectors = 0;
  bool preserve_l2 = true;
  bool flush_l2_on_reset = false;
  bool preserve_l1 = false;
  bool monotonic_sm = true;
  bool sort_by_timestamp = false;
  bool include_local = true;
  bool l1_fill_latency_set = false;
  bool l2_fill_latency_set = false;
  uint64_t l1_fill_latency = 0;
  uint64_t l2_fill_latency = 0;
  unsigned sector_size = 32;
  unsigned cache_line_size = 128;
  unsigned l1_line_size = 128;
  unsigned l2_line_size = 128;
  unsigned l1_size_bytes = 0;
  unsigned l2_size_bytes = 0;
  unsigned l1_assoc = 4;
  unsigned l2_assoc = 16;
  SetIndexFunction l1_set_index = SetIndexFunction::Linear;
  SetIndexFunction l2_set_index = SetIndexFunction::Linear;
  unsigned num_sms = 0;
  unsigned num_partitions = 0;
  unsigned num_memory_channels = 0;
  unsigned num_sub_partitions_per_channel = 1;
  unsigned num_banks = 16;
  unsigned partition_index_bit = 8;
  unsigned memory_partition_indexing = 0;
  unsigned mem_address_mask = 0;
  std::string mem_addr_mapping;
  unsigned issue_interval = 1;
  unsigned kernel_gap = 5000;
};

struct KernelMeta {
  int id = 0;
  std::string name = "unknown";
  std::string llm_phase = "unknown";
  fs::path semantic_file;
  unsigned grid_size = 0;
  unsigned block_size = 0;
  uint64_t local_base = 0;
  uint64_t local_warp_begin = 0;
  bool has_local_base = false;
};

struct HwParams {
  std::shared_ptr<const hyfiss_request_trace::HardwareProfile> profile;
  unsigned num_sms = 1;
  unsigned num_partitions = 1;
  unsigned num_memory_channels = 1;
  unsigned num_sub_partitions_per_channel = 1;
  unsigned num_banks = 16;
  unsigned memory_partition_indexing = 0;
  unsigned mem_address_mask = 0;
  std::string mem_addr_mapping;
  unsigned l1_size_bytes = 32 * 1024;
  unsigned l2_size_bytes = 4 * 1024 * 1024;
  unsigned l1_line_size = 128;
  unsigned l2_line_size = 128;
  unsigned l1_assoc = 4;
  unsigned l2_assoc = 16;
  SetIndexFunction l1_set_index = SetIndexFunction::Linear;
  SetIndexFunction l2_set_index = SetIndexFunction::Linear;
  uint64_t l1_fill_latency = 0;
  uint64_t l2_fill_latency = 0;
  unsigned kernel_gap = 5000;
};

struct LaneAddress {
  unsigned lane = 0;
  unsigned ref_id = 0;
  uint64_t addr = 0;  // Untouched captured address.
  bool is_local = false;
  uint64_t local_offset = 0;  // Derived model input, kept separate from addr.
};

struct MemoryInst {
  int kernel_id = 0;
  unsigned block_id = 0;
  unsigned sm_id = 0;
  uint64_t seq = 0;
  uint64_t pc = 0;
  std::string opcode;
  uint32_t mask = 0;
  uint64_t timestamp = 0;
  unsigned mem_width = 4;
  char op = 'R';
  bool has_space_metadata = false;
  uint64_t capture_seq = 0;
  uint64_t full_clock = 0;
  uint64_t local_warp_owner = 0;
  unsigned cta_warp = 0;
  unsigned function_id = 0;
  std::vector<LaneAddress> lanes;
};

struct IssueInfo {
  unsigned sm_id = 0;
  uint64_t cta_start = 0;
  bool has_cta_start = false;
};

struct SectorRequest {
  uint64_t addr = 0;
  unsigned size = 0;
  unsigned ref_id = 0;
  uint32_t lane_mask = 0;
  unsigned lane_count = 0;
  // Bit i is a requested byte at addr+i within this aligned 32-byte sector.
  // This describes one instruction's coverage, not cache-valid/dirty state.
  uint32_t byte_mask = 0;
};

struct CacheDirectionStats {
  uint64_t requests = 0;
  uint64_t hits = 0;
  uint64_t pending_hits = 0;
  uint64_t misses = 0;
  uint64_t line_misses = 0;
  uint64_t sector_misses = 0;
};

struct KernelStats {
  uint64_t mem_insts = 0;
  uint64_t lane_accesses = 0;
  uint64_t sector_requests = 0;
  uint64_t read_sector_requests = 0;
  uint64_t write_sector_requests = 0;
  uint64_t atomic_sector_requests = 0;
  uint64_t write_full_sector_requests = 0;
  uint64_t write_partial_sector_requests = 0;
  uint64_t write_covered_bytes = 0;
  uint64_t l1_requests = 0;
  uint64_t l1_hits = 0;
  uint64_t l1_pending_hits = 0;
  uint64_t l1_misses = 0;
  uint64_t l1_line_misses = 0;
  uint64_t l1_sector_misses = 0;
  uint64_t l2_requests = 0;
  uint64_t l2_hits = 0;
  uint64_t l2_pending_hits = 0;
  uint64_t l2_misses = 0;
  uint64_t l2_line_misses = 0;
  uint64_t l2_sector_misses = 0;
  // Program lookups that reach L2, in R/W/A order. These describe modeled
  // residency, not NCU's write-hit convention or internal writeback traffic.
  std::array<CacheDirectionStats, 3> l2_by_direction{};
  uint64_t dram_requests = 0;
  uint64_t dram_load_requests = 0;
  uint64_t dram_store_requests = 0;
  uint64_t dram_load_sectors = 0;
  uint64_t dram_store_sectors = 0;
  uint64_t dram_load_bytes = 0;
  uint64_t dram_store_bytes = 0;
  uint64_t l2_writeback_events = 0;
  uint64_t l2_writeback_dirty_sectors = 0;
  uint64_t l2_dirty_drain_events = 0;
  uint64_t l2_dirty_drain_sectors = 0;
  uint64_t ldg_ltc128b_insts = 0;
  uint64_t ldgsts_insts = 0;
  uint64_t wide_load_insts = 0;
  uint64_t wide_store_insts = 0;
  uint64_t reads = 0;
  uint64_t writes = 0;
  uint64_t atomics = 0;
};

std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

bool starts_with(const std::string &s, const std::string &prefix) {
  return s.rfind(prefix, 0) == 0;
}


bool contains_token(const std::string &s, const std::string &needle) {
  return s.find(needle) != std::string::npos;
}

bool kernel_is_cublas_gemm(const std::string &name) {
  return contains_token(name, "gemm") || contains_token(name, "ampere_");
}

bool kernel_is_softmax(const std::string &name) {
  return contains_token(name, "soft_max");
}

bool kernel_is_large_matmul(const std::string &name) {
  return contains_token(name, "mul_mat_q") ||
         contains_token(name, "mul_mat_vec_q") ||
         contains_token(name, "mul_mat_f") ||
         contains_token(name, "mul_mat_vec_f");
}

std::vector<std::string> split_ws(const std::string &line) {
  std::istringstream ss(line);
  std::vector<std::string> tokens;
  std::string tok;
  while (ss >> tok)
    tokens.push_back(tok);
  return tokens;
}

std::vector<std::string> split_char(const std::string &s, char delim) {
  std::vector<std::string> out;
  std::string item;
  std::istringstream ss(s);
  while (std::getline(ss, item, delim))
    out.push_back(trim(item));
  return out;
}

uint64_t parse_u64(const std::string &token, int base = 0) {
  size_t parsed = 0;
  uint64_t v = std::stoull(token, &parsed, base);
  if (parsed != token.size())
    throw std::runtime_error("bad integer token: " + token);
  return v;
}

int64_t parse_i64(const std::string &token, int base = 0) {
  size_t parsed = 0;
  int64_t v = std::stoll(token, &parsed, base);
  if (parsed != token.size())
    throw std::runtime_error("bad signed integer token: " + token);
  return v;
}

uint64_t parse_hexish(const std::string &token) {
  if (starts_with(token, "0x") || starts_with(token, "0X"))
    return parse_u64(token, 0);
  return parse_u64(token, 16);
}

uint64_t div_up_u64(uint64_t n, uint64_t d) {
  if (d == 0)
    return 0;
  return (n + d - 1) / d;
}

uint64_t sectors_for_bytes(uint64_t bytes, unsigned sector_size) {
  return div_up_u64(bytes, std::max(1u, sector_size));
}

uint64_t parse_size_bytes(std::string token) {
  token = trim(token);
  if (token.empty())
    throw std::runtime_error("empty size token");
  if (!token.empty() && (token.back() == 'B' || token.back() == 'b'))
    token.pop_back();
  if (token.empty())
    throw std::runtime_error("bad size token");
  char suffix = 0;
  if (!std::isdigit(static_cast<unsigned char>(token.back()))) {
    suffix = static_cast<char>(std::toupper(static_cast<unsigned char>(token.back())));
    token.pop_back();
  }
  uint64_t value = parse_u64(token, 0);
  switch (suffix) {
  case 0:
    return value;
  case 'K':
    return value << 10;
  case 'M':
    return value << 20;
  case 'G':
    return value << 30;
  case 'T':
    return value << 40;
  default:
    throw std::runtime_error("unknown size suffix: " + std::string(1, suffix));
  }
}

std::string hex_string(uint64_t v) {
  std::ostringstream os;
  os << "0x" << std::hex << v;
  return os.str();
}

std::string csv_escape(const std::string &s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos)
    return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"')
      out += '"';
    out += c;
  }
  out += "\"";
  return out;
}

std::string shell_escape_field(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == ',' || c == '\n' || c == '\r')
      out.push_back('_');
    else
      out.push_back(c);
  }
  return out;
}

std::unordered_map<std::string, std::string> parse_semantic_tag_fields(
    const std::string &tag) {
  std::unordered_map<std::string, std::string> fields;
  for (const auto &item : split_char(tag, ';')) {
    const auto pos = item.find('=');
    if (pos == std::string::npos)
      continue;
    fields[item.substr(0, pos)] = item.substr(pos + 1);
  }
  return fields;
}

std::unordered_map<std::string, std::string>
read_dash_config(const fs::path &path) {
  std::unordered_map<std::string, std::string> entries;
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open config: " + path.string());
  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos)
      line = line.substr(0, hash);
    line = trim(line);
    if (line.empty() || line[0] != '-')
      continue;
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    std::string value;
    std::getline(ss, value);
    entries[key] = trim(value);
  }
  return entries;
}

std::string get_string(const std::unordered_map<std::string, std::string> &m,
                       const std::string &key,
                       const std::string &fallback = "") {
  auto it = m.find(key);
  if (it == m.end() || it->second.empty())
    return fallback;
  std::istringstream ss(it->second);
  std::string tok;
  ss >> tok;
  return tok.empty() ? fallback : tok;
}

unsigned get_uint(const std::unordered_map<std::string, std::string> &m,
                  const std::string &key, unsigned fallback) {
  auto it = m.find(key);
  if (it == m.end() || it->second.empty())
    return fallback;
  std::istringstream ss(it->second);
  std::string tok;
  ss >> tok;
  if (tok.empty())
    return fallback;
  return static_cast<unsigned>(parse_u64(tok, 0));
}

HwParams read_hw_params(const fs::path &path,
    std::shared_ptr<const hyfiss_request_trace::HardwareProfile> profile={}) {
  HwParams hw;
  hw.profile=profile?profile:hyfiss_request_trace::HardwareProfile::load(path.string());
  if(hw.profile) {
    const auto &p=*hw.profile;
    hw.num_sms=p.sms;hw.num_memory_channels=p.channels;
    hw.num_sub_partitions_per_channel=p.subpartitions;
    hw.num_partitions=p.channels*p.subpartitions;hw.num_banks=p.banks;
    // Only the r4 cache is constructed; these legacy L1 fields are not used.
    hw.l1_size_bytes=0;hw.l1_assoc=0;hw.l1_line_size=128;
    hw.l2_size_bytes=p.l2_bytes;hw.l2_assoc=p.l2_ways;hw.l2_line_size=128;
    hw.l2_set_index=parse_set_index_function(p.l2_index);hw.kernel_gap=p.kernel_gap;
    return hw;
  }
  auto cfg = read_dash_config(path);
  const unsigned clusters = get_uint(cfg, "-gpgpu_num_clusters", 0);
  const unsigned sms_per_cluster =
      get_uint(cfg, "-gpgpu_num_sms_per_cluster", 1);
  if (clusters > 0)
    hw.num_sms = clusters * std::max(1u, sms_per_cluster);

  hw.num_memory_channels = get_uint(
      cfg, "-gpgpu_num_memory_controllers",
      get_uint(cfg, "-gpgpu_n_mem", hw.num_memory_channels));
  hw.num_sub_partitions_per_channel = get_uint(
      cfg, "-gpgpu_num_sub_partition_per_memory_channel",
      get_uint(cfg, "-gpgpu_n_sub_partition_per_mchannel",
               hw.num_sub_partitions_per_channel));
  hw.num_memory_channels = std::max(1u, hw.num_memory_channels);
  hw.num_sub_partitions_per_channel =
      std::max(1u, hw.num_sub_partitions_per_channel);
  hw.num_partitions = hw.num_memory_channels * hw.num_sub_partitions_per_channel;
  hw.memory_partition_indexing =
      get_uint(cfg, "-gpgpu_memory_partition_indexing",
               hw.memory_partition_indexing);
  hw.mem_address_mask =
      get_uint(cfg, "-gpgpu_mem_address_mask", hw.mem_address_mask);
  hw.mem_addr_mapping = get_string(cfg, "-gpgpu_mem_addr_mapping", "");

  const unsigned l1_kb = get_uint(cfg, "-gpgpu_unified_l1d_size", 0);
  const unsigned shmem_kb = get_uint(cfg, "-gpgpu_shmem_size_per_sm", 0);
  if (l1_kb > shmem_kb)
    hw.l1_size_bytes = (l1_kb - shmem_kb) * 1024;
  const unsigned l1_sets = get_uint(cfg, "-gpgpu_l1d_cache_sets", 0);
  const unsigned l1_block = get_uint(cfg, "-gpgpu_l1d_cache_block_size", 0);
  const unsigned l1_assoc = get_uint(cfg, "-gpgpu_l1d_cache_associative", 0);
  if (l1_block)
    hw.l1_line_size = l1_block;
  if (l1_assoc)
    hw.l1_assoc = l1_assoc;
  if (l1_sets && l1_assoc && hw.l1_line_size)
    hw.l1_size_bytes = l1_sets * l1_assoc * hw.l1_line_size;
  hw.l1_set_index = parse_set_index_function(
      get_string(cfg, "-gpgpu_l1d_cache_set_index_function", "L"));

  const unsigned l2_kb_per_subpart =
      get_uint(cfg, "-gpgpu_l2d_size_per_sub_partition", 0);
  if (l2_kb_per_subpart)
    hw.l2_size_bytes =
        l2_kb_per_subpart * 1024 * std::max(1u, hw.num_partitions);
  const unsigned l2_sets = get_uint(cfg, "-gpgpu_l2d_cache_sets", 0);
  const unsigned l2_block = get_uint(cfg, "-gpgpu_l2d_cache_block_size", 0);
  const unsigned l2_assoc = get_uint(cfg, "-gpgpu_l2d_cache_associative", 0);
  if (l2_block)
    hw.l2_line_size = l2_block;
  if (l2_assoc)
    hw.l2_assoc = l2_assoc;
  if (l2_sets && l2_assoc && hw.l2_line_size)
    hw.l2_size_bytes =
        l2_sets * l2_assoc * hw.l2_line_size * std::max(1u, hw.num_partitions);
  hw.l2_set_index = parse_set_index_function(
      get_string(cfg, "-gpgpu_l2d_cache_set_index_function", "L"));
  hw.kernel_gap = get_uint(cfg, "-gpgpu_kernel_launch_latency", hw.kernel_gap);
  hw.l1_fill_latency = get_uint(
      cfg, "-gpgpu_l1_fill_latency",
      get_uint(cfg, "-gpgpu_l2_rop_latency",
               get_uint(cfg, "-gpgpu_l1_latency",
                        get_uint(cfg, "-gpgpu_l1_cache_access_latency", 0))));
  hw.l2_fill_latency = get_uint(
      cfg, "-gpgpu_dram_mem_access_latency",
      get_uint(cfg, "-dram_latency", get_uint(cfg, "-gpgpu_dram_latency", 0)));
  return hw;
}

std::map<int, KernelMeta>
read_app_config_stream(std::istream &in, bool include_local,
                       const std::set<int> *kernel_filter) {
  if (include_local && kernel_filter)
    throw std::runtime_error(
        "filtered app.config cannot assign the local owner namespace");
  std::map<int, KernelMeta> kernels;
  const int max_selected_kernel =
      kernel_filter && !kernel_filter->empty() ? *kernel_filter->rbegin() : 0;
  int previous_kernel = -1;
  bool have_previous_kernel = false;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    const std::string prefix = "-kernel_";
    if (!starts_with(line, prefix))
      continue;
    size_t pos = prefix.size();
    size_t end_id = pos;
    while (end_id < line.size() && std::isdigit(static_cast<unsigned char>(line[end_id])))
      ++end_id;
    if (end_id == pos || end_id >= line.size() || line[end_id] != '_')
      continue;
    const int kid = static_cast<int>(parse_u64(line.substr(pos, end_id - pos), 10));
    if (kernel_filter) {
      if (have_previous_kernel && kid < previous_kernel)
        throw std::runtime_error(
            "app.config kernel ids are not nondecreasing");
      previous_kernel = kid;
      have_previous_kernel = true;
      if (kid > max_selected_kernel)
        break;
      if (!kernel_filter->count(kid))
        continue;
    }
    const size_t key_begin = end_id + 1;
    const size_t key_end = line.find_first_of(" \t", key_begin);
    const std::string key = line.substr(key_begin, key_end - key_begin);
    const std::string value =
        key_end == std::string::npos ? "" : trim(line.substr(key_end + 1));
    auto &k = kernels[kid];
    k.id = kid;
    if (key == "kernel_name")
      k.name = value;
    else if (key == "grid_size" && !value.empty())
      k.grid_size = static_cast<unsigned>(parse_u64(value, 0));
    else if (key == "block_size" && !value.empty())
      k.block_size = static_cast<unsigned>(parse_u64(value, 0));
    else if (key == "local_base_addr" && !value.empty()) {
      k.local_base = parse_u64(value, 0);
      k.has_local_base = true;
    }
    else if (key == "llama_phase" && !value.empty())
      k.llm_phase = value;
    else if (key == "llama_semantic_file" && !value.empty())
      k.semantic_file = value;
  }
  // The local address model reserves a 26-bit warp-owner namespace.  Global-
  // only replay neither emits nor consumes those addresses, so large launch
  // sets must not be rejected by an unused namespace.
  if (!include_local)
    return kernels;
  uint64_t owner = 0;
  for (auto &entry : kernels) {
    auto &k = entry.second;
    k.local_warp_begin = owner;
    const uint64_t warps = (uint64_t(k.block_size) + 31) / 32;
    const uint64_t count = warps * k.grid_size;
    if (owner > (uint64_t(1) << 26) || count > (uint64_t(1) << 26) - owner)
      throw std::runtime_error("configured launches exceed local model namespace");
    owner += count;
  }
  return kernels;
}

std::map<int, KernelMeta>
read_app_config(const fs::path &path, bool include_local,
                const std::set<int> *kernel_filter = nullptr) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open app config: " + path.string());
  return read_app_config_stream(in, include_local, kernel_filter);
}

uint64_t map_key(int kid, unsigned block) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(kid)) << 32) | block;
}

std::unordered_map<uint64_t, IssueInfo>
read_issue_config_stream(std::istream &in,
                         const std::set<int> &kernel_filter) {
  std::unordered_map<uint64_t, IssueInfo> out;
  if (kernel_filter.empty())
    return out;
  const int max_selected_kernel = *kernel_filter.rbegin();
  const std::string prefix = "-trace_issued_sm_id_";
  std::string key;
  while (in >> key) {
    if (!starts_with(key, prefix)) {
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    const std::string smid_text = key.substr(prefix.size());
    if (smid_text.empty() ||
        !std::all_of(smid_text.begin(), smid_text.end(), [](unsigned char c) {
          return std::isdigit(c) != 0;
        })) {
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    const unsigned smid =
        static_cast<unsigned>(parse_u64(smid_text, 10));
    int previous_kernel = -1;
    bool have_previous_kernel = false;
    char c = '\0';
    while (in.get(c)) {
      if (c == '\n')
        break;
      if (c != '(')
        continue;
      std::string tuple;
      bool closed = false;
      while (in.get(c)) {
        if (c == ')') {
          closed = true;
          break;
        }
        if (c == '\n' || tuple.size() >= 256)
          throw std::runtime_error("malformed/oversized issue.config tuple");
        tuple.push_back(c);
      }
      if (!closed)
        throw std::runtime_error("unterminated issue.config tuple");
      auto parts = split_char(tuple, ',');
      if (parts.size() < 2)
        throw std::runtime_error("malformed issue.config tuple");
      const int kid = static_cast<int>(parse_u64(parts[0], 0));
      // The tracer appends CTA identities to each SM vector while the global
      // kernel id only advances at kernel completion.  Enforce that producer
      // contract for the parsed prefix, then bulk-skip the irrelevant suffix.
      if (have_previous_kernel && kid < previous_kernel)
        throw std::runtime_error(
            "issue.config kernel ids are not nondecreasing within an SM");
      previous_kernel = kid;
      have_previous_kernel = true;
      if (kid > max_selected_kernel) {
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        break;
      }
      if (!kernel_filter.count(kid))
        continue;
      const unsigned block = static_cast<unsigned>(parse_u64(parts[1], 0));
      IssueInfo info;
      info.sm_id = smid;
      if (parts.size() >= 3 && !parts[2].empty()) {
        info.cta_start = parse_hexish(parts[2]);
        info.has_cta_start = true;
      }
      out[map_key(kid, block)] = info;
    }
  }
  if (in.bad())
    throw std::runtime_error("issue.config read error");
  return out;
}

std::unordered_map<uint64_t, IssueInfo>
read_issue_config(const fs::path &path, const std::set<int> &kernel_filter) {
  std::ifstream in(path);
  if (!in)
    return {};
  return read_issue_config_stream(in, kernel_filter);
}

char classify_opcode(const std::string &op, bool include_local) {
  if (starts_with(op, "LDG"))
    return 'R';
  if (starts_with(op, "STG"))
    return 'W';
  if (starts_with(op, "ATOM") || starts_with(op, "ATOMG") ||
      starts_with(op, "RED"))
    return 'A';
  if (include_local && starts_with(op, "LDL"))
    return 'R';
  if (include_local && starts_with(op, "STL"))
    return 'W';
  return 'N';
}

unsigned infer_width(const std::string &op) {
  auto has = [&](const std::string &s) {
    return op.find(s) != std::string::npos;
  };
  // SASS cache hints such as LTC128B are not element widths. Only
  // width-qualified opcode tokens should decide the per-lane byte count.
  if (has(".U128") || has(".S128") || has(".B128") || has(".128"))
    return 16;
  if (has(".U64") || has(".S64") || has(".B64") || has(".64"))
    return 8;
  if (has(".U32") || has(".S32") || has(".B32") || has(".32"))
    return 4;
  if (has(".U16") || has(".S16") || has(".B16") || has(".16"))
    return 2;
  if (has(".U8") || has(".S8") || has(".B8") || has(".8"))
    return 1;
  return 4;
}

void append_addr_group(std::vector<LaneAddress> &lanes, uint32_t mask,
                       uint64_t base, const std::vector<int64_t> &strides,
                       unsigned ref_id) {
  uint64_t last = base;
  for (unsigned lane = 0; lane < kWarpSize; ++lane) {
    if (lane > 0 && lane - 1 < strides.size())
      last = static_cast<uint64_t>(static_cast<int64_t>(last) +
                                   strides[lane - 1]);
    if ((mask >> lane) & 1u)
      lanes.push_back(LaneAddress{lane, ref_id, last});
  }
}

MemoryInst parse_memory_line(const std::string &line, int kernel_id,
                             unsigned block_id, bool raw_has_block,
                             uint64_t seq, bool include_local) {
  auto t = split_ws(line);
  size_t i = 0;
  if (raw_has_block) {
    if (t.size() < 8)
      throw std::runtime_error("short raw memory line: " + line);
    block_id = static_cast<unsigned>(parse_hexish(t[i++]));
  } else if (t.size() < 7) {
    throw std::runtime_error("short split memory line: " + line);
  }

  MemoryInst inst;
  inst.kernel_id = kernel_id;
  inst.block_id = block_id;
  inst.seq = seq;
  inst.pc = parse_hexish(t[i++]);
  inst.opcode = t[i++];
  inst.mask = static_cast<uint32_t>(parse_hexish(t[i++]));
  inst.timestamp = parse_hexish(t[i++]);
  const unsigned groups = static_cast<unsigned>(parse_hexish(t[i++]));
  inst.mem_width = infer_width(inst.opcode);
  inst.op = classify_opcode(inst.opcode, include_local);
  if (inst.op == 'N')
    return inst;

  for (unsigned g = 1; g <= groups; ++g) {
    if (i + 1 >= t.size())
      throw std::runtime_error("missing address group in line: " + line);
    const uint64_t base = parse_hexish(t[i++]);
    const unsigned pair_count = static_cast<unsigned>(parse_hexish(t[i++]));
    std::vector<int64_t> strides;
    for (unsigned p = 0; p < pair_count; ++p) {
      if (i >= t.size())
        throw std::runtime_error("missing stride pair in line: " + line);
      const auto pos = t[i].find(':');
      if (pos == std::string::npos)
        throw std::runtime_error("bad stride pair: " + t[i]);
      const int64_t stride = parse_i64(t[i].substr(0, pos), 10);
      const unsigned count =
          static_cast<unsigned>(parse_u64(t[i].substr(pos + 1), 10));
      for (unsigned c = 0; c < count; ++c)
        strides.push_back(stride);
      ++i;
    }
    append_addr_group(inst.lanes, inst.mask, base, strides, g);
  }
  return inst;
}

enum class CacheResult {
  Hit,
  HitReserved,
  LineMiss,
  SectorMiss,
};

enum class CacheOperation { Read, Store, Atomic };

void account_l2_lookup(KernelStats &stats, char operation, CacheResult result) {
  const unsigned index = operation == 'R' ? 0 : operation == 'W' ? 1 :
                         operation == 'A' ? 2 : 3;
  if (index == 3 || (result != CacheResult::Hit &&
                    result != CacheResult::HitReserved &&
                    result != CacheResult::LineMiss &&
                    result != CacheResult::SectorMiss))
    throw std::runtime_error("invalid directional L2 lookup");
  auto &direction = stats.l2_by_direction[index];
  ++stats.l2_requests;
  ++direction.requests;
  if (result == CacheResult::Hit) {
    ++stats.l2_hits;
    ++direction.hits;
  } else if (result == CacheResult::HitReserved) {
    ++stats.l2_pending_hits;
    ++direction.pending_hits;
  } else {
    ++stats.l2_misses;
    ++direction.misses;
    if (result == CacheResult::LineMiss) {
      ++stats.l2_line_misses;
      ++direction.line_misses;
    } else {
      ++stats.l2_sector_misses;
      ++direction.sector_misses;
    }
  }
}

void write_l2_direction_header(std::ostream &out) {
  for (const char *direction : {"read", "write", "atomic"})
    for (const char *field : {"requests", "hits", "pending_hits", "misses",
                              "line_misses", "sector_misses"})
      out << ",l2_" << direction << '_' << field;
}

void write_l2_direction_stats(std::ostream &out, const KernelStats &stats) {
  CacheDirectionStats total;
  for (const auto &d : stats.l2_by_direction) {
    if (d.requests != d.hits + d.pending_hits + d.misses ||
        d.misses != d.line_misses + d.sector_misses)
      throw std::runtime_error("directional L2 lookup counts do not conserve");
    total.requests += d.requests;
    total.hits += d.hits;
    total.pending_hits += d.pending_hits;
    total.misses += d.misses;
    total.line_misses += d.line_misses;
    total.sector_misses += d.sector_misses;
  }
  if (total.requests != stats.l2_requests || total.hits != stats.l2_hits ||
      total.pending_hits != stats.l2_pending_hits || total.misses != stats.l2_misses ||
      total.line_misses != stats.l2_line_misses || total.sector_misses != stats.l2_sector_misses)
    throw std::runtime_error("directional and aggregate L2 counts differ");
  for (const auto &d : stats.l2_by_direction)
    out << ',' << d.requests << ',' << d.hits << ',' << d.pending_hits << ','
        << d.misses << ',' << d.line_misses << ',' << d.sector_misses;
}

CacheOperation cache_operation(char op) {
  if (op == 'R') return CacheOperation::Read;
  if (op == 'W') return CacheOperation::Store;
  if (op == 'A') return CacheOperation::Atomic;
  throw std::runtime_error("unsupported cache operation");
}

bool bypass_l1_read(const MemoryInst &inst) {
  // Observed for ld.global.cg on sm_89. Keep the match restricted to the
  // measured LDG opcode family; do not infer all generic/local policies.
  return inst.op == 'R' && starts_with(inst.opcode, "LDG.") &&
         (inst.opcode.size() >= 11 &&
          inst.opcode.compare(inst.opcode.size() - 11, 11, ".STRONG.GPU") == 0);
}

struct EvictedLine {
  bool present = false;
  uint32_t valid_sectors = 0;
  bool dirty = false;
  // Canonical cache-line base; dirty_sectors uses bits relative to this base.
  uint64_t addr = 0;
  uint32_t dirty_sectors = 0;
  // Sum of dirty sectors, not the span from first to last dirty byte.
  unsigned dirty_bytes = 0;
  // Diagnostic payload coverage only. An incomplete sector writeback is not
  // evidence that a preservation read has been issued or completed.
  uint32_t incomplete_dirty_sectors = 0;
  unsigned known_dirty_bytes = 0;
  unsigned missing_dirty_bytes = 0;
};

struct WritebackSpan {
  uint64_t addr = 0;
  unsigned size = 0;
  // Relative to this span's addr, also for every expanded RLE unit.
  uint32_t sector_mask = 0;
};

template <class Consume>
void for_each_writeback_span(const EvictedLine &evicted, unsigned sector_size,
                             unsigned line_size, Consume consume) {
  if (!evicted.dirty) {
    if (evicted.dirty_sectors || evicted.dirty_bytes)
      throw std::runtime_error("non-dirty writeback has a dirty payload");
    return;
  }
  if (!sector_size || !line_size || line_size % sector_size ||
      line_size / sector_size > 32 || evicted.addr % line_size ||
      evicted.addr > UINT64_MAX - (line_size - 1))
    throw std::runtime_error("invalid writeback geometry or line base");
  const unsigned sectors = line_size / sector_size;
  const uint32_t allowed = sectors == 32 ? UINT32_MAX : (uint32_t(1) << sectors) - 1;
  if (!evicted.dirty_sectors || (evicted.dirty_sectors & ~allowed) ||
      evicted.dirty_bytes != static_cast<unsigned>(__builtin_popcount(evicted.dirty_sectors)) * sector_size)
    throw std::runtime_error("invalid writeback sector mask or byte count");
  uint32_t remaining = evicted.dirty_sectors;
  while (remaining) {
    const unsigned first = static_cast<unsigned>(__builtin_ctz(remaining));
    unsigned end = first;
    uint32_t run = 0;
    while (end < sectors && (remaining & (uint32_t(1) << end))) {
      run |= uint32_t(1) << end;
      ++end;
    }
    consume(WritebackSpan{evicted.addr + uint64_t(first) * sector_size,
                         (end - first) * sector_size, run >> first});
    remaining &= ~run;
  }
}

struct CacheAccess {
  uint32_t valid_before=0, dirty_before=0, valid_after=0, dirty_after=0;
  CacheResult result = CacheResult::Hit;
  EvictedLine evicted;
};

struct CacheOccupancy {
  uint64_t allocated_lines=0, clean_lines=0, dirty_lines=0;
  uint64_t partial_dirty_lines=0, valid_sectors=0, reserved_sectors=0;
  uint64_t dirty_sectors=0, incomplete_dirty_sectors=0;
  uint64_t known_bytes=0, known_dirty_bytes=0, missing_dirty_bytes=0;
  std::vector<uint64_t> sets_by_allocated_lines, sets_by_dirty_lines;
  // Joint census of stored masks, including reserved-only tags with V=D=0.
  std::map<std::pair<uint32_t,uint32_t>,uint64_t> lines_by_valid_dirty_masks;
};

struct DiagnosticDirtyLimitResult {
  std::vector<EvictedLine> emitted;
  unsigned dirty_lines_before=0, dirty_lines_after=0;
  bool satisfied=false;
};

class SectorLruCache {
public:
  SectorLruCache() = default;
  SectorLruCache(uint64_t size_bytes, unsigned line_size, unsigned assoc,
                 SetIndexFunction set_index_function = SetIndexFunction::Linear) {
    reset(size_bytes, line_size, assoc, set_index_function);
  }

  void reset(uint64_t size_bytes, unsigned line_size, unsigned assoc,
             SetIndexFunction set_index_function = SetIndexFunction::Linear) {
    line_size_ = std::max(1u, line_size);
    assoc_ = std::max(1u, assoc);
    const uint64_t lines = std::max<uint64_t>(1, size_bytes / line_size_);
    sets_ = std::max<uint64_t>(1, lines / assoc_);
    line_shift_ = 0;
    for (unsigned v = line_size_; v > 1; v >>= 1)
      ++line_shift_;
    set_bits_ = 0;
    for (uint64_t v = sets_; v > 1; v >>= 1)
      ++set_bits_;
    set_index_function_ = set_index_function;
    sets_data_.assign(static_cast<size_t>(sets_), {});
  }

  void clear() {
    for (auto &s : sets_data_)
      s.clear();
  }

  std::vector<EvictedLine> drain_dirty(unsigned sector_size, uint64_t now) {
    std::vector<EvictedLine> evicted_lines;
    for (auto &dq : sets_data_) {
      for (auto &entry : dq) {
        refresh(entry, now);
        EvictedLine evicted;
        capture_eviction(entry, 0, sector_size, evicted);
        if (evicted.dirty)
          evicted_lines.push_back(evicted);
      }
      dq.clear();
    }
    return evicted_lines;
  }

  uint64_t dirty_sector_count() const {
    uint64_t count = 0;
    for (const auto &dq : sets_data_) {
      for (const auto &entry : dq)
        count += static_cast<unsigned>(__builtin_popcount(entry.dirty_sectors));
    }
    return count;
  }

  // Observe stored state without refreshing reservations or changing LRU order.
  // Allocated tags include partial sectors with no readable valid bit.
  CacheOccupancy occupancy() const {
    CacheOccupancy out;
    out.sets_by_allocated_lines.resize(assoc_+1);
    out.sets_by_dirty_lines.resize(assoc_+1);
    for (const auto &dq : sets_data_) {
      if (dq.size()>assoc_) throw std::runtime_error("cache occupancy exceeds associativity");
      ++out.sets_by_allocated_lines[dq.size()];
      unsigned dirty_lines=0;
      for (const auto &entry : dq) {
        ++out.allocated_lines;
        ++out.lines_by_valid_dirty_masks[{entry.valid_sectors,entry.dirty_sectors}];
        if (entry.dirty_sectors) { ++out.dirty_lines; ++dirty_lines; }
        else ++out.clean_lines;
        const uint32_t incomplete=entry.dirty_sectors & ~entry.valid_sectors;
        out.partial_dirty_lines+=incomplete!=0;
        out.valid_sectors+=__builtin_popcount(entry.valid_sectors);
        out.reserved_sectors+=__builtin_popcount(entry.reserved_sectors);
        out.dirty_sectors+=__builtin_popcount(entry.dirty_sectors);
        out.incomplete_dirty_sectors+=__builtin_popcount(incomplete);
        for (unsigned s=0;s<line_size_/32;++s) {
          const unsigned known=__builtin_popcount(entry.known_bytes[s]);
          if (bool(entry.valid_sectors & (1u<<s)) != (known==32))
            throw std::runtime_error("cache valid/known-byte inconsistency");
          out.known_bytes+=known;
          if (entry.dirty_sectors & (1u<<s)) {
            out.known_dirty_bytes+=known; out.missing_dirty_bytes+=32-known;
          }
        }
      }
      ++out.sets_by_dirty_lines[dirty_lines];
    }
    if (out.allocated_lines!=out.clean_lines+out.dirty_lines ||
        out.dirty_sectors*32!=out.known_dirty_bytes+out.missing_dirty_bytes)
      throw std::runtime_error("cache occupancy conservation failed");
    uint64_t joint_lines=0,joint_valid=0,joint_dirty=0,joint_clean=0,joint_partial=0;
    for(const auto &[m,n]:out.lines_by_valid_dirty_masks) {
      joint_lines+=n;joint_valid+=n*__builtin_popcount(m.first);
      joint_dirty+=n*__builtin_popcount(m.second);joint_clean+=n*(m.second==0);
      joint_partial+=n*((m.second & ~m.first)!=0);
    }
    if(joint_lines!=out.allocated_lines||joint_valid!=out.valid_sectors||
       joint_dirty!=out.dirty_sectors||joint_clean!=out.clean_lines||joint_partial!=out.partial_dirty_lines)
      throw std::runtime_error("cache joint valid/dirty mask census failed");
    return out;
  }

  std::vector<EvictedLine> drain_eligible_dirty(
      unsigned sector_size, uint64_t now, uint64_t min_dirty_age,
      uint64_t max_sectors) {
    std::vector<EvictedLine> drained;
    const unsigned sec_size = std::max(1u, sector_size);
    const unsigned sectors_per_line = std::min<unsigned>(
        32, (line_size_ + sec_size - 1) / sec_size);
    uint64_t remaining = max_sectors ? max_sectors : ~uint64_t{0};
    for (auto &dq : sets_data_) {
      if (remaining == 0)
        break;
      for (auto &entry : dq) {
        if (remaining == 0)
          break;
        refresh(entry, now);
        uint32_t eligible = 0;
        for (unsigned s = 0; s < sectors_per_line; ++s) {
          const uint32_t bit = 1u << s;
          if ((entry.dirty_sectors & bit) == 0)
            continue;
          // Retained cleaning emits a complete sector, with no preservation
          // read or byte-enable transfer modeled here. Unreadable dirty bytes
          // must retain their dirty obligation, even without a pending read.
          // A full overwrite can supply data despite an older pending fill.
          if (!(entry.valid_sectors & bit))
            continue;
          if (now >= entry.dirty_since[s] && now - entry.dirty_since[s] >= min_dirty_age)
            eligible |= bit;
        }
        unsigned s = 0;
        while (s < sectors_per_line && remaining > 0) {
          while (s < sectors_per_line && (eligible & (1u << s)) == 0)
            ++s;
          if (s >= sectors_per_line)
            break;
          const unsigned start = s;
          uint32_t run_mask = 0;
          while (s < sectors_per_line && remaining > 0 &&
                 (eligible & (1u << s))) {
            run_mask |= 1u << s;
            ++s;
            --remaining;
          }
          EvictedLine evicted;
          capture_dirty_run(entry, start, run_mask, sec_size, evicted);
          if (evicted.dirty) {
            drained.push_back(evicted);
            entry.dirty_sectors &= ~run_mask;
            for (unsigned b = 0; b < sectors_per_line; ++b) {
              if (run_mask & (1u << b))
                entry.dirty_since[b] = 0;
            }
          }
        }
      }
    }
    return drained;
  }

  // Explicit hypothesis experiment only; not called by the production stream.
  // The caller supplies this cache's index address, not the canonical tag.
  // Tag LRU already reflects both reads and writes. Cleaning neither changes
  // that order nor advances a pending fill. Partial dirty obligations survive.
  DiagnosticDirtyLimitResult diagnostic_clean_set_dirty_overflow(
      uint64_t index_addr, unsigned max_dirty_lines) {
    if (line_size_%32 || line_size_/32>32 || max_dirty_lines>assoc_)
      throw std::runtime_error("invalid diagnostic dirty-line budget");
    auto &dq=sets_data_.at(static_cast<size_t>(set_index(index_addr)));
    DiagnosticDirtyLimitResult out;
    for(const auto &entry:dq)out.dirty_lines_before+=entry.dirty_sectors!=0;
    out.dirty_lines_after=out.dirty_lines_before;
    for(auto it=dq.rbegin();it!=dq.rend()&&out.dirty_lines_after>max_dirty_lines;++it) {
      if(!it->dirty_sectors || (it->dirty_sectors & ~it->valid_sectors))continue;
      EvictedLine emission;
      capture_dirty_run(*it,0,it->dirty_sectors,32,emission);
      out.emitted.push_back(emission);
      for(unsigned s=0;s<line_size_/32;++s)
        if(it->dirty_sectors&(1u<<s))it->dirty_since[s]=0;
      it->dirty_sectors=0;
      --out.dirty_lines_after;
    }
    out.satisfied=out.dirty_lines_after<=max_dirty_lines;
    return out;
  }

  CacheAccess access(uint64_t addr, unsigned access_size, unsigned sector_size,
                     uint64_t now, uint64_t fill_latency,
                     CacheOperation operation = CacheOperation::Read,
                     bool mark_dirty = false, uint64_t index_addr = 0,
                     bool use_index_addr = false,
                     bool streaming_fill = false,
                     uint32_t byte_mask = UINT32_MAX) {
    if (sector_size != 32 || line_size_ % sector_size || line_size_ / sector_size > 32 ||
        !access_size || access_size > line_size_ ||
        addr % line_size_ > line_size_ - access_size ||
        addr > UINT64_MAX - (access_size - 1) || !byte_mask ||
        (byte_mask != UINT32_MAX && (addr % sector_size || access_size != sector_size)) ||
        (mark_dirty && operation == CacheOperation::Read) || fill_latency > UINT64_MAX - now)
      throw std::runtime_error("invalid byte-valid cache access");
    CacheAccess out;
    const uint64_t line = addr / line_size_;
    const uint64_t set = set_index(use_index_addr ? index_addr : addr);
    const uint64_t tag = line;
    const uint32_t sectors = sector_mask(addr, access_size, sector_size);
    auto &dq = sets_data_[static_cast<size_t>(set)];

    for (auto it = dq.begin(); it != dq.end(); ++it) {
      refresh(*it, now);
      if (it->tag == tag) {
        LineEntry entry = *it;
        out.valid_before=entry.valid_sectors; out.dirty_before=entry.dirty_sectors;
        dq.erase(it);
        // Accel-Sim's lazy-fetch policy accepts a store hitting MODIFIED
        // sectors even when they are not yet fully readable. Reads/RMW still
        // require the valid byte coverage or a pending fill.
        if ((entry.valid_sectors & sectors) == sectors ||
            (operation == CacheOperation::Store &&
             (entry.dirty_sectors & sectors) == sectors)) {
          out.result = CacheResult::Hit;
        } else if (((entry.valid_sectors | entry.reserved_sectors) & sectors) == sectors) {
          out.result = CacheResult::HitReserved;
        } else if (operation == CacheOperation::Store ||
                   (streaming_fill && operation == CacheOperation::Read)) {
          out.result = CacheResult::SectorMiss;
        } else {
          const uint32_t missing =
              sectors & ~(entry.valid_sectors | entry.reserved_sectors);
          reserve_or_fill(entry, missing, now, fill_latency);
          out.result = CacheResult::SectorMiss;
        }
        if (operation == CacheOperation::Store)
          mark_written_bytes(entry, addr, access_size, byte_mask);
        if (mark_dirty)
          mark_dirty_sectors(entry, sectors, now);
        out.valid_after=entry.valid_sectors; out.dirty_after=entry.dirty_sectors;
        if (streaming_fill)
          dq.push_back(entry);
        else
          dq.push_front(entry);
        return out;
      }
    }

    if (streaming_fill && operation == CacheOperation::Read) {
      out.result = CacheResult::LineMiss;
      return out;
    }

    LineEntry entry;
    entry.tag = tag;
    if (operation != CacheOperation::Store)
      reserve_or_fill(entry, sectors, now, fill_latency);
    if (operation == CacheOperation::Store)
      mark_written_bytes(entry, addr, access_size, byte_mask);
    if (mark_dirty)
      mark_dirty_sectors(entry, sectors, now);
    out.valid_after=entry.valid_sectors; out.dirty_after=entry.dirty_sectors;
    if (streaming_fill)
      dq.push_back(entry);
    else
      dq.push_front(entry);
    if (dq.size() > assoc_)
      evict_one(dq, set, sector_size, now, out.evicted);
    out.result = CacheResult::LineMiss;
    return out;
  }

  void save(std::ostream &os) const {
    const uint32_t magic = 0x534c5232u; // SLR2: byte coverage, independent pending reads
    write_pod(os, magic);
    write_pod(os, line_size_);
    write_pod(os, assoc_);
    write_pod(os, sets_);
    write_pod(os, line_shift_);
    write_pod(os, set_bits_);
    const uint32_t fn = static_cast<uint32_t>(set_index_function_);
    write_pod(os, fn);

    uint64_t non_empty_sets = 0;
    for (const auto &dq : sets_data_) {
      if (!dq.empty())
        ++non_empty_sets;
    }
    write_pod(os, non_empty_sets);

    for (uint64_t set = 0; set < sets_data_.size(); ++set) {
      const auto &dq = sets_data_[static_cast<size_t>(set)];
      if (dq.empty())
        continue;
      write_pod(os, set);
      const uint64_t entries = dq.size();
      write_pod(os, entries);
      for (const auto &entry : dq) {
        write_pod(os, entry.tag);
        write_pod(os, entry.valid_sectors);
        write_pod(os, entry.reserved_sectors);
        write_pod(os, entry.dirty_sectors);
        for (auto mask : entry.known_bytes) write_pod(os, mask);
        for (unsigned s = 0; s < 32; ++s) {
          if (entry.reserved_sectors & (1u << s))
            write_pod(os, entry.reserved_until[s]);
        }
        for (unsigned s = 0; s < 32; ++s) {
          if (entry.dirty_sectors & (1u << s))
            write_pod(os, entry.dirty_since[s]);
        }
      }
    }
  }

  void load(std::istream &is, unsigned sector_size = 32,
            const std::function<uint64_t(uint64_t)> &index_address = {}) {
    const uint32_t magic = read_pod<uint32_t>(is, "cache magic");
    if (magic != 0x534c5232u)
      throw std::runtime_error("bad SectorLruCache checkpoint magic");
    const unsigned line_size = read_pod<unsigned>(is, "line_size");
    const unsigned assoc = read_pod<unsigned>(is, "assoc");
    const uint64_t sets = read_pod<uint64_t>(is, "sets");
    const unsigned line_shift = read_pod<unsigned>(is, "line_shift");
    const unsigned set_bits = read_pod<unsigned>(is, "set_bits");
    const uint32_t fn = read_pod<uint32_t>(is, "set_index_function");
    if (line_size != line_size_ || assoc != assoc_ || sets != sets_ ||
        line_shift != line_shift_ || set_bits != set_bits_ ||
        fn != static_cast<uint32_t>(set_index_function_)) {
      throw std::runtime_error("checkpoint cache geometry does not match current config");
    }

    std::vector<std::deque<LineEntry>> restored(sets_data_.size());
    std::set<uint64_t> seen_sets, seen_tags;
    if (sector_size != 32 || line_size_ % sector_size || line_size_ / sector_size > 32)
      throw std::runtime_error("invalid checkpoint sector geometry");
    const unsigned sector_count = line_size_ / sector_size;
    const uint32_t allowed_mask = sector_count == 32 ? UINT32_MAX : (1u << sector_count) - 1;
    const uint64_t non_empty_sets = read_pod<uint64_t>(is, "non_empty_sets");
    if (non_empty_sets > sets_)
      throw std::runtime_error("checkpoint non-empty set count out of range");
    for (uint64_t i = 0; i < non_empty_sets; ++i) {
      const uint64_t set = read_pod<uint64_t>(is, "set index");
      const uint64_t entries = read_pod<uint64_t>(is, "set entry count");
      if (set >= sets_data_.size() || !seen_sets.insert(set).second || entries == 0 || entries > assoc_)
        throw std::runtime_error("checkpoint set index/count or associativity invalid");
      auto &dq = restored[static_cast<size_t>(set)];
      for (uint64_t e = 0; e < entries; ++e) {
        LineEntry entry;
        entry.tag = read_pod<uint64_t>(is, "tag");
        entry.valid_sectors = read_pod<uint32_t>(is, "valid sectors");
        entry.reserved_sectors = read_pod<uint32_t>(is, "reserved sectors");
        entry.dirty_sectors = read_pod<uint32_t>(is, "dirty sectors");
        uint32_t known_sectors = 0, complete_sectors = 0;
        for (unsigned s = 0; s < 32; ++s) {
          entry.known_bytes[s] = read_pod<uint32_t>(is, "known bytes");
          if (entry.known_bytes[s]) known_sectors |= uint32_t(1) << s;
          if (entry.known_bytes[s] == UINT32_MAX) complete_sectors |= uint32_t(1) << s;
        }
        if (entry.tag > UINT64_MAX / line_size_ || !seen_tags.insert(entry.tag).second ||
            ((known_sectors | entry.valid_sectors | entry.reserved_sectors | entry.dirty_sectors) & ~allowed_mask) ||
            !(known_sectors | entry.reserved_sectors) ||
            entry.valid_sectors != complete_sectors ||
            (entry.dirty_sectors & ~(known_sectors | entry.reserved_sectors)))
          throw std::runtime_error("checkpoint tag or sector masks invalid");
        const uint64_t address = entry.tag * line_size_;
        if (set_index(index_address ? index_address(address) : address) != set)
          throw std::runtime_error("checkpoint tag in wrong cache set");
        for (unsigned s = 0; s < 32; ++s) {
          if (entry.reserved_sectors & (1u << s))
            entry.reserved_until[s] = read_pod<uint64_t>(is, "reserved until");
        }
        for (unsigned s = 0; s < 32; ++s) {
          if (entry.dirty_sectors & (1u << s))
            entry.dirty_since[s] = read_pod<uint64_t>(is, "dirty since");
        }
        dq.push_back(entry);
      }
    }
    sets_data_.swap(restored);
  }

private:
  struct LineEntry {
    uint64_t tag = 0;
    uint32_t valid_sectors = 0;
    uint32_t reserved_sectors = 0;
    uint32_t dirty_sectors = 0;
    std::array<uint32_t, 32> known_bytes{};
    std::array<uint64_t, 32> reserved_until{};
    std::array<uint64_t, 32> dirty_since{};
  };

  void refresh(LineEntry &entry, uint64_t now) const {
    if (entry.reserved_sectors == 0)
      return;
    uint32_t ready = 0;
    for (unsigned s = 0; s < 32; ++s) {
      const uint32_t bit = 1u << s;
      if ((entry.reserved_sectors & bit) && now >= entry.reserved_until[s]) {
        ready |= bit;
        entry.known_bytes[s] = UINT32_MAX;
        entry.reserved_until[s] = 0;
      }
    }
    entry.reserved_sectors &= ~ready;
    entry.valid_sectors |= ready;
  }

  void reserve_or_fill(LineEntry &entry, uint32_t sectors, uint64_t now,
                       uint64_t fill_latency) const {
    if (fill_latency == 0) {
      entry.valid_sectors |= sectors;
      for (unsigned s = 0; s < 32; ++s)
        if (sectors & (uint32_t(1) << s)) entry.known_bytes[s] = UINT32_MAX;
      return;
    }
    entry.reserved_sectors |= sectors;
    const uint64_t ready_time = now + fill_latency;
    for (unsigned s = 0; s < 32; ++s) {
      if (sectors & (1u << s))
        entry.reserved_until[s] = std::max(entry.reserved_until[s], ready_time);
    }
  }

  void mark_written_bytes(LineEntry &entry, uint64_t addr, unsigned size,
                          uint32_t byte_mask) const {
    const unsigned first = addr % line_size_, end = first + size;
    for (unsigned s = first / 32; s <= (end - 1) / 32; ++s) {
      const unsigned lo = std::max(first, s * 32) - s * 32;
      const unsigned hi = std::min(end, (s + 1) * 32) - s * 32;
      const uint32_t covered = static_cast<uint32_t>(((uint64_t(1) << (hi - lo)) - 1) << lo);
      entry.known_bytes[s] |= covered & byte_mask;
      if (entry.known_bytes[s] == UINT32_MAX)
        entry.valid_sectors |= uint32_t(1) << s;
    }
    // A full overwrite can satisfy a later read, but an earlier memory read
    // remains in flight until its recorded completion; its traffic is retained.
  }

  void mark_dirty_sectors(LineEntry &entry, uint32_t sectors,
                          uint64_t now) const {
    const uint32_t newly_dirty = sectors & ~entry.dirty_sectors;
    entry.dirty_sectors |= sectors;
    for (unsigned s = 0; s < 32; ++s) {
      if (newly_dirty & (1u << s))
        entry.dirty_since[s] = now;
    }
  }

  void evict_one(std::deque<LineEntry> &dq, uint64_t set,
                 unsigned sector_size, uint64_t now, EvictedLine &evicted) const {
    auto evict_at = [&](std::deque<LineEntry>::reverse_iterator victim) {
      capture_eviction(*victim, set, sector_size, evicted);
      dq.erase(std::next(victim).base());
    };

    for (auto victim = dq.rbegin(); victim != dq.rend(); ++victim) {
      refresh(*victim, now);
      if (victim->reserved_sectors == 0) {
        evict_at(victim);
        return;
      }
    }
    capture_eviction(dq.back(), set, sector_size, evicted);
    dq.pop_back();
  }

  void capture_eviction(const LineEntry &entry, uint64_t set,
                        unsigned sector_size, EvictedLine &evicted) const {
    evicted.present=true; evicted.addr=entry.tag*line_size_;
    evicted.valid_sectors=entry.valid_sectors;
    if (entry.dirty_sectors == 0)
      return;
    if (entry.dirty_sectors & entry.reserved_sectors & ~entry.valid_sectors)
      throw std::runtime_error("cannot evict incomplete dirty data with an outstanding read; completion scheduling is required");
    const uint64_t line = entry.tag;
    (void)set;
    evicted.dirty = true;
    evicted.addr = line * line_size_;
    evicted.dirty_sectors = entry.dirty_sectors;
    evicted.dirty_bytes = dirty_size(entry.dirty_sectors, sector_size);
    capture_coverage(entry, evicted);
  }

  void capture_dirty_run(const LineEntry &entry, unsigned start_sector,
                         uint32_t run_mask, unsigned sector_size,
                         EvictedLine &evicted) const {
    if (run_mask == 0)
      return;
    evicted.dirty = true;
    (void)start_sector;
    evicted.addr = entry.tag * line_size_;
    evicted.dirty_sectors = run_mask;
    evicted.valid_sectors = entry.valid_sectors;
    evicted.dirty_bytes = dirty_size(run_mask, sector_size);
    capture_coverage(entry, evicted);
  }

  void capture_coverage(const LineEntry &entry, EvictedLine &evicted) const {
    evicted.incomplete_dirty_sectors=evicted.dirty_sectors & ~entry.valid_sectors;
    for (unsigned s=0;s<line_size_/32;++s) if (evicted.dirty_sectors & (1u<<s))
      evicted.known_dirty_bytes+=__builtin_popcount(entry.known_bytes[s]);
    evicted.missing_dirty_bytes=evicted.dirty_bytes-evicted.known_dirty_bytes;
  }

  unsigned dirty_size(uint32_t sectors, unsigned sector_size) const {
    unsigned bytes = 0;
    const unsigned sec_size = std::max(1u, sector_size);
    for (unsigned s = 0; s < 32; ++s) {
      if (sectors & (1u << s))
        bytes += sec_size;
    }
    return std::min(bytes, line_size_);
  }

  uint64_t set_index(uint64_t addr) const {
    const uint64_t block = addr >> line_shift_;
    const unsigned index = static_cast<unsigned>(block % sets_);
    const uint64_t higher_bits = block >> set_bits_;
    switch (set_index_function_) {
    case SetIndexFunction::Ipoly:
      return ipoly_hash(higher_bits, index);
    case SetIndexFunction::BitwiseXor:
      return (index ^ (higher_bits & (sets_ - 1))) % sets_;
    case SetIndexFunction::Fermi:
      return fermi_hash(addr);
    case SetIndexFunction::Linear:
    default:
      return index;
    }
  }

  unsigned line_size_log2() const {
    unsigned bits = 0;
    unsigned v = line_size_;
    while (v > 1) {
      ++bits;
      v >>= 1;
    }
    return bits;
  }

  unsigned set_bits() const {
    unsigned bits = 0;
    uint64_t v = sets_;
    while (v > 1) {
      ++bits;
      v >>= 1;
    }
    return bits;
  }

  uint64_t ipoly_hash(uint64_t higher_bits, unsigned index) const {
    if (sets_ == 16) {
      std::bitset<64> a(higher_bits);
      std::bitset<4> b(index);
      std::bitset<4> n(index);
      n[0] = a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[6] ^ a[4] ^ a[3] ^ a[0] ^ b[0];
      n[1] = a[12] ^ a[8] ^ a[7] ^ a[6] ^ a[5] ^ a[3] ^ a[1] ^ a[0] ^ b[1];
      n[2] = a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[4] ^ a[2] ^ a[1] ^ b[2];
      n[3] = a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[3] ^ a[2] ^ b[3];
      return n.to_ulong();
    }
    if (sets_ == 32) {
      std::bitset<64> a(higher_bits);
      std::bitset<5> b(index);
      std::bitset<5> n(index);
      n[0] = a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[6] ^ a[5] ^ a[3] ^ a[0] ^ b[0];
      n[1] = a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[6] ^ a[4] ^ a[1] ^ b[1];
      n[2] = a[14] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[3] ^ a[2] ^ a[0] ^ b[2];
      n[3] = a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[4] ^ a[3] ^ a[1] ^ b[3];
      n[4] = a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ a[2] ^ b[4];
      return n.to_ulong();
    }
    if (sets_ == 64) {
      std::bitset<64> a(higher_bits);
      std::bitset<6> b(index);
      std::bitset<6> n(index);
      n[0] = a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[12] ^ a[10] ^ a[6] ^ a[5] ^ a[0] ^ b[0];
      n[1] = a[15] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[5] ^ a[1] ^ a[0] ^ b[1];
      n[2] = a[16] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[8] ^ a[6] ^ a[2] ^ a[1] ^ b[2];
      n[3] = a[17] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[9] ^ a[7] ^ a[3] ^ a[2] ^ b[3];
      n[4] = a[18] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[10] ^ a[8] ^ a[4] ^ a[3] ^ b[4];
      n[5] = a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[11] ^ a[9] ^ a[5] ^ a[4] ^ b[5];
      return n.to_ulong();
    }
    return index;
  }

  uint64_t fermi_hash(uint64_t addr) const {
    if (sets_ != 32 && sets_ != 64)
      return (addr >> line_shift_) & (sets_ - 1);
    unsigned lower_xor = static_cast<unsigned>((addr >> line_shift_) & 0x1f);
    unsigned upper_xor = static_cast<unsigned>((addr & 0xe000) >> 13);
    upper_xor |= static_cast<unsigned>((addr & 0x20000) >> 14);
    upper_xor |= static_cast<unsigned>((addr & 0x80000) >> 15);
    unsigned idx = lower_xor ^ upper_xor;
    if (sets_ == 64)
      idx |= static_cast<unsigned>((addr & 0x1000) >> 7);
    return idx & (sets_ - 1);
  }

  uint32_t sector_mask(uint64_t addr, unsigned access_size,
                       unsigned sector_size) const {
    const unsigned sec_size = std::max(1u, sector_size);
    const uint64_t offset = addr % line_size_;
    const uint64_t end = offset + std::max(1u, access_size) - 1;
    const unsigned first = static_cast<unsigned>(offset / sec_size);
    const unsigned last = static_cast<unsigned>(std::min<uint64_t>(
        line_size_ - 1, end) / sec_size);
    uint32_t mask = 0;
    for (unsigned s = first; s <= last && s < 32; ++s)
      mask |= (1u << s);
    return mask ? mask : 1u;
  }

  unsigned line_size_ = 128;
  unsigned assoc_ = 4;
  uint64_t sets_ = 1;
  unsigned line_shift_ = 7;
  unsigned set_bits_ = 0;
  SetIndexFunction set_index_function_ = SetIndexFunction::Linear;
  std::vector<std::deque<LineEntry>> sets_data_;
};

fs::path checkpoint_path_for(const fs::path &dir, size_t position, int kernel_id) {
  std::ostringstream name;
  name << "checkpoint_pos" << std::setfill('0') << std::setw(8) << position
       << "_k" << std::setw(8) << kernel_id << ".bin";
  return dir / name.str();
}

unsigned dram_partition_index(uint64_t addr, const Options &opt);
uint64_t l2_cache_index_addr(uint64_t addr, const Options &opt);
#include "cache_checkpoint.h"

void assign_coalesced_sectors(const MemoryInst &inst, unsigned sector_size,
                              std::vector<SectorRequest> &reqs) {
  if (sector_size != 32)
    throw std::runtime_error("coalescer requires 32-byte sectors");
  reqs.clear();
  if (reqs.capacity() < inst.lanes.size() * 2)
    reqs.reserve(inst.lanes.size() * 2);
  auto add_range = [&](const LaneAddress &la, uint64_t begin, unsigned bytes) {
    if (bytes == 0 || begin > UINT64_MAX - (bytes - 1))
      throw std::runtime_error("memory access address overflow");
    const uint64_t first = begin / sector_size;
    const uint64_t last = (begin + bytes - 1) / sector_size;
    for (uint64_t sec = first; ; ++sec) {
      const uint64_t base = sec * sector_size;
      auto it = std::find_if(reqs.begin(), reqs.end(), [&](const SectorRequest &r) {
        return r.ref_id == la.ref_id && r.addr == base;
      });
      if (it == reqs.end()) {
        reqs.push_back(SectorRequest{base, sector_size, la.ref_id, 0, 0});
        it = std::prev(reqs.end());
      }
      if ((it->lane_mask & (1u << la.lane)) == 0) {
        it->lane_mask |= (1u << la.lane);
        ++it->lane_count;
      }
      const unsigned first_byte = static_cast<unsigned>(std::max(begin, base) - base);
      const unsigned last_byte = static_cast<unsigned>(
          std::min<uint64_t>(begin + bytes - 1 - base, sector_size - 1));
      const unsigned covered = last_byte - first_byte + 1;
      const uint32_t mask = covered == 32 ? UINT32_MAX : ((uint32_t(1) << covered) - 1) << first_byte;
      it->byte_mask |= mask;
      if (sec == last) break;
    }
  };
  for (const auto &la : inst.lanes) {
    if (la.lane >= kWarpSize) throw std::runtime_error("invalid lane id");
    if (la.is_local) {
      uint64_t offset = la.local_offset;
      unsigned remaining = std::max(1u, inst.mem_width);
      if (offset > UINT32_MAX || remaining - 1 > UINT32_MAX - offset)
        throw std::runtime_error("local access exceeds private offset range");
      while (remaining) {
        const unsigned n = std::min(remaining, 4u - unsigned(offset % 4));
        add_range(la, hyfiss_memc::local_model_address(inst.local_warp_owner,
                                                     la.lane, offset), n);
        remaining -= n;
        offset += n;
      }
    } else {
      if (inst.has_space_metadata &&
          (la.addr >= (uint64_t(1) << 63) ||
           std::max(1u, inst.mem_width) - 1 >= (uint64_t(1) << 63) - la.addr))
        throw std::runtime_error("global address overlaps reserved local model namespace");
      add_range(la, la.addr, std::max(1u, inst.mem_width));
    }
  }
  std::sort(reqs.begin(), reqs.end(), [](const SectorRequest &a,
                                         const SectorRequest &b) {
    return a.ref_id != b.ref_id ? a.ref_id < b.ref_id : a.addr < b.addr;
  });
}

std::vector<SectorRequest> coalesce_to_sectors(const MemoryInst &inst,
                                               unsigned sector_size) {
  std::vector<SectorRequest> reqs;
  assign_coalesced_sectors(inst, sector_size, reqs);
  return reqs;
}

void account_write_coverage(KernelStats &stats, const MemoryInst &inst,
                            const std::vector<SectorRequest> &requests) {
  if (inst.op != 'W') return;
  for (const auto &request : requests) {
    if (request.size != 32 || request.byte_mask == 0)
      throw std::runtime_error("invalid write sector byte coverage");
    if (request.byte_mask == UINT32_MAX) ++stats.write_full_sector_requests;
    else ++stats.write_partial_sector_requests;
    stats.write_covered_bytes += static_cast<unsigned>(__builtin_popcount(request.byte_mask));
  }
}

unsigned low_mask_index(uint64_t value, unsigned shift, unsigned count) {
  if (count == 0)
    return 0;
  return static_cast<unsigned>((value >> shift) % count);
}

unsigned bits_for_count(unsigned count) {
  unsigned bits = 0;
  unsigned v = count > 1 ? count - 1 : 0;
  while (v) {
    ++bits;
    v >>= 1;
  }
  return bits;
}

unsigned ilog2_u32(unsigned v) {
  unsigned r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

unsigned next_power_of_two(unsigned n) {
  if (n <= 1)
    return 1;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

uint64_t pack_bits(uint64_t mask, uint64_t val, unsigned high, unsigned low) {
  unsigned pos = 0;
  uint64_t result = 0;
  for (unsigned i = low; i < high; ++i) {
    const uint64_t bit = uint64_t{1} << i;
    if (mask & bit) {
      result |= ((val & bit) >> i) << pos;
      ++pos;
    }
  }
  return result;
}

void mask_limit(uint64_t mask, unsigned &high, unsigned &low) {
  high = 64;
  low = 0;
  bool seen = false;
  for (unsigned i = 0; i < 64; ++i) {
    if (mask & (uint64_t{1} << i)) {
      if (!seen) {
        low = i;
        seen = true;
      }
      high = i + 1;
    }
  }
}

uint64_t ipoly_hash_function(uint64_t higher_bits, unsigned index,
                             unsigned bank_set_num) {
  if (bank_set_num == 16) {
    std::bitset<64> a(higher_bits);
    std::bitset<4> b(index);
    std::bitset<4> n(index);
    n[0] = a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[6] ^ a[4] ^ a[3] ^ a[0] ^ b[0];
    n[1] = a[12] ^ a[8] ^ a[7] ^ a[6] ^ a[5] ^ a[3] ^ a[1] ^ a[0] ^ b[1];
    n[2] = a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[4] ^ a[2] ^ a[1] ^ b[2];
    n[3] = a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[3] ^ a[2] ^ b[3];
    return n.to_ulong();
  }
  if (bank_set_num == 32) {
    std::bitset<64> a(higher_bits);
    std::bitset<5> b(index);
    std::bitset<5> n(index);
    n[0] = a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[6] ^ a[5] ^ a[3] ^ a[0] ^ b[0];
    n[1] = a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[6] ^ a[4] ^ a[1] ^ b[1];
    n[2] = a[14] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[3] ^ a[2] ^ a[0] ^ b[2];
    n[3] = a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[4] ^ a[3] ^ a[1] ^ b[3];
    n[4] = a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ a[2] ^ b[4];
    return n.to_ulong();
  }
  if (bank_set_num == 64) {
    std::bitset<64> a(higher_bits);
    std::bitset<6> b(index);
    std::bitset<6> n(index);
    n[0] = a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[12] ^ a[10] ^ a[6] ^ a[5] ^ a[0] ^ b[0];
    n[1] = a[15] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[5] ^ a[1] ^ a[0] ^ b[1];
    n[2] = a[16] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[8] ^ a[6] ^ a[2] ^ a[1] ^ b[2];
    n[3] = a[17] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[9] ^ a[7] ^ a[3] ^ a[2] ^ b[3];
    n[4] = a[18] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[10] ^ a[8] ^ a[4] ^ a[3] ^ b[4];
    n[5] = a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[11] ^ a[9] ^ a[5] ^ a[4] ^ b[5];
    return n.to_ulong();
  }
  return index % std::max(1u, bank_set_num);
}

class AccelSimAddressMapping {
public:
  void init(const Options &opt) {
    n_channel_ = std::max(1u, opt.num_memory_channels);
    n_sub_partition_ = std::max(1u, opt.num_sub_partitions_per_channel);
    total_sub_partitions_ = n_channel_ * n_sub_partition_;
    memory_partition_indexing_ = opt.memory_partition_indexing;
    mem_address_mask_ = opt.mem_address_mask;
    mem_addr_mapping_ = opt.mem_addr_mapping;
    log2_channel_ = ilog2_u32(n_channel_);
    log2_sub_partition_ = ilog2_u32(n_sub_partition_);
    next_power2_channel_ = next_power_of_two(n_channel_);
    gap_ = n_channel_ != (1u << log2_channel_);
    init_masks();
    enabled_ = !mem_addr_mapping_.empty() || memory_partition_indexing_ != 0 ||
               mem_address_mask_ != 0;
    if (!enabled_ && (opt.partition_index_bit >= 64 ||
        (uint64_t{1} << opt.partition_index_bit) < opt.l2_line_size))
      throw std::runtime_error("fallback partition stride must contain a whole L2 line");
  }

  bool enabled() const { return enabled_; }

  unsigned sub_partition(uint64_t addr) const {
    if (!enabled_)
      return low_mask_index(addr, 8, std::max(1u, total_sub_partitions_));
    Decoded d = decode(addr);
    return d.sub_partition % std::max(1u, total_sub_partitions_);
  }

  uint64_t partition_address(uint64_t addr) const {
    if (!enabled_)
      return addr;
    if (!gap_)
      return pack_bits(~(mask_chip_ | sub_partition_id_mask_), addr, 64, 0);
    const uint64_t low_mask = (uint64_t{1} << addr_chip_s_) - 1;
    uint64_t partition_addr = ((addr >> addr_chip_s_) / n_channel_) << addr_chip_s_;
    partition_addr |= addr & low_mask;
    return pack_bits(~sub_partition_id_mask_, partition_addr, 64, 0);
  }

private:
  struct Decoded {
    unsigned chip = 0;
    unsigned bank = 0;
    unsigned sub_partition = 0;
  };

  void init_masks() {
    switch (mem_address_mask_) {
    case 1:
      addr_chip_s_ = 13;
      mask_chip_ = 0x0;
      mask_bank_ = 0x0000000000001800ULL;
      mask_row_ = 0x0000000007FFE000ULL;
      mask_col_ = 0x00000000000007FFULL;
      mask_burst_ = 0x0;
      break;
    default:
      addr_chip_s_ = 10;
      mask_chip_ = 0x0;
      mask_bank_ = 0x0000000000000300ULL;
      mask_row_ = 0x0000000007FFE000ULL;
      mask_col_ = 0x0000000000001CFFULL;
      mask_burst_ = 0x0;
      break;
    }
    if (!mem_addr_mapping_.empty())
      parse_mapping(mem_addr_mapping_);

    if (addr_chip_s_ >= 0 && !gap_) {
      const uint64_t low_mask = (uint64_t{1} << addr_chip_s_) - 1;
      const unsigned nchipbits = log2_channel_;
      mask_bank_ = ((mask_bank_ & ~low_mask) << nchipbits) |
                   (mask_bank_ & low_mask);
      mask_row_ = ((mask_row_ & ~low_mask) << nchipbits) |
                  (mask_row_ & low_mask);
      mask_col_ = ((mask_col_ & ~low_mask) << nchipbits) |
                  (mask_col_ & low_mask);
      mask_burst_ = ((mask_burst_ & ~low_mask) << nchipbits) |
                    (mask_burst_ & low_mask);
      for (int i = addr_chip_s_; i < addr_chip_s_ + static_cast<int>(nchipbits); ++i)
        mask_chip_ |= uint64_t{1} << i;
    }

    mask_limit(mask_chip_, high_chip_, low_chip_);
    mask_limit(mask_bank_, high_bank_, low_bank_);
    mask_limit(mask_row_, high_row_, low_row_);
    mask_limit(mask_col_, high_col_, low_col_);
    mask_limit(mask_burst_, high_burst_, low_burst_);

    sub_partition_id_mask_ = 0;
    if (n_sub_partition_ > 1) {
      const unsigned sub_bits = ilog2_u32(n_sub_partition_);
      unsigned pos = 0;
      for (unsigned i = low_bank_; i < high_bank_; ++i) {
        if (mask_bank_ & (uint64_t{1} << i)) {
          sub_partition_id_mask_ |= uint64_t{1} << i;
          if (++pos >= sub_bits)
            break;
        }
      }
    }
  }

  void parse_mapping(const std::string &mapping) {
    size_t semi = mapping.find(';');
    std::string head = semi == std::string::npos ? mapping : mapping.substr(0, semi);
    std::string body = semi == std::string::npos ? mapping : mapping.substr(semi + 1);
    if (head.rfind("dramid@", 0) == 0)
      addr_chip_s_ = static_cast<int>(parse_u64(head.substr(7), 10));
    else
      addr_chip_s_ = -1;
    mask_chip_ = mask_bank_ = mask_row_ = mask_col_ = mask_burst_ = 0;
    int ofs = 63;
    for (char c : body) {
      switch (c) {
      case 'D':
      case 'd':
        if (addr_chip_s_ != -1)
          break;
        mask_chip_ |= uint64_t{1} << ofs--;
        break;
      case 'B':
      case 'b':
        mask_bank_ |= uint64_t{1} << ofs--;
        break;
      case 'R':
      case 'r':
        mask_row_ |= uint64_t{1} << ofs--;
        break;
      case 'C':
      case 'c':
        mask_col_ |= uint64_t{1} << ofs--;
        break;
      case 'S':
      case 's':
        mask_burst_ |= uint64_t{1} << ofs;
        mask_col_ |= uint64_t{1} << ofs--;
        break;
      case '0':
        --ofs;
        break;
      case '|':
      case ' ':
      case '.':
        break;
      default:
        break;
      }
    }
  }

  Decoded decode(uint64_t addr) const {
    Decoded d;
    uint64_t rest_high_bits = 0;
    if (!gap_) {
      d.chip = static_cast<unsigned>(pack_bits(mask_chip_, addr, high_chip_, low_chip_));
      d.bank = static_cast<unsigned>(pack_bits(mask_bank_, addr, high_bank_, low_bank_));
      rest_high_bits = addr >> (addr_chip_s_ + log2_channel_ + log2_sub_partition_);
    } else {
      d.chip = static_cast<unsigned>((addr >> addr_chip_s_) % n_channel_);
      const uint64_t rest = ((addr >> addr_chip_s_) / n_channel_) << addr_chip_s_;
      d.bank = static_cast<unsigned>(pack_bits(mask_bank_, rest, high_bank_, low_bank_));
      rest_high_bits = (addr >> addr_chip_s_) / n_channel_;
    }

    const unsigned sub_mask = n_sub_partition_ - 1;
    unsigned sub = d.chip * n_sub_partition_ + (d.bank & sub_mask);
    if (memory_partition_indexing_ == 2) {
      sub = static_cast<unsigned>(ipoly_hash_function(
          rest_high_bits, sub, next_power2_channel_ * n_sub_partition_));
      if (gap_)
        sub %= total_sub_partitions_;
    } else if (memory_partition_indexing_ == 1) {
      sub ^= static_cast<unsigned>(rest_high_bits &
                                   (next_power2_channel_ * n_sub_partition_ - 1));
      sub %= total_sub_partitions_;
    }
    d.sub_partition = sub % std::max(1u, total_sub_partitions_);
    return d;
  }

  bool enabled_ = false;
  bool gap_ = false;
  int addr_chip_s_ = 10;
  unsigned n_channel_ = 1;
  unsigned n_sub_partition_ = 1;
  unsigned total_sub_partitions_ = 1;
  unsigned log2_channel_ = 0;
  unsigned log2_sub_partition_ = 0;
  unsigned next_power2_channel_ = 1;
  unsigned memory_partition_indexing_ = 0;
  unsigned mem_address_mask_ = 0;
  std::string mem_addr_mapping_;
  uint64_t mask_chip_ = 0;
  uint64_t mask_bank_ = 0;
  uint64_t mask_row_ = 0;
  uint64_t mask_col_ = 0;
  uint64_t mask_burst_ = 0;
  uint64_t sub_partition_id_mask_ = 0;
  unsigned high_chip_ = 64, low_chip_ = 0;
  unsigned high_bank_ = 64, low_bank_ = 0;
  unsigned high_row_ = 64, low_row_ = 0;
  unsigned high_col_ = 64, low_col_ = 0;
  unsigned high_burst_ = 64, low_burst_ = 0;
};

const AccelSimAddressMapping *g_addr_mapping = nullptr;

unsigned dram_partition_index(uint64_t addr, const Options &opt) {
  if (g_addr_mapping && g_addr_mapping->enabled())
    return g_addr_mapping->sub_partition(addr);
  return low_mask_index(addr, opt.partition_index_bit,
                        std::max(1u, opt.num_partitions));
}

uint64_t l2_cache_index_addr(uint64_t addr, const Options &opt) {
  if (g_addr_mapping && g_addr_mapping->enabled())
    return g_addr_mapping->partition_address(addr);
  // Pair the fallback partition remainder with its quotient coordinate.
  // A non-power-of-two partition count must not be approximated by bit removal.
  // Original addresses remain the cache tags and emitted request addresses.
  const unsigned bit = opt.partition_index_bit;
  if (bit >= 64 || (uint64_t{1} << bit) < opt.l2_line_size)
    throw std::runtime_error("fallback partition stride must contain a whole L2 line");
  const uint64_t low_mask = (uint64_t{1} << bit) - 1;
  return (((addr >> bit) / std::max(1u, opt.num_partitions)) << bit) |
         (addr & low_mask);
}

bool wants_accelsim_compact(const Options &opt);

class RotatingOutput {
public:
  RotatingOutput() = default;

  void open(const fs::path &base_path, std::string header,
            uint64_t rotate_bytes, bool binary = false) {
    base_path_ = base_path;
    header_ = std::move(header);
    rotate_bytes_ = rotate_bytes;
    binary_ = binary;
    part_ = 0;
    paths_.clear();
    open_current_part();
  }

  explicit operator bool() const { return out_.is_open() && out_.good(); }

  std::ofstream &stream() {
    if (pending_rotate_) {
      out_.flush();
      out_.close();
      ++part_;
      pending_rotate_ = false;
      open_current_part();
    }
    return out_;
  }

  void rotate_if_needed() {
    if (!out_.is_open() || rotate_bytes_ == 0)
      return;
    const std::streampos pos = out_.tellp();
    if (pos == std::streampos(-1))
      return;
    if (static_cast<uint64_t>(static_cast<std::streamoff>(pos)) <
        rotate_bytes_)
      return;
    pending_rotate_ = true;
  }

  const std::vector<fs::path> &paths() const { return paths_; }

private:
  fs::path current_path() const {
    if (part_ == 0)
      return base_path_;
    std::ostringstream name;
    name << base_path_.filename().string() << ".part"
         << std::setfill('0') << std::setw(6) << part_;
    return base_path_.parent_path() / name.str();
  }

  void open_current_part() {
    const fs::path path = current_path();
    auto mode = std::ios::out | std::ios::trunc;
    if (binary_)
      mode |= std::ios::binary;
    out_.open(path, mode);
    if (!out_)
      throw std::runtime_error("cannot create output file: " + path.string());
    paths_.push_back(path);
    if (!header_.empty())
      out_ << header_;
  }

  fs::path base_path_;
  std::string header_;
  uint64_t rotate_bytes_ = 0;
  unsigned part_ = 0;
  bool pending_rotate_ = false;
  bool binary_ = false;
  std::ofstream out_;
  std::vector<fs::path> paths_;
};

std::string join_paths(const std::vector<fs::path> &paths) {
  std::ostringstream os;
  for (size_t i = 0; i < paths.size(); ++i) {
    if (i)
      os << ',';
    os << paths[i];
  }
  return os.str();
}

struct SemanticInfo {
  unsigned id = 0;
  std::string tag = "unknown";
  std::string kind = "unknown";
  std::string tensor = "unknown";
  std::string layer = "-1";
  std::string device = "unknown";
  std::string usage = "unknown";
};

class SemanticDatabase {
public:
  void load(const fs::path &path) {
    if (path.empty())
      return;
    source_path_ = path;
    std::ifstream in(path);
    if (!in)
      throw std::runtime_error("cannot open semantic file: " + path.string());

    std::vector<Range> ranges;
    std::string record_type;
    std::string line;
    while (in >> record_type) {
      if (record_type == "RANGE") {
        std::string start_s;
        std::string end_s;
        std::string tag;
        in >> start_s >> end_s >> tag;
        const uint64_t start = parse_u64(start_s, 0);
        const uint64_t end = parse_u64(end_s, 0);
        if (start < end && !tag.empty()) {
          const unsigned id = intern_tag(tag);
          ranges.push_back(Range{start, end, end - start, id});
        }
      }
      std::getline(in, line);
    }
    build_segments(ranges);
  }

  const SemanticInfo &lookup(uint64_t addr) const {
    if (segments_.empty())
      return unknown_;
    auto it = std::upper_bound(
        segments_.begin(), segments_.end(), addr,
        [](uint64_t value, const Segment &seg) { return value < seg.start; });
    if (it == segments_.begin())
      return unknown_;
    --it;
    if (addr >= it->start && addr < it->end && it->id < infos_.size())
      return infos_[it->id];
    return unknown_;
  }

  bool enabled() const { return !segments_.empty(); }
  const fs::path &source_path() const { return source_path_; }

  void write_dictionary(const fs::path &output_dir) const {
    if (infos_.size() <= 1)
      return;
    std::ofstream out(output_dir / "semantic_tags.csv");
    out << "semantic_id,kind,layer,tensor,device,usage,tag\n";
    for (size_t i = 1; i < infos_.size(); ++i) {
      const auto &s = infos_[i];
      out << s.id << ',' << csv_escape(s.kind) << ',' << csv_escape(s.layer)
          << ',' << csv_escape(s.tensor) << ',' << csv_escape(s.device) << ','
          << csv_escape(s.usage) << ',' << csv_escape(s.tag) << '\n';
    }
  }

private:
  struct Range {
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t size = 0;
    unsigned id = 0;
  };
  struct Segment {
    uint64_t start = 0;
    uint64_t end = 0;
    unsigned id = 0;
  };

  unsigned intern_tag(const std::string &tag) {
    auto it = id_by_tag_.find(tag);
    if (it != id_by_tag_.end())
      return it->second;
    SemanticInfo info;
    info.id = static_cast<unsigned>(infos_.size());
    info.tag = tag;
    const auto fields = parse_semantic_tag_fields(tag);
    auto get = [&](const std::string &key, const std::string &fallback) {
      auto fit = fields.find(key);
      return fit == fields.end() || fit->second.empty() ? fallback : fit->second;
    };
    info.kind = get("kind", "unknown");
    info.tensor = get("tensor", "unknown");
    info.layer = get("layer", "-1");
    info.device = get("device", "unknown");
    info.usage = get("usage", "unknown");
    infos_.push_back(info);
    id_by_tag_[tag] = info.id;
    return info.id;
  }

  void build_segments(const std::vector<Range> &ranges) {
    struct Event {
      uint64_t addr = 0;
      bool start = false;
      size_t index = 0;
    };
    std::vector<Event> events;
    events.reserve(ranges.size() * 2);
    for (size_t i = 0; i < ranges.size(); ++i) {
      events.push_back(Event{ranges[i].start, true, i});
      events.push_back(Event{ranges[i].end, false, i});
    }
    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
      if (a.addr != b.addr)
        return a.addr < b.addr;
      return a.start < b.start;
    });

    std::set<std::pair<uint64_t, size_t>> active;
    uint64_t cursor = 0;
    bool have_cursor = false;
    size_t i = 0;
    while (i < events.size()) {
      const uint64_t addr = events[i].addr;
      if (have_cursor && cursor < addr && !active.empty()) {
        const size_t best = active.begin()->second;
        segments_.push_back(Segment{cursor, addr, ranges[best].id});
      }
      while (i < events.size() && events[i].addr == addr) {
        const auto &r = ranges[events[i].index];
        const auto key = std::make_pair(r.size, events[i].index);
        if (events[i].start)
          active.insert(key);
        else
          active.erase(key);
        ++i;
      }
      cursor = addr;
      have_cursor = true;
    }
  }

  fs::path source_path_;
  SemanticInfo unknown_;
  std::vector<SemanticInfo> infos_ = {unknown_};
  std::unordered_map<std::string, unsigned> id_by_tag_;
  std::vector<Segment> segments_;
};

struct SemanticTrafficCounter {
  uint64_t source_read_bytes = 0;
  uint64_t source_write_bytes = 0;
  uint64_t source_atomic_bytes = 0;
  uint64_t l1_lookup_bytes = 0;
  uint64_t l1_hit_bytes = 0;
  uint64_t l1_pending_hit_bytes = 0;
  uint64_t l1_miss_bytes = 0;
  uint64_t l1_bypass_bytes = 0;
  uint64_t l2_lookup_bytes = 0;
  uint64_t l2_hit_bytes = 0;
  uint64_t l2_pending_hit_bytes = 0;
  uint64_t l2_miss_bytes = 0;
  uint64_t dram_read_bytes = 0;
  uint64_t dram_write_bytes = 0;
};

// Compact in-memory accounting only. It never emits request addresses. Dirty
// sectors retain their last modeled producer kernel and semantic id so delayed
// L2 writeback can be separated into producer, eviction trigger, and service
// scopes without attributing the bytes to the later kernel by accident.
class SemanticTrafficLedger {
public:
  void add_source(int kernel, const std::string &phase,
                  const SemanticInfo &sem,
                  char op, uint64_t bytes) {
    auto &row = rows_[key(kernel, phase, sem)];
    if (op == 'R') row.source_read_bytes += bytes;
    else if (op == 'W') row.source_write_bytes += bytes;
    else if (op == 'A') row.source_atomic_bytes += bytes;
  }

  void add_cache(int kernel, const std::string &phase,
                 const SemanticInfo &sem, bool l1_lookup,
                 CacheResult l1_result, bool l2_lookup,
                 CacheResult l2_result, uint64_t bytes) {
    auto &row = rows_[key(kernel, phase, sem)];
    if (l1_lookup) {
      row.l1_lookup_bytes += bytes;
      if (l1_result == CacheResult::Hit) row.l1_hit_bytes += bytes;
      else if (l1_result == CacheResult::HitReserved)
        row.l1_pending_hit_bytes += bytes;
      else row.l1_miss_bytes += bytes;
    } else {
      row.l1_bypass_bytes += bytes;
    }
    if (l2_lookup) {
      row.l2_lookup_bytes += bytes;
      if (l2_result == CacheResult::Hit) row.l2_hit_bytes += bytes;
      else if (l2_result == CacheResult::HitReserved)
        row.l2_pending_hit_bytes += bytes;
      else row.l2_miss_bytes += bytes;
    }
  }

  void add_dram_read(int kernel, const std::string &phase,
                     const SemanticInfo &sem,
                     uint64_t bytes) {
    rows_[key(kernel, phase, sem)].dram_read_bytes += bytes;
  }

  void add_dram_write(int kernel, const std::string &phase,
                      const SemanticInfo &sem,
                      uint64_t bytes) {
    rows_[key(kernel, phase, sem)].dram_write_bytes += bytes;
    direct_dram_write_bytes_ += bytes;
  }

  void mark_dirty(int kernel, const std::string &phase,
                  const SemanticInfo &sem, uint64_t addr,
                  uint64_t bytes, unsigned sector_size) {
    if (!sector_size || addr % sector_size || bytes % sector_size)
      throw std::runtime_error("dirty producer geometry is not sector aligned");
    register_kernel(kernel, phase, "");
    semantic_kinds_[sem.id] = sem.kind.empty() ? "unknown" : sem.kind;
    const DirtyOwner replacement{kernel, sem.id};
    for (uint64_t offset = 0; offset < bytes; offset += sector_size) {
      const uint64_t sector = addr + offset;
      auto found = dirty_owners_.find(sector);
      if (found != dirty_owners_.end()) {
        xor_owner(sector, found->second);
        found->second = replacement;
        xor_owner(sector, found->second);
      } else {
        dirty_owners_.emplace(sector, replacement);
        xor_owner(sector, replacement);
      }
    }
  }

  void add_dram_writeback(int service_kernel,
                          const std::string &service_phase,
                          const SemanticInfo &static_address_sem,
                          const SemanticInfo *trigger_sem,
                          const WritebackSpan &span,
                          unsigned sector_size) {
    if (!sector_size || span.addr % sector_size || span.size % sector_size)
      throw std::runtime_error("writeback producer geometry is not sector aligned");
    const SemanticInfo unknown;
    const SemanticInfo &trigger = trigger_sem ? *trigger_sem : unknown;
    register_kernel(service_kernel, service_phase, "");
    semantic_kinds_[trigger.id] = trigger.kind.empty() ? "unknown" : trigger.kind;
    semantic_kinds_[static_address_sem.id] =
        static_address_sem.kind.empty() ? "unknown" : static_address_sem.kind;
    for (uint64_t offset = 0; offset < span.size; offset += sector_size) {
      const uint64_t sector = span.addr + offset;
      auto found = dirty_owners_.find(sector);
      if (found == dirty_owners_.end())
        throw std::runtime_error("writeback sector has no retained dirty producer");
      const DirtyOwner owner = found->second;
      const SemanticInfo producer_sem = semantic_from_id(owner.semantic_id);
      rows_[key(owner.kernel, kernel_phase(owner.kernel), producer_sem)];
      rows_[key(service_kernel, service_phase, producer_sem)].dram_write_bytes +=
          sector_size;
      WritebackKey writeback_key{owner.kernel, owner.semantic_id,
                                 service_kernel, trigger.id,
                                 service_kernel};
      writeback_rows_[writeback_key] += sector_size;
      if (owner.semantic_id != static_address_sem.id)
        static_address_owner_mismatch_bytes_ += sector_size;
      xor_owner(sector, owner);
      dirty_owners_.erase(found);
      writeback_dram_write_bytes_ += sector_size;
    }
  }

  void begin_kernel(int kernel, const std::string &name,
                    const std::string &phase, uint64_t resident_dirty) {
    register_kernel(kernel, phase, name);
    check_dirty_count(resident_dirty, "kernel entry");
    boundaries_.push_back(CacheBoundary{kernel, name, phase, resident_dirty,
                                        digest_a_, digest_b_, 0, 0, 0, false});
  }

  void end_kernel(int kernel, uint64_t resident_dirty) {
    if (boundaries_.empty() || boundaries_.back().kernel != kernel ||
        boundaries_.back().has_exit)
      throw std::runtime_error("semantic cache-state kernel boundary mismatch");
    check_dirty_count(resident_dirty, "kernel exit");
    auto &boundary = boundaries_.back();
    boundary.exit_dirty = resident_dirty;
    boundary.exit_digest_a = digest_a_;
    boundary.exit_digest_b = digest_b_;
    boundary.has_exit = true;
  }

  void discard_dirty_without_service() {
    dirty_owners_.clear();
    digest_a_ = 0;
    digest_b_ = 0;
  }

  void write(const fs::path &output_dir,
             const std::map<int, KernelStats> &stats) const {
    SemanticTrafficCounter ledger_total;
    for (const auto &entry : rows_) add(ledger_total, entry.second);
    uint64_t stats_read = 0, stats_write = 0;
    uint64_t stats_source_read = 0, stats_source_write = 0;
    uint64_t stats_source_atomic = 0;
    uint64_t stats_l1_lookup = 0, stats_l1_hit = 0;
    uint64_t stats_l1_pending = 0, stats_l1_miss = 0;
    uint64_t stats_l2_lookup = 0, stats_l2_hit = 0;
    uint64_t stats_l2_pending = 0, stats_l2_miss = 0;
    for (const auto &entry : stats) {
      stats_read += entry.second.dram_load_bytes;
      stats_write += entry.second.dram_store_bytes;
      stats_source_read += entry.second.read_sector_requests * 32;
      stats_source_write += entry.second.write_sector_requests * 32;
      stats_source_atomic += entry.second.atomic_sector_requests * 32;
      stats_l1_lookup += entry.second.l1_requests * 32;
      stats_l1_hit += entry.second.l1_hits * 32;
      stats_l1_pending += entry.second.l1_pending_hits * 32;
      stats_l1_miss += entry.second.l1_misses * 32;
      stats_l2_lookup += entry.second.l2_requests * 32;
      stats_l2_hit += entry.second.l2_hits * 32;
      stats_l2_pending += entry.second.l2_pending_hits * 32;
      stats_l2_miss += entry.second.l2_misses * 32;
    }
    const uint64_t ledger_source = ledger_total.source_read_bytes +
        ledger_total.source_write_bytes + ledger_total.source_atomic_bytes;
    const uint64_t stats_source = stats_source_read + stats_source_write +
        stats_source_atomic;
    if (ledger_total.dram_read_bytes != stats_read ||
        ledger_total.dram_write_bytes != stats_write ||
        ledger_total.source_read_bytes != stats_source_read ||
        ledger_total.source_write_bytes != stats_source_write ||
        ledger_total.source_atomic_bytes != stats_source_atomic ||
        ledger_source != stats_source ||
        ledger_total.l1_lookup_bytes != stats_l1_lookup ||
        ledger_total.l1_hit_bytes != stats_l1_hit ||
        ledger_total.l1_pending_hit_bytes != stats_l1_pending ||
        ledger_total.l1_miss_bytes != stats_l1_miss ||
        ledger_total.l1_lookup_bytes + ledger_total.l1_bypass_bytes != stats_source ||
        ledger_total.l2_lookup_bytes != stats_l2_lookup ||
        ledger_total.l2_hit_bytes != stats_l2_hit ||
        ledger_total.l2_pending_hit_bytes != stats_l2_pending ||
        ledger_total.l2_miss_bytes != stats_l2_miss ||
        writeback_dram_write_bytes_ + direct_dram_write_bytes_ != stats_write)
      throw std::runtime_error("semantic traffic ledger does not conserve backend totals");

    std::ofstream csv(output_dir / "semantic_traffic.csv");
    csv << "kernel_id,service_phase,semantic_id,semantic_kind,source_read_bytes,"
           "source_write_bytes,source_atomic_bytes,l1_lookup_bytes,l1_hit_bytes,"
           "l1_pending_hit_bytes,l1_miss_bytes,l1_bypass_bytes,l2_lookup_bytes,"
           "l2_hit_bytes,l2_pending_hit_bytes,l2_miss_bytes,dram_read_bytes,"
           "dram_write_bytes\n";
    std::vector<uint64_t> ordered_keys;
    ordered_keys.reserve(rows_.size());
    for (const auto &entry : rows_) ordered_keys.push_back(entry.first);
    std::sort(ordered_keys.begin(), ordered_keys.end());
    for (uint64_t packed : ordered_keys) {
      const auto &c = rows_.at(packed);
      const int kernel = static_cast<int>(packed >> 32);
      const unsigned semantic_id = static_cast<unsigned>(packed);
      csv << kernel << ',' << csv_escape(kernel_phase(kernel)) << ','
          << semantic_id << ',' << csv_escape(semantic_kind(semantic_id)) << ','
          << c.source_read_bytes << ',' << c.source_write_bytes << ','
          << c.source_atomic_bytes << ',' << c.l1_lookup_bytes << ','
          << c.l1_hit_bytes << ',' << c.l1_pending_hit_bytes << ','
          << c.l1_miss_bytes << ',' << c.l1_bypass_bytes << ','
          << c.l2_lookup_bytes << ',' << c.l2_hit_bytes << ','
          << c.l2_pending_hit_bytes << ',' << c.l2_miss_bytes << ','
          << c.dram_read_bytes << ',' << c.dram_write_bytes << '\n';
    }
    if (!csv) throw std::runtime_error("failed writing semantic_traffic.csv");

    std::ofstream writeback_csv(output_dir / "semantic_writeback.csv");
    writeback_csv << "producer_kernel_id,producer_phase,producer_semantic_id,"
                     "producer_semantic_kind,trigger_kernel_id,trigger_phase,"
                     "trigger_semantic_id,trigger_semantic_kind,service_kernel_id,"
                     "service_phase,dram_write_bytes\n";
    for (const auto &entry : writeback_rows_) {
      const auto &[producer_kernel, producer_semantic, trigger_kernel,
                   trigger_semantic, service_kernel] = entry.first;
      writeback_csv << producer_kernel << ','
                    << csv_escape(kernel_phase(producer_kernel)) << ','
                    << producer_semantic << ','
                    << csv_escape(semantic_kind(producer_semantic)) << ','
                    << trigger_kernel << ','
                    << csv_escape(kernel_phase(trigger_kernel)) << ','
                    << trigger_semantic << ','
                    << csv_escape(semantic_kind(trigger_semantic)) << ','
                    << service_kernel << ','
                    << csv_escape(kernel_phase(service_kernel)) << ','
                    << entry.second << '\n';
    }
    if (!writeback_csv)
      throw std::runtime_error("failed writing semantic_writeback.csv");

    std::ofstream state_csv(output_dir / "semantic_cache_state.csv");
    state_csv << "kernel_id,kernel_name,phase,entry_resident_dirty_sectors,"
                 "entry_state_digest_a,entry_state_digest_b,"
                 "exit_resident_dirty_sectors,exit_state_digest_a,"
                 "exit_state_digest_b\n";
    for (const auto &boundary : boundaries_) {
      if (!boundary.has_exit)
        throw std::runtime_error("semantic cache-state boundary has no exit");
      state_csv << boundary.kernel << ',' << csv_escape(boundary.name) << ','
                << csv_escape(boundary.phase) << ',' << boundary.entry_dirty
                << ',' << hex_string(boundary.entry_digest_a) << ','
                << hex_string(boundary.entry_digest_b) << ','
                << boundary.exit_dirty << ','
                << hex_string(boundary.exit_digest_a) << ','
                << hex_string(boundary.exit_digest_b) << '\n';
    }
    if (!state_csv)
      throw std::runtime_error("failed writing semantic_cache_state.csv");

    SemanticTrafficCounter unknown;
    for (const auto &entry : rows_)
      if (semantic_kind(static_cast<unsigned>(entry.first)) == "unknown")
        add(unknown, entry.second);
    std::ofstream receipt(output_dir / "semantic_conservation.json");
    receipt << "{\n"
            << "  \"schema\": \"hyfiss_semantic_traffic_v1\",\n"
            << "  \"status\": \"PASS_EXACT_INTERNAL_CONSERVATION\",\n"
            << "  \"attribution_basis\": \"source and read service use request kernel plus static address range; delayed writeback uses retained last dirty producer semantic id plus explicit trigger and service kernels\",\n"
            << "  \"source_bytes\": " << ledger_source << ",\n"
            << "  \"backend_source_bytes\": " << stats_source << ",\n"
            << "  \"source_read_bytes\": " << ledger_total.source_read_bytes << ",\n"
            << "  \"backend_source_read_bytes\": " << stats_source_read << ",\n"
            << "  \"source_write_bytes\": " << ledger_total.source_write_bytes << ",\n"
            << "  \"backend_source_write_bytes\": " << stats_source_write << ",\n"
            << "  \"source_atomic_bytes\": " << ledger_total.source_atomic_bytes << ",\n"
            << "  \"backend_source_atomic_bytes\": " << stats_source_atomic << ",\n"
            << "  \"dram_read_bytes\": " << ledger_total.dram_read_bytes << ",\n"
            << "  \"backend_dram_read_bytes\": " << stats_read << ",\n"
            << "  \"dram_write_bytes\": " << ledger_total.dram_write_bytes << ",\n"
            << "  \"backend_dram_write_bytes\": " << stats_write << ",\n"
            << "  \"writeback_dram_write_bytes\": " << writeback_dram_write_bytes_ << ",\n"
            << "  \"direct_dram_write_bytes\": " << direct_dram_write_bytes_ << ",\n"
            << "  \"static_address_owner_mismatch_bytes\": " << static_address_owner_mismatch_bytes_ << ",\n"
            << "  \"final_resident_dirty_sectors\": " << dirty_owners_.size() << ",\n"
            << "  \"final_resident_dirty_state_digest_a\": \"" << hex_string(digest_a_) << "\",\n"
            << "  \"final_resident_dirty_state_digest_b\": \"" << hex_string(digest_b_) << "\",\n"
            << "  \"l1_lookup_bytes\": " << ledger_total.l1_lookup_bytes << ",\n"
            << "  \"backend_l1_lookup_bytes\": " << stats_l1_lookup << ",\n"
            << "  \"l1_bypass_bytes\": " << ledger_total.l1_bypass_bytes << ",\n"
            << "  \"l2_lookup_bytes\": " << ledger_total.l2_lookup_bytes << ",\n"
            << "  \"backend_l2_lookup_bytes\": " << stats_l2_lookup << ",\n"
            << "  \"unknown_dram_read_bytes\": " << unknown.dram_read_bytes << ",\n"
            << "  \"unknown_dram_write_bytes\": " << unknown.dram_write_bytes << ",\n"
            << "  \"materialized_request_trace_bytes\": 0,\n"
            << "  \"claim_boundary\": \"all source/cache/DRAM category sums exactly conserve modeled totals; semantic id is static absolute-VA range binding without allocation epoch; kernel family remains inferred from kernel name\"\n"
            << "}\n";
    if (!receipt) throw std::runtime_error("failed writing semantic_conservation.json");
  }

private:
  struct DirtyOwner {
    int kernel = 0;
    unsigned semantic_id = 0;
  };
  using WritebackKey = std::tuple<int, unsigned, int, unsigned, int>;
  struct CacheBoundary {
    int kernel = 0;
    std::string name;
    std::string phase;
    uint64_t entry_dirty = 0;
    uint64_t entry_digest_a = 0;
    uint64_t entry_digest_b = 0;
    uint64_t exit_dirty = 0;
    uint64_t exit_digest_a = 0;
    uint64_t exit_digest_b = 0;
    bool has_exit = false;
  };

  uint64_t key(int kernel, const std::string &phase,
               const SemanticInfo &sem) {
    if (kernel <= 0)
      throw std::runtime_error("semantic traffic kernel id is not positive");
    register_kernel(kernel, phase, "");
    semantic_kinds_[sem.id] = sem.kind.empty() ? "unknown" : sem.kind;
    return (uint64_t(static_cast<uint32_t>(kernel)) << 32) | sem.id;
  }

  void register_kernel(int kernel, const std::string &phase,
                       const std::string &name) {
    const std::string normalized_phase = phase.empty() ? "unknown" : phase;
    auto [phase_it, phase_inserted] = kernel_phases_.emplace(kernel, normalized_phase);
    if (!phase_inserted && phase_it->second != normalized_phase)
      throw std::runtime_error("semantic kernel phase changed");
    if (!name.empty()) {
      auto [name_it, name_inserted] = kernel_names_.emplace(kernel, name);
      if (!name_inserted && name_it->second != name)
        throw std::runtime_error("semantic kernel name changed");
    }
  }

  const std::string &kernel_phase(int kernel) const {
    static const std::string unknown = "unknown";
    const auto found = kernel_phases_.find(kernel);
    return found == kernel_phases_.end() ? unknown : found->second;
  }

  const std::string &semantic_kind(unsigned semantic_id) const {
    static const std::string unknown = "unknown";
    const auto found = semantic_kinds_.find(semantic_id);
    return found == semantic_kinds_.end() ? unknown : found->second;
  }

  SemanticInfo semantic_from_id(unsigned semantic_id) const {
    SemanticInfo result;
    result.id = semantic_id;
    result.kind = semantic_kind(semantic_id);
    return result;
  }

  static uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  void xor_owner(uint64_t sector, const DirtyOwner &owner) {
    const uint64_t identity = sector ^
        (uint64_t(static_cast<uint32_t>(owner.kernel)) << 17) ^
        (uint64_t(owner.semantic_id) << 49);
    digest_a_ ^= mix64(identity);
    digest_b_ ^= mix64(identity ^ 0xd6e8feb86659fd93ULL);
  }

  void check_dirty_count(uint64_t backend_count, const char *where) const {
    if (backend_count != dirty_owners_.size())
      throw std::runtime_error(std::string(where) +
                               " resident dirty owner count differs");
  }
  static void add(SemanticTrafficCounter &dst,
                  const SemanticTrafficCounter &src) {
    dst.source_read_bytes += src.source_read_bytes;
    dst.source_write_bytes += src.source_write_bytes;
    dst.source_atomic_bytes += src.source_atomic_bytes;
    dst.l1_lookup_bytes += src.l1_lookup_bytes;
    dst.l1_hit_bytes += src.l1_hit_bytes;
    dst.l1_pending_hit_bytes += src.l1_pending_hit_bytes;
    dst.l1_miss_bytes += src.l1_miss_bytes;
    dst.l1_bypass_bytes += src.l1_bypass_bytes;
    dst.l2_lookup_bytes += src.l2_lookup_bytes;
    dst.l2_hit_bytes += src.l2_hit_bytes;
    dst.l2_pending_hit_bytes += src.l2_pending_hit_bytes;
    dst.l2_miss_bytes += src.l2_miss_bytes;
    dst.dram_read_bytes += src.dram_read_bytes;
    dst.dram_write_bytes += src.dram_write_bytes;
  }
  std::unordered_map<uint64_t, SemanticTrafficCounter> rows_;
  std::map<int, std::string> kernel_phases_;
  std::map<int, std::string> kernel_names_;
  std::map<unsigned, std::string> semantic_kinds_ = {{0, "unknown"}};
  std::unordered_map<uint64_t, DirtyOwner> dirty_owners_;
  std::map<WritebackKey, uint64_t> writeback_rows_;
  std::vector<CacheBoundary> boundaries_;
  uint64_t digest_a_ = 0;
  uint64_t digest_b_ = 0;
  uint64_t writeback_dram_write_bytes_ = 0;
  uint64_t direct_dram_write_bytes_ = 0;
  uint64_t static_address_owner_mismatch_bytes_ = 0;
};

SemanticTrafficLedger *g_semantic_traffic_ledger = nullptr;
#include "cache_observation.h"
#include "cache_input_census.h"

bool wants_semantics(const Options &opt) {
  return opt.semantic_output != "none";
}


bool has_semantic_policy_file(const Options &opt) {
  return !opt.semantic_file.empty();
}

bool is_model_weight_semantic(const SemanticInfo &sem) {
  return sem.kind == "model_weight";
}

bool should_stream_l2_fill(const KernelMeta &meta, const MemoryInst &inst,
                           const SemanticInfo &sem) {
  if (inst.op == 'A')
    return true;
  if (inst.op == 'R' && is_model_weight_semantic(sem))
    return true;
  if (kernel_is_cublas_gemm(meta.name))
    return true;
  if (inst.op == 'R' && kernel_is_softmax(meta.name))
    return true;
  if (inst.op == 'R' && kernel_is_large_matmul(meta.name) &&
      inst.opcode.find("LTC128B") != std::string::npos)
    return true;
  return false;
}

std::string request_csv_header(const Options &opt) {
  std::string header =
      "timestamp,kernel_id,kernel_name,sm_id,block_id,inst_seq,pc,opcode,op,ref_id,cache_level,addr,access_size,lane_mask,lane_count,mem_width,l1_hit,l2_hit,sector_addr,line_addr,l1_line_addr,l2_line_addr,dram_partition,dram_bank,dram_row,dram_col";
  if (wants_semantics(opt)) {
    header += ",llm_phase,semantic_id";
    if (opt.semantic_output == "fields")
      header += ",semantic_kind,semantic_layer,semantic_tensor,semantic_device,semantic_usage";
    else if (opt.semantic_output == "tag")
      header += ",semantic_tag";
  }
  header += "\n";
  return header;
}

void write_request(RotatingOutput &out, uint64_t timestamp,
                   const KernelMeta &meta, const MemoryInst &inst,
                   const SectorRequest &req, const std::string &level,
                   bool l1_hit, bool l2_hit, const Options &opt,
                   const SemanticInfo *sem = nullptr) {
  const uint64_t line_addr = req.addr / opt.cache_line_size;
  const uint64_t l1_line = req.addr / opt.l1_line_size;
  const uint64_t l2_line = req.addr / opt.l2_line_size;
  const unsigned partition = dram_partition_index(req.addr, opt);
  const unsigned partition_bits = bits_for_count(std::max(1u, opt.num_partitions));
  const unsigned bank_shift = opt.partition_index_bit + partition_bits;
  const unsigned bank =
      low_mask_index(req.addr, bank_shift, std::max(1u, opt.num_banks));
  const unsigned bank_bits = bits_for_count(std::max(1u, opt.num_banks));
  const uint64_t row = req.addr >> (bank_shift + bank_bits);
  const uint64_t col = (req.addr >> 5) & 0x7f;

  auto &os = out.stream();
  os << timestamp << ',' << meta.id << ',' << csv_escape(meta.name) << ','
      << inst.sm_id << ',' << inst.block_id << ',' << inst.seq << ','
      << hex_string(inst.pc) << ',' << csv_escape(inst.opcode) << ',' << inst.op
      << ',' << req.ref_id << ',' << level << ',' << hex_string(req.addr) << ','
      << req.size << ',' << hex_string(req.lane_mask) << ',' << req.lane_count
      << ',' << inst.mem_width << ',' << (l1_hit ? 1 : 0) << ','
      << (l2_hit ? 1 : 0) << ',' << (req.addr / opt.sector_size) << ','
      << line_addr << ',' << l1_line << ',' << l2_line << ',' << partition
      << ',' << bank << ',' << row << ',' << col;
  if (wants_semantics(opt)) {
    const SemanticInfo unknown;
    const auto &s = sem ? *sem : unknown;
    os << ',' << csv_escape(meta.llm_phase) << ',' << s.id;
    if (opt.semantic_output == "fields") {
      os << ',' << csv_escape(s.kind) << ',' << csv_escape(s.layer) << ','
         << csv_escape(s.tensor) << ',' << csv_escape(s.device) << ','
         << csv_escape(s.usage);
    } else if (opt.semantic_output == "tag") {
      os << ',' << csv_escape(s.tag);
    }
  }
  os << '\n';
  out.rotate_if_needed();
}

void write_writeback_request(RotatingOutput &out, uint64_t timestamp,
                             const KernelMeta &meta, const MemoryInst &inst,
                             const WritebackSpan &span, bool l1_hit,
                             bool l2_hit, const Options &opt,
                             const SemanticInfo *sem = nullptr) {
  const uint64_t line_addr = span.addr / opt.cache_line_size;
  const uint64_t l1_line = span.addr / opt.l1_line_size;
  const uint64_t l2_line = span.addr / opt.l2_line_size;
  const unsigned partition = dram_partition_index(span.addr, opt);
  const unsigned partition_bits = bits_for_count(std::max(1u, opt.num_partitions));
  const unsigned bank_shift = opt.partition_index_bit + partition_bits;
  const unsigned bank =
      low_mask_index(span.addr, bank_shift, std::max(1u, opt.num_banks));
  const unsigned bank_bits = bits_for_count(std::max(1u, opt.num_banks));
  const uint64_t row = span.addr >> (bank_shift + bank_bits);
  const uint64_t col = (span.addr >> 5) & 0x7f;

  auto &os = out.stream();
  os << timestamp << ',' << meta.id << ',' << csv_escape(meta.name) << ','
      << inst.sm_id << ',' << inst.block_id << ',' << inst.seq << ','
      << hex_string(inst.pc) << ",L2_WRBK,W,0,DRAM," << hex_string(span.addr)
      << ',' << span.size << ',' << hex_string(span.sector_mask)
      << ",0," << span.size << ',' << (l1_hit ? 1 : 0) << ','
      << (l2_hit ? 1 : 0) << ',' << (span.addr / opt.sector_size) << ','
      << line_addr << ',' << l1_line << ',' << l2_line << ',' << partition
      << ',' << bank << ',' << row << ',' << col;
  if (wants_semantics(opt)) {
    const SemanticInfo unknown;
    const auto &s = sem ? *sem : unknown;
    os << ',' << csv_escape(meta.llm_phase) << ',' << s.id;
    if (opt.semantic_output == "fields") {
      os << ',' << csv_escape(s.kind) << ',' << csv_escape(s.layer) << ','
         << csv_escape(s.tensor) << ',' << csv_escape(s.device) << ','
         << csv_escape(s.usage);
    } else if (opt.semantic_output == "tag") {
      os << ',' << csv_escape(s.tag);
    }
  }
  os << '\n';
  out.rotate_if_needed();
}

void append_accelsim_semantic(std::ofstream &os, const Options &opt,
                              const KernelMeta &meta,
                              const SemanticInfo *sem) {
  if (!wants_semantics(opt))
    return;
  const SemanticInfo unknown;
  const auto &s = sem ? *sem : unknown;
  os << ", kid=" << meta.id
     << ", llm_phase=" << shell_escape_field(meta.llm_phase)
     << ", sem_id=" << s.id;
  if (opt.semantic_output == "fields") {
    os << ", sem_kind=" << shell_escape_field(s.kind)
       << ", sem_layer=" << shell_escape_field(s.layer)
       << ", sem_tensor=" << shell_escape_field(s.tensor)
       << ", sem_device=" << shell_escape_field(s.device)
       << ", sem_usage=" << shell_escape_field(s.usage);
  } else if (opt.semantic_output == "tag") {
    os << ", semantic=" << shell_escape_field(s.tag);
  }
}

void write_accelsim_writeback(RotatingOutput &out, uint64_t uid,
                              uint64_t timestamp, const KernelMeta &meta,
                              const WritebackSpan &span,
                              const Options &opt,
                              const SemanticInfo *sem = nullptr) {
  const unsigned partition = dram_partition_index(span.addr, opt);
  auto &os = out.stream();
  if (wants_accelsim_compact(opt)) {
    os << "ts=" << timestamp << ", part=" << partition
        << ", sid=4294967295, wid=4294967295"
        << ", addr=" << hex_string(span.addr)
        << ", op=store, size=" << span.size
        << ", type=L2_WRBK";
    append_accelsim_semantic(os, opt, meta, sem);
    os << '\n';
    out.rotate_if_needed();
    return;
  }
  os << "mf: uid=" << std::setw(6) << uid
      << ", sid4294967295:w4294967295, part=" << std::setw(2)
      << std::setfill('0') << partition << std::setfill(' ')
      << ", addr=" << hex_string(span.addr)
      << ", store, size=" << span.size
      << ", L2_WRBK  status = IN_PARTITION_MC_INTERFACE_QUEUE ("
      << timestamp << ")";
  append_accelsim_semantic(os, opt, meta, sem);
  os << ",\n";
  out.rotate_if_needed();
}

bool should_emit(const Options &opt, const std::string &level) {
  return opt.emit_level == "all" || opt.emit_level == level;
}

bool should_emit_phase(const Options &opt, const KernelMeta &meta) {
  if (opt.output_phase != "all" && meta.llm_phase != opt.output_phase)
    return false;
  return opt.emit_all_kernels || opt.emit_kernel_ids.count(meta.id) != 0;
}

bool wants_accelsim(const Options &opt) {
  return opt.output_format == "accelsim" ||
         opt.output_format == "accelsim-compact" ||
         opt.output_format == "both" ||
         opt.output_format == "both-compact" ||
         opt.output_format == "both-footprint";
}

bool wants_accelsim_compact(const Options &opt) {
  return opt.output_format == "accelsim-compact" ||
         opt.output_format == "both-compact";
}

bool wants_footprint(const Options &opt) {
  return opt.output_format == "footprint" ||
         opt.output_format == "request-footprint" ||
         opt.output_format == "both-footprint";
}

bool wants_csv(const Options &opt) {
  return opt.output_format == "csv" || opt.output_format == "both" ||
         opt.output_format == "both-compact";
}

std::string mask_to_accelsim_bits(uint32_t mask) {
  std::string bits;
  bits.reserve(kWarpSize);
  for (int lane = static_cast<int>(kWarpSize) - 1; lane >= 0; --lane)
    bits.push_back(((mask >> lane) & 1u) ? '1' : '0');
  return bits;
}

std::string accel_mem_type(const MemoryInst &inst) {
  if (starts_with(inst.opcode, "LDL"))
    return "LOCAL_R ";
  if (starts_with(inst.opcode, "STL"))
    return "LOCAL_W ";
  if (inst.op == 'W' || inst.op == 'A')
    return "GLOBAL_W";
  return "GLOBAL_R";
}

class FootprintWriter {
public:
  void open(const fs::path &path, const Options &opt, uint64_t rotate_bytes) {
    format_ = parse_footprint_format(opt.footprint_format);
    semantic_output_ =
        format_ == FootprintFormat::FullCsv ? opt.semantic_output : "id";

    std::string header;
    if (format_ == FootprintFormat::Binary) {
      header =
          "#HYFISS_REQUEST_FOOTPRINT_BINARY_V1 "
          "record=timestamp:u64,addr:u64,bytes:u64,count:u32,"
          "kernel_id:u32,semantic_id:u32,sm_id:u16,unit_bytes:u16,"
          "op:u8,phase:u8,reserved:u16 little_endian\n";
    } else if (format_ == FootprintFormat::MinimalCsv) {
      header =
          "timestamp,kernel_id,llm_phase,sm_id,op,addr,bytes,unit_bytes,count,semantic_id\n";
    } else {
      header =
          "timestamp,kernel_id,llm_phase,sm_id,pc,op,type,opcode,addr,bytes,unit_bytes,count,lane_mask_or,lane_count_sum";
      if (wants_semantics(opt)) {
        header += ",semantic_id";
        if (opt.semantic_output == "fields")
          header += ",semantic_kind,semantic_layer,semantic_tensor,semantic_device,semantic_usage";
        else if (opt.semantic_output == "tag")
          header += ",semantic_tag";
      }
      header += "\n";
    }

    out_.open(path, header, rotate_bytes, format_ == FootprintFormat::Binary);
    enabled_ = true;
  }

  bool enabled() const { return enabled_; }

  void write_request(uint64_t timestamp, const KernelMeta &meta,
                     const MemoryInst &inst, const SectorRequest &req,
                     const Options &opt, const SemanticInfo *sem) {
    const bool is_write = inst.op == 'W' || inst.op == 'A';
    append(timestamp, meta.id, meta.llm_phase, inst.sm_id, inst.pc,
           is_write ? "store" : "load", accel_mem_type(inst), inst.opcode,
           req.addr, req.size, req.lane_mask, req.lane_count, opt, sem);
  }

  void write_writeback(uint64_t timestamp, const KernelMeta &meta,
                       const MemoryInst &inst, const WritebackSpan &span,
                       const Options &opt, const SemanticInfo *sem) {
    append(timestamp, meta.id, meta.llm_phase, inst.sm_id, inst.pc,
           "store", "L2_WRBK", "L2_WRBK", span.addr, span.size,
           span.sector_mask, 0, opt, sem);
  }

  void flush() {
    if (!pending_.valid)
      return;
    if (format_ == FootprintFormat::Binary) {
      write_binary_record();
      return;
    }
    if (format_ == FootprintFormat::MinimalCsv) {
      write_minimal_record();
      return;
    }
    auto &os = out_.stream();
    os << pending_.timestamp << ',' << pending_.kernel_id << ','
       << csv_escape(pending_.phase) << ',' << pending_.sm_id << ','
       << hex_string(pending_.pc) << ',' << pending_.op << ','
       << pending_.type << ',' << csv_escape(pending_.opcode) << ','
       << hex_string(pending_.addr) << ',' << pending_.bytes << ','
       << pending_.unit_bytes << ',' << pending_.count << ','
       << hex_string(pending_.lane_mask_or) << ',' << pending_.lane_count_sum;
    if (pending_.has_semantics) {
      os << ',' << pending_.sem_id;
      if (semantic_output_ == "fields") {
        os << ',' << csv_escape(pending_.sem_kind) << ','
           << csv_escape(pending_.sem_layer) << ','
           << csv_escape(pending_.sem_tensor) << ','
           << csv_escape(pending_.sem_device) << ','
           << csv_escape(pending_.sem_usage);
      } else if (semantic_output_ == "tag") {
        os << ',' << csv_escape(pending_.sem_tag);
      }
    }
    os << '\n';
    out_.rotate_if_needed();
    ++records_;
    pending_ = Pending();
  }

  uint64_t records() const { return records_; }
  uint64_t expanded_requests() const { return expanded_requests_; }
  const std::vector<fs::path> &paths() const { return out_.paths(); }

private:
  struct Pending {
    bool valid = false;
    uint64_t timestamp = 0;
    int kernel_id = 0;
    std::string phase;
    unsigned sm_id = 0;
    uint64_t pc = 0;
    std::string op;
    std::string type;
    std::string opcode;
    uint64_t addr = 0;
    uint64_t bytes = 0;
    unsigned unit_bytes = 0;
    uint64_t count = 0;
    uint32_t lane_mask_or = 0;
    uint64_t lane_count_sum = 0;
    bool has_semantics = false;
    unsigned sem_id = 0;
    std::string sem_kind;
    std::string sem_layer;
    std::string sem_tensor;
    std::string sem_device;
    std::string sem_usage;
    std::string sem_tag;
  };

  static bool same_semantic(const Pending &p, const SemanticInfo *sem,
                            bool has_semantics, const std::string &semantic_output) {
    if (p.has_semantics != has_semantics)
      return false;
    if (!has_semantics)
      return true;
    const SemanticInfo unknown;
    const auto &s = sem ? *sem : unknown;
    if (p.sem_id != s.id)
      return false;
    if (semantic_output == "fields")
      return p.sem_kind == s.kind && p.sem_layer == s.layer &&
             p.sem_tensor == s.tensor && p.sem_device == s.device &&
             p.sem_usage == s.usage;
    if (semantic_output == "tag")
      return p.sem_tag == s.tag;
    return true;
  }

  bool can_merge(uint64_t timestamp, int kernel_id, const std::string &phase,
                 unsigned sm_id, uint64_t pc, const std::string &op,
                 const std::string &type, const std::string &opcode,
                 uint64_t addr, unsigned unit_bytes, const SemanticInfo *sem,
                 bool has_semantics) const {
    if (!pending_.valid)
      return false;
    const bool full_key = format_ == FootprintFormat::FullCsv;
    return pending_.timestamp == timestamp && pending_.kernel_id == kernel_id &&
           pending_.phase == phase && pending_.sm_id == sm_id &&
           pending_.op == op && pending_.unit_bytes == unit_bytes &&
           (!full_key || (pending_.pc == pc && pending_.type == type &&
                          pending_.opcode == opcode)) &&
           pending_.addr + pending_.bytes == addr &&
           same_semantic(pending_, sem, has_semantics, semantic_output_);
  }

  void append(uint64_t timestamp, int kernel_id, const std::string &phase,
              unsigned sm_id, uint64_t pc, const std::string &op,
              const std::string &type, const std::string &opcode,
              uint64_t addr, unsigned unit_bytes, uint32_t lane_mask,
              unsigned lane_count, const Options &opt, const SemanticInfo *sem) {
    if (!enabled_ || unit_bytes == 0)
      return;
    const bool has_semantics = wants_semantics(opt);
    ++expanded_requests_;
    if (can_merge(timestamp, kernel_id, phase, sm_id, pc, op, type, opcode,
                  addr, unit_bytes, sem, has_semantics)) {
      pending_.bytes += unit_bytes;
      pending_.count++;
      pending_.lane_mask_or |= lane_mask;
      pending_.lane_count_sum += lane_count;
      return;
    }

    flush();
    pending_.valid = true;
    pending_.timestamp = timestamp;
    pending_.kernel_id = kernel_id;
    pending_.phase = phase;
    pending_.sm_id = sm_id;
    pending_.pc = pc;
    pending_.op = op;
    pending_.type = type;
    pending_.opcode = opcode;
    pending_.addr = addr;
    pending_.bytes = unit_bytes;
    pending_.unit_bytes = unit_bytes;
    pending_.count = 1;
    pending_.lane_mask_or = lane_mask;
    pending_.lane_count_sum = lane_count;
    pending_.has_semantics = has_semantics;
    if (has_semantics) {
      const SemanticInfo unknown;
      const auto &s = sem ? *sem : unknown;
      pending_.sem_id = s.id;
      pending_.sem_kind = s.kind;
      pending_.sem_layer = s.layer;
      pending_.sem_tensor = s.tensor;
      pending_.sem_device = s.device;
      pending_.sem_usage = s.usage;
      pending_.sem_tag = s.tag;
    }
  }

  void finish_record() {
    out_.rotate_if_needed();
    ++records_;
    pending_ = Pending();
  }

  void write_minimal_record() {
    auto &os = out_.stream();
    os << pending_.timestamp << ',' << pending_.kernel_id << ','
       << csv_escape(pending_.phase) << ',' << pending_.sm_id << ','
       << pending_.op << ',' << hex_string(pending_.addr) << ','
       << pending_.bytes << ',' << pending_.unit_bytes << ','
       << pending_.count << ',' << pending_.sem_id << '\n';
    finish_record();
  }

  static uint8_t phase_code(const std::string &phase) {
    if (phase == "prefill")
      return 1;
    if (phase == "decode")
      return 2;
    return 0;
  }

  template <typename T> static void write_binary_value(std::ofstream &os, T v) {
    os.write(reinterpret_cast<const char *>(&v), sizeof(v));
  }

  void write_binary_record() {
    auto &os = out_.stream();
    const uint64_t timestamp = pending_.timestamp;
    const uint64_t addr = pending_.addr;
    const uint64_t bytes = pending_.bytes;
    const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(
        pending_.count, std::numeric_limits<uint32_t>::max()));
    const uint32_t kernel_id = pending_.kernel_id < 0
                                   ? 0
                                   : static_cast<uint32_t>(pending_.kernel_id);
    const uint32_t semantic_id = pending_.sem_id;
    const uint16_t sm_id = static_cast<uint16_t>(std::min<unsigned>(
        pending_.sm_id, std::numeric_limits<uint16_t>::max()));
    const uint16_t unit_bytes = static_cast<uint16_t>(std::min<unsigned>(
        pending_.unit_bytes, std::numeric_limits<uint16_t>::max()));
    const uint8_t op = pending_.op == "store" ? 1 : 0;
    const uint8_t phase = phase_code(pending_.phase);
    const uint16_t reserved = 0;

    write_binary_value(os, timestamp);
    write_binary_value(os, addr);
    write_binary_value(os, bytes);
    write_binary_value(os, count);
    write_binary_value(os, kernel_id);
    write_binary_value(os, semantic_id);
    write_binary_value(os, sm_id);
    write_binary_value(os, unit_bytes);
    write_binary_value(os, op);
    write_binary_value(os, phase);
    write_binary_value(os, reserved);
    finish_record();
  }

  bool enabled_ = false;
  std::string semantic_output_ = "none";
  FootprintFormat format_ = FootprintFormat::FullCsv;
  Pending pending_;
  RotatingOutput out_;
  uint64_t records_ = 0;
  uint64_t expanded_requests_ = 0;
};

// One path accounts and emits each contiguous dirty-sector span. Cache event
// counts remain distinct from transfer request counts.
void emit_l2_writeback(const EvictedLine &evicted, uint64_t timestamp,
                       const KernelMeta &meta, const MemoryInst &inst,
                       bool l1_hit, bool l2_hit, const Options &opt,
                       SemanticDatabase &semantic_db,
                       const SemanticInfo *trigger_sem, KernelStats &stats,
                       RotatingOutput &req_out, RotatingOutput &accel_out,
                       FootprintWriter &footprint_out, uint64_t &total_records,
                       uint64_t &accelsim_records, uint64_t &accelsim_uid) {
  for_each_writeback_span(evicted, opt.sector_size, opt.l2_line_size,
      [&](const WritebackSpan &span) {
    ++stats.dram_requests;
    ++stats.dram_store_requests;
    stats.dram_store_bytes += span.size;
    stats.dram_store_sectors += span.size / opt.sector_size;
    const SemanticInfo &sem = semantic_db.lookup(span.addr);
    if (g_semantic_traffic_ledger)
      g_semantic_traffic_ledger->add_dram_writeback(
          meta.id, meta.llm_phase, sem, trigger_sem, span, opt.sector_size);
    if (!should_emit_phase(opt, meta)) return;
    if (wants_csv(opt) && should_emit(opt, "DRAM")) {
      write_writeback_request(req_out, timestamp, meta, inst, span,
                              l1_hit, l2_hit, opt, &sem);
      ++total_records;
    }
    if (wants_accelsim(opt)) {
      write_accelsim_writeback(accel_out, accelsim_uid++, timestamp,
                               meta, span, opt, &sem);
      ++accelsim_records;
    }
    if (wants_footprint(opt))
      footprint_out.write_writeback(timestamp, meta, inst, span, opt, &sem);
  });
  if (evicted.dirty) {
    ++stats.l2_writeback_events;
    stats.l2_writeback_dirty_sectors += static_cast<unsigned>(__builtin_popcount(evicted.dirty_sectors));
  }
}

void write_accelsim_request(RotatingOutput &out, uint64_t uid,
                            uint64_t timestamp, const KernelMeta &meta,
                            const MemoryInst &inst,
                            const SectorRequest &req, const Options &opt,
                            const SemanticInfo *sem = nullptr) {
  const unsigned partition = dram_partition_index(req.addr, opt);
  const bool is_write = inst.op == 'W' || inst.op == 'A';
  auto &os = out.stream();
  if (wants_accelsim_compact(opt)) {
    os << "ts=" << timestamp << ", part=" << partition
        << ", sid=" << inst.sm_id << ", wid=0"
        << ", addr=" << hex_string(req.addr)
        << ", op=" << (is_write ? "store" : "load")
        << ", size=" << req.size
        << ", type=" << accel_mem_type(inst)
        << ", pc=" << hex_string(inst.pc)
        << ", mask=" << hex_string(req.lane_mask)
        << ", kernel=" << csv_escape(inst.opcode);
    append_accelsim_semantic(os, opt, meta, sem);
    os << '\n';
    out.rotate_if_needed();
    return;
  }
  os << "mf: uid=" << std::setw(6) << uid << ", sid" << std::setfill('0')
      << std::setw(2) << inst.sm_id << ":w" << std::setw(2) << 0
      << std::setfill(' ') << ", part=" << std::setw(2) << std::setfill('0')
      << partition << std::setfill(' ') << ", addr=" << hex_string(req.addr)
      << ", " << (is_write ? "store" : "load ") << ", size=" << req.size
      << ", " << accel_mem_type(inst)
      << " status = IN_PARTITION_MC_INTERFACE_QUEUE (" << timestamp << "), "
      << hex_string(inst.pc) << " w" << std::setfill('0') << std::setw(2) << 0
      << std::setfill(' ') << "[" << mask_to_accelsim_bits(req.lane_mask)
      << "]: ";
  append_accelsim_semantic(os, opt, meta, sem);
  os << '\n';
  out.rotate_if_needed();
}

std::vector<int> parse_kernel_list(const std::string &s,
                                   const std::map<int, KernelMeta> &available) {
  std::vector<int> out;
  if (s == "all") {
    for (const auto &kv : available)
      out.push_back(kv.first);
    return out;
  }
  for (auto part : split_char(s, ',')) {
    if (part.empty())
      continue;
    const auto dash = part.find('-');
    if (dash != std::string::npos) {
      const int first = static_cast<int>(parse_u64(part.substr(0, dash), 10));
      const int last = static_cast<int>(parse_u64(part.substr(dash + 1), 10));
      if (last < first)
        throw std::runtime_error("bad kernel range: " + part);
      for (int kid = first; kid <= last; ++kid)
        out.push_back(kid);
    } else {
      out.push_back(static_cast<int>(parse_u64(part, 10)));
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<fs::path> list_files(const fs::path &dir) {
  std::vector<fs::path> files;
  if (!fs::exists(dir))
    return files;
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file())
      files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  return files;
}

bool parse_kernel_dash_dir(const std::string &name, int &kid) {
  const std::string prefix = "kernel-";
  if (!starts_with(name, prefix))
    return false;
  const std::string id = name.substr(prefix.size());
  if (id.empty() || !std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      }))
    return false;
  kid = static_cast<int>(parse_u64(id, 10));
  return true;
}

bool parse_raw_mem_filename(const std::string &name, int &kid) {
  const std::string prefix = "kernel_";
  const std::string suffix = ".mem";
  if (!starts_with(name, prefix) || name.size() <= prefix.size() + suffix.size() ||
      name.substr(name.size() - suffix.size()) != suffix)
    return false;
  const std::string id = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
  if (id.empty() || !std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      }))
    return false;
  kid = static_cast<int>(parse_u64(id, 10));
  return true;
}

bool parse_raw_mem_part_filename(const std::string &name, int &kid,
                                 unsigned &part) {
  const std::string prefix = "kernel_";
  const std::string middle = ".mem.part";
  if (!starts_with(name, prefix))
    return false;
  const size_t mid = name.find(middle, prefix.size());
  if (mid == std::string::npos)
    return false;
  const std::string id = name.substr(prefix.size(), mid - prefix.size());
  const std::string part_s = name.substr(mid + middle.size());
  if (id.empty() || part_s.empty() ||
      !std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      }) ||
      !std::all_of(part_s.begin(), part_s.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      }))
    return false;
  kid = static_cast<int>(parse_u64(id, 10));
  part = static_cast<unsigned>(parse_u64(part_s, 10));
  return true;
}

std::vector<fs::path> raw_memory_files_for_kernel(const fs::path &memory_dir,
                                                  int kernel_id) {
  std::vector<std::pair<unsigned, fs::path>> parts;
  const fs::path base = memory_dir / ("kernel_" + std::to_string(kernel_id) + ".mem");
  if (fs::exists(base))
    parts.push_back({0, base});
  if (fs::exists(memory_dir)) {
    for (const auto &entry : fs::directory_iterator(memory_dir)) {
      if (!entry.is_regular_file())
        continue;
      int kid = 0;
      unsigned part = 0;
      if (parse_raw_mem_part_filename(entry.path().filename().string(), kid,
                                      part) &&
          kid == kernel_id) {
        parts.push_back({part, entry.path()});
      }
    }
  }
  std::sort(parts.begin(), parts.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  std::vector<fs::path> paths;
  for (const auto &part : parts)
    paths.push_back(part.second);
  return paths;
}

bool parse_raw_memc_filename(const std::string &name, int &kid) {
  const std::string prefix = "kernel_";
  const std::string suffix = ".memc";
  std::string parse_name = name;
  const std::string zst_suffix = ".zst";
  if (parse_name.size() > zst_suffix.size() &&
      parse_name.substr(parse_name.size() - zst_suffix.size()) == zst_suffix)
    parse_name.resize(parse_name.size() - zst_suffix.size());
  if (!starts_with(parse_name, prefix) ||
      parse_name.size() <= prefix.size() + suffix.size() ||
      parse_name.substr(parse_name.size() - suffix.size()) != suffix)
    return false;
  const std::string id = parse_name.substr(
      prefix.size(), parse_name.size() - prefix.size() - suffix.size());
  if (id.empty() || !std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      }))
    return false;
  kid = static_cast<int>(parse_u64(id, 10));
  return true;
}

bool parse_raw_memc_part_filename(const std::string &name, int &kid,
                                  unsigned &part) {
  const std::string prefix = "kernel_";
  const std::string middle = ".memc.part";
  std::string parse_name = name;
  const std::string zst_suffix = ".zst";
  if (parse_name.size() > zst_suffix.size() &&
      parse_name.substr(parse_name.size() - zst_suffix.size()) == zst_suffix)
    parse_name.resize(parse_name.size() - zst_suffix.size());
  if (!starts_with(parse_name, prefix))
    return false;
  const size_t mid = parse_name.find(middle, prefix.size());
  if (mid == std::string::npos)
    return false;
  const std::string id = parse_name.substr(prefix.size(), mid - prefix.size());
  const std::string part_s = parse_name.substr(mid + middle.size());
  if (id.empty() || part_s.empty() ||
      !std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      }) ||
      !std::all_of(part_s.begin(), part_s.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      }))
    return false;
  kid = static_cast<int>(parse_u64(id, 10));
  part = static_cast<unsigned>(parse_u64(part_s, 10));
  return true;
}

std::vector<fs::path> memc_memory_files_for_kernel(const fs::path &memory_dir,
                                                   int kernel_id) {
  std::vector<std::pair<unsigned, fs::path>> parts;
  const fs::path base = memory_dir / ("kernel_" + std::to_string(kernel_id) + ".memc");
  if (fs::exists(base))
    parts.push_back({0, base});
  const fs::path base_zst =
      memory_dir / ("kernel_" + std::to_string(kernel_id) + ".memc.zst");
  if (fs::exists(base_zst))
    parts.push_back({0, base_zst});
  if (fs::exists(memory_dir)) {
    for (const auto &entry : fs::directory_iterator(memory_dir)) {
      if (!entry.is_regular_file())
        continue;
      int kid = 0;
      unsigned part = 0;
      if (parse_raw_memc_part_filename(entry.path().filename().string(), kid,
                                       part) &&
          kid == kernel_id) {
        parts.push_back({part, entry.path()});
      }
    }
  }
  std::sort(parts.begin(), parts.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  std::vector<fs::path> paths;
  for (const auto &part : parts)
    paths.push_back(part.second);
  return paths;
}

bool path_ends_with(const fs::path &path, const std::string &suffix) {
  const std::string s = path.string();
  return s.size() >= suffix.size() &&
         s.substr(s.size() - suffix.size()) == suffix;
}

std::string shell_single_quote(const std::string &s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

fs::path materialize_memc_zst(const fs::path &path, const Options &opt,
                              int kernel_id, size_t path_index,
                              std::vector<fs::path> &temps) {
  if (!path_ends_with(path, ".zst"))
    return path;
  const fs::path stage_dir = opt.output_dir / ".memc_staging";
  fs::create_directories(stage_dir);
  const std::string stem =
      "kernel_" + std::to_string(kernel_id) + "_" +
      std::to_string(path_index) + "_" +
      std::to_string(std::hash<std::string>{}(path.string())) + ".memc";
  const fs::path tmp = stage_dir / stem;
  const fs::path tmp_part = stage_dir / (stem + ".tmp");
  if (fs::exists(tmp) || fs::exists(tmp_part))
    throw std::runtime_error("MEMC staging collision; use a fresh output directory");
  const std::string cmd = "set -C; zstd -q -dc -- " +
                          shell_single_quote(path.string()) + " > " +
                          shell_single_quote(tmp_part.string());
  const int rc = std::system(cmd.c_str());
  if (rc != 0)
    throw std::runtime_error("failed to decompress memc zstd trace: " +
                             path.string());
  fs::rename(tmp_part, tmp);
  temps.push_back(tmp);
  return tmp;
}

struct MemcAdmissionStats {
  uint64_t records = 0, v3_records = 0, legacy_records = 0;
  uint64_t global_lanes = 0, local_lanes = 0, shared_lanes = 0;
  uint64_t filtered_local_lanes = 0;
};
MemcAdmissionStats g_memc_admission;
std::map<int, KernelMeta> g_memc_metadata;
std::set<unsigned> g_memc_function_ids;

struct V3InputGuard {
  bool admitted = false;
  uint64_t records = 0;
  std::map<int, uint64_t> kernel_records;
  std::map<std::pair<int, unsigned>, uint64_t> sm_clock_origins;
  std::map<fs::path, std::pair<uintmax_t, fs::file_time_type>> file_states;
};
V3InputGuard g_v3_guard;

// A full-capture streaming pass binds even a selected-kernel run to its receipt.
// It also finds per-SM 64-bit origins without treating SM clocks as synchronized.
// File size/mtime are checked on reuse; experiment runners additionally hash
// immutable source files. This is not an adversarial filesystem snapshot.
void preflight_v3_capture(const Options &opt,
                         const std::map<int, KernelMeta> &metadata) {
  g_v3_guard = V3InputGuard{};
  const auto receipt_path = opt.configs_dir / "capture_receipt.json";
  if (!fs::exists(receipt_path)) return;
  if (fs::file_size(receipt_path) > 1024 * 1024)
    throw std::runtime_error("capture receipt exceeds size bound");
  boost::property_tree::ptree receipt;
  boost::property_tree::read_json(receipt_path.string(), receipt);
  if (receipt.get<std::string>("lane_format", "") != "memc_v3") return;
  auto need = [&](bool ok) {
    if (!ok) throw std::runtime_error("MEMCv3 capture receipt/input conservation failure");
  };
  std::set<std::string> keys;
  for (const auto &entry : receipt) need(keys.insert(entry.first).second);
  need(receipt.get<std::string>("schema") == "hyfiss_nvbit_capture_receipt_v1" &&
       receipt.get<std::string>("status") == "PASS" &&
       receipt.get<std::string>("trace_mode") == "lane" &&
       receipt.get<bool>("space_metadata_persisted") &&
       receipt.get<bool>("space_classification_complete"));
  for (const auto key : {"unknown_space_lane_references", "dropped_records",
                         "malformed_packets", "sequence_errors", "io_errors"})
    need(receipt.get<uint64_t>(key) == 0);
  const uint64_t expected = receipt.get<uint64_t>("host_persisted_records");
  need(expected <= UINT64_MAX / 64);
  need(receipt.get<uint64_t>("device_pushed_records") == expected &&
       receipt.get<uint64_t>("host_received_records") == expected &&
       receipt.get<uint64_t>("kernel_count") == metadata.size());
  uint64_t globals = 0, locals = 0, shared = 0;
  std::set<fs::path> expected_files;
  for (const auto &entry : metadata) {
    const int kid = entry.first;
    const auto paths = memc_memory_files_for_kernel(opt.memory_dir, kid);
    need(!paths.empty());
    uint64_t count = 0;
    for (size_t index = 0; index < paths.size(); ++index) {
      const auto &path = paths[index];
      // New capture currently writes raw MEMC; refuse an unverified compressed
      // transport rather than allowing it to bypass the completeness pass.
      need(!path_ends_with(path, ".zst"));
      const auto name = path.filename().string();
      const auto base = "kernel_" + std::to_string(kid) + ".memc";
      int parsed_kid = 0;
      unsigned parsed_part = 0;
      need(index == 0 ? name == base :
           (parse_raw_memc_part_filename(name, parsed_kid, parsed_part) &&
            parsed_kid == kid && parsed_part == index));
      need(expected_files.insert(path).second && fs::file_size(path) != 0);
      const auto before = std::make_pair(fs::file_size(path), fs::last_write_time(path));
      std::ifstream in(path, std::ios::binary);
      hyfiss_memc::Reader reader(in);
      need(reader.version == 3);
      hyfiss_memc::Record r;
      while (reader.next(r)) {
        need(g_v3_guard.records < expected && r.sequence == g_v3_guard.records);
        ++g_v3_guard.records; ++count;
        const auto key = std::make_pair(kid, r.sm);
        auto origin = g_v3_guard.sm_clock_origins.find(key);
        if (origin == g_v3_guard.sm_clock_origins.end())
          g_v3_guard.sm_clock_origins.emplace(key, r.full_clock);
        else origin->second = std::min(origin->second, r.full_clock);
        for (unsigned ref = 0; ref < r.ref_count; ++ref) {
          need(r.refs[ref].unknown == 0);
          globals += __builtin_popcount(r.refs[ref].global);
          locals += __builtin_popcount(r.refs[ref].local);
          shared += __builtin_popcount(r.refs[ref].shared);
        }
      }
      need(before == std::make_pair(fs::file_size(path), fs::last_write_time(path)));
      g_v3_guard.file_states.emplace(path, before);
    }
    g_v3_guard.kernel_records.emplace(kid, count);
  }
  // Detect a whole extra kernel/file that the app.config does not enumerate.
  for (const auto &entry : fs::directory_iterator(opt.memory_dir)) {
    const auto name = entry.path().filename().string();
    if (entry.is_regular_file() && name.find(".memc") != std::string::npos)
      need(expected_files.count(entry.path()) == 1);
  }
  need(g_v3_guard.records == expected &&
       globals == receipt.get<uint64_t>("global_space_lane_references") &&
       locals == receipt.get<uint64_t>("local_space_lane_references") &&
       shared == receipt.get<uint64_t>("shared_space_lane_references"));
  g_v3_guard.admitted = true;
}

std::set<unsigned> read_function_ids(const fs::path &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("MEMCv3 requires function.config");
  std::set<unsigned> ids;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto tab = line.find('\t');
    if (tab == std::string::npos || tab + 1 == line.size())
      throw std::runtime_error("invalid MEMCv3 function table row");
    const uint64_t id = parse_u64(line.substr(0, tab), 10);
    if (id == 0 || id > UINT32_MAX || !ids.insert(static_cast<unsigned>(id)).second)
      throw std::runtime_error("invalid/duplicate MEMCv3 function id");
  }
  if (!in.eof() || ids.empty()) throw std::runtime_error("invalid MEMCv3 function table");
  return ids;
}

MemoryInst admit_memc_record(const hyfiss_memc::Record &r, unsigned version,
                            const std::string &opcode, int kernel_id, uint64_t seq,
                            const Options &opt, const KernelMeta &meta) {
  MemoryInst mi;
  mi.kernel_id = kernel_id; mi.block_id = r.cta; mi.seq = seq;
  mi.capture_seq = r.sequence; mi.pc = r.pc; mi.opcode = opcode;
  mi.mask = r.mask; mi.timestamp = r.relative_clock;
  mi.mem_width = infer_width(opcode);
  mi.has_space_metadata = version == 3;
  if (version == 3) {
    // Space completeness precedes direction inference or any filtering.
    for (unsigned ref = 0; ref < r.ref_count; ++ref)
      if (r.refs[ref].unknown)
        throw std::runtime_error("MEMCv3 unknown address space; record=" +
                                 std::to_string(r.sequence));
    if (r.ref_count != 1)
      throw std::runtime_error("MEMCv3 dual-reference roles are not established");
    const auto stem = opcode.substr(0, opcode.find('.'));
    if (stem == "LD" || stem == "LDG" || stem == "LDL" || stem == "LDS") mi.op = 'R';
    else if (stem == "ST" || stem == "STG" || stem == "STL" || stem == "STS") mi.op = 'W';
    else if (stem == "ATOM" || stem == "ATOMG" || stem == "RED") mi.op = 'A';
    else throw std::runtime_error("unsupported MEMCv3 memory opcode: " + opcode);
    mi.sm_id = r.sm; mi.cta_warp = r.cta_warp; mi.function_id = r.function;
    mi.full_clock = r.full_clock;
    mi.local_warp_owner = meta.local_warp_begin +
                         uint64_t(r.cta) * ((uint64_t(meta.block_size) + 31) / 32) + r.cta_warp;
  } else {
    mi.op = classify_opcode(opcode, opt.include_local);
  }
  unsigned retained_lanes = 0;
  for (unsigned ref = 0; ref < r.ref_count; ++ref) {
    if (version == 3)
      retained_lanes += __builtin_popcount(r.refs[ref].global |
                           (opt.include_local ? r.refs[ref].local : 0));
    else if (mi.op != 'N') retained_lanes += __builtin_popcount(r.mask);
  }
  mi.lanes.reserve(retained_lanes);
  for (unsigned ref = 0; ref < r.ref_count; ++ref) {
    const auto &rr = r.refs[ref];
    for (unsigned lane = 0; lane < 32; ++lane) {
      const uint32_t bit = uint32_t(1) << lane;
      if (!(r.mask & bit)) continue;
      if (version == 3) {
        if (rr.shared & bit) { ++g_memc_admission.shared_lanes; continue; }
        if (rr.local & bit) {
          ++g_memc_admission.local_lanes;
          if (!opt.include_local) { ++g_memc_admission.filtered_local_lanes; continue; }
          const uint64_t raw = rr.addresses[lane];
          const bool explicit_local = starts_with(opcode, "LDL") || starts_with(opcode, "STL");
          if (!explicit_local && (!meta.has_local_base || raw < meta.local_base))
            throw std::runtime_error("generic local address lacks compatible captured base");
          const uint64_t offset = explicit_local ? raw : raw - meta.local_base;
          if (offset > UINT32_MAX) throw std::runtime_error("local offset exceeds uint32");
          mi.lanes.push_back(LaneAddress{lane, ref + 1, raw, true, offset});
        } else if (rr.global & bit) {
          ++g_memc_admission.global_lanes;
          mi.lanes.push_back(LaneAddress{lane, ref + 1, rr.addresses[lane]});
        }
      } else if (mi.op != 'N') {
        mi.lanes.push_back(LaneAddress{lane, ref + 1, rr.addresses[lane]});
      }
    }
  }
  return mi;
}

void for_each_kernel_inst_memc(
    int kernel_id, const Options &opt,
    const std::unordered_map<uint64_t, IssueInfo> &issue_map,
    const std::function<void(MemoryInst &)> &consume) {
  uint64_t seq = 0, previous_capture = 0, first_clock = 0;
  bool have_capture = false, have_clock = false;
  unsigned kernel_version = 0;
  const auto &metadata = g_memc_metadata;
  const auto meta_it = metadata.find(kernel_id);
  const KernelMeta meta = meta_it != metadata.end() ? meta_it->second : KernelMeta{};
  const auto paths = memc_memory_files_for_kernel(opt.memory_dir, kernel_id);
  std::vector<fs::path> temp_paths;
  for (size_t path_index = 0; path_index < paths.size(); ++path_index) {
    const auto read_path = materialize_memc_zst(paths[path_index], opt, kernel_id,
                                               path_index, temp_paths);
    if (g_v3_guard.admitted) {
      const auto guard = g_v3_guard.file_states.find(read_path);
      if (guard == g_v3_guard.file_states.end() ||
          guard->second != std::make_pair(fs::file_size(read_path), fs::last_write_time(read_path)))
        throw std::runtime_error("MEMCv3 input changed after preflight");
    }
    // Historical empty files represented kernels without memory instructions.
    // For new v3 capture every file has a header, including an empty kernel.
    if (fs::file_size(read_path) == 0) continue;
    std::ifstream in(read_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open memc trace: " + read_path.string());
    hyfiss_memc::Reader reader(in);
    if (kernel_version && kernel_version != reader.version)
      throw std::runtime_error("mixed MEMC versions within a kernel");
    kernel_version = reader.version;
    if (reader.version == 3) {
      if (!g_v3_guard.admitted)
        throw std::runtime_error("MEMCv3 requires full capture receipt preflight");
      if (!meta.grid_size || !meta.block_size)
        throw std::runtime_error("MEMCv3 requires actual grid/block dimensions");
      if (g_memc_function_ids.empty())
        g_memc_function_ids = read_function_ids(opt.configs_dir / "function.config");
    }
    if (reader.version == 3) {
      const auto guard = g_v3_guard.file_states.find(read_path);
      if (guard == g_v3_guard.file_states.end() ||
          guard->second != std::make_pair(fs::file_size(read_path), fs::last_write_time(read_path)))
        throw std::runtime_error("MEMCv3 input changed after preflight");
    }
    hyfiss_memc::Record r;
    while (reader.next(r)) {
      ++g_memc_admission.records;
      if (reader.version >= 2) {
        if (have_capture && (previous_capture == UINT64_MAX || r.sequence != previous_capture + 1))
          throw std::runtime_error("MEMC capture sequence gap/reorder within kernel");
        previous_capture = r.sequence; have_capture = true;
      }
      const auto issue = issue_map.find(map_key(kernel_id, r.cta));
      if (reader.version == 3) {
        ++g_memc_admission.v3_records;
        if (r.cta >= meta.grid_size || r.cta_warp >= (uint64_t(meta.block_size) + 31) / 32 ||
            r.sm >= opt.num_sms || !g_memc_function_ids.count(r.function))
          throw std::runtime_error("MEMCv3 spatial identity is inconsistent with configuration");
        for (unsigned lane = 0; lane < 32; ++lane)
          if ((r.mask & (uint32_t(1) << lane)) && uint64_t(r.cta_warp) * 32 + lane >= meta.block_size)
            throw std::runtime_error("MEMCv3 effective lane outside CTA");
        if (issue == issue_map.end() || issue->second.sm_id != r.sm)
          throw std::runtime_error("MEMCv3 SM disagrees with captured issue.config");
        if (!have_clock) { first_clock = r.full_clock; have_clock = true; }
        if (static_cast<uint32_t>(r.full_clock - first_clock) != r.relative_clock)
          throw std::runtime_error("MEMCv3 full/relative clock disagreement");
      } else ++g_memc_admission.legacy_records;
      MemoryInst mi = admit_memc_record(r, reader.version, reader.opcodes[r.opcode_id],
                                         kernel_id, seq++, opt, meta);
      if (mi.op == 'N' || mi.lanes.empty()) continue;
      if (reader.version == 3) {
        mi.timestamp = r.full_clock - g_v3_guard.sm_clock_origins.at({kernel_id, r.sm});
      } else {
        mi.sm_id = issue == issue_map.end() ? (mi.block_id % std::max(1u, opt.num_sms)) : issue->second.sm_id;
        if (issue != issue_map.end() && issue->second.has_cta_start) mi.timestamp += issue->second.cta_start;
      }
      // A per-SM 64-bit elapsed coordinate avoids the legacy 32-bit wrap. It does
      // not establish cross-SM synchronization; delivery order remains the input.
      consume(mi);
    }
    if (reader.version == 3 && g_v3_guard.file_states.at(read_path) !=
        std::make_pair(fs::file_size(read_path), fs::last_write_time(read_path)))
      throw std::runtime_error("MEMCv3 input changed while reading");
  }
  if (g_v3_guard.admitted && seq != g_v3_guard.kernel_records.at(kernel_id))
    throw std::runtime_error("MEMCv3 kernel record count changed after preflight");
  // Materialized inputs are evidence. Keep them; never delete preexisting files.
}

std::vector<MemoryInst>
load_kernel_insts_memc(int kernel_id, const Options &opt,
                       const std::unordered_map<uint64_t, IssueInfo> &issue_map) {
  std::vector<MemoryInst> insts;
  for_each_kernel_inst_memc(kernel_id, opt, issue_map,
                           [&](MemoryInst &mi) { insts.push_back(std::move(mi)); });
  return insts;
}

bool uses_memc_input(int kernel_id, const Options &opt) {
  const auto stem = "kernel_" + std::to_string(kernel_id) + ".memc";
  return opt.input_format == "memc" ||
         (opt.input_format == "auto" &&
          (fs::exists(opt.memory_dir / stem) || fs::exists(opt.memory_dir / (stem + ".zst"))));
}

bool parse_split_mem_filename(const std::string &name, int &kid,
                              unsigned &block) {
  const std::string prefix = "kernel_";
  const std::string marker = "_block_";
  const std::string suffix = ".mem";
  if (!starts_with(name, prefix) || name.size() <= prefix.size() + suffix.size() ||
      name.substr(name.size() - suffix.size()) != suffix)
    return false;
  const size_t marker_pos = name.find(marker, prefix.size());
  if (marker_pos == std::string::npos)
    return false;
  const std::string kid_s = name.substr(prefix.size(), marker_pos - prefix.size());
  const std::string block_s = name.substr(marker_pos + marker.size(),
                                          name.size() - marker_pos - marker.size() - suffix.size());
  if (kid_s.empty() || block_s.empty())
    return false;
  if (!std::all_of(kid_s.begin(), kid_s.end(), [](unsigned char c) { return std::isdigit(c) != 0; }) ||
      !std::all_of(block_s.begin(), block_s.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
    return false;
  kid = static_cast<int>(parse_u64(kid_s, 10));
  block = static_cast<unsigned>(parse_u64(block_s, 10));
  return true;
}

std::vector<int> discover_kernel_ids_from_memory(const fs::path &memory_dir) {
  std::vector<int> ids;
  if (!fs::exists(memory_dir))
    return ids;
  for (const auto &entry : fs::directory_iterator(memory_dir)) {
    const std::string name = entry.path().filename().string();
    int kid = 0;
    if (entry.is_directory() && parse_kernel_dash_dir(name, kid))
      ids.push_back(kid);
    else if (entry.is_regular_file() && parse_raw_mem_filename(name, kid))
      ids.push_back(kid);
    else if (entry.is_regular_file() && parse_raw_memc_filename(name, kid))
      ids.push_back(kid);
    else if (entry.is_regular_file()) {
      unsigned part = 0;
      if (parse_raw_mem_part_filename(name, kid, part))
        ids.push_back(kid);
      else if (parse_raw_memc_part_filename(name, kid, part))
        ids.push_back(kid);
    }
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

std::vector<MemoryInst>
load_kernel_insts(int kernel_id, const Options &opt,
                  const std::unordered_map<uint64_t, IssueInfo> &issue_map) {
  std::vector<MemoryInst> insts;
  uint64_t seq = 0;
  const fs::path split_dir =
      opt.memory_dir / ("kernel-" + std::to_string(kernel_id));
  const fs::path memc_path =
      opt.memory_dir / ("kernel_" + std::to_string(kernel_id) + ".memc");
  const fs::path memc_zst_path =
      opt.memory_dir / ("kernel_" + std::to_string(kernel_id) + ".memc.zst");
  const bool use_memc =
      opt.input_format == "memc" ||
      (opt.input_format == "auto" &&
       (fs::exists(memc_path) || fs::exists(memc_zst_path)));
  const bool use_split =
      opt.input_format == "split" ||
      (opt.input_format == "auto" && fs::exists(split_dir) &&
       fs::is_directory(split_dir));

  if (use_memc) {
    insts = load_kernel_insts_memc(kernel_id, opt, issue_map);
  } else if (use_split) {
    for (const auto &path : list_files(split_dir)) {
      const std::string name = path.filename().string();
      int kid = 0;
      unsigned block = 0;
      if (!parse_split_mem_filename(name, kid, block))
        continue;
      if (kid != kernel_id)
        continue;
      std::ifstream in(path);
      if (!in)
        throw std::runtime_error("cannot open memory trace: " + path.string());
      std::string line;
      while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty())
          continue;
        MemoryInst mi =
            parse_memory_line(line, kernel_id, block, false, seq++,
                              opt.include_local);
        if (mi.op == 'N' || mi.lanes.empty())
          continue;
        auto it = issue_map.find(map_key(kernel_id, block));
        mi.sm_id = (it == issue_map.end())
                       ? (block % std::max(1u, opt.num_sms))
                       : it->second.sm_id;
        if (it != issue_map.end() && it->second.has_cta_start)
          mi.timestamp += it->second.cta_start;
        insts.push_back(std::move(mi));
      }
    }
  } else {
    const auto paths = raw_memory_files_for_kernel(opt.memory_dir, kernel_id);
    if (paths.empty())
      return insts;
    for (const auto &path : paths) {
      std::ifstream in(path);
      if (!in)
        throw std::runtime_error("cannot open memory trace: " + path.string());
      std::string line;
      while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty())
          continue;
        MemoryInst mi =
            parse_memory_line(line, kernel_id, 0, true, seq++, opt.include_local);
        if (mi.op == 'N' || mi.lanes.empty())
          continue;
        auto it = issue_map.find(map_key(kernel_id, mi.block_id));
        mi.sm_id = (it == issue_map.end())
                       ? (mi.block_id % std::max(1u, opt.num_sms))
                       : it->second.sm_id;
        if (it != issue_map.end() && it->second.has_cta_start)
          mi.timestamp += it->second.cta_start;
        insts.push_back(std::move(mi));
      }
    }
  }

  if (opt.sort_by_timestamp) {
    std::sort(insts.begin(), insts.end(),
              [](const MemoryInst &a, const MemoryInst &b) {
                if (a.timestamp != b.timestamp)
                  return a.timestamp < b.timestamp;
                if (a.sm_id != b.sm_id)
                  return a.sm_id < b.sm_id;
                if (a.block_id != b.block_id)
                  return a.block_id < b.block_id;
                return a.seq < b.seq;
              });
  }
  return insts;
}

void for_each_kernel_inst_raw(
    int kernel_id, const Options &opt,
    const std::unordered_map<uint64_t, IssueInfo> &issue_map,
    const std::function<void(MemoryInst &)> &consume) {
  const auto paths = raw_memory_files_for_kernel(opt.memory_dir, kernel_id);
  if (paths.empty())
    throw std::runtime_error("missing raw memory trace for kernel " +
                             std::to_string(kernel_id));
  uint64_t seq = 0;
  for (const auto &path : paths) {
    std::ifstream input(path);
    if (!input)
      throw std::runtime_error("cannot open raw memory trace: " + path.string());
    std::string line;
    while (std::getline(input, line)) {
      line = trim(line);
      if (line.empty())
        continue;
      MemoryInst instruction = parse_memory_line(
          line, kernel_id, 0, true, seq++, opt.include_local);
      if (instruction.op == 'N' || instruction.lanes.empty())
        continue;
      const auto issue = issue_map.find(map_key(kernel_id, instruction.block_id));
      instruction.sm_id =
          issue == issue_map.end()
              ? instruction.block_id % std::max(1u, opt.num_sms)
              : issue->second.sm_id;
      if (issue != issue_map.end() && issue->second.has_cta_start)
        instruction.timestamp += issue->second.cta_start;
      consume(instruction);
    }
    if (!input.eof())
      throw std::runtime_error("failed while streaming raw memory trace: " +
                               path.string());
  }
}

void usage(std::ostream &os) {
  os << "Usage: hyfiss-request-trace --trace-root DIR --hw-config FILE [options]\n"
        "       or: hyfiss-request-trace --configs DIR --memory-traces DIR --hw-config FILE [options]\n\n"
        "Options:\n"
        "  --output-dir DIR             Output directory, default request_trace_out\n"
        "  --kernels all|1,2,3|1-9      Kernels to process, default all\n"
        "  --emit-kernels all|1,2,3|1-9 Kernels to write after processing, default all\n"
        "  --input-format auto|split|raw|memc\n"
        "  --emit-level all|L1|L2|DRAM  Request levels to emit, default all\n"
        "  --output-phase all|prefill|decode  Phase to write; cache/model still processes all phases, default all\n"
        "  --output-format csv|accelsim|accelsim-compact|footprint|request-footprint|both|both-compact|both-footprint|summary, default both\n"
        "  --output-rotate-size SIZE    Rotate request output files at SIZE bytes, default 1T; set 0 to disable\n"
        "  --footprint-format csv|minimal|binary  Footprint record encoding, default csv\n"
        "  --checkpoint-dir DIR         Write V4 L1/L2 checkpoints at kernel boundaries\n"
        "  --checkpoint-interval N      Write a checkpoint before every N selected kernels\n"
        "  --checkpoint-only            Build checkpoints without request output\n"
        "  --restore-checkpoint FILE    Restore V4 L1/L2 state; first selected kernel must match\n"
        "  --semantic-file FILE         llama.cpp semantic RANGE/PHASE sidecar file\n"
        "  --semantic-output none|id|fields|tag, default none; id writes sem_id plus semantic_tags.csv\n"
        "  --sector-size N              Coalescing sector bytes, default 32\n"
        "  --partition-index-bit N      Simple DRAM partition index bit, default 8\n"
        "  --l1-store-policy bypass|allocate, default bypass\n"
        "  --write-sector-policy line-miss-only|all, default line-miss-only\n"
        "  --dram-store-policy request|writeback, default writeback\n"
        "  --l1-fill-latency N          Cycles a missed L1 sector stays reserved\n"
        "  --l2-fill-latency N          Cycles a missed L2 sector stays reserved\n"
        "  --reset-l2-per-kernel        Disable cross-kernel L2 preservation\n"
        "  --flush-l2-on-reset          Emit dirty L2 writebacks before clearing L2 at kernel boundaries\n"
        "  --preserve-l1                Preserve L1 across kernels\n"
        "  --sort-by-timestamp          Sort by trace timestamp; with issue.config CTA starts, uses CTA-aware timestamps\n"
        "  --no-sort                    Preserve file order instead of timestamp order (default for raw traces)\n"
        "  --no-local                   Exclude local-memory lanes and do not allocate their owner namespace\n";
}

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(name + " requires value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage(std::cout);
      std::exit(0);
    } else if (arg == "--trace-root")
      opt.trace_root = need(arg);
    else if (arg == "--configs")
      opt.configs_dir = need(arg);
    else if (arg == "--memory-traces")
      opt.memory_dir = need(arg);
    else if (arg == "--hw-config")
      opt.hw_config = need(arg);
    else if (arg == "--output-dir")
      opt.output_dir = need(arg);
    else if (arg == "--kernels")
      opt.kernels = need(arg);
    else if (arg == "--emit-kernels")
      opt.emit_kernels = need(arg);
    else if (arg == "--input-format")
      opt.input_format = need(arg);
    else if (arg == "--emit-level")
      opt.emit_level = need(arg);
    else if (arg == "--output-phase")
      opt.output_phase = need(arg);
    else if (arg == "--output-format")
      opt.output_format = need(arg);
    else if (arg == "--output-rotate-size" || arg == "--output-rotate-bytes")
      opt.output_rotate_bytes = parse_size_bytes(need(arg));
    else if (arg == "--footprint-format")
      opt.footprint_format = need(arg);
    else if (arg == "--checkpoint-dir")
      opt.checkpoint_dir = need(arg);
    else if (arg == "--checkpoint-interval")
      opt.checkpoint_interval = parse_u64(need(arg), 0);
    else if (arg == "--checkpoint-only")
      opt.checkpoint_only = true;
    else if (arg == "--restore-checkpoint")
      opt.restore_checkpoint = need(arg);
    else if (arg == "--semantic-file")
      opt.semantic_file = need(arg);
    else if (arg == "--semantic-output")
      opt.semantic_output = need(arg);
    else if (arg == "--sector-size")
      opt.sector_size = static_cast<unsigned>(parse_u64(need(arg), 0));
    else if (arg == "--cache-line-size")
      opt.cache_line_size = static_cast<unsigned>(parse_u64(need(arg), 0));
    else if (arg == "--partition-index-bit")
      opt.partition_index_bit = static_cast<unsigned>(parse_u64(need(arg), 0));
    else if (arg == "--num-banks")
      opt.num_banks = static_cast<unsigned>(parse_u64(need(arg), 0));
    else if (arg == "--issue-interval")
      opt.issue_interval = static_cast<unsigned>(parse_u64(need(arg), 0));
    else if (arg == "--kernel-gap")
      opt.kernel_gap = static_cast<unsigned>(parse_u64(need(arg), 0));
    else if (arg == "--l1-store-policy")
      opt.l1_store_policy = need(arg);
    else if (arg == "--write-sector-policy")
      opt.write_sector_policy = need(arg);
    else if (arg == "--dram-store-policy")
      opt.dram_store_policy = need(arg);
    else if (arg == "--disable-l2-dirty-drain")
      opt.l2_dirty_drain = false;
    else if (arg == "--enable-l2-streaming-fill")
      opt.l2_streaming_fill = true;
    else if (arg == "--disable-l2-streaming-fill")
      opt.l2_streaming_fill = false;
    else if (arg == "--l2-dirty-drain-latency") {
      opt.l2_dirty_drain_latency = parse_u64(need(arg), 0);
      opt.l2_dirty_drain_latency_set = true;
    }
    else if (arg == "--l2-dirty-drain-max-sectors-per-kernel")
      opt.l2_dirty_drain_max_sectors_per_kernel = parse_u64(need(arg), 0);
    else if (arg == "--l2-dirty-drain-high-watermark-sectors")
      opt.l2_dirty_drain_high_watermark_sectors = parse_u64(need(arg), 0);
    else if (arg == "--l2-dirty-drain-target-sectors")
      opt.l2_dirty_drain_target_sectors = parse_u64(need(arg), 0);
    else if (arg == "--l1-fill-latency") {
      opt.l1_fill_latency = parse_u64(need(arg), 0);
      opt.l1_fill_latency_set = true;
    }
    else if (arg == "--l2-fill-latency") {
      opt.l2_fill_latency = parse_u64(need(arg), 0);
      opt.l2_fill_latency_set = true;
    }
    else if (arg == "--reset-l2-per-kernel")
      opt.preserve_l2 = false;
    else if (arg == "--flush-l2-on-reset")
      opt.flush_l2_on_reset = true;
    else if (arg == "--preserve-l1")
      opt.preserve_l1 = true;
    else if (arg == "--no-monotonic-sm")
      opt.monotonic_sm = false;
    else if (arg == "--sort-by-timestamp")
      opt.sort_by_timestamp = true;
    else if (arg == "--no-sort")
      opt.sort_by_timestamp = false;
    else if (arg == "--no-local")
      opt.include_local = false;
    else
      throw std::runtime_error("unknown argument: " + arg);
  }

  if (!opt.trace_root.empty()) {
    if (opt.configs_dir.empty())
      opt.configs_dir = opt.trace_root / "configs";
    if (opt.memory_dir.empty())
      opt.memory_dir = opt.trace_root / "memory_traces";
  }
  if (opt.configs_dir.empty() || opt.memory_dir.empty())
    throw std::runtime_error("provide --trace-root or both --configs and --memory-traces");
  if (opt.hw_config.empty())
    throw std::runtime_error("--hw-config is required");
  if (opt.input_format != "auto" && opt.input_format != "split" &&
      opt.input_format != "raw" && opt.input_format != "memc")
    throw std::runtime_error("--input-format must be auto, split, raw, or memc");
  if (opt.emit_level != "all" && opt.emit_level != "L1" &&
      opt.emit_level != "L2" && opt.emit_level != "DRAM")
    throw std::runtime_error("--emit-level must be all, L1, L2, or DRAM");
  if (opt.output_phase != "all" && opt.output_phase != "prefill" &&
      opt.output_phase != "decode")
    throw std::runtime_error("--output-phase must be all, prefill, or decode");
  if (opt.output_format != "csv" && opt.output_format != "accelsim" &&
      opt.output_format != "accelsim-compact" && opt.output_format != "footprint" &&
      opt.output_format != "request-footprint" && opt.output_format != "both" &&
      opt.output_format != "both-compact" && opt.output_format != "both-footprint" &&
      opt.output_format != "summary")
    throw std::runtime_error("--output-format must be csv, accelsim, accelsim-compact, footprint, request-footprint, both, both-compact, both-footprint, or summary");
  const auto footprint_format = parse_footprint_format(opt.footprint_format);
  if (opt.footprint_format != "csv" && opt.footprint_format != "minimal" &&
      opt.footprint_format != "minimal-csv" &&
      opt.footprint_format != "csv-minimal" &&
      opt.footprint_format != "binary" && opt.footprint_format != "bin" &&
      opt.footprint_format != "fbin")
    throw std::runtime_error("--footprint-format must be csv, minimal, or binary");
  opt.footprint_format = footprint_format_name(footprint_format);
  if (opt.semantic_output != "none" && opt.semantic_output != "id" &&
      opt.semantic_output != "fields" && opt.semantic_output != "tag")
    throw std::runtime_error("--semantic-output must be none, id, fields, or tag");
  if (opt.checkpoint_only)
    opt.output_format = "summary";
  if (opt.checkpoint_interval > 0 && opt.checkpoint_dir.empty())
    throw std::runtime_error("--checkpoint-interval requires --checkpoint-dir");
  if (opt.l1_store_policy != "bypass" && opt.l1_store_policy != "allocate")
    throw std::runtime_error("--l1-store-policy must be bypass or allocate");
  if (opt.write_sector_policy != "line-miss-only" &&
      opt.write_sector_policy != "all")
    throw std::runtime_error("--write-sector-policy must be line-miss-only or all");
  if (opt.dram_store_policy != "request" &&
      opt.dram_store_policy != "writeback")
    throw std::runtime_error("--dram-store-policy must be request or writeback");
  return opt;
}

void apply_hw_options(Options &opt, const HwParams &hw) {
  if(hw.profile) {
    const auto &p=*hw.profile;
    opt.sector_size=32;opt.num_banks=p.banks;opt.partition_index_bit=p.partition_bit;
    opt.l1_store_policy="bypass";opt.write_sector_policy=p.write_sector_policy;
    opt.dram_store_policy=p.dram_store_policy;opt.preserve_l1=false;
    opt.preserve_l2=p.preserve_l2;opt.flush_l2_on_reset=false;
    opt.l2_dirty_drain=false;opt.l2_streaming_fill=false;
    opt.l1_fill_latency_set=opt.l2_fill_latency_set=true;
    opt.l1_fill_latency=opt.l2_fill_latency=0;
    opt.l2_dirty_drain_latency_set=true;opt.l2_dirty_drain_latency=0;
    opt.l2_dirty_drain_max_sectors_per_kernel=0;
    opt.l2_dirty_drain_high_watermark_sectors=0;opt.l2_dirty_drain_target_sectors=0;
    opt.monotonic_sm=true;opt.issue_interval=p.issue_interval;
  }
  opt.num_sms = hw.num_sms;
  opt.num_partitions = hw.num_partitions;
  opt.num_memory_channels = hw.num_memory_channels;
  opt.num_sub_partitions_per_channel = hw.num_sub_partitions_per_channel;
  opt.memory_partition_indexing = hw.memory_partition_indexing;
  opt.mem_address_mask = hw.mem_address_mask;
  opt.mem_addr_mapping = hw.mem_addr_mapping;
  opt.l1_size_bytes = hw.l1_size_bytes;
  opt.l2_size_bytes = hw.l2_size_bytes;
  opt.l1_line_size = hw.l1_line_size;
  opt.l2_line_size = hw.l2_line_size;
  opt.cache_line_size = hw.l2_line_size;
  opt.l1_assoc = hw.l1_assoc;
  opt.l2_assoc = hw.l2_assoc;
  opt.l1_set_index = hw.l1_set_index;
  opt.l2_set_index = hw.l2_set_index;
  opt.kernel_gap = hw.kernel_gap;
  if (!opt.l1_fill_latency_set)
    opt.l1_fill_latency = hw.l1_fill_latency;
  if (!opt.l2_fill_latency_set)
    opt.l2_fill_latency = hw.l2_fill_latency;
  if (!opt.l2_dirty_drain_latency_set)
    opt.l2_dirty_drain_latency = opt.l2_fill_latency;
  const uint64_t l2_total_sectors =
      opt.sector_size ? opt.l2_size_bytes / opt.sector_size : 0;
  if (l2_total_sectors > 0) {
    if (opt.l2_dirty_drain_high_watermark_sectors == 0)
      opt.l2_dirty_drain_high_watermark_sectors =
          std::max<uint64_t>(1, (l2_total_sectors * 3) / 4);
    if (opt.l2_dirty_drain_target_sectors == 0)
      opt.l2_dirty_drain_target_sectors =
          std::max<uint64_t>(1, l2_total_sectors / 2);
    if (opt.l2_dirty_drain_target_sectors >=
        opt.l2_dirty_drain_high_watermark_sectors)
      opt.l2_dirty_drain_target_sectors =
          opt.l2_dirty_drain_high_watermark_sectors - 1;
  }
}

} // namespace

namespace hyfiss_request_trace {

namespace {

unsigned popcount32(uint32_t v) {
  unsigned c = 0;
  while (v) {
    v &= v - 1;
    ++c;
  }
  return c;
}

MemoryInst make_memory_inst_from_hyfiss(const mem_instn &src, int kernel_id,
                                        unsigned sm_id, uint64_t seq,
                                        bool include_local) {
  MemoryInst inst;
  inst.kernel_id = kernel_id;
  inst.block_id = src.block_id;
  inst.sm_id = src.sm_id ? src.sm_id : sm_id;
  inst.seq = src.sequence ? src.sequence : seq;
  inst.pc = src.pc;
  inst.opcode = src.opcode;
  inst.mask = src.mask;
  inst.timestamp = src.time_stamp;
  inst.mem_width = infer_width(inst.opcode);
  inst.op = classify_opcode(inst.opcode, include_local);
  if (inst.op == 'N')
    return inst;

  inst.lanes.reserve(src.addr.size());
  std::vector<unsigned> active_lanes;
  active_lanes.reserve(kWarpSize);
  for (unsigned lane = 0; lane < kWarpSize; ++lane) {
    if ((inst.mask >> lane) & 1u)
      active_lanes.push_back(lane);
  }
  if (active_lanes.empty()) {
    for (unsigned lane = 0; lane < kWarpSize && lane < src.addr.size(); ++lane)
      active_lanes.push_back(lane);
  }
  const unsigned active_count = std::max(1u, popcount32(inst.mask));
  unsigned groups = static_cast<unsigned>((src.addr.size() + active_count - 1) /
                                          active_count);
  groups = std::max(1u, groups);
  size_t addr_index = 0;
  for (unsigned ref = 1; ref <= groups && addr_index < src.addr.size(); ++ref) {
    for (unsigned lane : active_lanes) {
      if (addr_index >= src.addr.size())
        break;
      inst.lanes.push_back(LaneAddress{lane, ref, src.addr[addr_index++]});
    }
  }
  return inst;
}

void assign_memory_inst_from_ordered(
    const hyfiss_request_trace::OrderedMemoryInst &src, int kernel_id,
    uint64_t seq, bool include_local, MemoryInst &inst) {
  inst.kernel_id = kernel_id;
  inst.block_id = src.block_id;
  inst.sm_id = src.sm_id;
  inst.seq = src.sequence ? src.sequence : seq;
  inst.pc = src.pc;
  inst.opcode = src.opcode;
  inst.mask = src.mask;
  inst.timestamp = src.timestamp;
  inst.mem_width = infer_width(inst.opcode);
  inst.op = classify_opcode(inst.opcode, include_local);
  inst.lanes.clear();
  if (inst.op == 'N')
    return;
  std::array<unsigned, kWarpSize> active_lanes{};
  unsigned active_lane_count = 0;
  for (unsigned lane = 0; lane < kWarpSize; ++lane)
    if ((inst.mask >> lane) & 1u)
      active_lanes[active_lane_count++] = lane;
  if (active_lane_count == 0)
    for (unsigned lane = 0; lane < kWarpSize && lane < src.addr.size(); ++lane)
      active_lanes[active_lane_count++] = lane;
  const unsigned active_count = std::max(1u, popcount32(inst.mask));
  unsigned groups = static_cast<unsigned>(
      (src.addr.size() + active_count - 1) / active_count);
  groups = std::max(1u, groups);
  if (inst.lanes.capacity() < src.addr.size())
    inst.lanes.reserve(src.addr.size());
  std::size_t address = 0;
  for (unsigned ref = 1; ref <= groups && address < src.addr.size(); ++ref)
    for (unsigned lane_index = 0; lane_index < active_lane_count; ++lane_index) {
      if (address >= src.addr.size()) break;
      inst.lanes.push_back(LaneAddress{
          active_lanes[lane_index], ref, src.addr[address++]});
    }
}

MemoryInst make_memory_inst_from_ordered(
    const hyfiss_request_trace::OrderedMemoryInst &src, int kernel_id,
    uint64_t seq, bool include_local) {
  MemoryInst inst;
  assign_memory_inst_from_ordered(src, kernel_id, seq, include_local, inst);
  return inst;
}

void apply_backend_options(Options &opt, const BackendOptions &backend_opt) {
  opt.hw_config = backend_opt.hw_config;
  opt.output_dir = backend_opt.output_dir.empty() ? fs::path("request_trace_out")
                                                  : fs::path(backend_opt.output_dir);
  opt.semantic_file = backend_opt.semantic_file;
  opt.semantic_summary = backend_opt.semantic_summary;
  opt.output_format = backend_opt.output_format.empty() ? "summary"
                                                        : backend_opt.output_format;
  opt.emit_level = backend_opt.emit_level.empty() ? "DRAM" : backend_opt.emit_level;
  opt.l1_store_policy = backend_opt.l1_store_policy.empty()
                            ? "bypass"
                            : backend_opt.l1_store_policy;
  opt.write_sector_policy = backend_opt.write_sector_policy.empty()
                                ? "line-miss-only"
                                : backend_opt.write_sector_policy;
  opt.dram_store_policy = backend_opt.dram_store_policy.empty()
                              ? "writeback"
                              : backend_opt.dram_store_policy;
  opt.l2_dirty_drain = backend_opt.l2_dirty_drain;
  opt.l2_streaming_fill = backend_opt.l2_streaming_fill;
  opt.l2_dirty_drain_latency = backend_opt.l2_dirty_drain_latency;
  opt.l2_dirty_drain_latency_set = backend_opt.l2_dirty_drain_latency_set;
  opt.l2_dirty_drain_max_sectors_per_kernel =
      backend_opt.l2_dirty_drain_max_sectors_per_kernel;
  opt.l2_dirty_drain_high_watermark_sectors =
      backend_opt.l2_dirty_drain_high_watermark_sectors;
  opt.l2_dirty_drain_target_sectors =
      backend_opt.l2_dirty_drain_target_sectors;
  opt.preserve_l2 = backend_opt.preserve_l2;
  opt.preserve_l1 = backend_opt.preserve_l1;
  opt.monotonic_sm = backend_opt.monotonic_sm;
  opt.include_local = backend_opt.include_local;
  opt.sector_size = backend_opt.sector_size;
  opt.issue_interval = backend_opt.issue_interval;
  opt.kernel_gap = backend_opt.kernel_gap;
  opt.l1_fill_latency_set = backend_opt.l1_fill_latency_set;
  opt.l2_fill_latency_set = backend_opt.l2_fill_latency_set;
  opt.l1_fill_latency = backend_opt.l1_fill_latency;
  opt.l2_fill_latency = backend_opt.l2_fill_latency;
}

bool valid_backend_order(const std::string &order) {
  return order == "hyfiss-sm" || order == "sm-major" ||
         order == "timestamp";
}

} // namespace

int run_from_sm_trace_source(
    const BackendOptions &supplied_options,
    const std::function<bool(KernelTraceRef &)> &next_kernel) {
  BackendOptions backend_opt=supplied_options;
  try {
    if (backend_opt.hw_config.empty())
      throw std::runtime_error("request backend requires hw_config");
    if (backend_opt.output_dir.empty())
      throw std::runtime_error("request backend requires output_dir");
    if (!valid_backend_order(backend_opt.order))
      throw std::runtime_error("request backend order must be hyfiss-sm, sm-major, or timestamp");

    Options opt;
    apply_backend_options(opt, backend_opt);
    const HwParams hw = read_hw_params(opt.hw_config,backend_opt.hardware_profile);
    apply_hw_options(opt, hw);
    if(hw.profile) {
      if(backend_opt.r4_l1_read_filter && backend_opt.r4_model_id!=hw.profile->values.at("context_model_id"))
        throw std::runtime_error("hardware config and kernel context model identities differ");
      backend_opt.r4_l1_read_filter=true;
      backend_opt.r4_model_id=hw.profile->values.at("context_model_id");
      backend_opt.order="timestamp";
    }

    if (backend_opt.r4_l1_read_filter && (opt.preserve_l1 ||
        opt.l1_store_policy!="bypass" || opt.l1_fill_latency!=0 || opt.sector_size!=32))
      throw std::runtime_error("r4 requires per-kernel L1 reset, store bypass and zero L1 fill latency");
    if(backend_opt.r4_l1_read_filter &&
       backend_opt.r4_model_id!="CLOCK_u128_s16_h2_c1062" &&
       backend_opt.r4_model_id!="r4-small-shared-20260922")
      throw std::runtime_error("unknown r4 model identity");
    std::unique_ptr<R4L1ReadFilter> r4;
    if (backend_opt.r4_l1_read_filter)
      r4=std::make_unique<R4L1ReadFilter>(std::max(1u,opt.num_sms),hw.profile);
    std::ofstream r4_ledger;

    if (backend_opt.observe_cache && (opt.output_format!="summary" || opt.l2_line_size!=128 ||
        opt.sector_size!=32 || !opt.preserve_l2 || opt.l2_dirty_drain ||
        opt.dram_store_policy!="writeback" || opt.l2_streaming_fill))
      throw std::runtime_error("observation requires summary, 128B/32B, persistent writeback L2 without drain/streaming");
    CacheObservation observation;
    CacheInputCensus input_census;
    fs::create_directories(opt.output_dir);
    if(hw.profile) {
      const auto snapshot=opt.output_dir/"hardware.config";
      const auto resolved=opt.output_dir/"hardware.resolved.json";
      if(fs::exists(snapshot)||fs::exists(resolved))throw std::runtime_error("hardware receipt exists; use fresh output directory");
      std::ofstream copy(snapshot,std::ios::binary);copy<<hw.profile->source_text;
      if(!copy)throw std::runtime_error("hardware snapshot write failed");
      boost::property_tree::write_json(resolved.string(),hw.profile->resolved());
    }
    if (r4) {
      if(fs::exists(opt.output_dir/"r4_l1_profiles.csv"))
        throw std::runtime_error("r4 ledger exists; use fresh output directory");
      r4_ledger.open(opt.output_dir/"r4_l1_profiles.csv");
      if(!r4_ledger)throw std::runtime_error("cannot open r4 ledger");
      r4_ledger<<"kernel_id,shared_kib,effective_bytes,sets,ways,allocations,model\n";
    }
    RotatingOutput req_out;
    if (wants_csv(opt)) {
      req_out.open(opt.output_dir / "requests.csv", request_csv_header(opt),
                   opt.output_rotate_bytes);
    }
    RotatingOutput accel_out;
    if (wants_accelsim(opt)) {
      accel_out.open(opt.output_dir / "accelsim_mem.trace", "",
                     opt.output_rotate_bytes);
    }
    FootprintWriter footprint_out;
    if (wants_footprint(opt)) {
      footprint_out.open(opt.output_dir / "request_footprint.trace", opt,
                         opt.output_rotate_bytes);
    }

    std::map<int, KernelStats> stats;
    AccelSimAddressMapping addr_mapping;
    addr_mapping.init(opt);
    g_addr_mapping = &addr_mapping;
    SemanticDatabase semantic_db;
    if ((wants_semantics(opt) || has_semantic_policy_file(opt)) &&
        !opt.semantic_file.empty())
      semantic_db.load(opt.semantic_file);
    if (opt.semantic_summary && !semantic_db.enabled())
      throw std::runtime_error(
          "semantic summary requires a non-empty semantic RANGE sidecar");
    SemanticTrafficLedger semantic_traffic;
    g_semantic_traffic_ledger = opt.semantic_summary ? &semantic_traffic : nullptr;

    std::vector<SectorLruCache> l1_caches;
    for (unsigned sm = 0; !r4 && sm < std::max(1u, opt.num_sms); ++sm)
      l1_caches.emplace_back(opt.l1_size_bytes, opt.l1_line_size,
                             opt.l1_assoc, opt.l1_set_index);
    const unsigned l2_partitions = std::max(1u, opt.num_partitions);
    const uint64_t l2_size_per_partition =
        std::max<uint64_t>(opt.l2_line_size * opt.l2_assoc,
                           opt.l2_size_bytes / l2_partitions);
    std::vector<SectorLruCache> l2_caches;
    for (unsigned p = 0; p < l2_partitions; ++p)
      l2_caches.emplace_back(l2_size_per_partition, opt.l2_line_size,
                             opt.l2_assoc, opt.l2_set_index);

    uint64_t kernel_base = 0;
    uint64_t total_records = 0;
    uint64_t accelsim_records = 0;
    uint64_t accelsim_uid = 1;
    auto drain_l2_dirty = [&](const KernelMeta &meta, uint64_t timestamp) {
      if (!opt.l2_dirty_drain || opt.dram_store_policy != "writeback" ||
          !opt.preserve_l2)
        return;
      auto &ks = stats[meta.id];
      uint64_t total_dirty = 0;
      for (const auto &cache : l2_caches)
        total_dirty += cache.dirty_sector_count();
      if (total_dirty <= opt.l2_dirty_drain_high_watermark_sectors)
        return;
      uint64_t remaining =
          total_dirty > opt.l2_dirty_drain_target_sectors
              ? total_dirty - opt.l2_dirty_drain_target_sectors
              : 0;
      if (opt.l2_dirty_drain_max_sectors_per_kernel)
        remaining = std::min(remaining,
                             opt.l2_dirty_drain_max_sectors_per_kernel);
      if (remaining == 0)
        return;
      MemoryInst drain_inst;
      drain_inst.kernel_id = meta.id;
      drain_inst.opcode = "L2_DIRTY_DRAIN";
      drain_inst.op = 'W';
      drain_inst.timestamp = timestamp;
      for (auto &cache : l2_caches) {
        const uint64_t per_cache_budget = remaining;
        for (const auto &evicted : cache.drain_eligible_dirty(
                 opt.sector_size, timestamp, opt.l2_dirty_drain_latency,
                 per_cache_budget)) {
          const unsigned dirty_sectors = popcount32(evicted.dirty_sectors);
          if (dirty_sectors == 0)
            continue;
          ks.l2_dirty_drain_events++;
          ks.l2_dirty_drain_sectors += dirty_sectors;
          if (remaining) {
            if (dirty_sectors >= remaining)
              remaining = 0;
            else
              remaining -= dirty_sectors;
          }
          emit_l2_writeback(evicted, timestamp, meta, drain_inst, false, false, opt,
              semantic_db, nullptr, ks, req_out, accel_out, footprint_out, total_records,
              accelsim_records, accelsim_uid);
        }
        if (remaining == 0)
          break;
      }
    };

    std::vector<SectorRequest> reusable_sector_requests;
    auto process_inst = [&](const KernelMeta &meta, MemoryInst &inst,
                            std::vector<uint64_t> &last_ts,
                            uint64_t &kernel_max_ts) {
      if (r4 && inst.sm_id >= last_ts.size())
        throw std::runtime_error("r4 SM id outside hardware configuration");
      if (inst.sm_id >= last_ts.size())
        inst.sm_id %= last_ts.size();
      uint64_t ts = kernel_base + inst.timestamp;
      if (opt.monotonic_sm) {
        uint64_t &last = last_ts[inst.sm_id];
        if (last != 0 && ts <= last)
          ts = last + opt.issue_interval;
        last = ts;
      }
      kernel_max_ts = std::max(kernel_max_ts, ts);
      auto &ks = stats[meta.id];
      ks.mem_insts++;
      ks.lane_accesses += inst.lanes.size();
      if (inst.op == 'R')
        ks.reads++;
      else if (inst.op == 'W')
        ks.writes++;
      else if (inst.op == 'A')
        ks.atomics++;

      assign_coalesced_sectors(
          inst, opt.sector_size, reusable_sector_requests);
      const auto &reqs = reusable_sector_requests;
      if (backend_opt.observe_cache) observation.instruction(meta,inst,reqs,semantic_db);
      account_write_coverage(ks, inst, reqs);
      ks.sector_requests += reqs.size();
      if (inst.op == 'R')
        ks.read_sector_requests += reqs.size();
      else if (inst.op == 'W')
        ks.write_sector_requests += reqs.size();
      else if (inst.op == 'A')
        ks.atomic_sector_requests += reqs.size();
      if (inst.op == 'R' && inst.opcode.find("LTC128B") != std::string::npos)
        ks.ldg_ltc128b_insts++;
      if (starts_with(inst.opcode, "LDGSTS"))
        ks.ldgsts_insts++;
      if (inst.op == 'R' && inst.mem_width >= 16)
        ks.wide_load_insts++;
      if (inst.op == 'W' && inst.mem_width >= 16)
        ks.wide_store_insts++;
      for (const auto &req : reqs) {
        if(r4)r4->require_sector(req.addr,req.byte_mask);
        const SemanticInfo &req_sem = semantic_db.lookup(req.addr);
        if (g_semantic_traffic_ledger)
          g_semantic_traffic_ledger->add_source(
              meta.id, meta.llm_phase, req_sem, inst.op, req.size);
        bool l1_hit = false;
        bool l2_hit = false;
        CacheAccess l1_access, l2_access;
        CacheResult l1_result = CacheResult::Hit;
        CacheResult l2_result = CacheResult::Hit;
        const bool bypass_l1 =
            (inst.op == 'W' && opt.l1_store_policy == "bypass") ||
            inst.op == 'A';
        if (!bypass_l1) {
          ks.l1_requests++;
          if (bypass_l1_read(inst)) l1_access.result=CacheResult::SectorMiss;
          else if (r4) {
            if(inst.op!='R' || req.size!=32)
              throw std::runtime_error("r4 accepts read sectors only");
            auto a=r4->access(inst.sm_id,req.addr,req.byte_mask);
            l1_access.result=a.outcome==0?CacheResult::Hit:
                             a.outcome==2?CacheResult::LineMiss:CacheResult::SectorMiss;
            l1_access.valid_before=a.before;l1_access.valid_after=a.after;
            l1_access.evicted.present=a.victim;l1_access.evicted.addr=a.victim_addr;
            l1_access.evicted.valid_sectors=a.victim_valid;
          }
          else l1_access=l1_caches[inst.sm_id].access(req.addr, req.size, opt.sector_size, ts,
                          opt.l1_fill_latency, cache_operation(inst.op),
                          false, 0, false, false, req.byte_mask);
          l1_result=l1_access.result;
          if (backend_opt.observe_l1_access && !bypass_l1_read(inst)) {
            const auto &a=l1_access; const auto &v=a.evicted;
            const int outcome=a.result==CacheResult::Hit?0:a.result==CacheResult::HitReserved?1:a.result==CacheResult::LineMiss?2:3;
            backend_opt.observe_l1_access(L2AccessObservation{meta.id,inst.sm_id,req.addr,req.addr,inst.op,req.byte_mask,outcome,
              a.valid_before,a.dirty_before,a.valid_after,a.dirty_after,v.present,v.addr,v.valid_sectors,v.dirty_sectors});
          }
          l1_hit = l1_result == CacheResult::Hit ||
                   l1_result == CacheResult::HitReserved;
          if (l1_result == CacheResult::Hit)
            ks.l1_hits++;
          else if (l1_result == CacheResult::HitReserved)
            ks.l1_pending_hits++;
          else {
            ks.l1_misses++;
            if (l1_result == CacheResult::LineMiss)
              ks.l1_line_misses++;
            else
              ks.l1_sector_misses++;
          }
          if (wants_csv(opt) && should_emit(opt, "L1") &&
              should_emit_phase(opt, meta)) {
            write_request(req_out, ts, meta, inst, req, "L1", l1_hit, false,
                          opt, &req_sem);
            ++total_records;
          }
        }

        const bool write_like = inst.op == 'W' || inst.op == 'A';
        const bool needs_dram_read = inst.op != 'W';
        const bool l2_lookup = write_like || bypass_l1 || !l1_hit;
        EvictedLine l2_evicted;
        if (l2_lookup) {
          const unsigned l2_partition =
              dram_partition_index(req.addr, opt) % l2_partitions;
          const uint64_t l2_index_addr = l2_cache_index_addr(req.addr, opt);
          const bool stream_l2 = opt.l2_streaming_fill &&
                                 should_stream_l2_fill(meta, inst, req_sem);
          l2_access = l2_caches[l2_partition].access(
              req.addr, req.size, opt.sector_size, ts, opt.l2_fill_latency,
              cache_operation(inst.op), opt.dram_store_policy == "writeback" && write_like,
              l2_index_addr, true, stream_l2, req.byte_mask);
          l2_result = l2_access.result;
          l2_evicted = l2_access.evicted;
          if (backend_opt.observe_l2_access) {
            const auto &a=l2_access; const auto &v=a.evicted;
            const int outcome=a.result==CacheResult::Hit?0:a.result==CacheResult::HitReserved?1:a.result==CacheResult::LineMiss?2:3;
            backend_opt.observe_l2_access(L2AccessObservation{meta.id,l2_partition,req.addr,l2_index_addr,inst.op,req.byte_mask,outcome,
              a.valid_before,a.dirty_before,a.valid_after,a.dirty_after,v.present,v.addr,v.valid_sectors,v.dirty_sectors});
          }
          l2_hit = l2_result == CacheResult::Hit ||
                   l2_result == CacheResult::HitReserved;
          account_l2_lookup(ks, inst.op, l2_result);
          if (wants_csv(opt) && should_emit(opt, "L2") &&
              should_emit_phase(opt, meta)) {
            write_request(req_out, ts, meta, inst, req, "L2", l1_hit, l2_hit,
                          opt, &req_sem);
            ++total_records;
          }
        }

        if (backend_opt.observe_cache)
          observation.access(meta,inst,req,req_sem.id,!bypass_l1,l1_access,l2_lookup,l2_access);
        if (g_semantic_traffic_ledger) {
          g_semantic_traffic_ledger->add_cache(
              meta.id, meta.llm_phase, req_sem, !bypass_l1, l1_result,
              l2_lookup, l2_result, req.size);
          if (opt.dram_store_policy == "writeback" && write_like)
            g_semantic_traffic_ledger->mark_dirty(
                meta.id, meta.llm_phase, req_sem, req.addr, req.size,
                opt.sector_size);
        }

        if (opt.dram_store_policy == "writeback") {
          if ((write_like || bypass_l1 || !l1_hit) && !l2_hit && needs_dram_read) {
            // Atomic origin remains in opcode; DRAM fill direction is read.
            std::optional<MemoryInst> atomic_read;
            if (inst.op == 'A') { atomic_read = inst; atomic_read->op = 'R'; }
            const MemoryInst &transfer = atomic_read ? *atomic_read : inst;
            ks.dram_requests++;
            ks.dram_load_requests++;
            ks.dram_load_bytes += req.size;
            ks.dram_load_sectors += sectors_for_bytes(req.size, opt.sector_size);
            if (g_semantic_traffic_ledger)
              g_semantic_traffic_ledger->add_dram_read(
                  meta.id, meta.llm_phase, req_sem, req.size);
            if (wants_csv(opt) && should_emit(opt, "DRAM") &&
                should_emit_phase(opt, meta)) {
              write_request(req_out, ts, meta, transfer, req, "DRAM", l1_hit,
                            l2_hit, opt, &req_sem);
              ++total_records;
            }
            if (wants_accelsim(opt) && should_emit_phase(opt, meta)) {
              write_accelsim_request(accel_out, accelsim_uid++, ts, meta, transfer,
                                     req, opt, &req_sem);
              ++accelsim_records;
            }
            if (wants_footprint(opt) && should_emit_phase(opt, meta)) {
              footprint_out.write_request(ts, meta, transfer, req, opt, &req_sem);
            }
          }
          emit_l2_writeback(l2_evicted, ts, meta, inst, l1_hit, l2_hit, opt,
              semantic_db, &req_sem, ks, req_out, accel_out, footprint_out, total_records,
              accelsim_records, accelsim_uid);
        } else {
          const bool emit_write_sector =
              opt.write_sector_policy == "all" ||
              l2_result == CacheResult::LineMiss;
          if ((write_like || bypass_l1 || !l1_hit) && !l2_hit &&
              (!write_like || emit_write_sector)) {
            ks.dram_requests++;
            if (write_like) {
              ks.dram_store_requests++;
              ks.dram_store_bytes += req.size;
              ks.dram_store_sectors += sectors_for_bytes(req.size, opt.sector_size);
              if (g_semantic_traffic_ledger)
                g_semantic_traffic_ledger->add_dram_write(
                    meta.id, meta.llm_phase, req_sem, req.size);
            } else {
              ks.dram_load_requests++;
              ks.dram_load_bytes += req.size;
              ks.dram_load_sectors += sectors_for_bytes(req.size, opt.sector_size);
              if (g_semantic_traffic_ledger)
                g_semantic_traffic_ledger->add_dram_read(
                    meta.id, meta.llm_phase, req_sem, req.size);
            }
            if (wants_csv(opt) && should_emit(opt, "DRAM") &&
                should_emit_phase(opt, meta)) {
              write_request(req_out, ts, meta, inst, req, "DRAM", l1_hit,
                            l2_hit, opt, &req_sem);
              ++total_records;
            }
            if (wants_accelsim(opt) && should_emit_phase(opt, meta)) {
              write_accelsim_request(accel_out, accelsim_uid++, ts, meta, inst,
                                     req, opt, &req_sem);
              ++accelsim_records;
            }
            if (wants_footprint(opt) && should_emit_phase(opt, meta)) {
              footprint_out.write_request(ts, meta, inst, req, opt, &req_sem);
            }
          }
        }
      }
    };

    KernelTraceRef kernel_ref;
    while (next_kernel(kernel_ref)) {
      const bool has_ordered_source = static_cast<bool>(kernel_ref.next_ordered_inst);
      if (backend_opt.observe_cache && !has_ordered_source)
        throw std::runtime_error("cache observation requires the ordered streaming source");
      if (kernel_ref.sm_traces == nullptr && !has_ordered_source)
        continue;
      if (kernel_ref.sm_traces != nullptr && has_ordered_source)
        throw std::runtime_error(
            "HyFiSS backend kernel supplied both materialized and ordered sources");
      KernelMeta meta;
      meta.id = kernel_ref.kernel_id;
      meta.name = kernel_ref.kernel_name.empty() ? "unknown" : kernel_ref.kernel_name;
      meta.llm_phase = kernel_ref.llm_phase.empty() ? "unknown" : kernel_ref.llm_phase;
      if(r4) {
        if((kernel_ref.r4_shared_kib==8 || kernel_ref.r4_shared_kib==16) &&
           backend_opt.r4_model_id!="r4-small-shared-20260922")
          throw std::runtime_error("small shared candidate requires explicit model identity");
        r4->begin_kernel(kernel_ref.r4_shared_kib,kernel_ref.r4_allocations);
        r4_ledger<<meta.id<<','<<kernel_ref.r4_shared_kib<<','<<r4->sets()*128*r4->ways()
                 <<','<<r4->sets()<<','<<r4->ways()<<','<<kernel_ref.r4_allocations.size()
                 <<','<<backend_opt.r4_model_id<<"\n";
        if(!r4_ledger)throw std::runtime_error("r4 ledger write failed");
      }
      if (!opt.preserve_l1) {
        for (auto &c : l1_caches)
          c.clear();
      }
      if (!opt.preserve_l2) {
        for (auto &c : l2_caches)
          c.clear();
        if (g_semantic_traffic_ledger)
          g_semantic_traffic_ledger->discard_dirty_without_service();
      }

      if (g_semantic_traffic_ledger) {
        uint64_t entry_dirty = 0;
        for (const auto &cache : l2_caches)
          entry_dirty += cache.dirty_sector_count();
        g_semantic_traffic_ledger->begin_kernel(
            meta.id, meta.name, meta.llm_phase, entry_dirty);
      }

      if (backend_opt.observe_cache) {
        uint64_t count=0; for(const auto &c:l2_caches) count+=c.dirty_sector_count();
        observation.begin(meta.id,count);
        observation.occupancy(meta.id,"entry",l2_caches);
      }
      std::vector<uint64_t> last_ts(std::max(1u, opt.num_sms), 0);
      uint64_t kernel_max_ts = 0;
      uint64_t seq = 0;

      if (has_ordered_source) {
        if (backend_opt.order != "timestamp")
          throw std::runtime_error(
              "ordered instruction source requires backend order=timestamp");
        hyfiss_request_trace::OrderedMemoryInst source;
        MemoryInst inst;
        while (kernel_ref.next_ordered_inst(source)) {
          if (backend_opt.observe_cache || opt.include_local) input_census.observe(meta,source,opt.include_local);
          assign_memory_inst_from_ordered(
              source, meta.id, seq++, opt.include_local, inst);
          if (inst.op != 'N' && !inst.lanes.empty())
            process_inst(meta, inst, last_ts, kernel_max_ts);
        }
      } else if (backend_opt.order == "timestamp") {
        std::vector<MemoryInst> insts;
        for (const auto &sm_pair : *kernel_ref.sm_traces) {
          for (const auto &src : sm_pair.second) {
            MemoryInst inst = make_memory_inst_from_hyfiss(
                src, meta.id, static_cast<unsigned>(sm_pair.first), seq++,
                opt.include_local);
            if (inst.op != 'N' && !inst.lanes.empty())
              insts.push_back(std::move(inst));
          }
        }
        std::sort(insts.begin(), insts.end(), [](const MemoryInst &a,
                                                const MemoryInst &b) {
          if (a.timestamp != b.timestamp)
            return a.timestamp < b.timestamp;
          if (a.sm_id != b.sm_id)
            return a.sm_id < b.sm_id;
          if (a.block_id != b.block_id)
            return a.block_id < b.block_id;
          return a.seq < b.seq;
        });
        for (auto &inst : insts)
          process_inst(meta, inst, last_ts, kernel_max_ts);
      } else if (backend_opt.order == "sm-major") {
        for (const auto &sm_pair : *kernel_ref.sm_traces) {
          for (const auto &src : sm_pair.second) {
            MemoryInst inst = make_memory_inst_from_hyfiss(
                src, meta.id, static_cast<unsigned>(sm_pair.first), seq++,
                opt.include_local);
            if (inst.op != 'N' && !inst.lanes.empty())
              process_inst(meta, inst, last_ts, kernel_max_ts);
          }
        }
      } else {
        size_t max_inst_count = 0;
        for (const auto &sm_pair : *kernel_ref.sm_traces)
          max_inst_count = std::max(max_inst_count, sm_pair.second.size());
        for (size_t inst_index = 0; inst_index < max_inst_count; ++inst_index) {
          for (const auto &sm_pair : *kernel_ref.sm_traces) {
            if (inst_index >= sm_pair.second.size())
              continue;
            MemoryInst inst = make_memory_inst_from_hyfiss(
                sm_pair.second[inst_index], meta.id,
                static_cast<unsigned>(sm_pair.first), seq++, opt.include_local);
            if (inst.op != 'N' && !inst.lanes.empty())
              process_inst(meta, inst, last_ts, kernel_max_ts);
          }
        }
      }

      drain_l2_dirty(meta, kernel_max_ts + 1);
      if (backend_opt.observe_cache) {
        uint64_t count=0; for(const auto &c:l2_caches) count+=c.dirty_sector_count();
        observation.end(meta.id,count);
        observation.occupancy(meta.id,"exit",l2_caches);
      }
      if (g_semantic_traffic_ledger) {
        uint64_t exit_dirty = 0;
        for (const auto &cache : l2_caches)
          exit_dirty += cache.dirty_sector_count();
        g_semantic_traffic_ledger->end_kernel(meta.id, exit_dirty);
      }
      std::cerr << "hyfiss backend kernel " << meta.id << " " << meta.name
                << ": " << stats[meta.id].mem_insts
                << " memory instructions, " << stats[meta.id].dram_requests
                << " DRAM requests\n";
      kernel_base = std::max(kernel_base, kernel_max_ts + opt.kernel_gap);
    }

    footprint_out.flush();
    if (backend_opt.observe_cache) {
      input_census.write(opt.output_dir,stats);
      observation.write(opt.output_dir,stats);
    }
    if (opt.semantic_summary)
      semantic_traffic.write(opt.output_dir, stats);
    g_semantic_traffic_ledger = nullptr;

    std::ofstream sum(opt.output_dir / "kernel_summary.csv");
    sum << "kernel_id,mem_insts,lane_accesses,sector_requests,read_sector_requests,write_sector_requests,atomic_sector_requests,l1_requests,l1_hits,l1_pending_hits,l1_misses,l1_line_misses,l1_sector_misses,l1_hit_rate,l2_requests,l2_hits,l2_pending_hits,l2_misses,l2_line_misses,l2_sector_misses,l2_hit_rate,dram_requests,dram_load_requests,dram_store_requests,dram_load_sectors,dram_store_sectors,dram_load_bytes,dram_store_bytes,l2_writeback_events,l2_writeback_dirty_sectors,l2_dirty_drain_events,l2_dirty_drain_sectors,ldg_ltc128b_insts,ldgsts_insts,wide_load_insts,wide_store_insts,reads,writes,atomics,write_full_sector_requests,write_partial_sector_requests,write_covered_bytes";
    write_l2_direction_header(sum);
    sum << '\n';
    for (const auto &kv : stats) {
      const auto &s = kv.second;
      const double l1_hr =
          s.l1_requests ? static_cast<double>(s.l1_hits + s.l1_pending_hits) / s.l1_requests : 0.0;
      const double l2_hr =
          s.l2_requests ? static_cast<double>(s.l2_hits + s.l2_pending_hits) / s.l2_requests : 0.0;
      sum << kv.first << ',' << s.mem_insts << ',' << s.lane_accesses << ','
          << s.sector_requests << ',' << s.read_sector_requests << ','
          << s.write_sector_requests << ',' << s.atomic_sector_requests << ','
          << s.l1_requests << ',' << s.l1_hits << ',' << s.l1_pending_hits
          << ',' << s.l1_misses << ',' << s.l1_line_misses << ','
          << s.l1_sector_misses << ',' << std::fixed << std::setprecision(6)
          << l1_hr << ',' << s.l2_requests << ',' << s.l2_hits << ','
          << s.l2_pending_hits << ',' << s.l2_misses << ','
          << s.l2_line_misses << ',' << s.l2_sector_misses << ','
          << std::fixed << std::setprecision(6) << l2_hr << ','
          << s.dram_requests << ',' << s.dram_load_requests << ','
          << s.dram_store_requests << ',' << s.dram_load_sectors << ','
          << s.dram_store_sectors << ',' << s.dram_load_bytes << ','
          << s.dram_store_bytes << ',' << s.l2_writeback_events << ','
          << s.l2_writeback_dirty_sectors << ',' << s.l2_dirty_drain_events << ','
          << s.l2_dirty_drain_sectors << ',' << s.ldg_ltc128b_insts << ','
          << s.ldgsts_insts << ',' << s.wide_load_insts << ','
          << s.wide_store_insts << ',' << s.reads << ',' << s.writes << ','
          << s.atomics << ',' << s.write_full_sector_requests << ',' << s.write_partial_sector_requests << ',' << s.write_covered_bytes;
      write_l2_direction_stats(sum, s);
      sum << '\n';
    }

    std::ofstream run(opt.output_dir / "run_summary.txt");
    run << "HyFiSS internal request trace backend\n"
        << "hw_config=" << opt.hw_config << "\n"
        << "output_format=" << opt.output_format << "\n"
        << "footprint_format=" << opt.footprint_format << "\n"
        << "semantic_output=" << opt.semantic_output << "\n"
        << "semantic_file=" << opt.semantic_file << "\n"
        << "emit_level=" << opt.emit_level << "\n"
        << "output_phase=" << opt.output_phase << "\n"
        << "order=" << backend_opt.order << "\n"
        << "l1_filter_model=" << (r4?backend_opt.r4_model_id:"legacy") << "\n"
        << "l1_geometry_source=" << (r4?(hw.profile?"hardware.resolved.json;r4_l1_profiles.csv":"r4_l1_profiles.csv;config_geometry_below_unused"):"hw_config") << "\n"
        << "include_local=" << (opt.include_local ? 1 : 0) << "\n"
        << "num_sms=" << opt.num_sms << "\n"
        << "sector_size=" << opt.sector_size << "\n"
        << "l1_size_bytes=" << opt.l1_size_bytes
        << " l1_line_size=" << opt.l1_line_size
        << " l1_assoc=" << opt.l1_assoc
        << " l1_set_index=" << set_index_name(opt.l1_set_index) << "\n"
        << "l2_size_bytes=" << opt.l2_size_bytes
        << " l2_size_per_partition=" << l2_size_per_partition
        << " l2_line_size=" << opt.l2_line_size
        << " l2_assoc=" << opt.l2_assoc
        << " l2_set_index=" << set_index_name(opt.l2_set_index) << "\n"
        << "cache_data_validity=known_bytes_union_v1;pending_read_independent_v1\n"
        << "l2_lookup_statistics=directional_residency_v1;write_not_ncu_hit_v1\n"
        << "l1_read_policy=ldg_strong_gpu_bypass_v1_sm89_validated\n"
        << "l2_writeback_transfer=dirty_sector_runs_v1\n"
        << "l2_dirty_drain_budget=global_remaining_v2\n"
        << "l2_index_coordinate="
        << (addr_mapping.enabled() ? "explicit_accelsim_v1" : "fallback_quotient_v2") << "\n"
        << "num_partitions=" << opt.num_partitions
        << " num_memory_channels=" << opt.num_memory_channels
        << " num_sub_partitions_per_channel=" << opt.num_sub_partitions_per_channel
        << " num_banks=" << opt.num_banks
        << " partition_index_bit=" << opt.partition_index_bit << "\n"
        << "address_mapping=" << (addr_mapping.enabled() ? "accelsim" : "low-mask")
        << " memory_partition_indexing=" << opt.memory_partition_indexing
        << " mem_address_mask=" << opt.mem_address_mask
        << " mem_addr_mapping=" << opt.mem_addr_mapping << "\n"
        << "preserve_l2=" << (opt.preserve_l2 ? 1 : 0)
        << " flush_l2_on_reset=" << (opt.flush_l2_on_reset ? 1 : 0)
        << " preserve_l1=" << (opt.preserve_l1 ? 1 : 0) << "\n"
        << "l1_store_policy=" << opt.l1_store_policy << "\n"
        << "write_sector_policy=" << opt.write_sector_policy << "\n"
        << "dram_store_policy=" << opt.dram_store_policy << "\n"
        << "l1_fill_latency=" << opt.l1_fill_latency << "\n"
        << "l2_fill_latency=" << opt.l2_fill_latency << "\n"
        << "l2_dirty_drain=" << (opt.l2_dirty_drain ? 1 : 0) << "\n"
        << "l2_streaming_fill=" << (opt.l2_streaming_fill ? 1 : 0) << "\n"
        << "l2_dirty_drain_latency=" << opt.l2_dirty_drain_latency << "\n"
        << "l2_dirty_drain_max_sectors_per_kernel="
        << opt.l2_dirty_drain_max_sectors_per_kernel << "\n"
        << "l2_dirty_drain_high_watermark_sectors="
        << opt.l2_dirty_drain_high_watermark_sectors << "\n"
        << "l2_dirty_drain_target_sectors="
        << opt.l2_dirty_drain_target_sectors << "\n"
        << "output_rotate_bytes=" << opt.output_rotate_bytes << "\n"
        << "records_written=" << total_records << "\n"
        << "accelsim_records_written=" << accelsim_records << "\n"
        << "footprint_records_written=" << footprint_out.records() << "\n"
        << "footprint_expanded_requests=" << footprint_out.expanded_requests() << "\n"
        << "request_output_files=" << join_paths(req_out.paths()) << "\n"
        << "accelsim_output_files=" << join_paths(accel_out.paths()) << "\n"
        << "footprint_output_files=" << join_paths(footprint_out.paths()) << "\n";

    if (wants_csv(opt))
      std::cerr << "wrote " << total_records << " request records to "
                << join_paths(req_out.paths()) << "\n";
    if (wants_accelsim(opt))
      std::cerr << "wrote " << accelsim_records
                << " AccelSim-style DRAM records to "
                << join_paths(accel_out.paths()) << "\n";
    if (wants_footprint(opt))
      std::cerr << "wrote " << footprint_out.records()
                << " compressed footprint records for "
                << footprint_out.expanded_requests() << " DRAM requests to "
                << join_paths(footprint_out.paths()) << "\n";
    if (wants_semantics(opt) || opt.semantic_summary)
      semantic_db.write_dictionary(opt.output_dir);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "hyfiss-request-backend: " << e.what() << "\n";
    return 1;
  }
}


int run_from_sm_traces(const std::vector<KernelTraceRef> &kernels,
                       const BackendOptions &backend_opt) {
  size_t next = 0;
  return run_from_sm_trace_source(
      backend_opt, [&](KernelTraceRef &kernel_ref) -> bool {
        if (next >= kernels.size())
          return false;
        kernel_ref = kernels[next++];
        return true;
      });
}

int run_cli(int argc, char **argv) {
  g_memc_admission = MemcAdmissionStats{};
  g_memc_metadata.clear();
  g_memc_function_ids.clear();
  try {
    Options opt = parse_args(argc, argv);
    const HwParams hw = read_hw_params(opt.hw_config);
    if(hw.profile)throw std::runtime_error("unified r4 hardware requires the ordered backend with per-kernel context; use hbserve-profile-stream-cache");
    apply_hw_options(opt, hw);

    const bool filter_app_metadata =
        !opt.include_local && opt.kernels != "all" &&
        !fs::exists(opt.configs_dir / "capture_receipt.json");
    std::vector<int> requested_kernel_ids;
    std::set<int> requested_kernel_filter;
    if (filter_app_metadata) {
      const std::map<int, KernelMeta> no_available_kernels;
      requested_kernel_ids =
          parse_kernel_list(opt.kernels, no_available_kernels);
      requested_kernel_filter.insert(requested_kernel_ids.begin(),
                                     requested_kernel_ids.end());
    }
    auto kernels_meta = read_app_config(
        opt.configs_dir / "app.config", opt.include_local,
        filter_app_metadata ? &requested_kernel_filter : nullptr);
    preflight_v3_capture(opt, kernels_meta);
    if (filter_app_metadata) {
      for (int kid : requested_kernel_ids) {
        auto &meta = kernels_meta[kid];
        meta.id = kid;
      }
    } else {
      for (int kid : discover_kernel_ids_from_memory(opt.memory_dir)) {
        auto &meta = kernels_meta[kid];
        meta.id = kid;
      }
    }
    g_memc_metadata = kernels_meta;
    const auto kernel_ids = filter_app_metadata
                                ? requested_kernel_ids
                                : parse_kernel_list(opt.kernels, kernels_meta);
    if (kernel_ids.empty())
      throw std::runtime_error("no kernels selected");
    const std::set<int> kernel_filter(kernel_ids.begin(), kernel_ids.end());
    const auto issue_map =
        read_issue_config(opt.configs_dir / "issue.config", kernel_filter);
    if (opt.emit_kernels != "all") {
      const auto emit_ids = parse_kernel_list(opt.emit_kernels, kernels_meta);
      opt.emit_all_kernels = false;
      opt.emit_kernel_ids.insert(emit_ids.begin(), emit_ids.end());
    }

    std::map<int, KernelStats> stats;
    if (opt.semantic_file.empty()) {
      for (const auto &kv : kernels_meta) {
        if (!kv.second.semantic_file.empty()) {
          opt.semantic_file = kv.second.semantic_file;
          break;
        }
      }
      if (opt.semantic_file.empty() &&
          fs::exists(opt.trace_root / "semantic" / "llama_semantics.tsv")) {
        opt.semantic_file = opt.trace_root / "semantic" / "llama_semantics.tsv";
      }
    }
    AccelSimAddressMapping addr_mapping;
    addr_mapping.init(opt);
    g_addr_mapping = &addr_mapping;
    SemanticDatabase semantic_db;
    if ((wants_semantics(opt) || has_semantic_policy_file(opt)) &&
        !opt.semantic_file.empty())
      semantic_db.load(opt.semantic_file);

    std::vector<SectorLruCache> l1_caches;
    for (unsigned sm = 0; sm < std::max(1u, opt.num_sms); ++sm)
      l1_caches.emplace_back(opt.l1_size_bytes, opt.l1_line_size,
                             opt.l1_assoc, opt.l1_set_index);
    const unsigned l2_partitions = std::max(1u, opt.num_partitions);
    const uint64_t l2_size_per_partition =
        std::max<uint64_t>(opt.l2_line_size * opt.l2_assoc,
                           opt.l2_size_bytes / l2_partitions);
    std::vector<SectorLruCache> l2_caches;
    for (unsigned p = 0; p < l2_partitions; ++p)
      l2_caches.emplace_back(l2_size_per_partition, opt.l2_line_size,
                             opt.l2_assoc, opt.l2_set_index);

    const std::string checkpoint_contract =
        (!opt.restore_checkpoint.empty() || !opt.checkpoint_dir.empty())
        ? cache_checkpoint_contract(opt) : std::string{};
    uint64_t kernel_base = 0;
    if (!opt.restore_checkpoint.empty()) {
      kernel_base = load_cache_checkpoint(opt.restore_checkpoint, opt,
          checkpoint_contract, kernel_ids.front(), l1_caches, l2_caches);
      std::cerr << "restored checkpoint " << opt.restore_checkpoint
                << " kernel_base=" << kernel_base << "\n";
    }
    std::ofstream checkpoint_manifest;
    if (!opt.checkpoint_dir.empty() && opt.checkpoint_interval > 0) {
      fs::create_directories(opt.checkpoint_dir);
      if (fs::exists(opt.checkpoint_dir / "manifest.csv"))
        throw std::runtime_error("checkpoint manifest exists; use a fresh directory");
      checkpoint_manifest.open(opt.checkpoint_dir / "manifest.csv");
      checkpoint_manifest << "position,kernel_id,kernel_base,path\n";
    }
    auto maybe_save_checkpoint = [&](size_t position, int next_kernel_id) {
      if (opt.checkpoint_dir.empty() || opt.checkpoint_interval == 0)
        return;
      if (position % opt.checkpoint_interval != 0)
        return;
      const fs::path path =
          checkpoint_path_for(opt.checkpoint_dir, position, next_kernel_id);
      save_cache_checkpoint(path, kernel_base, position, next_kernel_id,
                            checkpoint_contract, l1_caches, l2_caches);
      if (checkpoint_manifest) {
        checkpoint_manifest << position << ',' << next_kernel_id << ','
                            << kernel_base << ',' << path << '\n';
        checkpoint_manifest.flush();
      }
      std::cerr << "checkpoint position " << position << " kernel "
                << next_kernel_id << " -> " << path << "\n";
    };
    fs::create_directories(opt.output_dir);
    RotatingOutput req_out;
    if (wants_csv(opt)) {
      req_out.open(opt.output_dir / "requests.csv", request_csv_header(opt),
                   opt.output_rotate_bytes);
    }
    RotatingOutput accel_out;
    if (wants_accelsim(opt)) {
      accel_out.open(opt.output_dir / "accelsim_mem.trace", "",
                     opt.output_rotate_bytes);
    }
    FootprintWriter footprint_out;
    if (wants_footprint(opt)) {
      footprint_out.open(opt.output_dir / "request_footprint.trace", opt,
                         opt.output_rotate_bytes);
    }

    uint64_t total_records = 0;
    uint64_t accelsim_records = 0;
    uint64_t accelsim_uid = 1;
    auto drain_l2_dirty = [&](const KernelMeta &meta, uint64_t timestamp) {
      if (!opt.l2_dirty_drain || opt.dram_store_policy != "writeback" ||
          !opt.preserve_l2)
        return;
      auto &ks = stats[meta.id];
      uint64_t total_dirty = 0;
      for (const auto &cache : l2_caches)
        total_dirty += cache.dirty_sector_count();
      if (total_dirty <= opt.l2_dirty_drain_high_watermark_sectors)
        return;
      uint64_t remaining =
          total_dirty > opt.l2_dirty_drain_target_sectors
              ? total_dirty - opt.l2_dirty_drain_target_sectors
              : 0;
      if (opt.l2_dirty_drain_max_sectors_per_kernel)
        remaining = std::min(remaining,
                             opt.l2_dirty_drain_max_sectors_per_kernel);
      if (remaining == 0)
        return;
      MemoryInst drain_inst;
      drain_inst.kernel_id = meta.id;
      drain_inst.opcode = "L2_DIRTY_DRAIN";
      drain_inst.op = 'W';
      drain_inst.timestamp = timestamp;
      for (auto &cache : l2_caches) {
        const uint64_t per_cache_budget = remaining;
        for (const auto &evicted : cache.drain_eligible_dirty(
                 opt.sector_size, timestamp, opt.l2_dirty_drain_latency,
                 per_cache_budget)) {
          const unsigned dirty_sectors = popcount32(evicted.dirty_sectors);
          if (dirty_sectors == 0)
            continue;
          ks.l2_dirty_drain_events++;
          ks.l2_dirty_drain_sectors += dirty_sectors;
          if (remaining) {
            if (dirty_sectors >= remaining)
              remaining = 0;
            else
              remaining -= dirty_sectors;
          }
          emit_l2_writeback(evicted, timestamp, meta, drain_inst, false, false, opt,
              semantic_db, nullptr, ks, req_out, accel_out, footprint_out, total_records,
              accelsim_records, accelsim_uid);
        }
        if (remaining == 0)
          break;
      }
    };

    auto flush_l2_dirty = [&](const KernelMeta &meta, uint64_t timestamp) {
      if (opt.preserve_l2 || !opt.flush_l2_on_reset)
        return;
      auto &ks = stats[meta.id];
      MemoryInst flush_inst;
      flush_inst.kernel_id = meta.id;
      flush_inst.opcode = "L2_FLUSH";
      flush_inst.op = 'W';
      flush_inst.timestamp = timestamp;
      for (auto &cache : l2_caches) {
        for (const auto &evicted : cache.drain_dirty(opt.sector_size, timestamp)) {
          emit_l2_writeback(evicted, timestamp, meta, flush_inst, false, false, opt,
              semantic_db, nullptr, ks, req_out, accel_out, footprint_out, total_records,
              accelsim_records, accelsim_uid);
        }
      }
    };
    for (size_t kid_pos = 0; kid_pos < kernel_ids.size(); ++kid_pos) {
      const int kid = kernel_ids[kid_pos];
      maybe_save_checkpoint(kid_pos, kid);
      auto meta_it = kernels_meta.find(kid);
      KernelMeta meta;
      if (meta_it == kernels_meta.end())
        meta.id = kid;
      else
        meta = meta_it->second;

      if (!opt.preserve_l1) {
        for (auto &c : l1_caches)
          c.clear();
      }
      if (!opt.preserve_l2) {
        for (auto &c : l2_caches)
          c.clear();
      }

      std::vector<uint64_t> last_ts(std::max(1u, opt.num_sms), 0);
      uint64_t kernel_max_ts = 0;

      auto process_inst = [&](MemoryInst &inst) {
        if (inst.sm_id >= last_ts.size())
          inst.sm_id %= last_ts.size();
        uint64_t ts = kernel_base + inst.timestamp;
        if (opt.monotonic_sm) {
          uint64_t &last = last_ts[inst.sm_id];
          if (last != 0 && ts <= last)
            ts = last + opt.issue_interval;
          last = ts;
        }
        kernel_max_ts = std::max(kernel_max_ts, ts);
        auto &ks = stats[kid];
        ks.mem_insts++;
        ks.lane_accesses += inst.lanes.size();
        if (inst.op == 'R')
          ks.reads++;
        else if (inst.op == 'W')
          ks.writes++;
        else if (inst.op == 'A')
          ks.atomics++;

        const auto reqs = coalesce_to_sectors(inst, opt.sector_size);
        account_write_coverage(ks, inst, reqs);
        ks.sector_requests += reqs.size();
        if (inst.op == 'R')
          ks.read_sector_requests += reqs.size();
        else if (inst.op == 'W')
          ks.write_sector_requests += reqs.size();
        else if (inst.op == 'A')
          ks.atomic_sector_requests += reqs.size();
        if (inst.op == 'R' && inst.opcode.find("LTC128B") != std::string::npos)
          ks.ldg_ltc128b_insts++;
        if (starts_with(inst.opcode, "LDGSTS"))
          ks.ldgsts_insts++;
        if (inst.op == 'R' && inst.mem_width >= 16)
          ks.wide_load_insts++;
        if (inst.op == 'W' && inst.mem_width >= 16)
          ks.wide_store_insts++;
      for (const auto &req : reqs) {
          const SemanticInfo &req_sem = semantic_db.lookup(req.addr);
          bool l1_hit = false;
          bool l2_hit = false;
          CacheResult l2_result = CacheResult::Hit;
          const bool bypass_l1 =
              (inst.op == 'W' && opt.l1_store_policy == "bypass") ||
              inst.op == 'A';
          if (!bypass_l1) {
            ks.l1_requests++;
            const CacheResult l1_result =
                bypass_l1_read(inst) ? CacheResult::SectorMiss :
                l1_caches[inst.sm_id]
                    .access(req.addr, req.size, opt.sector_size, ts,
                            opt.l1_fill_latency, cache_operation(inst.op),
                            false, 0, false, false, req.byte_mask)
                    .result;
            l1_hit = l1_result == CacheResult::Hit ||
                     l1_result == CacheResult::HitReserved;
            if (l1_result == CacheResult::Hit) {
              ks.l1_hits++;
            } else if (l1_result == CacheResult::HitReserved) {
              ks.l1_pending_hits++;
            } else {
              ks.l1_misses++;
              if (l1_result == CacheResult::LineMiss)
                ks.l1_line_misses++;
              else
                ks.l1_sector_misses++;
            }
            if (wants_csv(opt) && should_emit(opt, "L1") &&
                should_emit_phase(opt, meta)) {
              write_request(req_out, ts, meta, inst, req, "L1", l1_hit, false,
                            opt, &req_sem);
              ++total_records;
            }
          }

          const bool write_like = inst.op == 'W' || inst.op == 'A';
          const bool needs_dram_read = inst.op != 'W';
          EvictedLine l2_evicted;
          if (write_like || bypass_l1 || !l1_hit) {
            const unsigned l2_partition = dram_partition_index(req.addr, opt) % l2_partitions;
            const uint64_t l2_index_addr = l2_cache_index_addr(req.addr, opt);
            const bool stream_l2 = opt.l2_streaming_fill &&
                                   should_stream_l2_fill(meta, inst, req_sem);
            const CacheAccess l2_access = l2_caches[l2_partition].access(
                req.addr, req.size, opt.sector_size, ts, opt.l2_fill_latency,
                cache_operation(inst.op), opt.dram_store_policy == "writeback" && write_like,
                l2_index_addr, true, stream_l2, req.byte_mask);
            l2_result = l2_access.result;
            l2_evicted = l2_access.evicted;
            l2_hit = l2_result == CacheResult::Hit ||
                     l2_result == CacheResult::HitReserved;
            account_l2_lookup(ks, inst.op, l2_result);
            if (wants_csv(opt) && should_emit(opt, "L2") &&
                should_emit_phase(opt, meta)) {
              write_request(req_out, ts, meta, inst, req, "L2", l1_hit, l2_hit,
                            opt, &req_sem);
              ++total_records;
            }
          }

          if (opt.dram_store_policy == "writeback") {
            if ((write_like || bypass_l1 || !l1_hit) && !l2_hit && needs_dram_read) {
              // Atomic origin remains in opcode; DRAM fill direction is read.
              std::optional<MemoryInst> atomic_read;
              if (inst.op == 'A') { atomic_read = inst; atomic_read->op = 'R'; }
              const MemoryInst &transfer = atomic_read ? *atomic_read : inst;
              ks.dram_requests++;
              ks.dram_load_requests++;
              ks.dram_load_bytes += req.size;
              ks.dram_load_sectors += sectors_for_bytes(req.size, opt.sector_size);
              if (wants_csv(opt) && should_emit(opt, "DRAM") &&
                  should_emit_phase(opt, meta)) {
                write_request(req_out, ts, meta, transfer, req, "DRAM", l1_hit,
                              l2_hit, opt, &req_sem);
                ++total_records;
              }
              if (wants_accelsim(opt) && should_emit_phase(opt, meta)) {
                write_accelsim_request(accel_out, accelsim_uid++, ts, meta,
                                       transfer, req, opt, &req_sem);
                ++accelsim_records;
              }
              if (wants_footprint(opt) && should_emit_phase(opt, meta)) {
                footprint_out.write_request(ts, meta, transfer, req, opt, &req_sem);
              }
            }
            emit_l2_writeback(l2_evicted, ts, meta, inst, l1_hit, l2_hit, opt,
              semantic_db, &req_sem, ks, req_out, accel_out, footprint_out, total_records,
              accelsim_records, accelsim_uid);
          } else {
            const bool emit_write_sector =
                opt.write_sector_policy == "all" ||
                l2_result == CacheResult::LineMiss;
            if ((write_like || bypass_l1 || !l1_hit) && !l2_hit &&
                (!write_like || emit_write_sector)) {
              ks.dram_requests++;
              if (write_like) {
                ks.dram_store_requests++;
                ks.dram_store_bytes += req.size;
                ks.dram_store_sectors += sectors_for_bytes(req.size, opt.sector_size);
              } else {
                ks.dram_load_requests++;
                ks.dram_load_bytes += req.size;
                ks.dram_load_sectors += sectors_for_bytes(req.size, opt.sector_size);
              }
              if (wants_csv(opt) && should_emit(opt, "DRAM") &&
                  should_emit_phase(opt, meta)) {
                write_request(req_out, ts, meta, inst, req, "DRAM", l1_hit,
                              l2_hit, opt, &req_sem);
                ++total_records;
              }
              if (wants_accelsim(opt) && should_emit_phase(opt, meta)) {
                write_accelsim_request(accel_out, accelsim_uid++, ts, meta,
                                       inst, req, opt, &req_sem);
                ++accelsim_records;
              }
              if (wants_footprint(opt) && should_emit_phase(opt, meta)) {
                footprint_out.write_request(ts, meta, inst, req, opt, &req_sem);
              }
            }
          }
        }
      };
      if (uses_memc_input(kid, opt) && !opt.sort_by_timestamp) {
        // Default delivery-order replay holds only one instruction. Sorting is
        // an explicit alternate semantic operation and retains its vector path.
        for_each_kernel_inst_memc(kid, opt, issue_map, process_inst);
      } else {
        auto insts = load_kernel_insts(kid, opt, issue_map);
        for (auto &inst : insts) process_inst(inst);
      }
      drain_l2_dirty(meta, kernel_max_ts + 1);
      flush_l2_dirty(meta, kernel_max_ts + 1);
      std::cerr << "kernel " << kid << " " << meta.name << ": "
                << stats[kid].mem_insts << " memory instructions, "
                << stats[kid].dram_requests << " DRAM requests\n";
      kernel_base = std::max(kernel_base, kernel_max_ts + opt.kernel_gap);
    }

    if (!opt.checkpoint_dir.empty() && opt.checkpoint_interval > 0) {
      const int final_kernel_id = -1; // Explicit terminal state, not a guessed successor.
      const fs::path path =
          checkpoint_path_for(opt.checkpoint_dir, kernel_ids.size(),
                              final_kernel_id);
      save_cache_checkpoint(path, kernel_base, kernel_ids.size(),
                            final_kernel_id, checkpoint_contract, l1_caches, l2_caches);
      if (checkpoint_manifest) {
        checkpoint_manifest << kernel_ids.size() << ',' << final_kernel_id
                            << ',' << kernel_base << ',' << path << '\n';
      }
      std::cerr << "checkpoint final position " << kernel_ids.size()
                << " -> " << path << "\n";
    }

    footprint_out.flush();

    std::ofstream sum(opt.output_dir / "kernel_summary.csv");
    sum << "kernel_id,mem_insts,lane_accesses,sector_requests,read_sector_requests,write_sector_requests,atomic_sector_requests,l1_requests,l1_hits,l1_pending_hits,l1_misses,l1_line_misses,l1_sector_misses,l1_hit_rate,l2_requests,l2_hits,l2_pending_hits,l2_misses,l2_line_misses,l2_sector_misses,l2_hit_rate,dram_requests,dram_load_requests,dram_store_requests,dram_load_sectors,dram_store_sectors,dram_load_bytes,dram_store_bytes,l2_writeback_events,l2_writeback_dirty_sectors,l2_dirty_drain_events,l2_dirty_drain_sectors,ldg_ltc128b_insts,ldgsts_insts,wide_load_insts,wide_store_insts,reads,writes,atomics,write_full_sector_requests,write_partial_sector_requests,write_covered_bytes";
    write_l2_direction_header(sum);
    sum << '\n';
    for (const auto &kv : stats) {
      const auto &s = kv.second;
      const double l1_hr =
          s.l1_requests ? static_cast<double>(s.l1_hits + s.l1_pending_hits) / s.l1_requests : 0.0;
      const double l2_hr =
          s.l2_requests ? static_cast<double>(s.l2_hits + s.l2_pending_hits) / s.l2_requests : 0.0;
      sum << kv.first << ',' << s.mem_insts << ',' << s.lane_accesses << ','
          << s.sector_requests << ',' << s.read_sector_requests << ','
          << s.write_sector_requests << ',' << s.atomic_sector_requests << ','
          << s.l1_requests << ',' << s.l1_hits << ',' << s.l1_pending_hits
          << ',' << s.l1_misses << ',' << s.l1_line_misses << ','
          << s.l1_sector_misses << ',' << std::fixed << std::setprecision(6)
          << l1_hr << ',' << s.l2_requests << ',' << s.l2_hits << ','
          << s.l2_pending_hits << ',' << s.l2_misses << ','
          << s.l2_line_misses << ',' << s.l2_sector_misses << ','
          << std::fixed << std::setprecision(6) << l2_hr << ','
          << s.dram_requests << ',' << s.dram_load_requests << ','
          << s.dram_store_requests << ',' << s.dram_load_sectors << ','
          << s.dram_store_sectors << ',' << s.dram_load_bytes << ','
          << s.dram_store_bytes << ',' << s.l2_writeback_events << ','
          << s.l2_writeback_dirty_sectors << ',' << s.l2_dirty_drain_events << ','
          << s.l2_dirty_drain_sectors << ',' << s.ldg_ltc128b_insts << ','
          << s.ldgsts_insts << ',' << s.wide_load_insts << ','
          << s.wide_store_insts << ',' << s.reads << ',' << s.writes << ','
          << s.atomics << ',' << s.write_full_sector_requests << ',' << s.write_partial_sector_requests << ',' << s.write_covered_bytes;
      write_l2_direction_stats(sum, s);
      sum << '\n';
    }

    std::ofstream run(opt.output_dir / "run_summary.txt");
    run << "HyFiSS request-level trace generator\n"
        << "configs_dir=" << opt.configs_dir << "\n"
        << "memory_dir=" << opt.memory_dir << "\n"
        << "hw_config=" << opt.hw_config << "\n"
        << "kernels=" << opt.kernels << "\n"
        << "emit_level=" << opt.emit_level << "\n"
        << "output_phase=" << opt.output_phase << "\n"
        << "output_format=" << opt.output_format << "\n"
        << "footprint_format=" << opt.footprint_format << "\n"
        << "semantic_output=" << opt.semantic_output << "\n"
        << "semantic_file=" << opt.semantic_file << "\n"
        << "input_format=" << opt.input_format << "\n"
        << "include_local=" << (opt.include_local ? 1 : 0) << "\n"
        << "app_config_metadata_mode="
        << (filter_app_metadata ? "selected_global_only_v1" : "full") << "\n"
        << "app_config_metadata_entries=" << kernels_meta.size() << "\n"
        << "selected_kernel_count=" << kernel_ids.size() << "\n"
        << "selected_issue_map_entries=" << issue_map.size() << "\n"
        << "issue_config_parser=streaming_selected_kernel_prefix_v2\n"
        << "issue_config_kernel_order_contract=nondecreasing_per_sm_producer_append_v1\n"
        << "memc_records_decoded=" << g_memc_admission.records << "\n"
        << "memcv3_capture_preflight_records=" << g_v3_guard.records << "\n"
        << "memcv3_capture_preflight_passed=" << g_v3_guard.admitted << "\n"
        << "memcv3_time_coordinate=per_sm_64bit_elapsed_not_globally_synchronized\n"
        << "memc_instruction_storage=" << (opt.sort_by_timestamp ? "whole_kernel_sorted" : "streaming_one_instruction") << "\n"
        << "memcv3_records=" << g_memc_admission.v3_records << "\n"
        << "memc_legacy_records=" << g_memc_admission.legacy_records << "\n"
        << "memcv3_global_lane_references=" << g_memc_admission.global_lanes << "\n"
        << "memcv3_local_lane_references=" << g_memc_admission.local_lanes << "\n"
        << "memcv3_shared_lane_references=" << g_memc_admission.shared_lanes << "\n"
        << "memcv3_filtered_local_lane_references=" << g_memc_admission.filtered_local_lanes << "\n"
        << "local_address_model=launch_cta_warp_private_word_interleaving_not_physical\n"
        << "generic_local_offset_model=captured_pointer_minus_captured_local_base_unvalidated_general_mapping\n"
        << "hardware_accuracy_accepted=0\n"
        << "sort_by_timestamp=" << (opt.sort_by_timestamp ? 1 : 0) << "\n"
        << "num_sms=" << opt.num_sms << "\n"
        << "sector_size=" << opt.sector_size << "\n"
        << "l1_size_bytes=" << opt.l1_size_bytes
        << " l1_line_size=" << opt.l1_line_size
        << " l1_assoc=" << opt.l1_assoc
        << " l1_set_index=" << set_index_name(opt.l1_set_index) << "\n"
        << "l2_size_bytes=" << opt.l2_size_bytes
        << " l2_size_per_partition=" << l2_size_per_partition
        << " l2_line_size=" << opt.l2_line_size
        << " l2_assoc=" << opt.l2_assoc
        << " l2_set_index=" << set_index_name(opt.l2_set_index) << "\n"
        << "cache_data_validity=known_bytes_union_v1;pending_read_independent_v1\n"
        << "l2_lookup_statistics=directional_residency_v1;write_not_ncu_hit_v1\n"
        << "l1_read_policy=ldg_strong_gpu_bypass_v1_sm89_validated\n"
        << "l2_writeback_transfer=dirty_sector_runs_v1\n"
        << "l2_dirty_drain_budget=global_remaining_v2\n"
        << "l2_index_coordinate="
        << (addr_mapping.enabled() ? "explicit_accelsim_v1" : "fallback_quotient_v2") << "\n"
        << "num_partitions=" << opt.num_partitions
        << " num_memory_channels=" << opt.num_memory_channels
        << " num_sub_partitions_per_channel=" << opt.num_sub_partitions_per_channel
        << " num_banks=" << opt.num_banks
        << " partition_index_bit=" << opt.partition_index_bit << "\n"
        << "address_mapping=" << (addr_mapping.enabled() ? "accelsim" : "low-mask")
        << " memory_partition_indexing=" << opt.memory_partition_indexing
        << " mem_address_mask=" << opt.mem_address_mask
        << " mem_addr_mapping=" << opt.mem_addr_mapping << "\n"
        << "preserve_l2=" << (opt.preserve_l2 ? 1 : 0)
        << " flush_l2_on_reset=" << (opt.flush_l2_on_reset ? 1 : 0)
        << " preserve_l1=" << (opt.preserve_l1 ? 1 : 0) << "\n"
        << "l1_store_policy=" << opt.l1_store_policy << "\n"
        << "write_sector_policy=" << opt.write_sector_policy << "\n"
        << "dram_store_policy=" << opt.dram_store_policy << "\n"
        << "l1_fill_latency=" << opt.l1_fill_latency << "\n"
        << "l2_fill_latency=" << opt.l2_fill_latency << "\n"
        << "l2_dirty_drain=" << (opt.l2_dirty_drain ? 1 : 0) << "\n"
        << "l2_streaming_fill=" << (opt.l2_streaming_fill ? 1 : 0) << "\n"
        << "l2_dirty_drain_latency=" << opt.l2_dirty_drain_latency << "\n"
        << "l2_dirty_drain_max_sectors_per_kernel="
        << opt.l2_dirty_drain_max_sectors_per_kernel << "\n"
        << "l2_dirty_drain_high_watermark_sectors="
        << opt.l2_dirty_drain_high_watermark_sectors << "\n"
        << "l2_dirty_drain_target_sectors="
        << opt.l2_dirty_drain_target_sectors << "\n"
        << "output_rotate_bytes=" << opt.output_rotate_bytes << "\n"
        << "records_written=" << total_records << "\n"
        << "accelsim_records_written=" << accelsim_records << "\n"
        << "footprint_records_written=" << footprint_out.records() << "\n"
        << "footprint_expanded_requests=" << footprint_out.expanded_requests() << "\n"
        << "request_output_files=" << join_paths(req_out.paths()) << "\n"
        << "accelsim_output_files=" << join_paths(accel_out.paths()) << "\n"
        << "footprint_output_files=" << join_paths(footprint_out.paths()) << "\n";

    if (wants_csv(opt)) {
      std::cerr << "wrote " << total_records << " request records to "
                << join_paths(req_out.paths()) << "\n";
    }
    if (wants_accelsim(opt)) {
      std::cerr << "wrote " << accelsim_records
                << " AccelSim-style DRAM records to "
                << join_paths(accel_out.paths()) << "\n";
    }
    if (wants_footprint(opt)) {
      std::cerr << "wrote " << footprint_out.records()
                << " compressed footprint records for "
                << footprint_out.expanded_requests() << " DRAM requests to "
                << join_paths(footprint_out.paths()) << "\n";
    }
    if (wants_semantics(opt) || opt.semantic_summary)
      semantic_db.write_dictionary(opt.output_dir);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "hyfiss-request-trace: " << e.what() << "\n";
    usage(std::cerr);
    return 1;
  }
}

} // namespace hyfiss_request_trace

#ifndef HYFISS_REQUEST_TRACE_NO_MAIN
int main(int argc, char **argv) {
  return hyfiss_request_trace::run_cli(argc, argv);
}
#endif
