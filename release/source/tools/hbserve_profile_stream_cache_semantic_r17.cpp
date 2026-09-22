// Stream an admitted full-workload HBServe profile directly into Memgen or
// HBServe's 12-byte naïve-cache wire format without materializing raw SASS.

#define HYFISS_REQUEST_TRACE_NO_MAIN
#include "hyfiss_request_trace_generator_stream_r4_semantic.cc"

#include <openssl/evp.h>
#include <cstddef>

// xmu has the stable libzstd runtime ABI but not the development header.
// Declare only the small, public decompression surface used here so the
// executable can link directly to libzstd.so.1 without copying a header tree.
extern "C" {
unsigned long long ZSTD_getFrameContentSize(const void *src, std::size_t src_size);
std::size_t ZSTD_decompress(void *dst, std::size_t dst_capacity,
                            const void *src, std::size_t compressed_size);
unsigned ZSTD_isError(std::size_t code);
const char *ZSTD_getErrorName(std::size_t code);
}
static constexpr unsigned long long ZSTD_CONTENTSIZE_UNKNOWN = ~0ULL;
static constexpr unsigned long long ZSTD_CONTENTSIZE_ERROR = ~1ULL;

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>

namespace hbserve_profile_stream {

using boost::property_tree::ptree;
using hyfiss_request_trace::OrderedMemoryInst;

#pragma pack(push, 1)
struct CompactRecord {
  std::uint16_t object_index;
  std::uint32_t object_offset;
  std::uint16_t kernel_ordinal;
  std::uint16_t bytes;
  std::uint8_t operation;
  std::uint8_t flags;
};
#pragma pack(pop)
static_assert(sizeof(CompactRecord) == 12, "compact wire record must be 12 bytes");

struct Arguments {
  std::string mode;
  fs::path profile_index;
  fs::path app_config;
  fs::path issue_config;
  fs::path hw_config;
  fs::path stats;
  fs::path output_dir;
  fs::path semantic_file;
  fs::path r4_context;
  bool observe_cache=false;
  bool include_local=false;
};

[[noreturn]] void usage_error(const std::string &message) {
  throw std::runtime_error(
      message +
      "\nusage: hbserve-profile-stream-cache --mode memgen|compact"
      " --profile-index PATH --app-config PATH --issue-config PATH"
      " --hw-config PATH --stats PATH [--output-dir PATH]"
      " [--semantic-file PATH]");
}

Arguments parse_arguments(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    if (index + 1 >= argc)
      usage_error("missing value for " + key);
    const std::string value = argv[++index];
    if (key == "--mode") result.mode = value;
    else if (key == "--profile-index") result.profile_index = value;
    else if (key == "--app-config") result.app_config = value;
    else if (key == "--issue-config") result.issue_config = value;
    else if (key == "--hw-config") result.hw_config = value;
    else if (key == "--stats") result.stats = value;
    else if (key == "--output-dir") result.output_dir = value;
    else if (key == "--semantic-file") result.semantic_file = value;
    else if (key == "--r4-context") result.r4_context = value;
    else if (key == "--include-local") {
      if(value!="true" && value!="false") usage_error("include-local must be true or false");
      result.include_local=(value=="true");
    }
    else if (key == "--observe-cache") {
      if(value!="true" && value!="false") usage_error("observe-cache must be true or false");
      result.observe_cache=(value=="true");
    }
    else usage_error("unknown option " + key);
  }
  if (result.mode != "memgen" && result.mode != "compact")
    usage_error("mode must be memgen or compact");
  if (result.profile_index.empty() || result.app_config.empty() ||
      result.issue_config.empty() || result.hw_config.empty() ||
      result.stats.empty())
    usage_error("profile index, app/issue configs, hardware config, and stats are required");
  if (result.mode == "memgen" && result.output_dir.empty())
    usage_error("memgen mode requires --output-dir");
  if (!result.r4_context.empty() && result.mode!="memgen")
    usage_error("r4 context requires memgen mode");
  return result;
}

void need(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::string sha256_file(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  need(static_cast<bool>(input), "cannot hash profile: " + path.string());
  EVP_MD_CTX *raw = EVP_MD_CTX_new();
  need(raw != nullptr, "cannot allocate SHA-256 context");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      raw, &EVP_MD_CTX_free);
  need(EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1,
       "cannot initialize SHA-256");
  std::array<char, 1 << 20> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0)
      need(EVP_DigestUpdate(context.get(), buffer.data(),
                            static_cast<std::size_t>(count)) == 1,
           "cannot update SHA-256");
  }
  need(input.eof(), "profile read failed while hashing: " + path.string());
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned digest_size = 0;
  need(EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) == 1 &&
           digest_size == 32,
       "cannot finalize SHA-256");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned index = 0; index < digest_size; ++index)
    output << std::setw(2) << static_cast<unsigned>(digest[index]);
  return output.str();
}

std::string sha256_bytes(const char *data, std::size_t size) {
  EVP_MD_CTX *raw = EVP_MD_CTX_new();
  need(raw != nullptr, "cannot allocate SHA-256 context");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      raw, &EVP_MD_CTX_free);
  need(EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1 &&
           EVP_DigestUpdate(context.get(), data, size) == 1,
       "cannot hash packed profile record");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned digest_size = 0;
  need(EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) == 1 &&
           digest_size == 32,
       "cannot finalize packed profile SHA-256");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned index = 0; index < digest_size; ++index)
    output << std::setw(2) << static_cast<unsigned>(digest[index]);
  return output.str();
}

std::uint64_t parse_hex_value(const std::string &text) {
  std::size_t consumed = 0;
  const std::uint64_t value = std::stoull(
      text, &consumed,
      text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')
          ? 0 : 16);
  need(consumed == text.size(), "invalid hexadecimal value: " + text);
  return value;
}

std::int64_t parse_signed_decimal(const std::string &text) {
  std::size_t consumed = 0;
  const std::int64_t value = std::stoll(text, &consumed, 10);
  need(consumed == text.size(), "invalid signed value: " + text);
  return value;
}

std::string escape_json(const std::string &value) {
  std::ostringstream out;
  for (char item : value) {
    switch (item) {
    case '\\': out << "\\\\"; break;
    case '"': out << "\\\""; break;
    case '\n': out << "\\n"; break;
    case '\r': out << "\\r"; break;
    case '\t': out << "\\t"; break;
    default: out << item;
    }
  }
  return out.str();
}

struct Launch {
  std::string name;
  std::string phase;
  std::uint32_t grid_x = 0;
  std::uint32_t grid_y = 0;
  std::uint32_t grid_z = 0;
  std::uint32_t grid_size = 0;
  std::uint32_t block_size = 0;
};

std::vector<Launch> read_launches(const fs::path &path) {
  std::ifstream input(path);
  need(static_cast<bool>(input), "cannot open app.config: " + path.string());
  const std::regex pattern(R"(^-kernel_([0-9]+)_([^[:space:]]+)[[:space:]]+(.*)$)");
  std::map<int, std::map<std::string, std::string>> fields;
  std::string line;
  std::smatch match;
  while (std::getline(input, line)) {
    if (!std::regex_match(line, match, pattern)) continue;
    fields[std::stoi(match[1].str())][match[2].str()] = trim(match[3].str());
  }
  need(!fields.empty(), "app.config contains no kernels");
  need(fields.begin()->first == 1 && fields.rbegin()->first == static_cast<int>(fields.size()),
       "app.config kernel ids are not dense 1..N");
  std::vector<Launch> result(fields.size() + 1);
  for (const auto &kernel : fields) {
    const auto &value = kernel.second;
    auto required = [&](const char *key) -> const std::string & {
      const auto found = value.find(key);
      need(found != value.end() && !found->second.empty(),
           "app.config missing field " + std::string(key));
      return found->second;
    };
    Launch item;
    item.name = required("kernel_name");
    item.phase = required("llama_phase");
    item.grid_x = static_cast<std::uint32_t>(std::stoul(required("grid_dim_x")));
    item.grid_y = static_cast<std::uint32_t>(std::stoul(required("grid_dim_y")));
    item.grid_z = static_cast<std::uint32_t>(std::stoul(required("grid_dim_z")));
    item.grid_size = static_cast<std::uint32_t>(std::stoul(required("grid_size")));
    item.block_size = static_cast<std::uint32_t>(std::stoul(required("block_size")));
    need(item.grid_x && item.grid_y && item.grid_z && item.block_size,
         "zero launch geometry");
    need(std::uint64_t(item.grid_x) * item.grid_y * item.grid_z == item.grid_size,
         "grid dimensions disagree with grid size");
    result[kernel.first] = std::move(item);
  }
  return result;
}

struct ProfileIndexRow {
  int kernel_id = 0;
  fs::path path;
  std::string sha256;
  std::string status;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::string encoding = "identity";
  std::uint64_t uncompressed_bytes = 0;
  std::string uncompressed_sha256;
  bool packed = false;
};

std::vector<ProfileIndexRow> read_profile_index(const fs::path &path) {
  std::ifstream input(path);
  need(static_cast<bool>(input), "cannot open profile index: " + path.string());
  std::vector<ProfileIndexRow> result;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::istringstream stream(line);
    ptree row;
    boost::property_tree::read_json(stream, row);
    ProfileIndexRow value;
    value.kernel_id = row.get<int>("kernel_id");
    value.path = row.get<std::string>("path");
    value.sha256 = row.get<std::string>("sha256");
    value.status = row.get<std::string>("status");
    const auto offset = row.get_optional<std::uint64_t>("offset");
    const auto bytes = row.get_optional<std::uint64_t>("bytes");
    need(static_cast<bool>(offset) == static_cast<bool>(bytes),
         "packed profile index requires both offset and bytes");
    if (offset) {
      value.packed = true;
      value.offset = *offset;
      value.bytes = *bytes;
      need(value.bytes != 0, "packed profile record is empty");
      value.encoding = row.get<std::string>("encoding", "identity");
      need(value.encoding == "identity" || value.encoding == "zstd_frame",
           "unsupported packed profile encoding");
      if (value.encoding == "zstd_frame") {
        value.uncompressed_bytes = row.get<std::uint64_t>("uncompressed_bytes");
        value.uncompressed_sha256 = row.get<std::string>("uncompressed_sha256");
        need(value.uncompressed_bytes != 0,
             "zstd profile record has zero uncompressed bytes");
        need(value.uncompressed_sha256.size() == 64 &&
                 std::all_of(value.uncompressed_sha256.begin(),
                             value.uncompressed_sha256.end(),
                     [](unsigned char item) {
                       return std::isdigit(item) || (item >= 'a' && item <= 'f');
                     }),
             "zstd profile index has invalid uncompressed SHA-256");
      }
    }
    need(value.kernel_id == static_cast<int>(result.size()) + 1,
         "profile index is not dense ordered 1..N");
    need(value.status.rfind("PASS_", 0) == 0, "profile index contains non-PASS row");
    need(value.sha256.size() == 64 &&
             std::all_of(value.sha256.begin(), value.sha256.end(),
                 [](unsigned char item) {
                   return std::isdigit(item) || (item >= 'a' && item <= 'f');
                 }),
         "profile index has invalid lowercase SHA-256");
    need(fs::is_regular_file(value.path), "indexed profile is missing: " + value.path.string());
    if (value.packed) {
      const std::uint64_t file_bytes = fs::file_size(value.path);
      need(value.offset <= file_bytes && value.bytes <= file_bytes - value.offset,
           "packed profile byte range escapes pack file");
    }
    result.push_back(std::move(value));
  }
  need(!result.empty(), "profile index is empty");
  return result;
}

struct Placement {
  std::uint64_t start = 0;
  std::uint8_t sm = 0;
};

class PlacementIndex {
public:
  PlacementIndex(const fs::path &path, const std::vector<Launch> &launches) {
    offsets_.resize(launches.size() + 1, 0);
    for (std::size_t kernel = 1; kernel < launches.size(); ++kernel) {
      need(UINT64_MAX - offsets_[kernel] >= launches[kernel].grid_size,
           "placement offset overflow");
      offsets_[kernel + 1] = offsets_[kernel] + launches[kernel].grid_size;
    }
    const std::uint64_t total = offsets_[launches.size()];
    need(total <= std::numeric_limits<std::size_t>::max(), "placement too large");
    starts_.resize(static_cast<std::size_t>(total));
    sms_.resize(static_cast<std::size_t>(total));
    seen_.resize(static_cast<std::size_t>((total + 7) / 8), 0);

    std::ifstream input(path);
    need(static_cast<bool>(input), "cannot open issue.config: " + path.string());
    const std::string prefix = "-trace_issued_sm_id_";
    std::string key;
    std::uint64_t count = 0;
    while (input >> key) {
      if (key.rfind(prefix, 0) != 0) {
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        continue;
      }
      const std::string sm_text = key.substr(prefix.size());
      need(!sm_text.empty() && std::all_of(sm_text.begin(), sm_text.end(),
           [](unsigned char c) { return std::isdigit(c) != 0; }),
           "invalid issue.config SM key");
      const unsigned sm = static_cast<unsigned>(std::stoul(sm_text));
      need(sm < 256, "SM id exceeds compact placement range");
      char character = '\0';
      int previous_kernel = -1;
      while (input.get(character)) {
        if (character == '\n') break;
        if (character != '(') continue;
        std::string tuple;
        bool closed = false;
        while (input.get(character)) {
          if (character == ')') { closed = true; break; }
          need(character != '\n' && tuple.size() < 256,
               "malformed/oversized issue tuple");
          tuple.push_back(character);
        }
        need(closed, "unterminated issue tuple");
        const auto parts = split_char(tuple, ',');
        need(parts.size() == 3, "issue tuple does not have kernel,CTA,start");
        const int kernel = std::stoi(parts[0]);
        need(kernel >= 1 && kernel < static_cast<int>(launches.size()),
             "issue tuple kernel is out of range");
        need(previous_kernel <= kernel, "issue kernel ids decrease within an SM line");
        previous_kernel = kernel;
        const std::uint32_t block = static_cast<std::uint32_t>(std::stoul(parts[1]));
        need(block < launches[kernel].grid_size, "issue CTA is out of range");
        const std::uint64_t flat = offsets_[kernel] + block;
        const std::size_t byte = static_cast<std::size_t>(flat >> 3);
        const std::uint8_t bit = static_cast<std::uint8_t>(1u << (flat & 7));
        need((seen_[byte] & bit) == 0, "duplicate issue kernel/CTA tuple");
        seen_[byte] |= bit;
        starts_[static_cast<std::size_t>(flat)] = parse_hex_value(parts[2]);
        sms_[static_cast<std::size_t>(flat)] = static_cast<std::uint8_t>(sm);
        ++count;
      }
    }
    need(!input.bad(), "issue.config read error");
    need(count == total, "issue tuple count differs from total grid CTAs");
    for (std::uint64_t flat = 0; flat < total; ++flat)
      need((seen_[static_cast<std::size_t>(flat >> 3)] & (1u << (flat & 7))) != 0,
           "issue mapping is not complete");
    tuple_count_ = count;
  }

  Placement at(int kernel, std::uint32_t block) const {
    const std::uint64_t flat = offsets_.at(static_cast<std::size_t>(kernel)) + block;
    return {starts_.at(static_cast<std::size_t>(flat)),
            sms_.at(static_cast<std::size_t>(flat))};
  }

  std::uint64_t tuple_count() const { return tuple_count_; }

private:
  std::vector<std::uint64_t> offsets_;
  std::vector<std::uint64_t> starts_;
  std::vector<std::uint8_t> sms_;
  std::vector<std::uint8_t> seen_;
  std::uint64_t tuple_count_ = 0;
};

struct AddressRule {
  bool exact = false;
  bool x_floor_quotient = false;
  bool x_tiled = false;
  bool x_axis_permutation = false;
  std::int64_t intercept = 0;
  std::int64_t x_stride = 0;
  std::int64_t y_stride = 0;
  std::int64_t z_stride = 0;
  std::vector<std::int64_t> y_offsets;
  bool has_z_partition = false;
  std::uint32_t z_partition = 0;
  std::int64_t z_partition_stride = 0;
  std::uint32_t x_divisor = 0;
  std::int64_t x_quotient_stride = 0;
  std::int64_t x_remainder_stride = 0;
  std::array<std::uint32_t, 3> x_input_extents{};
  std::array<std::uint32_t, 3> x_output_axis_order{};
  std::int64_t element_stride = 0;
  std::unordered_map<std::uint32_t, std::uint64_t> exact_bases;
};

struct AddressGroup {
  std::array<std::int64_t, 32> lane_offsets{};
  AddressRule rule;
};

struct Entry {
  std::uint32_t ordinal = 0;
  std::uint32_t pc = 0;
  std::uint32_t mask = 0;
  std::string opcode;
  char cache_operation = 'N';
  unsigned memory_width = 4;
  bool evict_first = false;
  std::uint64_t default_delta = 0;
  std::unordered_map<std::uint32_t, std::uint64_t> exact_delta;
  std::vector<AddressGroup> groups;
};

bool has_opcode_token(const std::string &opcode, const std::string &wanted) {
  for (const std::string &token : split_char(opcode, '.')) {
    std::string upper;
    for (unsigned char item : token)
      upper.push_back(static_cast<char>(std::toupper(item)));
    if (upper == wanted) return true;
  }
  return false;
}

struct StructuralClass {
  std::string id;
  std::vector<Entry> entries;
  std::vector<std::uint32_t> shared_order;
  bool per_cta_order = false;
};

std::vector<std::int64_t> read_int_array(const ptree &node) {
  std::vector<std::int64_t> result;
  for (const auto &item : node)
    result.push_back(parse_signed_decimal(item.second.get_value<std::string>()));
  return result;
}

AddressRule parse_rule(const ptree &node) {
  AddressRule result;
  const std::string kind = node.get<std::string>("kind", "coordinate_affine");
  if (kind == "exact_cta_base_table") {
    result.exact = true;
    for (const auto &item : node.get_child("bases_by_cta"))
      result.exact_bases.emplace(
          static_cast<std::uint32_t>(std::stoul(item.first)),
          item.second.get_value<std::uint64_t>());
    need(!result.exact_bases.empty(), "empty exact CTA base table");
    return result;
  }
  if (kind == "coordinate_x_quotient_remainder_y_table_z_partition") {
    result.x_tiled = true;
    result.intercept = node.get<std::int64_t>("intercept");
    result.x_divisor = node.get<std::uint32_t>("cta_x_divisor");
    result.x_quotient_stride = node.get<std::int64_t>("cta_x_quotient_stride");
    result.x_remainder_stride = node.get<std::int64_t>("cta_x_remainder_stride");
    result.z_stride = node.get<std::int64_t>("cta_z_stride");
    result.y_offsets = read_int_array(node.get_child("cta_y_offsets"));
    need(result.x_divisor >= 2, "invalid tiled CTA x divisor");
    need(!result.y_offsets.empty(), "tiled CTA x rule has no y-offset table");
    if (const auto value = node.get_optional<std::uint32_t>("cta_z_partition")) {
      result.has_z_partition = true;
      result.z_partition = *value;
      result.z_partition_stride =
          node.get<std::int64_t>("cta_z_partition_stride");
    }
    return result;
  }
  if (kind == "coordinate_x_floor_quotient") {
    result.x_floor_quotient = true;
    result.intercept = node.get<std::int64_t>("intercept");
    result.x_divisor = node.get<std::uint32_t>("cta_x_divisor");
    result.x_quotient_stride = node.get<std::int64_t>("cta_x_quotient_stride");
    need(result.x_divisor >= 2, "invalid CTA x-floor divisor");
    return result;
  }
  if (kind == "coordinate_x_axis_permutation") {
    result.x_axis_permutation = true;
    result.intercept = node.get<std::int64_t>("intercept");
    std::size_t position = 0;
    for (const auto &item : node.get_child("cta_x_input_extents")) {
      need(position < 3, "too many CTA axis-permutation extents");
      result.x_input_extents[position++] = item.second.get_value<std::uint32_t>();
    }
    need(position == 3, "CTA axis-permutation requires three extents");
    position = 0;
    std::array<bool, 3> seen_axes{};
    for (const auto &item : node.get_child("cta_x_output_axis_order")) {
      need(position < 3, "too many CTA axis-permutation axes");
      const auto axis = item.second.get_value<std::uint32_t>();
      need(axis < 3 && !seen_axes[axis], "invalid CTA axis permutation");
      seen_axes[axis] = true;
      result.x_output_axis_order[position++] = axis;
    }
    need(position == 3, "CTA axis-permutation requires three axes");
    result.element_stride = node.get<std::int64_t>("element_stride");
    need(result.element_stride != 0, "zero CTA axis-permutation element stride");
    return result;
  }
  result.intercept = node.get<std::int64_t>("intercept");
  result.x_stride = node.get<std::int64_t>("cta_x_stride");
  result.y_stride = node.get<std::int64_t>("cta_y_stride", 0);
  result.z_stride = node.get<std::int64_t>("cta_z_stride", 0);
  if (const auto value = node.get_child_optional("cta_y_offsets"))
    result.y_offsets = read_int_array(*value);
  if (const auto value = node.get_optional<std::uint32_t>("cta_z_partition")) {
    result.has_z_partition = true;
    result.z_partition = *value;
    result.z_partition_stride = node.get<std::int64_t>("cta_z_partition_stride");
  }
  return result;
}

std::array<std::int64_t, 32> parse_lane_offsets(const ptree &group) {
  std::array<std::int64_t, 32> result{};
  unsigned lane = 1;
  std::int64_t offset = 0;
  for (const auto &item : group.get_child("pairs")) {
    const std::string token = item.second.get_value<std::string>();
    const std::size_t colon = token.find(':');
    need(colon != std::string::npos, "stride pair lacks colon");
    const std::int64_t stride = parse_signed_decimal(token.substr(0, colon));
    const unsigned count = static_cast<unsigned>(std::stoul(token.substr(colon + 1)));
    need(count && lane + count <= 32, "invalid stride run");
    for (unsigned repeat = 0; repeat < count; ++repeat) {
      offset += stride;
      result[lane++] = offset;
    }
  }
  need(lane == 32, "stride runs do not reconstruct 32 lanes");
  return result;
}

Entry parse_entry(const ptree &node) {
  Entry result;
  result.ordinal = node.get<std::uint32_t>("ordinal");
  result.pc = static_cast<std::uint32_t>(parse_hex_value(node.get<std::string>("pc")));
  result.mask = static_cast<std::uint32_t>(parse_hex_value(node.get<std::string>("mask")));
  need(result.mask != 0, "profile entry has empty lane mask");
  result.opcode = node.get<std::string>("opcode");
  result.cache_operation = classify_opcode(result.opcode, false);
  result.memory_width = infer_width(result.opcode);
  result.evict_first = result.cache_operation == 'R' &&
                       has_opcode_token(result.opcode, "EF");
  result.default_delta = node.get<std::uint64_t>("sampled_timestamp_delta");
  if (const auto table = node.get_child_optional("sampled_timestamp_delta_by_cta")) {
    for (const auto &item : *table)
      result.exact_delta.emplace(
          static_cast<std::uint32_t>(std::stoul(item.first)),
          item.second.get_value<std::uint64_t>());
  }
  const auto &groups = node.get_child("groups");
  const auto &rules = node.get_child("address_rules");
  need(std::distance(groups.begin(), groups.end()) ==
           std::distance(rules.begin(), rules.end()),
       "profile group/rule count differs");
  auto rule = rules.begin();
  for (auto group = groups.begin(); group != groups.end(); ++group, ++rule) {
    AddressGroup value;
    value.lane_offsets = parse_lane_offsets(group->second);
    value.rule = parse_rule(rule->second);
    result.groups.push_back(std::move(value));
  }
  need(!result.groups.empty(), "profile entry has no address group");
  return result;
}

StructuralClass parse_class(const std::string &id, const ptree &entries) {
  StructuralClass result;
  result.id = id;
  for (const auto &item : entries)
    result.entries.push_back(parse_entry(item.second));
  result.per_cta_order = std::any_of(
      result.entries.begin(), result.entries.end(),
      [](const Entry &entry) { return !entry.exact_delta.empty(); });
  result.shared_order.resize(result.entries.size());
  std::iota(result.shared_order.begin(), result.shared_order.end(), 0);
  std::stable_sort(result.shared_order.begin(), result.shared_order.end(),
      [&](std::uint32_t left, std::uint32_t right) {
        const Entry &a = result.entries[left];
        const Entry &b = result.entries[right];
        return std::tie(a.default_delta, a.ordinal) <
               std::tie(b.default_delta, b.ordinal);
      });
  return result;
}

std::string decode_profile_record(const ProfileIndexRow &index,
                                  const std::string &encoded) {
  if (index.encoding == "identity") return encoded;
  need(index.encoding == "zstd_frame", "unsupported profile record encoding");
  need(index.uncompressed_bytes <= std::numeric_limits<std::size_t>::max(),
       "uncompressed profile record exceeds addressable memory");
  const unsigned long long frame_size =
      ZSTD_getFrameContentSize(encoded.data(), encoded.size());
  need(frame_size != ZSTD_CONTENTSIZE_ERROR,
       "packed profile record is not a valid zstd frame");
  need(frame_size != ZSTD_CONTENTSIZE_UNKNOWN,
       "packed profile zstd frame omits its content size");
  need(frame_size == index.uncompressed_bytes,
       "packed profile zstd frame size differs from index");
  std::string decoded(static_cast<std::size_t>(index.uncompressed_bytes), '\0');
  const std::size_t written = ZSTD_decompress(
      decoded.data(), decoded.size(), encoded.data(), encoded.size());
  need(!ZSTD_isError(written),
       std::string("packed profile zstd decompression failed: ") +
           ZSTD_getErrorName(written));
  need(written == decoded.size(),
       "packed profile zstd decompressed byte count differs");
  need(sha256_bytes(decoded.data(), decoded.size()) ==
           index.uncompressed_sha256,
       "packed profile uncompressed SHA-256 differs");
  return decoded;
}

class KernelGenerator {
public:
  KernelGenerator(const ProfileIndexRow &index, const Launch &launch,
                  const PlacementIndex &placement, bool materialize_opcode, bool reject_modeled_rebinding = false)
      : kernel_(index.kernel_id), launch_(launch), placement_(placement),
        materialize_opcode_(materialize_opcode) {
    ptree root;
    if (index.packed) {
      need(index.bytes <= std::numeric_limits<std::size_t>::max(),
           "packed profile record exceeds addressable memory");
      std::string encoded(static_cast<std::size_t>(index.bytes), '\0');
      std::ifstream input(index.path, std::ios::binary);
      need(static_cast<bool>(input), "cannot open profile pack: " + index.path.string());
      input.seekg(static_cast<std::streamoff>(index.offset));
      need(static_cast<bool>(input), "cannot seek profile pack");
      input.read(encoded.data(), static_cast<std::streamsize>(encoded.size()));
      need(input.gcount() == static_cast<std::streamsize>(encoded.size()),
           "short read from profile pack");
      need(sha256_bytes(encoded.data(), encoded.size()) == index.sha256,
           "packed profile record SHA-256 differs");
      std::string decoded = decode_profile_record(index, encoded);
      std::istringstream stream(decoded);
      boost::property_tree::read_json(stream, root);
    } else {
      need(sha256_file(index.path) == index.sha256,
           "indexed profile SHA-256 differs: " + index.path.string());
      boost::property_tree::read_json(index.path.string(), root);
    }
    need(root.get<std::string>("schema.name") ==
             "hbserve.hyfiss_sampled_sass_profile",
         "unexpected profile schema name");
    const int version = root.get<int>("schema.version");
    need(version >= 4 && version <= 12, "unsupported profile schema version");
    need(root.get<int>("kernel.id") == kernel_, "profile kernel id differs");
    need(root.get<std::string>("kernel.name") == launch_.name,
         "profile/app kernel name differs");
    need(root.get<std::uint32_t>("kernel.grid_size") == launch_.grid_size,
         "profile/app grid size differs");
    const std::string status = root.get<std::string>("status");
    need(status.rfind("PASS_", 0) == 0, "profile status is not PASS");
    if (reject_modeled_rebinding) {
      // Check the already decoded, SHA-verified payload; no extra profile pass.
      const auto observed = root.get_optional<bool>("model.target_addresses_hardware_observed");
      need(status != "PASS_MODELED_LAYER_PROFILE_BINDING" &&
           index.status != "PASS_MODELED_LAYER_PROFILE_BINDING" &&
           !root.get_child_optional("model.layer_rebinding") &&
           (!observed || *observed),
           "r4 original allocation context cannot consume modeled layer rebinding");
    }

    if (version == 6 || version == 9 || version == 10 || version == 11 || version == 12) {
      std::unordered_map<std::string, std::uint32_t> class_index;
      for (const auto &item : root.get_child("structural_classes")) {
        const std::string id = item.second.get<std::string>("class_id");
        need(!id.empty() && !class_index.count(id), "duplicate structural class id");
        class_index[id] = static_cast<std::uint32_t>(classes_.size());
        classes_.push_back(parse_class(id, item.second.get_child("template")));
      }
      need(!classes_.empty(), "structural profile has no classes");
      if (version == 6 || version == 9) {
        for (const auto &item : root.get_child("cta_class_by_id")) {
          const std::string id = item.second.get_value<std::string>();
          const auto found = class_index.find(id);
          need(found != class_index.end(), "CTA refers to unknown structural class");
          class_by_block_.push_back(found->second);
        }
        need(class_by_block_.size() == launch_.grid_size,
             "CTA structural class map size differs from grid");
      } else if (version == 12) {
        const auto member_set = [&](const std::string &path) {
          std::unordered_map<std::uint32_t, bool> values;
          for (const auto &item : root.get_child(path)) {
            const auto value = item.second.get_value<std::uint32_t>();
            need(value < launch_.grid_size && values.emplace(value, true).second,
                 "invalid declared structural sample set");
          }
          return values;
        };
        const auto training = member_set("sampling.training_ctas");
        const auto holdout = member_set("sampling.holdout_ctas");
        need(training == member_set("source.launch.fit_ctas") &&
             holdout == member_set("source.launch.holdout_ctas"),
             "structural samples differ from source launch");
        for (const auto &entry : training)
          need(!holdout.count(entry.first), "declared training and holdout overlap");
        std::size_t total_training = 0, total_holdout = 0;
        need(root.get<std::string>("structural_class_selector.kind") ==
                 "linear_cta_intervals", "unsupported schema-12 selector");
        class_by_block_.reserve(launch_.grid_size);
        std::vector<bool> used(classes_.size(), false);
        for (const auto &item : root.get_child("structural_class_selector.intervals")) {
          const auto lo = item.second.get<std::uint32_t>("start");
          const auto hi = item.second.get<std::uint32_t>("stop");
          const auto found = class_index.find(item.second.get<std::string>("class_id"));
          need(lo == class_by_block_.size() && lo < hi && hi <= launch_.grid_size,
               "linear CTA intervals overlap, leave a hole, or escape the grid");
          need(found != class_index.end(), "linear CTA interval uses unknown class");
          class_by_block_.insert(class_by_block_.end(), hi-lo, found->second);
          used[found->second] = true;
        }
        need(class_by_block_.size() == launch_.grid_size &&
             std::all_of(used.begin(), used.end(), [](bool x) { return x; }),
             "linear CTA intervals do not cover the grid and every class");
        for (const auto &item : root.get_child("structural_classes")) {
          const auto expected = class_index.at(item.second.get<std::string>("class_id"));
          const auto count = std::count(class_by_block_.begin(), class_by_block_.end(), expected);
          need(item.second.get<std::uint32_t>("domain_cta_count") == count,
               "structural class domain count differs");
          std::unordered_map<std::uint32_t, bool> members;
          for (const auto &cta : item.second.get_child("ctas")) {
            const auto block = cta.second.get_value<std::uint32_t>();
            need(block < launch_.grid_size && class_by_block_[block] == expected &&
                 training.count(block) && members.emplace(block, true).second,
                 "invalid structural training member");
          }
          need(!members.empty(), "structural class has no training members");
          std::unordered_map<std::uint32_t, bool> held;
          for (const auto &cta : item.second.get_child("independent_holdout_ctas")) {
            const auto block = cta.second.get_value<std::uint32_t>();
            need(block < launch_.grid_size && class_by_block_[block] == expected &&
                 holdout.count(block) && !members.count(block) && held.emplace(block, true).second,
                 "invalid structural holdout member");
          }
          if (item.second.get<bool>("observed_domain_complete"))
            need(members.size() == static_cast<std::size_t>(count) && held.empty(),
                 "complete class domain is not fully observed");
          else
            need(members.size() >= 2 && held.size() >= 2,
                 "extrapolated class has insufficient independent evidence");
          total_training += members.size(); total_holdout += held.size();
        }
        need(total_training == training.size() && total_holdout == holdout.size(),
             "structural class samples do not cover declared samples");
      } else {
        need(root.get<std::string>("structural_class_selector.kind") ==
                 "categorical_y",
             "unsupported sampled structural class selector");
        std::vector<std::uint32_t> class_by_y;
        std::vector<bool> class_used(classes_.size(), false);
        for (const auto &item :
             root.get_child("structural_class_selector.class_by_y")) {
          const std::string id = item.second.get_value<std::string>();
          const auto found = class_index.find(id);
          need(found != class_index.end(),
               "categorical y selector refers to unknown structural class");
          class_by_y.push_back(found->second);
          class_used[found->second] = true;
        }
        need(class_by_y.size() == launch_.grid_y,
             "categorical y selector size differs from grid y");
        need(std::all_of(class_used.begin(), class_used.end(),
                         [](bool used) { return used; }),
             "categorical y selector leaves a structural class unused");
        class_by_block_.reserve(launch_.grid_size);
        for (std::uint32_t block = 0; block < launch_.grid_size; ++block) {
          const std::uint32_t y = (block / launch_.grid_x) % launch_.grid_y;
          class_by_block_.push_back(class_by_y.at(y));
        }
        for (const auto &item : root.get_child("structural_classes")) {
          const std::string id = item.second.get<std::string>("class_id");
          const std::uint32_t expected = class_index.at(id);
          const auto ctas = item.second.get_child_optional("ctas");
          need(static_cast<bool>(ctas),
               "sampled structural class has no training CTAs");
          for (const auto &cta : *ctas) {
            const std::uint32_t block = cta.second.get_value<std::uint32_t>();
            need(block < launch_.grid_size,
                 "sampled structural class CTA is outside grid");
            need(class_by_block_.at(block) == expected,
                 "sampled structural class CTA disagrees with categorical y selector");
          }
        }
      }
    } else {
      classes_.push_back(parse_class("class-00", root.get_child("template")));
      need(!classes_[0].entries.empty(), "sampled profile has empty template");
      class_by_block_.assign(launch_.grid_size, 0);
    }
    initialize_cursors();
  }

  bool next(OrderedMemoryInst &output) {
    if (heap_.empty()) return false;
    Cursor cursor = heap_.top();
    heap_.pop();
    const StructuralClass &klass = classes_.at(class_by_block_[cursor.block]);
    const auto &order = order_for(cursor.block, klass);
    need(cursor.position < order.size(), "cursor escaped template order");
    const Entry &entry = klass.entries.at(order[cursor.position]);
    const Placement placed = placement_.at(kernel_, cursor.block);
    const std::uint64_t delta = timestamp_delta(entry, cursor.block);
    need(placed.start <= UINT64_MAX - delta, "generated timestamp overflows u64");

    output.pc = entry.pc;
    output.timestamp = placed.start + delta;
    output.mask = entry.mask;
    if (materialize_opcode_) output.opcode = entry.opcode;
    else output.opcode.clear();
    output.block_id = cursor.block;
    output.sm_id = placed.sm;
    output.sequence = sequence_++;
    output.addr.clear();
    const auto active_lanes = static_cast<std::size_t>(
        __builtin_popcount(static_cast<unsigned>(entry.mask)));
    const auto required_addresses = active_lanes * entry.groups.size();
    if (output.addr.capacity() < required_addresses)
      output.addr.reserve(required_addresses);
    for (const AddressGroup &group : entry.groups) {
      const std::uint64_t base = address_base(group.rule, cursor.block);
      for (unsigned lane = 0; lane < 32; ++lane) {
        if ((entry.mask & (std::uint32_t(1) << lane)) == 0) continue;
        const __int128 address = static_cast<__int128>(base) + group.lane_offsets[lane];
        need(address >= 0 && address <= UINT64_MAX, "generated lane address overflows");
        output.addr.push_back(static_cast<std::uint64_t>(address));
      }
    }
    need(!output.addr.empty(), "generated memory instruction has no active address");
    update_digest(output, entry.opcode);
    last_cache_operation_ = entry.cache_operation;
    last_memory_width_ = entry.memory_width;
    last_evict_first_ = entry.evict_first;

    ++cursor.position;
    if (cursor.position < order.size()) {
      const Entry &next_entry = klass.entries.at(order[cursor.position]);
      cursor.time = placed.start + timestamp_delta(next_entry, cursor.block);
      heap_.push(cursor);
    }
    return true;
  }

  std::uint64_t instruction_count() const { return instruction_count_; }
  std::uint64_t lane_count() const { return lane_count_; }
  std::uint64_t digest_a() const { return digest_a_; }
  std::uint64_t digest_b() const { return digest_b_; }
  char last_cache_operation() const { return last_cache_operation_; }
  unsigned last_memory_width() const { return last_memory_width_; }
  bool last_evict_first() const { return last_evict_first_; }

private:
  struct Cursor {
    std::uint64_t time = 0;
    std::uint16_t sm = 0;
    std::uint32_t block = 0;
    std::uint32_t position = 0;
  };
  struct CursorGreater {
    bool operator()(const Cursor &a, const Cursor &b) const {
      return std::tie(a.time, a.sm, a.block, a.position) >
             std::tie(b.time, b.sm, b.block, b.position);
    }
  };

  std::uint64_t timestamp_delta(const Entry &entry, std::uint32_t block) const {
    if (entry.exact_delta.empty()) return entry.default_delta;
    const auto found = entry.exact_delta.find(block);
    need(found != entry.exact_delta.end(), "CTA absent from exact timestamp table");
    return found->second;
  }

  const std::vector<std::uint32_t> &order_for(
      std::uint32_t block, const StructuralClass &klass) const {
    if (!klass.per_cta_order) return klass.shared_order;
    return per_block_order_.at(block);
  }

  std::uint64_t address_base(const AddressRule &rule, std::uint32_t block) const {
    if (rule.exact) {
      const auto found = rule.exact_bases.find(block);
      need(found != rule.exact_bases.end(), "CTA absent from exact address table");
      return found->second;
    }
    const std::uint32_t x = block % launch_.grid_x;
    const std::uint32_t quotient = block / launch_.grid_x;
    const std::uint32_t y = quotient % launch_.grid_y;
    const std::uint32_t z = quotient / launch_.grid_y;
    if (rule.x_axis_permutation) {
      need(launch_.grid_y == 1 && launch_.grid_z == 1,
           "CTA axis-permutation address rule requires one-dimensional launch geometry");
      const std::uint64_t extent_product =
          static_cast<std::uint64_t>(rule.x_input_extents[0]) *
          rule.x_input_extents[1] * rule.x_input_extents[2];
      need(extent_product == launch_.grid_x,
           "CTA axis-permutation extents differ from launch grid x");
      std::array<std::uint32_t, 3> coordinate{
          x / (rule.x_input_extents[1] * rule.x_input_extents[2]),
          (x / rule.x_input_extents[2]) % rule.x_input_extents[1],
          x % rule.x_input_extents[2],
      };
      std::uint64_t mapped = 0;
      for (const auto axis : rule.x_output_axis_order)
        mapped = mapped * rule.x_input_extents[axis] + coordinate[axis];
      __int128 value = rule.intercept;
      value += static_cast<__int128>(rule.element_stride) * mapped;
      need(value >= 0 && value <= UINT64_MAX,
           "predicted CTA axis-permutation address base overflows");
      return static_cast<std::uint64_t>(value);
    }
    if (rule.x_tiled) {
      need(y < rule.y_offsets.size(), "CTA y outside tiled x-rule offset table");
      __int128 value = rule.intercept;
      value += static_cast<__int128>(rule.x_quotient_stride) *
               (x / rule.x_divisor);
      value += static_cast<__int128>(rule.x_remainder_stride) *
               (x % rule.x_divisor);
      value += rule.y_offsets[y];
      value += static_cast<__int128>(rule.z_stride) * z;
      if (rule.has_z_partition && z >= rule.z_partition)
        value += rule.z_partition_stride;
      need(value >= 0 && value <= UINT64_MAX,
           "predicted tiled CTA x address base overflows");
      return static_cast<std::uint64_t>(value);
    }
    if (rule.x_floor_quotient) {
      need(launch_.grid_y == 1 && launch_.grid_z == 1,
           "CTA x-floor address rule requires one-dimensional launch geometry");
      __int128 value = rule.intercept;
      value += static_cast<__int128>(rule.x_quotient_stride) *
               (x / rule.x_divisor);
      need(value >= 0 && value <= UINT64_MAX,
           "predicted CTA x-floor address base overflows");
      return static_cast<std::uint64_t>(value);
    }
    const std::int64_t y_term = rule.y_offsets.empty()
        ? rule.y_stride * static_cast<std::int64_t>(y)
        : (need(y < rule.y_offsets.size(), "CTA y outside offset table"), rule.y_offsets[y]);
    __int128 value = rule.intercept;
    value += static_cast<__int128>(rule.x_stride) * x;
    value += y_term;
    value += static_cast<__int128>(rule.z_stride) * z;
    if (rule.has_z_partition && z >= rule.z_partition)
      value += rule.z_partition_stride;
    need(value >= 0 && value <= UINT64_MAX, "predicted address base overflows");
    return static_cast<std::uint64_t>(value);
  }

  void initialize_cursors() {
    per_block_order_.resize(launch_.grid_size);
    for (std::uint32_t block = 0; block < launch_.grid_size; ++block) {
      const StructuralClass &klass = classes_.at(class_by_block_[block]);
      if (klass.entries.empty()) continue;
      if (klass.per_cta_order) {
        auto &order = per_block_order_[block];
        order.resize(klass.entries.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::uint32_t left, std::uint32_t right) {
          const Entry &a = klass.entries[left];
          const Entry &b = klass.entries[right];
          return std::make_tuple(timestamp_delta(a, block), a.ordinal) <
                 std::make_tuple(timestamp_delta(b, block), b.ordinal);
        });
      }
      const auto &order = order_for(block, klass);
      const Placement placed = placement_.at(kernel_, block);
      heap_.push(Cursor{
          placed.start + timestamp_delta(klass.entries[order[0]], block),
          placed.sm, block, 0});
    }
  }

  void mix(std::uint64_t value) {
    digest_a_ ^= value;
    digest_a_ *= UINT64_C(1099511628211);
    digest_b_ += value + UINT64_C(0x9e3779b97f4a7c15) +
                 (digest_b_ << 6) + (digest_b_ >> 2);
  }

  void update_digest(const OrderedMemoryInst &item,
                     const std::string &source_opcode) {
    mix(static_cast<std::uint64_t>(kernel_));
    mix(item.block_id);
    mix(item.sm_id);
    mix(item.timestamp);
    mix(item.pc);
    mix(item.mask);
    for (unsigned char character : source_opcode) mix(character);
    mix(item.addr.size());
    for (std::uint64_t address : item.addr) mix(address);
    ++instruction_count_;
    lane_count_ += item.addr.size();
  }

  int kernel_;
  Launch launch_;
  const PlacementIndex &placement_;
  bool materialize_opcode_ = true;
  std::vector<StructuralClass> classes_;
  std::vector<std::uint32_t> class_by_block_;
  std::vector<std::vector<std::uint32_t>> per_block_order_;
  std::priority_queue<Cursor, std::vector<Cursor>, CursorGreater> heap_;
  std::uint64_t sequence_ = 1;
  std::uint64_t instruction_count_ = 0;
  std::uint64_t lane_count_ = 0;
  std::uint64_t digest_a_ = UINT64_C(1469598103934665603);
  std::uint64_t digest_b_ = UINT64_C(0x243f6a8885a308d3);
  char last_cache_operation_ = 'N';
  unsigned last_memory_width_ = 4;
  bool last_evict_first_ = false;
};

struct SourceStats {
  std::uint64_t kernels = 0;
  std::uint64_t instructions = 0;
  std::uint64_t lanes = 0;
  std::uint64_t digest_a = UINT64_C(1469598103934665603);
  std::uint64_t digest_b = UINT64_C(0x243f6a8885a308d3);
  std::uint64_t sector_requests = 0;
  std::uint64_t read_sector_requests = 0;
  std::uint64_t write_sector_requests = 0;
  std::uint64_t compact_bytes = 0;
};

class WorkloadSource {
public:
  WorkloadSource(const fs::path &profile_index, const fs::path &app_config,
                 const fs::path &issue_config, bool materialize_opcode)
      : launches_(read_launches(app_config)),
        profiles_(read_profile_index(profile_index)),
        placement_(issue_config, launches_),
        materialize_opcode_(materialize_opcode) {
    need(profiles_.size() + 1 == launches_.size(),
         "profile index and app.config kernel counts differ");
  }

  bool begin_next(int &kernel, std::string &name, std::string &phase) {
    if (next_ >= profiles_.size()) return false;
    const ProfileIndexRow &row = profiles_[next_++];
    current_ = std::make_unique<KernelGenerator>(
        row, launches_.at(static_cast<std::size_t>(row.kernel_id)), placement_,
        materialize_opcode_, reject_modeled_rebinding_);
    kernel = row.kernel_id;
    name = launches_.at(static_cast<std::size_t>(row.kernel_id)).name;
    phase = launches_.at(static_cast<std::size_t>(row.kernel_id)).phase;
    ++stats_.kernels;
    return true;
  }

  bool next_inst(OrderedMemoryInst &item) {
    need(static_cast<bool>(current_), "no current kernel generator");
    if (current_->next(item)) return true;
    stats_.instructions += current_->instruction_count();
    stats_.lanes += current_->lane_count();
    stats_.digest_a ^= current_->digest_a() + UINT64_C(0x9e3779b97f4a7c15) +
                       (stats_.digest_a << 6) + (stats_.digest_a >> 2);
    stats_.digest_b ^= current_->digest_b() + UINT64_C(0x517cc1b727220a95) +
                       (stats_.digest_b << 7) + (stats_.digest_b >> 3);
    current_.reset();
    return false;
  }

  char last_cache_operation() const {
    need(static_cast<bool>(current_), "no current kernel generator");
    return current_->last_cache_operation();
  }

  unsigned last_memory_width() const {
    need(static_cast<bool>(current_), "no current kernel generator");
    return current_->last_memory_width();
  }

  bool last_evict_first() const {
    need(static_cast<bool>(current_), "no current kernel generator");
    return current_->last_evict_first();
  }

  SourceStats &stats() { return stats_; }
  const SourceStats &stats() const { return stats_; }
  std::uint64_t placement_tuples() const { return placement_.tuple_count(); }
  void reject_modeled_rebinding() { reject_modeled_rebinding_ = true; }
  void validate_hardware(const hyfiss_request_trace::HardwareProfile &hardware) const {
    for(size_t kernel=1;kernel<launches_.size();++kernel)
      for(uint32_t block=0;block<launches_[kernel].grid_size;++block)
        need(placement_.at(kernel,block).sm==block%hardware.sms,
             "issue.config disagrees with hardware round-robin CTA placement");
  }

private:
  std::vector<Launch> launches_;
  std::vector<ProfileIndexRow> profiles_;
  PlacementIndex placement_;
  bool materialize_opcode_ = true;
  bool reject_modeled_rebinding_ = false;
  std::size_t next_ = 0;
  std::unique_ptr<KernelGenerator> current_;
  SourceStats stats_;
};

class BufferedWorkloadSource {
public:
  explicit BufferedWorkloadSource(WorkloadSource &source,
                                  std::size_t capacity = 4096)
      : source_(source), slots_(capacity), producer_([this] { produce(); }) {
    need(capacity >= 2, "buffered source capacity is too small");
  }

  ~BufferedWorkloadSource() {
    cancel();
    join_noexcept();
  }

  bool begin_next(int &kernel, std::string &name, std::string &phase) {
    need(!in_kernel_ && !held_.has_value(),
         "buffered source began a kernel before the prior kernel ended");
    const auto index = acquire_ready();
    if (!index.has_value()) {
      join_and_rethrow();
      return false;
    }
    Slot &slot = slots_.at(*index);
    if (slot.kind == Kind::Done) {
      release(*index);
      saw_done_ = true;
      join_and_rethrow();
      return false;
    }
    need(slot.kind == Kind::KernelStart,
         "buffered source expected a kernel-start record");
    kernel = slot.kernel;
    name = slot.name;
    phase = slot.phase;
    slot.name.clear();
    slot.phase.clear();
    release(*index);
    in_kernel_ = true;
    return true;
  }

  bool next_inst(OrderedMemoryInst &output) {
    need(in_kernel_, "buffered source has no current kernel");
    release_held(output);
    const auto index = acquire_ready();
    need(index.has_value(), "buffered source ended inside a kernel");
    Slot &slot = slots_.at(*index);
    if (slot.kind == Kind::KernelEnd) {
      release(*index);
      in_kernel_ = false;
      return false;
    }
    need(slot.kind == Kind::Instruction,
         "buffered source expected an instruction record");
    std::swap(output, slot.instruction);
    held_ = *index;
    return true;
  }

  void finish() {
    need(!in_kernel_ && !held_.has_value(),
         "buffered source finish occurred inside a kernel");
    join_and_rethrow();
    need(saw_done_, "buffered source consumer did not reach end-of-stream");
  }

  void cancel_and_join() {
    cancel();
    join_noexcept();
  }

private:
  enum class Kind { KernelStart, Instruction, KernelEnd, Done };

  struct Slot {
    Kind kind = Kind::Done;
    int kernel = 0;
    std::string name;
    std::string phase;
    OrderedMemoryInst instruction;
  };

  std::size_t reserve_write() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [&] { return cancelled_ || count_ < slots_.size(); });
    if (cancelled_) throw std::runtime_error("buffered source cancelled");
    return write_index_;
  }

  void publish(std::size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    need(!cancelled_, "buffered source cancelled before publish");
    need(index == write_index_ && count_ < slots_.size(),
         "buffered source publish order differs");
    write_index_ = (write_index_ + 1) % slots_.size();
    ++count_;
    not_empty_.notify_one();
  }

  std::optional<std::size_t> acquire_ready() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&] { return count_ > 0 || producer_done_; });
    if (count_ == 0) {
      const auto error = producer_error_;
      lock.unlock();
      if (error) std::rethrow_exception(error);
      return std::nullopt;
    }
    return read_index_;
  }

  void release(std::size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    need(index == read_index_ && count_ > 0,
         "buffered source release order differs");
    read_index_ = (read_index_ + 1) % slots_.size();
    --count_;
    not_full_.notify_one();
  }

  void release_held(OrderedMemoryInst &output) {
    if (!held_.has_value()) return;
    Slot &slot = slots_.at(*held_);
    std::swap(output, slot.instruction);
    const std::size_t index = *held_;
    held_.reset();
    release(index);
  }

  void produce() {
    try {
      int kernel = 0;
      std::string name;
      std::string phase;
      while (source_.begin_next(kernel, name, phase)) {
        std::size_t index = reserve_write();
        Slot &start = slots_.at(index);
        start.kind = Kind::KernelStart;
        start.kernel = kernel;
        start.name = name;
        start.phase = phase;
        publish(index);

        while (true) {
          index = reserve_write();
          Slot &item = slots_.at(index);
          if (source_.next_inst(item.instruction)) {
            item.kind = Kind::Instruction;
            publish(index);
          } else {
            item.kind = Kind::KernelEnd;
            publish(index);
            break;
          }
        }
      }
      const std::size_t index = reserve_write();
      slots_.at(index).kind = Kind::Done;
      publish(index);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        producer_done_ = true;
      }
      not_empty_.notify_all();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!cancelled_) producer_error_ = std::current_exception();
        producer_done_ = true;
      }
      not_empty_.notify_all();
      not_full_.notify_all();
    }
  }

  void cancel() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  void join_and_rethrow() {
    if (producer_.joinable()) producer_.join();
    if (producer_error_) std::rethrow_exception(producer_error_);
  }

  void join_noexcept() noexcept {
    try {
      if (producer_.joinable()) producer_.join();
    } catch (...) {
    }
  }

  WorkloadSource &source_;
  std::vector<Slot> slots_;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::size_t read_index_ = 0;
  std::size_t write_index_ = 0;
  std::size_t count_ = 0;
  bool producer_done_ = false;
  bool cancelled_ = false;
  std::exception_ptr producer_error_;
  std::thread producer_;
  std::optional<std::size_t> held_;
  bool in_kernel_ = false;
  bool saw_done_ = false;
};

class CompactOutput {
public:
  CompactOutput() { records_.reserve(std::size_t{1} << 20); }
  ~CompactOutput() { try { flush(); } catch (...) {} }
  void append(const CompactRecord &record) {
    records_.push_back(record);
    if (records_.size() == records_.capacity()) flush();
  }
  void flush() {
    if (records_.empty()) return;
    const std::size_t written = std::fwrite(
        records_.data(), sizeof(CompactRecord), records_.size(), stdout);
    need(written == records_.size(), "failed writing compact cache stream");
    records_.clear();
  }
private:
  std::vector<CompactRecord> records_;
};

void assign_ordered_sectors(const OrderedMemoryInst &inst,
                            unsigned memory_width,
                            std::vector<SectorRequest> &requests) {
  need(memory_width > 0, "zero memory width");
  std::array<unsigned, 32> active_lanes{};
  unsigned active_count = 0;
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((inst.mask & (std::uint32_t(1) << lane)) != 0)
      active_lanes[active_count++] = lane;
  need(active_count > 0, "ordered instruction has no active lanes");
  need(inst.addr.size() % active_count == 0,
       "ordered address groups are not lane-complete");

  requests.clear();
  if (requests.capacity() < inst.addr.size() * 2)
    requests.reserve(inst.addr.size() * 2);
  for (std::size_t address_index = 0; address_index < inst.addr.size();
       ++address_index) {
    const std::uint64_t address = inst.addr[address_index];
    need(address <= UINT64_MAX - (memory_width - 1),
         "memory access address overflows");
    const unsigned lane = active_lanes[address_index % active_count];
    const unsigned ref = static_cast<unsigned>(address_index / active_count) + 1;
    const std::uint64_t first = address / 32;
    const std::uint64_t last = (address + memory_width - 1) / 32;
    for (std::uint64_t sector = first; ; ++sector) {
      const std::uint64_t base = sector * 32;
      auto found = std::find_if(
          requests.begin(), requests.end(), [&](const SectorRequest &request) {
            return request.ref_id == ref && request.addr == base;
          });
      if (found == requests.end()) {
        requests.push_back(SectorRequest{base, 32, ref, 0, 0});
        found = std::prev(requests.end());
      }
      if ((found->lane_mask & (std::uint32_t(1) << lane)) == 0) {
        found->lane_mask |= std::uint32_t(1) << lane;
        ++found->lane_count;
      }
      if (sector == last) break;
    }
  }
  std::sort(requests.begin(), requests.end(),
            [](const SectorRequest &left, const SectorRequest &right) {
              return left.ref_id != right.ref_id
                  ? left.ref_id < right.ref_id
                  : left.addr < right.addr;
            });
}

void write_stats(const Arguments &args, const WorkloadSource &source,
                 std::uint64_t elapsed_ms, int backend_returncode) {
  need(!fs::exists(args.stats), "refusing to replace source stats: " + args.stats.string());
  std::ofstream output(args.stats);
  need(static_cast<bool>(output), "cannot create source stats");
  const SourceStats &stats = source.stats();
  output << "{\n"
         << "  \"schema\": \"hbserve_profile_stream_cache_v1\",\n"
         << "  \"status\": \"PASS\",\n"
         << "  \"mode\": \"" << escape_json(args.mode) << "\",\n"
         << "  \"profile_index\": \"" << escape_json(args.profile_index.string()) << "\",\n"
         << "  \"app_config\": \"" << escape_json(args.app_config.string()) << "\",\n"
         << "  \"issue_config\": \"" << escape_json(args.issue_config.string()) << "\",\n"
         << "  \"hw_config\": \"" << escape_json(args.hw_config.string()) << "\",\n"
         << "  \"kernel_count\": " << stats.kernels << ",\n"
         << "  \"placement_tuples\": " << source.placement_tuples() << ",\n"
         << "  \"generated_memory_instructions\": " << stats.instructions << ",\n"
         << "  \"generated_lane_addresses\": " << stats.lanes << ",\n"
         << "  \"semantic_digest_a\": \"" << std::hex << std::setw(16)
         << std::setfill('0') << stats.digest_a << "\",\n"
         << "  \"semantic_digest_b\": \"" << std::setw(16) << stats.digest_b
         << std::dec << "\",\n"
         << "  \"source_sector_stats_scope\": \"" << (args.mode=="compact"?"COMPACT_STREAM":"SEE_BACKEND_KERNEL_SUMMARY") << "\",\n"
         << "  \"sector_requests\": " << (args.mode=="compact"?std::to_string(stats.sector_requests):"null") << ",\n"
         << "  \"read_sector_requests\": " << (args.mode=="compact"?std::to_string(stats.read_sector_requests):"null") << ",\n"
         << "  \"write_sector_requests\": " << (args.mode=="compact"?std::to_string(stats.write_sector_requests):"null") << ",\n"
         << "  \"compact_bytes\": " << stats.compact_bytes << ",\n"
         << "  \"backend_returncode\": " << backend_returncode << ",\n"
         << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
         << "  \"ordering\": \"kernel id, then global merge of CTA start plus modeled intra-CTA delta\",\n"
         << "  \"materialized_raw_sass_bytes\": 0,\n"
         << "  \"claim_boundary\": \"HBServe generated address stream and cache-model input; not captured full SASS or hardware truth\"\n"
         << "}\n";
  need(static_cast<bool>(output), "failed writing source stats");
}

int run_compact(const Arguments &args, WorkloadSource &source) {
  CompactOutput output;
  int kernel = 0;
  std::string name;
  std::string phase;
  std::vector<SectorRequest> sectors;
  while (source.begin_next(kernel, name, phase)) {
    need(kernel <= UINT16_MAX, "kernel id exceeds compact u16");
    OrderedMemoryInst raw;
    while (source.next_inst(raw)) {
      const char operation = source.last_cache_operation();
      if (operation == 'N') continue;
      need(operation != 'A', "atomic/reduction request is unsupported by naïve cache");
      need(operation == 'R' || operation == 'W', "unsupported memory operation");
      assign_ordered_sectors(raw, source.last_memory_width(), sectors);
      const bool evict_first = source.last_evict_first();
      for (const auto &sector : sectors) {
        need(sector.addr < (UINT64_C(1) << 48), "address exceeds compact 48-bit boundary");
        need(sector.addr % 32 == 0 && sector.size == 32,
             "coalescer emitted non-sector request");
        output.append(CompactRecord{
            static_cast<std::uint16_t>(sector.addr >> 32),
            static_cast<std::uint32_t>(sector.addr),
            static_cast<std::uint16_t>(kernel),
            32,
            static_cast<std::uint8_t>(operation == 'R' ? 0 : 1),
            static_cast<std::uint8_t>(evict_first ? 1 : 0)});
        SourceStats &stats = source.stats();
        ++stats.sector_requests;
        stats.compact_bytes += sizeof(CompactRecord);
        if (operation == 'R') ++stats.read_sector_requests;
        else ++stats.write_sector_requests;
      }
    }
  }
  output.flush();
  need(std::fflush(stdout) == 0, "failed flushing compact stream");
  return 0;
}

int run_memgen(const Arguments &args, WorkloadSource &source,
    const std::function<void(const hyfiss_request_trace::L2AccessObservation&)> &observe_l2={},
    const std::function<void(const hyfiss_request_trace::L2AccessObservation&)> &observe_l1={}) {
  struct Context { unsigned shared; std::vector<hyfiss_request_trace::R4Allocation> allocations; };
  const auto hardware=hyfiss_request_trace::HardwareProfile::load(args.hw_config.string());
  if(hardware) {
    need(!args.r4_context.empty(),"unified r4 hardware requires --r4-context");
    source.validate_hardware(*hardware);
  }
  std::map<int,Context> contexts;
  std::string r4_model_id="CLOCK_u128_s16_h2_c1062";
  if(!args.r4_context.empty()) {
    source.reject_modeled_rebinding();
    ptree root;boost::property_tree::read_json(args.r4_context.string(),root);
    need(root.get<std::string>("schema")=="MEMGEN_R4_CONTEXT_V1","r4 context schema mismatch");
    r4_model_id=root.get<std::string>("model_id","CLOCK_u128_s16_h2_c1062");
    need(r4_model_id=="CLOCK_u128_s16_h2_c1062" || r4_model_id=="r4-small-shared-20260922","unknown r4 model identity");
    if(hardware)need(r4_model_id==hardware->values.at("context_model_id"),"hardware config and kernel context model identities differ");
    need(root.get<std::string>("profile_index_sha256")==sha256_file(args.profile_index),"r4 profile identity mismatch");
    need(root.get<std::string>("app_config_sha256")==sha256_file(args.app_config),"r4 launch identity mismatch");
    for(const auto &item:root.get_child("kernels")) {
      const auto &row=item.second;Context c;
      c.shared=row.get<unsigned>("shared_kib");
      if(c.shared==8 || c.shared==16)need(r4_model_id=="r4-small-shared-20260922","small shared candidate requires explicit model identity");
      if(hardware)hardware->ways_for(c.shared);else hyfiss_request_trace::R4L1ReadFilter::ways_for(c.shared);
      need(!row.get<std::string>("shared_evidence").empty(),"r4 shared profile provenance missing");
      for(const auto &entry:row.get_child("allocations")) {
        const auto &a=entry.second;
        c.allocations.push_back({a.get<uint64_t>("id"),a.get<uint64_t>("base"),a.get<uint64_t>("bytes")});
      }
      need(contexts.emplace(row.get<int>("kernel_id"),std::move(c)).second,"duplicate r4 kernel context");
    }
    need(!contexts.empty(),"empty r4 kernel context");
  }
  std::set<int> used_contexts;
  hyfiss_request_trace::BackendOptions options;
  options.r4_l1_read_filter=!args.r4_context.empty();
  options.r4_model_id=r4_model_id;
  options.hardware_profile=hardware;
  options.hw_config = args.hw_config.string();
  options.output_dir = args.output_dir.string();
  options.semantic_file = args.semantic_file.string();
  options.semantic_summary = !args.semantic_file.empty();
  options.output_format = "summary";
  options.emit_level = "DRAM";
  options.order = "timestamp";
  options.l1_store_policy = "bypass";
  options.write_sector_policy = "line-miss-only";
  options.dram_store_policy = "writeback";
  options.l2_dirty_drain = false;
  options.l2_streaming_fill = false;
  options.preserve_l2 = true;
  options.preserve_l1 = false;
  options.monotonic_sm = true;
  options.include_local = args.include_local;
  options.observe_cache = args.observe_cache;
  options.observe_l2_access = observe_l2;
  options.observe_l1_access = observe_l1;
  options.l1_fill_latency_set = true;
  options.l2_fill_latency_set = true;
  options.l1_fill_latency = 0;
  options.l2_fill_latency = 0;
  options.sector_size = 32;
  BufferedWorkloadSource buffered(source);
  try {
    const int code = hyfiss_request_trace::run_from_sm_trace_source(
        options, [&](hyfiss_request_trace::KernelTraceRef &ref) -> bool {
          int kernel = 0;
          std::string name;
          std::string phase;
          if (!buffered.begin_next(kernel, name, phase)) return false;
          ref = hyfiss_request_trace::KernelTraceRef{};
          ref.kernel_id = kernel;
          ref.kernel_name = std::move(name);
          ref.llm_phase = std::move(phase);
          if(options.r4_l1_read_filter) {
            auto it=contexts.find(kernel);
            need(it!=contexts.end(),"missing r4 kernel context");
            need(used_contexts.insert(kernel).second,"repeated r4 kernel context");
            ref.r4_shared_kib=it->second.shared;
            ref.r4_allocations=it->second.allocations;
          }
          ref.next_ordered_inst = [&](OrderedMemoryInst &item) {
            return buffered.next_inst(item);
          };
          return true;
        });
    if(code!=0) { buffered.cancel_and_join(); return code; }
    buffered.finish();
    if(hardware) {
      auto identity=hardware->resolved();
      identity.put("source_sha256",sha256_bytes(hardware->source_text.data(),hardware->source_text.size()));
      identity.put("context_sha256",sha256_file(args.r4_context));
      identity.put("cta_placement_validation","verified_by_hbserve_against_hardware_config");
      boost::property_tree::write_json((args.output_dir/"hardware.identity.json").string(),identity);
    }
    if(options.r4_l1_read_filter) {
      need(used_contexts.size()==contexts.size(),"unused r4 kernel context");
      std::ofstream receipt(args.output_dir/"r4_context_identity.json");
      need(bool(receipt),"cannot write r4 context identity");
      receipt<<"{\"schema\":\"MEMGEN_R4_CONTEXT_IDENTITY_V1\",\"sha256\":\""
             <<sha256_file(args.r4_context)<<"\",\"model_id\":\""<<r4_model_id<<"\",\"hardware_accuracy_accepted\":false}\n";
      need(bool(receipt),"r4 context identity write failed");
    }
    return code;
  } catch (...) {
    buffered.cancel_and_join();
    throw;
  }
}

int run(int argc, char **argv) {
  if(argc==3 && std::string(argv[1])=="--describe-hardware-config") {
    const auto hw=read_hw_params(argv[2]);boost::property_tree::ptree root;
    if(hw.profile) {
      root=hw.profile->resolved();
      root.put("source_sha256",sha256_bytes(hw.profile->source_text.data(),hw.profile->source_text.size()));
    } else {
      root.put("schema","LEGACY_MEMGEN_HARDWARE");root.put("num_sms",hw.num_sms);
      root.put("source_sha256",sha256_file(argv[2]));
      root.put("note","legacy file; backend policy options and r4 context may override cache behavior");
    }
    boost::property_tree::write_json(std::cout,root);return 0;
  }
  const auto began = std::chrono::steady_clock::now();
  const Arguments args = parse_arguments(argc, argv);
  for (const auto &path : {args.profile_index, args.app_config,
                           args.issue_config, args.hw_config})
    need(fs::is_regular_file(path), "missing input: " + path.string());
  if (!args.semantic_file.empty())
    need(fs::is_regular_file(args.semantic_file),
         "missing semantic sidecar: " + args.semantic_file.string());
  need(!fs::exists(args.stats), "stats output already exists: " + args.stats.string());
  if (args.mode == "memgen")
    need(!fs::exists(args.output_dir), "memgen output already exists: " + args.output_dir.string());
  WorkloadSource source(args.profile_index, args.app_config, args.issue_config,
                        args.mode == "memgen");
  const int code = args.mode == "compact" ? run_compact(args, source)
                                           : run_memgen(args, source);
  need(code == 0, "cache backend failed with code " + std::to_string(code));
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - began);
  write_stats(args, source, static_cast<std::uint64_t>(elapsed.count()), code);
  return code;
}

} // namespace hbserve_profile_stream

int main(int argc, char **argv) {
  try {
    return hbserve_profile_stream::run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "hbserve-profile-stream-cache: " << error.what() << '\n';
    return 1;
  }
}
