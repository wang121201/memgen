#ifndef HYFISS_MEMC_READER_H
#define HYFISS_MEMC_READER_H

#include <array>
#include <cstdint>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Raw wire decoder. Does not filter opcodes, infer spaces, or assign physical
// addresses. In particular, unknown references survive decoding for admission.
namespace hyfiss_memc {
inline void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
inline bool uvar(std::istream &in, uint64_t &value) {
  value = 0;
  for (unsigned n = 0; n < 10; ++n) {
    const int c = in.get();
    if (c == std::char_traits<char>::eof()) {
      require(!in.bad() && in.eof(), "MEMC input read error");
      if (n == 0) return false;
      throw std::runtime_error("truncated MEMC varint");
    }
    const auto b = static_cast<uint8_t>(c);
    require(n != 9 || b <= 1, "MEMC uint64 varint overflow");
    value |= uint64_t(b & 127) << (7 * n);
    if (!(b & 128)) return true;
  }
  throw std::runtime_error("MEMC varint too long");
}
inline uint64_t required(std::istream &in) {
  uint64_t v;
  require(uvar(in, v), "unexpected EOF inside MEMC record/header");
  return v;
}
// Return the two's-complement delta as unsigned; addition is modulo 2^64,
// without signed overflow or a shift beyond the width of uint64_t.
inline uint64_t delta(uint64_t v) { return (v >> 1) ^ (uint64_t(0) - (v & 1)); }
inline uint32_t u32(uint64_t v) {
  require(v <= UINT32_MAX, "MEMC field exceeds uint32");
  return static_cast<uint32_t>(v);
}
struct Reference {
  uint32_t global = 0, local = 0, shared = 0, unknown = 0;
  std::array<uint64_t, 32> addresses{};
};
struct Record {
  uint64_t sequence = 0, full_clock = 0;
  uint32_t cta = 0, pc = 0, mask = 0, relative_clock = 0;
  uint32_t sm = 0, cta_warp = 0, function = 0, opcode_id = 0;
  unsigned ref_count = 0;
  std::array<Reference, 2> refs{};
};
class Reader {
 public:
  unsigned version = 0;
  std::vector<std::string> opcodes;
  explicit Reader(std::istream &in) : in_(in) {
    char magic[9];
    in_.read(magic, 9);
    require(in_.gcount() == 9 && std::string(magic, 9) == "HYFMEMC1\n",
            "not a HyFiSS MEMC file");
    const uint64_t v = required(in_);
    require(v >= 1 && v <= 3, "unsupported MEMC version");
    version = static_cast<unsigned>(v);
    const uint64_t n = required(in_);
    require(n >= 1 && n <= 10000, "MEMC opcode count out of bounds");
    for (uint64_t i = 0; i < n; ++i) {
      const uint64_t len = required(in_);
      require(len >= 1 && len <= 4096, "MEMC opcode length out of bounds");
      std::string op(static_cast<size_t>(len), '\0');
      in_.read(op.data(), static_cast<std::streamsize>(len));
      require(in_.gcount() == static_cast<std::streamsize>(len), "truncated MEMC opcode");
      for (unsigned char c : op)
        require(c >= 33 && c <= 126, "invalid MEMC SASS opcode character");
      opcodes.push_back(std::move(op));
    }
  }
  bool next(Record &r) {
    uint64_t first;
    if (!uvar(in_, first)) return false;
    r = Record{};
    r.sequence = version >= 2 ? first : legacy_sequence_++;
    block_ += delta(version >= 2 ? required(in_) : first);
    pc_ += delta(required(in_));
    r.cta = u32(block_); r.pc = u32(pc_);
    const uint64_t op = required(in_);
    require(op < opcodes.size(), "MEMC invalid opcode id");
    r.opcode_id = static_cast<uint32_t>(op);
    r.mask = u32(required(in_));
    require(version < 3 || r.mask != 0, "MEMCv3 empty effective mask");
    clock_ += delta(required(in_));
    r.relative_clock = u32(clock_);
    const uint64_t refs = required(in_);
    require(refs >= 1 && refs <= 2, "MEMC reference count out of bounds");
    r.ref_count = static_cast<unsigned>(refs);
    if (version == 3) {
      r.sm = u32(required(in_)); r.cta_warp = u32(required(in_));
      r.function = u32(required(in_)); r.full_clock = required(in_);
      require(r.function != 0, "MEMCv3 zero function id");
    }
    for (unsigned i = 0; i < r.ref_count; ++i) {
      auto &ref = r.refs[i];
      const uint64_t tag = version == 3 ? required(in_) : 0;
      require(tag <= 4, "MEMCv3 invalid space tag");
      if (tag == 4) {
        ref.global = u32(required(in_)); ref.local = u32(required(in_));
        ref.shared = u32(required(in_));
      } else {
        if (tag == 1) ref.global = r.mask;
        if (tag == 2) ref.local = r.mask;
        if (tag == 3) ref.shared = r.mask;
      }
      require(((ref.global | ref.local | ref.shared) & ~r.mask) == 0 &&
              !(ref.global & ref.local) && !(ref.global & ref.shared) &&
              !(ref.local & ref.shared), "MEMCv3 invalid/overlapping space masks");
      ref.unknown = r.mask & ~(ref.global | ref.local | ref.shared);
      base_ += delta(required(in_));
      ref.addresses[0] = base_;
      const uint64_t pairs = required(in_);
      require(pairs >= 1 && pairs <= 31, "MEMC stride pair count out of bounds");
      unsigned lane = 1;
      for (uint64_t p = 0; p < pairs; ++p) {
        const uint64_t stride = delta(required(in_));
        const uint64_t run = required(in_);
        require(run >= 1 && run <= 31 && run <= 32 - lane,
                "MEMC stride run out of bounds");
        for (unsigned j = 0; j < run; ++j, ++lane)
          ref.addresses[lane] = ref.addresses[lane - 1] + stride;
      }
      require(lane == 32, "MEMC stride runs must reconstruct 32 lanes");
    }
    return true;
  }
 private:
  std::istream &in_;
  uint64_t block_ = 0, pc_ = 0, clock_ = 0, base_ = 0, legacy_sequence_ = 0;
};

// A MODEL coordinate, not a hardware physical address. Bit 63 isolates the
// local namespace. Owner is a unique warp index across all configured launches.
// 32-bit private offsets are word-interleaved across the 32 lanes. Call this
// separately for each contiguous fragment within a 4-byte word.
inline uint64_t local_model_address(uint64_t owner, unsigned lane, uint64_t offset) {
  require(owner < (uint64_t(1) << 26) && lane < 32 && offset <= UINT32_MAX,
          "local model coordinate out of bounds");
  return (uint64_t(1) << 63) | (owner << 37) |
         ((offset / 4) * 128 + lane * 4 + offset % 4);
}
} // namespace hyfiss_memc
#endif
