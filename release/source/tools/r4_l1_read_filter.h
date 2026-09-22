#ifndef MEMGEN_R4_L1_READ_FILTER_H
#define MEMGEN_R4_L1_READ_FILTER_H
#include "memgen_hardware_config.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace hyfiss_request_trace {
// Original allocation spans, not tensor views. IDs must identify the allocation
// generation for this kernel. Multi-allocation/SM use is an unvalidated extension.
struct R4Allocation { uint64_t id=0, base=0, bytes=0; };
struct R4ReadResult {
  unsigned outcome=0; // 0 hit, 2 line miss, 3 resident-tag sector miss
  uint32_t before=0, after=0, victim_valid=0;
  uint64_t victim_addr=0;
  bool victim=false;
};

// Frozen CLOCK_u128_s16_h2_c1062. Synchronous sector fills, no MSHR/timing.
class R4L1ReadFilter {
public:
  struct Node { uint64_t tag=0; uint32_t mask=0; bool ref=false; };
  struct Set {
    std::vector<Node> nodes;
    std::unordered_map<uint64_t,unsigned> lookup;
    unsigned hand=0;
  };
  explicit R4L1ReadFilter(unsigned sms, std::shared_ptr<const HardwareProfile> profile={}):sms_(sms),profile_(std::move(profile)) {
    if (!sms) throw std::runtime_error("r4 requires nonzero SM count");
  }
  static unsigned ways_for(unsigned shared_kib) {
    switch(shared_kib) {
    case 8:return 64; case 16:return 56; case 32:return 50; case 64:return 33; case 100:return 14;
    default:throw std::runtime_error("candidate requires declared shared profile 8/16/32/64/100 KiB");
    }
  }
  static unsigned index(uint32_t offset, unsigned sets=16) {
    uint32_t x=offset/128;
    x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16;
    return x&(sets-1);
  }
  void begin_kernel(unsigned shared_kib, std::vector<R4Allocation> allocations) {
    const unsigned ways=profile_?profile_->ways_for(shared_kib):ways_for(shared_kib);
    if (allocations.empty()) throw std::runtime_error("r4 allocation map is missing");
    std::sort(allocations.begin(),allocations.end(),[](auto a,auto b){return a.base<b.base;});
    std::unordered_map<uint64_t,bool> ids;
    uint64_t end=0;
    for (const auto &a:allocations) {
      if (!a.id || !a.base || a.base%128 || !a.bytes || a.bytes>(uint64_t{1}<<32) ||
          a.base>UINT64_MAX-a.bytes || a.base<end || !ids.emplace(a.id,true).second)
        throw std::runtime_error("r4 requires unique nonoverlapping aligned allocations within 4 GiB");
      end=a.base+a.bytes;
    }
    allocations_=std::move(allocations); ways_=ways; last_allocation_=0;
    // Retain bounded storage across kernels, but never retain cache contents.
    data_.resize(sms_);
    for(auto &sm:data_)sm.resize(sets());
    for(auto &sm:data_)for(auto &set:sm) {
      set.nodes.clear();set.lookup.clear();set.hand=0;
      if(set.nodes.capacity()<ways_)set.nodes.reserve(ways_);
      if(set.lookup.bucket_count()<2*ways_)set.lookup.reserve(2*ways_);
    }
    ready_=true;
  }
  void require_span(uint64_t address,uint64_t bytes) {
    if(!ready_ || !bytes || address>UINT64_MAX-bytes)
      throw std::runtime_error("r4 invalid allocation span check");
    // A run of sectors usually belongs to the same original allocation.
    // The half-open span check also rejects holes and address reuse mistakes.
    const auto *allocation=&allocations_[last_allocation_];
    if(address<allocation->base || address-allocation->base>=allocation->bytes) {
    auto it=std::upper_bound(allocations_.begin(),allocations_.end(),address,
                            [](uint64_t x,const R4Allocation&a){return x<a.base;});
    if(it==allocations_.begin())throw std::runtime_error("r4 address lacks allocation base");
    --it;
    if(address-it->base>=it->bytes)throw std::runtime_error("r4 address lacks allocation base");
    last_allocation_=it-allocations_.begin();allocation=&*it;
    }
    if(bytes>allocation->bytes-(address-allocation->base))
      throw std::runtime_error("r4 sector extends beyond original allocation");
  }
  void require_sector(uint64_t address,uint32_t requested_bytes) {
    if(address%32 || !requested_bytes)
      throw std::runtime_error("r4 requires nonempty aligned requested-byte mask");
    // Allocation bases are 128 B aligned, so a sector cannot straddle the
    // beginning of a span. The last requested byte, not the fetched sector
    // width, must remain within the original observed allocation.
    require_span(address,32u-static_cast<unsigned>(__builtin_clz(requested_bytes)));
  }
  R4ReadResult access(unsigned sm,uint64_t address,uint32_t requested_bytes=UINT32_MAX) {
    if (!ready_ || sm>=sms_ || address%32)
      throw std::runtime_error("r4 requires initialized, aligned sector input and valid SM");
    require_sector(address,requested_bytes);
    const auto *allocation=&allocations_[last_allocation_];
    const uint32_t offset=static_cast<uint32_t>(address-allocation->base);
    // Absolute tag keeps distinct allocations from aliasing at equal offsets.
    const uint64_t tag=address/128;
    const uint32_t bit=1u<<((offset%128)/32);
    auto &s=data_[sm][index(offset,sets())]; R4ReadResult out;
    auto found=s.lookup.find(tag);
    if(found!=s.lookup.end()) {
      auto &n=s.nodes[found->second];out.before=n.mask;
      out.outcome=(n.mask&bit)?0:3;n.mask|=bit;n.ref=true;out.after=n.mask;
      return out;
    }
    out.outcome=2;unsigned slot;
    if(s.nodes.size()<ways_) {slot=s.nodes.size();s.nodes.emplace_back();}
    else {
      while(s.nodes[s.hand].ref) {s.nodes[s.hand].ref=false;s.hand=(s.hand+1)%ways_;}
      slot=s.hand;s.hand=(s.hand+1)%ways_;
      const auto &old=s.nodes[slot];out.victim=true;out.victim_addr=old.tag*128;
      out.victim_valid=old.mask;s.lookup.erase(old.tag);
    }
    s.nodes[slot]=Node{tag,bit,true};s.lookup.emplace(tag,slot);out.after=bit;
    return out;
  }
  unsigned ways() const {return ways_;}
  unsigned sets() const {return profile_?profile_->l1_sets:16;}
  const Set& inspect(unsigned sm,unsigned set) const {return data_.at(sm).at(set);}
private:
  unsigned sms_,ways_=0;bool ready_=false;
  std::vector<R4Allocation> allocations_;
  size_t last_allocation_=0;
  std::shared_ptr<const HardwareProfile> profile_;
  std::vector<std::vector<Set>> data_;
};
}
#endif
