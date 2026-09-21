#include <stdint.h>
#include <stdio.h>

#include "memory_common.h"
#include "nvbit_reg_rw.h"
#include "utils/channel.hpp"
#include "utils/utils.h"

extern "C" __device__ __noinline__ void
instrument_cta_entry(uint32_t function_id,uint64_t pentry_function,
                     uint64_t pcount,uint64_t pids,uint64_t pcalls,uint64_t pmasks) {
  // All related functions have this hook, but only the actual launch entry
  // function can contribute. The original SASS predicate is not consulted.
  if(function_id!=*reinterpret_cast<const uint32_t*>(pentry_function))return;
  uint64_t cta=uint64_t(blockIdx.x)+uint64_t(gridDim.x)*(uint64_t(blockIdx.y)+uint64_t(gridDim.y)*blockIdx.z);
  uint32_t count=*reinterpret_cast<const uint32_t*>(pcount),lo=0,hi=count;
  const uint64_t*ids=reinterpret_cast<const uint64_t*>(pids);
  while(lo<hi){uint32_t mid=lo+(hi-lo)/2;if(ids[mid]<cta)lo=mid+1;else hi=mid;}
  if(lo==count||ids[lo]!=cta)return;
  uint32_t mask=__activemask(),warp=(threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z))/32;
  if(get_laneid()==__ffs(mask)-1){
    atomicAdd(reinterpret_cast<uint32_t*>(pcalls)+lo*32+warp,1u);
    atomicOr(reinterpret_cast<uint32_t*>(pmasks)+lo*32+warp,mask);
  }
}

extern "C" __device__ __noinline__ void
instrument_mem_fast(int pred, int pc, int opcode_id, uint64_t addr1,
                    int mref_id, uint64_t addr2, uint64_t pchannel_dev,
                    uint64_t pcapture_counter, int space_hint,
                    uint32_t function_id, int source_predicate, int source_predicate_not,
                    uint32_t transfer_width, uint32_t transfer_policy,
                    uint64_t psample_cta_count, uint64_t psample_cta_ids) {
  // Task-private CTA-uniform sampling gate. The packet ABI and the original
  // lane/predicate/space/source-control path below remain unchanged. No lane
  // within an admitted CTA is filtered. These tool metadata loads are not
  // application memory instructions and are not part of the returned sample.
  const uint64_t sample_cta = uint64_t(blockIdx.x) + uint64_t(gridDim.x) *
      (uint64_t(blockIdx.y) + uint64_t(gridDim.y) * uint64_t(blockIdx.z));
  const uint32_t sg_sample_cta_count = *reinterpret_cast<const uint32_t *>(psample_cta_count);
  const uint64_t *sg_sample_cta_ids = reinterpret_cast<const uint64_t *>(psample_cta_ids);
  uint32_t lo = 0, hi = sg_sample_cta_count;
  while (lo < hi) { const uint32_t mid = lo + (hi - lo) / 2;
    if (sg_sample_cta_ids[mid] < sample_cta) lo = mid + 1; else hi = mid; }
  if (lo == sg_sample_cta_count || sg_sample_cta_ids[lo] != sample_cta) return;
  const uint32_t active_mask = __activemask();
  const uint32_t predicate_mask = __ballot_sync(active_mask, pred);
  const uint32_t effective_mask = active_mask & predicate_mask;
  if (effective_mask == 0)
    return;

  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  fast_mem_access_t ma;
  uint64_t current_clk;
  asm("mov.u64 %0, %clock64;" : "=l"(current_clk));
  ma.curr_clk = current_clk;
  ma.transfer_width = transfer_width;
  ma.transfer_policy = transfer_policy;
  const unsigned selector = unsigned(source_predicate_not) >> 8;
  const bool predicate = selector ? ((unsigned(source_predicate) >> (selector - 1)) & 1u) : source_predicate != 0;
  const bool reads_source = predicate != ((source_predicate_not & 1) != 0);
  ma.source_read_mask = __ballot_sync(active_mask, pred && transfer_width && reads_source);


  ma.sm_id = get_smid();
  int4 cta = get_ctaid();
  ma.cta_id_x = cta.x;
  ma.cta_id_y = cta.y;
  ma.cta_id_z = cta.z;
  ma.warp_id = get_warpid();
  ma.gwarp_id = get_global_warp_id();
  ma.opcode_id = opcode_id;
  ma.pc = pc;
  ma.mref_id = mref_id;
  ma.function_id = function_id;
  // %warpid is an SM hardware slot, not a CTA-relative logical warp ID.
  ma.cta_warp_id = (threadIdx.x + blockDim.x *
                   (threadIdx.y + blockDim.y * threadIdx.z)) / 32;
  ma.active_mask = active_mask;
  ma.predicate_mask = predicate_mask;

  #pragma unroll
  for (int ref = 0; ref < 2; ++ref) {
    unsigned space = FAST_SPACE_UNKNOWN;
    if (pred && ref < mref_id) {
      if (space_hint == FAST_SPACE_ASYNC_G2S) {
        // Proven on sm_89 / NVBit 1.7.6. Spaces include ignored-source lanes;
        // whether the source is read is separate from its known address space.
        space = ref == 0 ? FAST_SPACE_SHARED : FAST_SPACE_GLOBAL;
      } else if (space_hint == FAST_SPACE_GENERIC) {
        const void *pointer = reinterpret_cast<const void *>(ref == 0 ? addr1 : addr2);
        const unsigned flags = pointer == nullptr ? 0U :
                               (__isGlobal(pointer) ? 1U : 0U) |
                               (__isLocal(pointer) ? 2U : 0U) |
                               (__isShared(pointer) ? 4U : 0U);
        space = flags == 1 ? FAST_SPACE_GLOBAL : flags == 2 ? FAST_SPACE_LOCAL :
                flags == 4 ? FAST_SPACE_SHARED : FAST_SPACE_UNKNOWN;
      } else {
        // Explicit local/shared operands can be offsets, so never apply generic
        // address predicates to them. Unresolved multi-ref instructions stay 0.
        space = space_hint;
      }
    }
    ma.global_mask[ref] = __ballot_sync(active_mask, space == FAST_SPACE_GLOBAL);
    ma.local_mask[ref] = __ballot_sync(active_mask, space == FAST_SPACE_LOCAL);
    ma.shared_mask[ref] = __ballot_sync(active_mask, space == FAST_SPACE_SHARED);
  }

  for (int i = 0; i < 32; i++) {
    const uint64_t shuffled1 = __shfl_sync(active_mask, addr1, i);
    const uint64_t shuffled2 = (mref_id == 2) ? __shfl_sync(active_mask, addr2, i) : 0;
    ma.mem_addrs1[i] = (effective_mask & (1U << i)) ? shuffled1 : 0;
    ma.mem_addrs2[i] = (effective_mask & (1U << i)) ? shuffled2 : 0;
  }

  if (first_laneid == laneid) {
    unsigned long long *capture_counter =
        (unsigned long long *)pcapture_counter;
    // This counter is a conservation count, not an ordering ticket. Concurrent
    // warps can reserve a ticket before another warp reaches ChannelDev::push,
    // so only the single host consumer can assign the persisted total order.
    atomicAdd(capture_counter, 1ull);
    ma.capture_seq = 0;
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&ma, sizeof(fast_mem_access_t));
  }
}
