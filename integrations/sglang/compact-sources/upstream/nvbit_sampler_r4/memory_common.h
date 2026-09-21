#ifndef HYFISS_MEMORY_COMMON_H
#define HYFISS_MEMORY_COMMON_H

#include <stdint.h>

// Instrumentation hints, independent of the NVBit enum's numeric values.
enum FastMemorySpace {
  FAST_SPACE_UNKNOWN = 0,
  FAST_SPACE_GLOBAL = 1,
  FAST_SPACE_LOCAL = 2,
  FAST_SPACE_SHARED = 3,
  FAST_SPACE_GENERIC = 4,
  FAST_SPACE_ASYNC_G2S = 5
};

typedef struct {
  uint64_t capture_seq;
  uint32_t transfer_width;
  uint32_t transfer_policy;
  uint32_t source_read_mask;
  uint32_t function_id;
  uint32_t cta_warp_id;
  uint32_t global_mask[2];
  uint32_t local_mask[2];
  uint32_t shared_mask[2];
  int sm_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warp_id;
  int gwarp_id;
  int opcode_id;
  int pc;
  int mref_id;
  uint64_t mem_addrs1[32];
  uint64_t mem_addrs2[32];
  uint64_t curr_clk;
  uint32_t active_mask;
  uint32_t predicate_mask;
} fast_mem_access_t;

#endif
