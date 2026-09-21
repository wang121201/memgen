#pragma once
// CPU-testable host completion evidence. No CUDA calls or packet changes.
#include <cstdint>
#include <string>
#include <vector>
namespace sgflush {
struct Row {
  std::string key;
  uint64_t ordinal=0;
  int launch_error=-1,sync_error=-1;
  bool sentinel=false,conservation=false;
  uint64_t pushed=0,received=0,selected=0,executed_ctas=0,packet_ctas=0,planned_ctas=0;
};
inline bool closed(const std::vector<Row>& rows,const std::vector<std::string>& planned,
                   uint64_t sampled,uint64_t submitted,uint64_t internal_dispatches,
                   uint64_t internal_dispatch_returns) {
  // R4 supports the observed NVBit mode only: tool dispatch callbacks are absent.
  // Nonzero callback counts require a separately qualified future protocol.
  if(internal_dispatches||internal_dispatch_returns||rows.size()!=planned.size()||
     rows.size()!=sampled||rows.size()!=submitted||rows.empty())return false;
  for(size_t i=0;i<rows.size();++i) {
    const Row&r=rows[i];
    if(r.key!=planned[i]||r.ordinal!=i||r.launch_error!=0||r.sync_error!=0||
       !r.sentinel||!r.conservation||r.pushed!=r.received||r.received!=r.selected||
       !r.planned_ctas||r.executed_ctas!=r.planned_ctas||r.packet_ctas>r.executed_ctas)return false;
  }
  return true;
}
} // namespace sgflush
