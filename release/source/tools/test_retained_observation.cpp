#define HYFISS_REQUEST_TRACE_NO_MAIN
#include "hyfiss_request_trace_generator_stream_r4_semantic.cc"
using namespace hyfiss_request_trace;

// Fixed, independently specified functional ledger scenario. No automatic
// cleaner selection or physical queue is enabled by this test.
int main(int argc,char**argv){
 if(argc!=2)return 2;
 try{
  fs::path output=argv[1];fs::create_directory(output);
  CacheObservation observer;SemanticDatabase db;
  std::vector<SectorLruCache> caches;caches.emplace_back(128,128,1);
  std::map<int,KernelStats> stats;KernelMeta m;
  auto begin=[&](int k){m.id=k;m.llm_phase=k==1?"prefill":"decode";observer.begin(k,caches[0].dirty_sector_count());observer.occupancy(k,"entry",caches);stats[k];};
  auto end=[&](){observer.occupancy(m.id,"exit",caches);observer.end(m.id,caches[0].dirty_sector_count());};
  auto store=[&](uint64_t addr,unsigned sm){
   MemoryInst i;i.kernel_id=m.id;i.op='W';i.opcode="STG.E.32";i.sm_id=sm;i.mask=255;
   for(unsigned n=0;n<8;++n)i.lanes.push_back({n,0,addr+4*n});
   auto requests=coalesce_to_sectors(i,32);if(requests.size()!=1)throw std::runtime_error("fixture coalescing");
   observer.instruction(m,i,requests,db);
   auto access=caches[0].access(addr,32,32,m.id,0,CacheOperation::Store,true);
   observer.access(m,i,requests[0],0,false,{},true,access);
  };
  auto clean=[&](unsigned count){
   auto payloads=caches[0].drain_eligible_dirty(32,100,0,count);
   for(const auto &p:payloads)observer.retained_clean(m,p);
   return payloads;
  };
  auto rejects=[&](const KernelMeta &epoch,const EvictedLine &p){
   bool failed=false;try{observer.retained_clean(epoch,p);}catch(const std::runtime_error&){failed=true;}
   if(!failed)throw std::runtime_error("invalid event accepted");
  };
  begin(1);store(0,2);store(64,3);end();
  begin(2);
  EvictedLine candidate;candidate.dirty=true;candidate.addr=0;candidate.valid_sectors=5;candidate.dirty_sectors=1;candidate.dirty_bytes=32;candidate.known_dirty_bytes=32;
  auto invalid=candidate;invalid.valid_sectors=7;invalid.dirty_sectors=3;invalid.dirty_bytes=64;invalid.known_dirty_bytes=64;rejects(m,invalid); // one owner missing: neither may be consumed
  invalid=candidate;invalid.valid_sectors=4;rejects(m,invalid);
  invalid=candidate;invalid.missing_dirty_bytes=4;rejects(m,invalid);
  invalid=candidate;invalid.present=true;rejects(m,invalid);
  auto other_epoch=m;other_epoch.id=99;rejects(other_epoch,candidate);
  auto first=clean(1);if(first.size()!=1||first[0].dirty_sectors!=1)throw std::runtime_error("partial clean fixture");
  rejects(m,first[0]);
  if(caches[0].access(0,32,32,101,0).result!=CacheResult::Hit)throw std::runtime_error("clean invalidated data");
  end();rejects(m,first[0]);
  begin(3);store(0,4);end(); // after cleaning this is a new dirty birth
  begin(4);auto rest=clean(0);if(rest.size()!=2)throw std::runtime_error("two disjoint producer spans");end();
  // Expected totals are specified from the scenario, not copied out of observer.
  for(int k:{1,3}){
   auto &s=stats[k];s.mem_insts=k==1?2:1;s.lane_accesses=s.mem_insts*8;
   s.sector_requests=s.write_sector_requests=s.l2_requests=s.mem_insts;
   s.l2_by_direction[1].requests=s.mem_insts;
   if(k==1){s.l2_line_misses=1;s.l2_sector_misses=1;}else s.l2_hits=1;
  }
  for(int k:{2,4}){auto &s=stats[k];unsigned n=k==2?1:2;s.dram_store_bytes=n*32;s.l2_writeback_dirty_sectors=n;s.l2_writeback_events=n;}
  observer.write(output,stats);
  std::cout<<"PASS explicit retained-clean ownership, epochs, rejection and conservation\n";
 }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
