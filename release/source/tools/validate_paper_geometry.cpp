// Validate the accepted geometry through the existing backend, not a hardware-policy claim.
#define HYFISS_REQUEST_TRACE_NO_MAIN
#include "hyfiss_request_trace_generator_stream_r4_semantic.cc"
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("config required");
 Options opt;opt.l2_dirty_drain=false;opt.l2_streaming_fill=false;opt.l1_fill_latency_set=opt.l2_fill_latency_set=true;apply_hw_options(opt,read_hw_params(argv[1]));
 if(opt.l2_size_bytes!=41943040||opt.l2_line_size!=128||opt.l2_assoc!=16||opt.num_partitions!=20||opt.l2_set_index!=SetIndexFunction::BitwiseXor||!opt.mem_addr_mapping.empty()||opt.memory_partition_indexing)throw std::runtime_error("accepted geometry/parser mismatch");
 std::vector<SectorLruCache> caches;
 for(unsigned p=0;p<20;p++)caches.emplace_back(2097152,128,16,opt.l2_set_index);
 // A set-associative XOR cache need not fit every unaligned contiguous capacity-sized span.
 // Use a full partition/set-index period and verify per-set address populations first.
 uint64_t lines=327680,hits=0,sectors=0;const uint64_t base=41943040ull*64;
 std::vector<unsigned> reference(20480);
 for(uint64_t i=0;i<lines;i++){
  const uint64_t address=base+i*128,local=l2_cache_index_addr(address,opt)/128;
  const unsigned p=dram_partition_index(address,opt),set=(local%1024)^((local>>10)&1023);
  ++reference.at(p*1024+set);
 }
 if(std::any_of(reference.begin(),reference.end(),[](unsigned x){return x!=16;}))throw std::runtime_error("test footprint does not distribute 16 tags per set");
 auto access=[&](uint64_t address){auto p=dram_partition_index(address,opt);if(p>=20)throw std::runtime_error("partition bounds");return caches[p].access(address,32,32,0,0,CacheOperation::Read,false,l2_cache_index_addr(address,opt),true);};
 for(uint64_t i=0;i<lines;i++){auto r=access(base+i*128);if(r.result==CacheResult::Hit||r.evicted.present)throw std::runtime_error("first-sector fill aliased/evicted");}
 for(auto &c:caches){auto o=c.occupancy();if(o.allocated_lines!=16384||o.sets_by_allocated_lines.at(16)!=1024||o.valid_sectors!=16384||o.dirty_sectors)throw std::runtime_error("all-set capacity not reachable");}
 for(uint64_t i=0;i<lines;i++){auto r=access(base+i*128);hits+=r.result==CacheResult::Hit;}
 if(hits!=lines)throw std::runtime_error("complete capacity reuse failed");
 for(uint64_t i=0;i<lines;i++)for(unsigned s=1;s<4;s++){auto r=access(base+i*128+s*32);if(r.result!=CacheResult::SectorMiss||r.evicted.present)throw std::runtime_error("independent sector fill failed");}
 for(auto &c:caches){auto o=c.occupancy();if(o.allocated_lines!=16384||o.sets_by_allocated_lines.at(16)!=1024||o.valid_sectors!=65536||o.dirty_sectors)throw std::runtime_error("full-sector capacity mismatch");sectors+=o.valid_sectors;}
 std::cout<<"{\"status\":\"PASS_ACCEPTED_STRUCTURAL_GEOMETRY\",\"hardware_acceptance\":false,\"logical_slices\":20,\"sets_per_slice\":1024,\"ways\":16,\"allocated_lines\":"<<lines<<",\"one_sector_reuse_hits\":"<<hits<<",\"full_valid_sectors\":"<<sectors<<",\"reachable_capacity_bytes\":"<<sectors*32<<",\"replacement_scope\":\"existing LRU reference; paper policy pending\"}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
