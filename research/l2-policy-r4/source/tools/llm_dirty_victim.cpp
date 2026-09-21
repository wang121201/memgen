// Candidate victim rules on the frozen ordered L2 request stream.
#define main unused_hbserve_main
#include "hbserve_profile_stream_cache_semantic_r17.cpp"
#undef main
#include <unordered_set>
#define L2_RETENTION_CANDIDATE_NO_MAIN
#include "l2_retention_candidate.cpp"
#include <memory>

void check_victim(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
int result_victim(const CacheAccess&a){return a.result==CacheResult::Hit?0:a.result==CacheResult::HitReserved?1:a.result==CacheResult::LineMiss?2:3;}
void reference_victim(const CacheAccess&a,const hyfiss_request_trace::L2AccessObservation&o){
 check_victim(result_victim(a)==o.outcome&&a.valid_before==o.valid_before&&a.valid_after==o.valid_after&&a.dirty_before==o.dirty_before&&a.dirty_after==o.dirty_after&&a.evicted.present==o.victim&&a.evicted.addr==o.victim_addr&&a.evicted.valid_sectors==o.victim_valid&&a.evicted.dirty_sectors==o.victim_dirty,"reference sidecar differs from frozen backend");
}
std::unique_ptr<RetentionReplay> candidate_victim(const char*config,unsigned i){
 auto p=std::make_unique<RetentionReplay>(config,i==2?"fixed_budget8_clean_retained_after_store_v1":"disabled",false);
 if(i){p->policy=i==1?"read_clean_first_lru_v1":"read_clean_first_budget8_v1";for(auto&c:p->caches)c.diagnostic_read_clean_first=true;}
 return p;
}
void fixture_victim(const char*config){
 auto ref=candidate_victim(config,0),clean=candidate_victim(config,1);std::vector<uint64_t> lines;
 for(uint64_t x=0;lines.size()<40 && x<(128ull<<20);x+=128)if(dram_partition_index(x,ref->opt)==0&&ref->caches[0].diagnostic_set_id(l2_cache_index_addr(x,ref->opt))==0)lines.push_back(x);
 check_victim(lines.size()==40,"fixture set population");
 for(auto*p:{ref.get(),clean.get()}){p->epoch=1;p->access(lines[0],true);for(unsigned i=1;i<16;++i)p->access(lines[i],false);p->epoch=2;p->access(lines[16],false);}
 check_victim(ref->last_access.evicted.addr==lines[0]&&ref->c.emitted==32,"LRU reference victim");
 check_victim(clean->last_access.evicted.addr==lines[1]&&clean->c.emitted==0&&clean->owners.count(lines[0]),"read clean-first retains oldest dirty and evicts oldest clean");
 clean->access(lines[0],false);check_victim(clean->last_access.result==CacheResult::Hit,"retained dirty read hit");
 auto store_mixed=candidate_victim(config,1);store_mixed->epoch=1;store_mixed->access(lines[0],true);for(unsigned i=1;i<16;++i)store_mixed->access(lines[i],false);store_mixed->access(lines[16],true);
 check_victim(store_mixed->last_access.evicted.addr==lines[0]&&store_mixed->c.emitted==32,"mixed-set store must keep original LRU victim");
 auto sector=candidate_victim(config,1);sector->epoch=1;sector->access(lines[0],true);for(unsigned i=1;i<16;++i)sector->access(lines[i],false);sector->access(lines[0]+32,false);
 check_victim(sector->last_access.result==CacheResult::SectorMiss&&!sector->last_access.evicted.present&&sector->c.emitted==0,"sector miss must not replace a tag");
 // Store misses preserve ordinary LRU. All old lines dirty: exclude incoming
 // clean allocation from preference and fall back to the original dirty LRU.
 for(bool store:{false,true}){auto p=candidate_victim(config,1);p->epoch=1;for(unsigned i=0;i<16;++i)p->access(lines[i],true);p->epoch=2;p->access(lines[16],store);
  check_victim(p->last_access.evicted.addr==lines[0]&&p->c.emitted==32&&p->versions[1].emitted==1,"all dirty/store fallback must evict old LRU, not incoming line");
 }
 auto partial=candidate_victim(config,1);partial->epoch=1;partial->access(lines[0],true,255);
 for(unsigned i=1;i<=16;++i)partial->access(lines[i],false);
 check_victim(partial->owners.count(lines[0])&&partial->c.emitted==0,"partial dirty retained");
 partial->access(lines[0],true,UINT32_MAX^255u);const auto fills=partial->c.fill;partial->access(lines[0],false);
 check_victim(partial->last_access.result==CacheResult::Hit&&partial->c.fill==fills,"complementary writes complete retained sector without a read");
 auto demand=candidate_victim(config,1);demand->epoch=1;demand->access(lines[0],true,255);demand->access(lines[0],false);
 check_victim(demand->last_access.result==CacheResult::SectorMiss&&demand->last_access.valid_after==1&&demand->dirty()==1&&demand->c.fill==32,"demand read completes partial dirty bytes while retaining dirty");
 auto fallback=candidate_victim(config,1);fallback->epoch=1;for(unsigned i=0;i<16;++i)fallback->access(lines[i],true,255);fallback->epoch=2;fallback->access(lines[16],false);
 check_victim(fallback->last_access.evicted.incomplete_dirty_sectors==1&&fallback->last_access.evicted.missing_dirty_bytes==24&&fallback->c.emitted==32&&fallback->c.fill==32,"partial victim preserves diagnostic coverage; no invented preservation read");
 auto budget=candidate_victim(config,2);budget->epoch=1;for(unsigned i=0;i<9;++i)budget->access(lines[i],true);
 check_victim(budget->dirty()==8&&budget->c.cleaned==1&&budget->c.emitted==32,"budget8 clean retained");
 budget->access(lines[0],false);check_victim(budget->last_access.result==CacheResult::Hit,"budget cleanup retained tag");
 for(auto*p:{ref.get(),clean.get(),partial.get(),fallback.get(),budget.get()}){p->assert_pressure_metadata();std::ostringstream a,b;p->write_ownership(p->policy,a,b);}
 std::cout<<"PASS clean-first oldest-clean, dirty retention, store/all-dirty fallback, partial/complementary coverage, producer ownership, budget8\n";
}

int main(int argc,char**argv){try{
 if(argc==3&&std::string(argv[1])=="--fixture"){fixture_victim(argv[2]);return 0;}
 using namespace hbserve_profile_stream;auto began=std::chrono::steady_clock::now();const Arguments args=parse_arguments(argc,argv);
 need(args.mode=="memgen"&&!fs::exists(args.output_dir)&&!fs::exists(args.stats),"fresh memgen outputs required");WorkloadSource source(args.profile_index,args.app_config,args.issue_config,true);
 std::vector<std::unique_ptr<RetentionReplay>> policies;for(unsigned i=0;i<3;++i)policies.push_back(candidate_victim(args.hw_config.c_str(),i));
 const auto parent=args.output_dir.parent_path();std::ofstream out(parent/"dirty-policies.csv"),snap(parent/"versions-at-kernel.csv");
 out<<"policy,kernel_id,requests,source_read_sectors,source_write_sectors,read_hits,read_misses,read_B,write_B,victim_lines,dirty_entry,dirty_created,redirty,dirty_evicted,cleaned_retained,dirty_exit,request_digest,tag_digest,read_state_equal,outcome_differences,victim_tag_differences,incomplete_victim_sectors,missing_victim_bytes,residual_B\n";
 snap<<"policy,boundary_kernel,producer,writes,superseded,emitted,resident,residual\n";
 int kernel=0;uint64_t requests=0;std::vector<uint64_t> entry(3),outcomes(3),victims(3),incomplete(3),missing(3);
 auto flush=[&](){if(!kernel)return;for(size_t i=0;i<policies.size();++i){auto&p=*policies[i];const auto&c=p.c;const auto after=p.dirty();const auto&ref=*policies[0];
  check_victim(entry[i]+c.created==after+c.evicted+c.cleaned&&c.created+c.redirty==c.writes&&c.emitted==32*(c.evicted+c.cleaned),"dirty/source conservation");
  check_victim(p.owners.size()==after&&c.reads+c.writes==requests&&c.read_hits+c.read_misses==c.reads&&c.fill==32*c.read_misses,"resident/request/read conservation");
  check_victim(ref.c.reads==c.reads&&ref.c.writes==c.writes&&ref.c.digest==c.digest,"candidate changed ordered input requests");p.assert_pressure_metadata();
  bool equal=true;for(size_t j=0;j<p.caches.size();++j)equal&=ref.caches[j].diagnostic_same_read_state(p.caches[j]);
  std::map<int,uint64_t> resident;for(const auto&x:p.owners)++resident[x.second];for(const auto&[id,v]:p.versions){check_victim(v.writes==v.superseded+v.emitted+resident[id],"version conservation");snap<<p.policy<<','<<kernel<<','<<id<<','<<v.writes<<','<<v.superseded<<','<<v.emitted<<','<<resident[id]<<",0\n";}
  out<<p.policy<<','<<kernel<<','<<requests<<','<<c.reads<<','<<c.writes<<','<<c.read_hits<<','<<c.read_misses<<','<<c.fill<<','<<c.emitted<<','<<c.victims<<','<<entry[i]<<','<<c.created<<','<<c.redirty<<','<<c.evicted<<','<<c.cleaned<<','<<after<<','<<c.digest<<','<<c.tag_digest<<','<<equal<<','<<outcomes[i]<<','<<victims[i]<<','<<incomplete[i]<<','<<missing[i]<<",0\n";
 }out.flush();snap.flush();check_victim(bool(out)&&bool(snap),"summary write failure");};
 int code=run_memgen(args,source,[&](const hyfiss_request_trace::L2AccessObservation&o){
  if(o.kernel_id!=kernel){flush();kernel=o.kernel_id;requests=0;for(size_t i=0;i<3;++i){policies[i]->epoch=kernel;policies[i]->c={};entry[i]=policies[i]->dirty();outcomes[i]=victims[i]=incomplete[i]=missing[i]=0;}}
  check_victim((o.operation=='R'||o.operation=='W')&&o.addr%32==0&&o.outcome!=1,"unsupported L2 request domain");++requests;
  for(size_t i=0;i<3;++i){auto&p=*policies[i];check_victim(dram_partition_index(o.addr,p.opt)==o.partition&&l2_cache_index_addr(o.addr,p.opt)==o.index_addr,"same address mapping required");p.access(o.addr,o.operation=='W',o.byte_mask);const auto&a=p.last_access;if(!i)reference_victim(a,o);
   outcomes[i]+=result_victim(a)!=o.outcome;victims[i]+=a.evicted.present!=o.victim||(a.evicted.present&&a.evicted.addr!=o.victim_addr);incomplete[i]+=__builtin_popcount(a.evicted.incomplete_dirty_sectors);missing[i]+=a.evicted.missing_dirty_bytes;
  }
 });flush();need(code==0,"backend failure");
 std::ofstream ownership(parent/"ownership.csv"),versions(parent/"versions.csv");ownership<<"policy,producer,trigger,reason,sectors,bytes,queue,admission,completion\n";versions<<"policy,producer,writes,superseded,emitted,resident,residual\n";
 for(auto&p:policies)p->write_ownership(p->policy,ownership,versions);
 write_stats(args,source,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-began).count(),code);
 std::cout<<"PASS_SAME_L2_INPUT_DIFFERENT_VICTIM_READ_OUTCOMES_AND_VERSION_CONSERVATION\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
