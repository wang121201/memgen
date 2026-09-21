// Functional dirty-policy sidecars on the exact live frozen L2 request stream.
#define main unused_hbserve_main
#include "hbserve_profile_stream_cache_semantic_r17.cpp"
#undef main
#include <unordered_set>
#define L2_RETENTION_CANDIDATE_NO_MAIN
#include "l2_retention_candidate.cpp"
#include <memory>

void require_pressure(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int outcome_pressure(const CacheAccess&a){return a.result==CacheResult::Hit?0:a.result==CacheResult::HitReserved?1:a.result==CacheResult::LineMiss?2:3;}
void match_pressure(const CacheAccess&a,const hyfiss_request_trace::L2AccessObservation&o,bool dirty){
 require_pressure(outcome_pressure(a)==o.outcome && a.valid_before==o.valid_before && a.valid_after==o.valid_after && a.evicted.present==o.victim && a.evicted.addr==o.victim_addr && a.evicted.valid_sectors==o.victim_valid,"candidate altered tag/read outcome");
 if(dirty)require_pressure(a.dirty_before==o.dirty_before && a.dirty_after==o.dirty_after && a.evicted.dirty_sectors==o.victim_dirty,"reference dirty state differs from backend");
}

void fixture_pressure(const char*config){
 RetentionReplay r(config,"read_allocation_age8_clean_retained_v1",false);
 std::vector<uint64_t> lines;
 for(uint64_t x=0;lines.size()<32 && x<(128ull<<20);x+=128)
  if(dram_partition_index(x,r.opt)==0 && r.caches[0].diagnostic_set_id(l2_cache_index_addr(x,r.opt))==0)lines.push_back(x);
 require_pressure(lines.size()==32,"fixture set population");
 r.epoch=1;r.access(lines[0],true);
 for(unsigned i=1;i<=7;++i)r.access(lines[i],false);
 require_pressure(r.c.cleaned==0 && r.c.pressure_reads==7 && r.dirty()==1,"seven allocations must not clean");
 r.access(lines[1],false);r.access(lines[1]+32,false);
 require_pressure(r.c.pressure_reads==7 && r.c.cleaned==0,"hit and sector miss must not age");
 r.epoch=2;r.access(lines[8],false);
 require_pressure(r.c.cleaned==1 && r.c.emitted==32 && r.c.evicted==0 && r.dirty()==0,"eighth allocation cleans without victim");
 require_pressure(r.versions[1].emitted==1 && r.lineage[{1,2,"clean_retained"}]==1,"cross-kernel producer/trigger");
 r.access(lines[0],false);require_pressure(r.last_access.result==CacheResult::Hit,"clean must retain valid tag");
 r.access(lines[9],false);require_pressure(r.c.emitted==32,"unchanged clean version must not re-emit");
 r.epoch=3;r.access(lines[0],true);require_pressure(r.dirty()==1 && r.versions[3].writes==1,"rewrite creates new obligation");
 r.epoch=4;
 for(unsigned i=10;i<=17;++i)r.access(lines[i],false);
 require_pressure(r.c.cleaned==2 && r.c.emitted==64 && r.c.evicted==0 && r.dirty()==0 && r.lineage[{3,4,"clean_retained"}]==1,"redirty version must clean once with new producer");
 r.assert_pressure_metadata();
 std::ostringstream events,versions;r.write_ownership("pressure",events,versions);

 RetentionReplay recent(config,"read_allocation_age8_clean_retained_v1",false);recent.epoch=1;recent.access(lines[0],true);
 for(unsigned i=1;i<=7;++i)recent.access(lines[i],false);
 recent.access(lines[0],true);recent.access(lines[8],false);
 require_pressure(recent.c.cleaned==0 && recent.c.redirty==1,"store resets line pressure age");
 RetentionReplay partial(config,"read_allocation_age8_clean_retained_v1",false);partial.epoch=1;partial.access(lines[0],true,255);
 for(unsigned i=1;i<=8;++i)partial.access(lines[i],false);
 require_pressure(partial.c.cleaned==0 && partial.c.pressure_partial_blocked>0 && partial.dirty()==1,"incomplete dirty bytes must survive");
 const auto fill_before=partial.c.fill;partial.access(lines[0],true,UINT32_MAX ^ 255u);
 for(unsigned i=9;i<=16;++i)partial.access(lines[i],false);
 require_pressure(partial.c.cleaned==1 && partial.c.emitted==32 && partial.dirty()==0 && partial.c.fill==fill_before+8*32,"complementary writes complete sector without a fill");
 partial.assert_pressure_metadata();
 RetentionReplay filled(config,"read_allocation_age8_clean_retained_v1",false);filled.epoch=1;filled.access(lines[0],true,255);
 for(unsigned i=1;i<=8;++i)filled.access(lines[i],false);
 filled.access(lines[0],false);require_pressure(filled.last_access.result==CacheResult::SectorMiss && filled.c.cleaned==0 && filled.c.fill==9*32,"partial read fill preserves dirty obligation");
 filled.access(lines[9],false);require_pressure(filled.c.cleaned==1 && filled.c.emitted==32 && filled.c.fill==10*32,"read completed sector becomes eligible once");
 filled.assert_pressure_metadata();
 RetentionReplay replaced(config,"read_allocation_age8_clean_retained_v1",false);replaced.epoch=1;
 for(unsigned i=0;i<=16;++i)replaced.access(lines[i],true);
 require_pressure(replaced.c.evicted==1 && !replaced.written_at_read_count.count(lines[0]/128),"tag eviction removes pressure metadata");
 replaced.epoch=2;replaced.access(lines[0],true);replaced.assert_pressure_metadata();
 replaced.access(lines[17],false);require_pressure(replaced.c.cleaned==0 && replaced.written_at_read_count.at(lines[0]/128)==0,"reinserted tag starts a fresh age");
 RetentionReplay oldest(config,"read_allocation_age8_clean_retained_v1",false);oldest.epoch=1;oldest.access(lines[0],true);oldest.access(lines[1],true);oldest.access(lines[0],false);
 for(unsigned i=2;i<=9;++i)oldest.access(lines[i],false);
 require_pressure(oldest.c.cleaned==1 && oldest.owners.count(lines[0]) && !oldest.owners.count(lines[1]),"choose tag-LRU eligible line, at most one per read allocation");
 oldest.assert_pressure_metadata();recent.assert_pressure_metadata();
 RetentionReplay sector(config,"read_allocation_age8_one_sector_v1",false);sector.epoch=1;
 for(unsigned s=0;s<4;++s)sector.access(lines[0]+s*32,true);
 for(unsigned i=1;i<=8;++i)sector.access(lines[i],false);
 require_pressure(sector.c.cleaned==1 && sector.c.emitted==32 && sector.dirty()==3 && sector.written_at_read_count.count(lines[0]/128) && !sector.owners.count(lines[0]),"sector rule cleans lowest sector only and retains remaining ages");
 sector.assert_pressure_metadata();
 std::ostringstream joint;r.write_joint(joint);partial.write_joint(joint);filled.write_joint(joint);recent.write_joint(joint);oldest.write_joint(joint);sector.write_joint(joint);
 RetentionReplay full(config,"read_allocation_age8_clean_retained_v1",false);full.epoch=1;
 for(unsigned i=0;i<16;++i)full.access(lines[i],false);
 require_pressure(full.last_access.result==CacheResult::LineMiss && !full.last_access.evicted.present,"15 to16 tags has no victim");
 full.access(lines[16],false);require_pressure(full.last_access.evicted.present,"full read set allocation has a victim");
 full.assert_pressure_metadata();full.write_joint(joint);
 for(unsigned s=0;s<4;++s){sector.access(lines[0]+s*32,false);require_pressure(sector.last_access.result==CacheResult::Hit,"partial cleaning preserves all read hits");}
 for(unsigned i=9;i<=11;++i)sector.access(lines[i],false);
 require_pressure(sector.c.cleaned==4 && sector.c.emitted==128 && sector.dirty()==0 && sector.c.pressure_dirty_read_hits==3,"four triggers clean four sectors without duplicate versions");
 sector.assert_pressure_metadata();
 std::cout<<"PASS pressure threshold, read exclusion, retained hit, producer lifecycle, redirty, eviction/reinsert, partial completion, metadata conservation, tag-LRU and sector granularity\n";
}

int main(int argc,char**argv){try{
 if(argc==3 && std::string(argv[1])=="--fixture"){fixture_pressure(argv[2]);return 0;}
 using namespace hbserve_profile_stream;
 auto began=std::chrono::steady_clock::now();const Arguments args=parse_arguments(argc,argv);
 need(args.mode=="memgen" && !fs::exists(args.output_dir) && !fs::exists(args.stats),"fresh memgen outputs required");
 WorkloadSource source(args.profile_index,args.app_config,args.issue_config,true);
 std::vector<std::unique_ptr<RetentionReplay>> policies;
 for(const char*p:{"disabled","fixed_budget8_clean_retained_after_store_v1","read_allocation_age8_clean_retained_v1","read_allocation_age8_one_sector_v1"})policies.emplace_back(new RetentionReplay(args.hw_config.c_str(),p,false));
 auto parent=args.output_dir.parent_path();std::ofstream out(parent/"dirty-policies.csv");
 std::ofstream snapshots(parent/"versions-at-kernel.csv");
 snapshots<<"policy,boundary_kernel,producer,writes,superseded,emitted,resident,residual\n";
 std::ofstream joint(parent/"joint.csv");
 joint<<"policy,kernel_id,set_tags_after,dirty_lines_after,dirty_sectors_after,partial_dirty_lines,eligible_lines,selected_oldest_rank,selected_age_bin,selected_dirty_sectors,partition_dirty_bin,victim_present,victim_dirty_sectors,count,partition_dirty_sum,partition_dirty_min,partition_dirty_max,selected_age_sum,selected_age_min,selected_age_max\n";
 out<<"policy,kernel_id,requests,source_read_sectors,source_write_sectors,read_hits,read_misses,read_B,write_B,victim_lines,dirty_entry,dirty_created,redirty,dirty_evicted,cleaned_retained,dirty_exit,read_allocations,pressure_dirty_seen,pressure_eligible_seen,pressure_partial_blocked,pressure_max_age,pressure_clean_events,pressure_dirty_under8_events,pressure_dirty_read_hits,pressure_selected_revisited,selected_dirty1,selected_dirty2,selected_dirty3,selected_dirty4,request_digest,tag_digest,read_state_equal,residual_B\n";
 int kernel=0;uint64_t requests=0;std::vector<uint64_t> entry(policies.size());
 auto flush=[&](){
  if(!kernel)return;
  for(size_t i=0;i<policies.size();++i){auto &p=*policies[i];const auto&c=p.c;const auto after=p.dirty();const auto &ref=*policies[0];
   require_pressure(entry[i]+c.created==after+c.evicted+c.cleaned && c.created+c.redirty==c.writes && c.emitted==32*(c.evicted+c.cleaned),"dirty/source conservation");
   require_pressure(p.owners.size()==after,"owner/resident conservation");
   p.assert_pressure_metadata();
   p.write_joint(joint);
   std::map<int,uint64_t> resident;for(const auto&x:p.owners)++resident[x.second];
   for(const auto&[id,v]:p.versions){
    require_pressure(v.writes==v.superseded+v.emitted+resident[id],"dirty version conservation");
    snapshots<<p.policy<<','<<kernel<<','<<id<<','<<v.writes<<','<<v.superseded<<','<<v.emitted<<','<<resident[id]<<",0\n";
   }
   for(size_t j=0;j<p.caches.size();++j)require_pressure(ref.caches[j].diagnostic_same_read_state(p.caches[j]),"tag/valid/known/LRU state changed");
   require_pressure(ref.c.reads==c.reads && ref.c.writes==c.writes && ref.c.fill==c.fill && ref.c.read_hits==c.read_hits && ref.c.read_misses==c.read_misses && ref.c.digest==c.digest && ref.c.tag_digest==c.tag_digest,"read or request equivalence");
   out<<p.policy<<','<<kernel<<','<<requests<<','<<c.reads<<','<<c.writes<<','<<c.read_hits<<','<<c.read_misses<<','<<c.fill<<','<<c.emitted<<','<<c.victims<<','<<entry[i]<<','<<c.created<<','<<c.redirty<<','<<c.evicted<<','<<c.cleaned<<','<<after<<','<<c.pressure_reads<<','<<c.pressure_dirty_seen<<','<<c.pressure_eligible_seen<<','<<c.pressure_partial_blocked<<','<<c.pressure_max_age<<','<<c.pressure_clean_events<<','<<c.pressure_dirty_under8_events<<','<<c.pressure_dirty_read_hits<<','<<c.pressure_selected_revisited;
   for(unsigned s=1;s<=4;++s)out<<','<<c.pressure_selected_sectors[s];
   out<<','<<c.digest<<','<<c.tag_digest<<",1,0\n";
  }
  out.flush();require_pressure(bool(out),"policy summary output");
  snapshots.flush();require_pressure(bool(snapshots),"version boundary snapshot output");
 };
 int code=run_memgen(args,source,[&](const hyfiss_request_trace::L2AccessObservation&o){
  if(o.kernel_id!=kernel){flush();kernel=o.kernel_id;requests=0;for(size_t i=0;i<policies.size();++i){policies[i]->epoch=kernel;policies[i]->c={};policies[i]->joint.clear();entry[i]=policies[i]->dirty();}}
  require_pressure((o.operation=='R'||o.operation=='W') && o.addr%32==0 && o.outcome!=1,"unsupported L2 request domain");
  ++requests;
  for(size_t i=0;i<policies.size();++i){auto&p=*policies[i];require_pressure(dram_partition_index(o.addr,p.opt)==o.partition && l2_cache_index_addr(o.addr,p.opt)==o.index_addr,"index/partition mismatch");p.access(o.addr,o.operation=='W',o.byte_mask);match_pressure(p.last_access,o,i==0);}
 });
 flush();need(code==0,"backend failure");
 std::ofstream ownership(parent/"ownership.csv"),versions(parent/"versions.csv");
 ownership<<"policy,producer,trigger,reason,sectors,bytes,queue,admission,completion\n";
 versions<<"policy,producer,writes,superseded,emitted,resident,residual\n";
 for(auto&p:policies)p->write_ownership(p->policy,ownership,versions);
 write_stats(args,source,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-began).count(),code);
 std::cout<<"PASS_DIRTY_PRESSURE_POLICIES_SAME_REQUESTS_READ_STATE_AND_VERSION_CONSERVATION\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
