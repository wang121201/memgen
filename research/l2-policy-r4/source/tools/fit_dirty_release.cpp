// Candidate victim rules on the frozen ordered L2 request stream.
#define main unused_hbserve_main
#include "hbserve_profile_stream_cache_semantic_r17.cpp"
#undef main
#include <unordered_set>
#define L2_RETENTION_CANDIDATE_NO_MAIN
#include "l2_dirty_release_replay.cpp"
#include <memory>
#include "native_dirty_census.hpp"

void check_victim(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
int result_victim(const CacheAccess&a){return a.result==CacheResult::Hit?0:a.result==CacheResult::HitReserved?1:a.result==CacheResult::LineMiss?2:3;}
void reference_victim(const CacheAccess&a,const hyfiss_request_trace::L2AccessObservation&o){
 check_victim(result_victim(a)==o.outcome&&a.valid_before==o.valid_before&&a.valid_after==o.valid_after&&a.dirty_before==o.dirty_before&&a.dirty_after==o.dirty_after&&a.evicted.present==o.victim&&a.evicted.addr==o.victim_addr&&a.evicted.valid_sectors==o.victim_valid&&a.evicted.dirty_sectors==o.victim_dirty,"reference sidecar differs from frozen backend");
}
unsigned selected_read_age=8;
unsigned selected_write_quota=2;
unsigned selected_phase_variant=0;
unsigned selected_behavior_arm=0;
bool selected_sector_age=false;
bool selected_disable_aged=false;
std::unique_ptr<RetentionReplay> candidate_victim(const char*config,unsigned i){
 auto p=std::make_unique<RetentionReplay>(config,i?"fixed_budget8_clean_retained_after_store_v1":"disabled",false);
 if(i){p->write_budget=selected_write_quota;p->policy="read_clean_first_budget"+std::to_string(selected_write_quota)+"_control_v1";for(auto&c:p->caches)c.diagnostic_read_clean_first=true;}
 if(i==2){p->release_mode=true;p->release_age=selected_read_age;p->policy="read_clean_first_budget"+std::to_string(selected_write_quota)+"_min1_age"+std::to_string(selected_read_age)+"_v1";}
 if(i==2){p->phase_variant=selected_phase_variant;if(selected_phase_variant)p->policy+="_phase"+std::to_string(selected_phase_variant);}
 if(i==2){p->behavior_arm=selected_behavior_arm;if(selected_behavior_arm)p->policy+="_windowarm"+std::to_string(selected_behavior_arm);}
 if(i==2){p->sector_age=selected_sector_age;if(selected_sector_age)p->policy+="_sectorage1";}
 if(i==2){p->disable_aged=selected_disable_aged;if(selected_disable_aged)p->policy+="_agedoff1";}
 return p;
}

void original_fixture_victim(const char*config){
 auto base=candidate_victim(config,0);std::vector<uint64_t> lines;uint64_t other=UINT64_MAX;
 for(uint64_t x=0;lines.size()<40 && x<(128ull<<20);x+=128){const auto p=dram_partition_index(x,base->opt);const auto s=base->caches[p].diagnostic_set_id(l2_cache_index_addr(x,base->opt));if(p==0&&s==0)lines.push_back(x);else if(p==0&&other==UINT64_MAX)other=x;}
 check_victim(lines.size()==40&&other!=UINT64_MAX,"fixture addresses");
 for(unsigned age:{1,4,8,16}){
  selected_read_age=age;auto p=candidate_victim(config,2);p->epoch=1;p->access(lines[0],true);
  p->access(lines[0],false);p->access(lines[0]+32,false);p->access(other,false);check_victim(p->c.emitted==0&&p->set_read_allocations.at({0,0})==0,"hits/sector misses/other sets not pressure");
  for(unsigned j=1;j<age;++j)p->access(lines[j],false);
  check_victim(p->c.emitted==0&&p->dirty()==1,"no early clean");p->epoch=2;p->access(lines[age],false);
  check_victim(p->c.emitted==32&&p->c.read_clean_sectors==1&&p->dirty()==0&&p->versions[1].emitted==1,"last dirty line released exactly at age");
  p->access(lines[0],false);check_victim(p->last_access.result==CacheResult::Hit,"tag retained");
  p->epoch=3;p->access(lines[0],true);check_victim(p->owners.at(lines[0])==3,"new dirty producer");
  for(unsigned j=age+1;j<2*age;++j)p->access(lines[j],false);
  check_victim(p->c.emitted==32,"rewrite refresh age");p->epoch=4;p->access(lines[2*age],false);
  check_victim(p->c.emitted==64&&p->versions[3].emitted==1&&p->dirty()==0,"rewrite pressure boundary");p->assert_pressure_metadata();
  std::ostringstream o,v;p->write_ownership(p->policy,o,v);
  auto partial=candidate_victim(config,2);partial->epoch=1;partial->access(lines[0],true,255);
  for(unsigned j=1;j<=age+1;++j)partial->access(lines[j],false);
  check_victim(partial->c.emitted==0&&partial->c.pressure_partial_blocked>0,"partial remains blocked");partial->assert_pressure_metadata();
  auto quota=candidate_victim(config,2);quota->epoch=1;for(unsigned j=0;j<3;++j)quota->access(lines[j],true);
  check_victim(quota->dirty()==2&&quota->c.store_clean_sectors==1,"same store quota");quota->assert_pressure_metadata();
 }
 std::cout<<"PASS min1 ages1/4/8/16 exact boundary, rewrite, no-age controls, producer, tag, partial and unchanged quota\n";
}


void fixture_victim(const char*config){
 selected_write_quota=2;original_fixture_victim(config);selected_read_age=16;
 auto base=candidate_victim(config,0);std::vector<uint64_t> lines;
 for(uint64_t x=0;lines.size()<40 && x<(128ull<<20);x+=128)if(dram_partition_index(x,base->opt)==0&&base->caches[0].diagnostic_set_id(l2_cache_index_addr(x,base->opt))==0)lines.push_back(x);
 check_victim(lines.size()==40,"quota fixture population");
 for(unsigned quota:{2,3,4,16})for(unsigned sectors:{1,4}){
  selected_write_quota=quota;auto p=candidate_victim(config,2);p->epoch=1;
  for(unsigned j=0;j<quota;++j)for(unsigned k=0;k<sectors;++k)p->access(lines[j]+32*k,true);
  check_victim(p->c.emitted==0&&p->dirty()==quota*sectors,"within quota no writeback");
  p->epoch=2;p->access(lines[quota],true);
  check_victim(p->c.emitted==32*sectors&&p->dirty()==(quota-1)*sectors+1,"boundary writes only victim dirty sectors");
  check_victim((quota<16?p->c.store_clean_sectors:p->c.evicted)==sectors,"quota cleaning vs ordinary eviction reason");
  check_victim(p->versions[1].emitted==sectors&&p->owners.at(lines[quota])==2,"boundary producer attribution");
  if(quota<16){p->access(lines[0],false);check_victim(p->last_access.result==CacheResult::Hit,"quota retains clean tag");}
  p->assert_pressure_metadata();std::ostringstream o,v;p->write_ownership(p->policy,o,v);
 }
 for(unsigned quota:{2,3,4,16}){
  selected_write_quota=quota;auto p=candidate_victim(config,2);p->epoch=1;p->access(lines[0],true);
  for(unsigned j=1;j<=15;++j)p->access(lines[j],false);
  check_victim(p->c.emitted==0,"15 reads below threshold");p->epoch=2;p->access(lines[0]+32,true);
  for(unsigned j=16;j<=30;++j)p->access(lines[j],false);
  check_victim(p->c.emitted==0,"other-sector store resets whole-line age");p->epoch=3;p->access(lines[31],false);
  check_victim(p->c.emitted==64&&p->dirty()==0&&p->versions[1].emitted==1&&p->versions[2].emitted==1,"two sectors retain distinct producers through one line clean");
  p->assert_pressure_metadata();std::ostringstream o,v;p->write_ownership(p->policy,o,v);
 }
 std::cout<<"PASS quota2/16 sector1/4 reasons and cross-sector age/version refresh\n";
}


void census_fixture(const char*config){
 selected_write_quota=4;selected_read_age=16;auto p=candidate_victim(config,2);NativeDirtyCensus n;
 std::vector<uint64_t> lines;for(uint64_t x=0;lines.size()<20&&x<(128ull<<20);x+=128)if(dram_partition_index(x,p->opt)==0&&p->caches[0].diagnostic_set_id(l2_cache_index_addr(x,p->opt))==0)lines.push_back(x);
 check_victim(lines.size()==20,"census fixture addresses");
 auto access=[&](uint64_t x,bool w){const auto before=p->c;auto it=p->owners.find(x);const int prior=it==p->owners.end()?0:it->second==p->epoch?1:2;p->access(x,w);n.observe(*p,x,w,UINT32_MAX,before,prior);};
 p->epoch=1;access(lines[0],false);access(lines[0],true);access(lines[0],true);access(lines[0]+32,true);access(lines[0]+64,false);access(lines[0],false);
 check_victim(n.requests==6&&n.created==2&&n.redirty==1&&n.read_allocations==1,"census fixture source population");std::ostringstream out;n.write(*p,out);
 n.reset_counts();p->c={};p->epoch=2;access(lines[0],true);for(unsigned j=1;j<=17;++j)access(lines[j],false);n.write(*p,out);
 check_victim(n.redirty==1&&p->c.read_clean_sectors==2,"census cross-kernel age and cleanup");
 std::cout<<"PASS passive census exact populations, cross-kernel age and producer persistence\n";
}

void phase_fixture(const char*config){
 selected_phase_variant=0;selected_write_quota=4;selected_read_age=16;auto base=candidate_victim(config,0);
 std::vector<uint64_t> lines;uint64_t other=UINT64_MAX;
 for(uint64_t x=0;lines.size()<40&&x<(128ull<<20);x+=128){auto part=dram_partition_index(x,base->opt);auto set=base->caches[part].diagnostic_set_id(l2_cache_index_addr(x,base->opt));if(part==0&&set==0)lines.push_back(x);else if(part==0&&other==UINT64_MAX)other=x;}
 check_victim(lines.size()==40&&other!=UINT64_MAX,"phase fixture addresses");
 auto model=[&](unsigned v){selected_phase_variant=v;auto p=candidate_victim(config,2);p->epoch=1;return p;};
 auto early=model(2);early->access(lines[0],true);early->access(lines[1],true);
 early->access(lines[1],false);early->access(lines[1]+32,false);early->access(other,false);
 check_victim(early->pc.closures==0&&early->c.emitted==0,"hits sector-miss and other set do not close");early->epoch=2;early->access(lines[2],false);
 check_victim(early->pc.closures==1&&early->pc.early_events==1&&early->c.emitted==32&&early->owners.count(lines[1])&& !early->owners.count(lines[0]),"early excludes latest and survives kernel boundary");
 early->access(lines[3],false);check_victim(early->pc.closures==1&&early->c.emitted==32,"closure consumed once");
 early->epoch=3;early->access(lines[0],true);early->access(lines[4],false);
 check_victim(early->pc.early_events==2&&early->owners.at(lines[0])==3&&early->versions[1].emitted==2,"clean then rewrite owner and latest exclusion");
 auto single=model(2);single->access(lines[0],true);single->access(lines[1],false);check_victim(single->c.emitted==0&&single->pc.closures==1,"one dirty cannot early release");
 auto partial=model(2);partial->access(lines[0],true,255);partial->access(lines[1],true);partial->access(lines[2],false);
 check_victim(partial->c.emitted==0&&partial->pc.closures==1&&partial->pc.early_qualified==0,"partial old line and latest full line ineligible");
 partial->access(lines[0],false);partial->access(lines[3],false);
 check_victim(partial->c.emitted==0&&partial->pc.closures==1,"failed closure does not retry after full read");
 auto paced=model(1);paced->access(lines[0],true);paced->access(lines[1],true);
 for(unsigned i=2;i<=16;++i)paced->access(lines[i],false);
 check_victim(paced->c.emitted==0,"age15 no release");paced->access(lines[17],false);
 check_victim(paced->c.emitted==32&&paced->pc.aged_events==1,"first release at age16 unrestricted by synthetic timestamp");
 paced->epoch=2;for(unsigned i=18;i<=32;++i)paced->access(lines[i],false);
 check_victim(paced->c.emitted==32&&paced->pc.spacing_blocked==15,"spacing15 survives kernel");paced->access(lines[33],false);
 check_victim(paced->c.emitted==64&&paced->pc.aged_events==2,"spacing16 exact release");
 auto both=model(3);both->access(lines[0],true);both->access(lines[1],true);both->access(lines[2],false);both->access(lines[3],true);both->access(lines[4],false);
 check_victim(both->pc.early_qualified==2&&both->pc.early_events==1&&both->pc.spacing_blocked==1&&both->c.emitted==32,"early also obeys spacing and blocked closure consumed");
 for(unsigned i=5;i<=19;++i)both->access(lines[i],false);
 check_victim(both->pc.aged_events==1&&both->c.emitted==64&&both->pc.closures==2,"fallback age release after spacing without retrying closure");
 auto joint=model(2);joint->access(lines[0],true);joint->access(lines[1],true);
 for(unsigned i=2;i<=16;++i){joint->access(lines[1],true);joint->access(lines[i],false);}
 // An independent simultaneous-qualification case uses a still-young latest line.
 auto qualified=model(0);qualified->access(lines[0],true);for(unsigned i=1;i<=15;++i)qualified->access(lines[i],false);qualified->phase_variant=2;qualified->epoch=2;qualified->access(lines[15],true);qualified->access(lines[16],false);
 check_victim(qualified->pc.early_events==1&&qualified->pc.aged_events==0&&qualified->c.read_clean_sectors==1,"both paths eligible emit only once");
 for(unsigned v:{4,5}){
  auto sparse=model(v);sparse->access(lines[0],true);sparse->access(lines[1],true);sparse->access(lines[2],false);check_victim(sparse->pc.early_events==1&&sparse->c.emitted==32,"exact-two early accepted");
  auto dense=model(v);dense->access(lines[0],true);dense->access(lines[1],true);dense->access(lines[2],true);dense->access(lines[3],false);check_victim(dense->pc.early_events==0&&dense->pc.closures==1&&dense->c.emitted==0,"three-dirty early rejected");
  for(auto*p:{sparse.get(),dense.get()}){p->assert_pressure_metadata();check_victim(p->pc.early_sectors+p->pc.aged_sectors==p->c.read_clean_sectors,"guard reason exact");std::ostringstream o,v;p->write_ownership(p->policy,o,v);}
 }
 for(unsigned v:{6,7}){
  auto p=model(v);p->access(lines[0],true);p->access(lines[1],true);p->access(lines[2],false);p->access(lines[3],true);p->access(lines[4],false);
  check_victim(p->pc.early_events==1&&p->pc.spacing_blocked==1&&p->c.emitted==32,"early-only spacing blocks immediate retry");
  p->epoch=2;for(unsigned i=5;i<=18;++i)p->access(lines[i],false);
  check_victim(p->pc.aged_events==1&&p->c.emitted==64,"aged path remains unthrottled at count16 after count1 early");
  p->assert_pressure_metadata();check_victim(p->pc.early_sectors+p->pc.aged_sectors==p->c.read_clean_sectors,"early-only reason conservation");std::ostringstream o,z;p->write_ownership(p->policy,o,z);
 }
 for(unsigned v:{8,9}){
  auto sparse=model(v);sparse->access(lines[0],true);sparse->access(lines[1],true);sparse->access(lines[2],false);check_victim(sparse->pc.early_qualified==0&&sparse->c.emitted==0,"three-tag sparse set cannot early release");
  auto below=model(v);for(unsigned i=2;i<=13;++i)below->access(lines[i],false);below->access(lines[0],true);below->access(lines[1],true);below->access(lines[14],false);check_victim(below->pc.early_qualified==0&&below->c.emitted==0,"fifteen tags cannot early release");
  auto full=model(v);for(unsigned i=2;i<=14;++i)full->access(lines[i],false);full->access(lines[0],true);full->access(lines[1],true);full->access(lines[15],false);check_victim(full->pc.early_events==1&&full->c.emitted==32,"sixteen tags first early release");
  full->epoch=2;full->access(lines[16],true);full->access(lines[17],false);check_victim(full->pc.early_events==(v==8?2:1)&&full->pc.spacing_blocked==(v==9?1:0)&&full->pc.aged_events==0,"full-set spacing arm differs on immediate second release");
  for(auto*p:{sparse.get(),below.get(),full.get()}){p->assert_pressure_metadata();check_victim(p->pc.early_sectors+p->pc.aged_sectors==p->c.read_clean_sectors,"full-set release accounting");std::ostringstream o,z;p->write_ownership(p->policy,o,z);}
 }
 for(unsigned v:{10,11,12,13}){
  const unsigned mark=v<=11?15:14;const bool paced=v%2;
  auto below=model(v);for(unsigned i=2;i<=mark-3;++i)below->access(lines[i],false);below->access(lines[0],true);below->access(lines[1],true);below->access(lines[mark-2],false);
  check_victim(below->pc.early_events==0&&below->c.emitted==0,"one tag below watermark rejects");
  auto at=model(v);for(unsigned i=2;i<=mark-2;++i)at->access(lines[i],false);at->access(lines[0],true);at->access(lines[1],true);at->access(lines[mark-1],false);
  check_victim(at->pc.early_events==1&&at->c.emitted==32,"exact watermark admits");at->epoch=2;at->access(lines[mark],true);at->access(lines[mark+1],false);
  check_victim(at->pc.early_events==(paced?1:2)&&at->pc.spacing_blocked==(paced?1:0)&&at->pc.aged_events==0,"watermark pacing branch distinction");
  for(auto*p:{below.get(),at.get()}){p->assert_pressure_metadata();check_victim(p->pc.early_qualified==p->pc.early_events+p->pc.spacing_blocked&&p->pc.early_sectors+p->pc.aged_sectors==p->c.read_clean_sectors,"watermark qualification and sector accounting");std::ostringstream o,z;p->write_ownership(p->policy,o,z);}
 }
 for(unsigned v:{14,15}){
  auto p=model(v);for(unsigned i=2;i<=14;++i)p->access(lines[i],false);p->access(lines[0],true);p->access(lines[1],true);p->access(lines[15],false);
  check_victim(p->pc.proposals==1&&p->pc.pressure_rejected==1&&p->pc.early_qualified==0&&p->c.emitted==0,"sixteenth tag has no replacement and rejects early");
  p->access(lines[1],true);p->epoch=2;p->access(lines[16],false);
  check_victim(p->last_access.evicted.present&&!p->last_access.evicted.dirty&&p->pc.early_events==1&&p->c.emitted==32,"seventeenth tag replaces clean and admits early across kernel");
  p->access(lines[17],true);const auto prior=p->pc.proposals;p->access(lines[17],false);p->access(lines[17]+32,false);p->access(other,false);
  check_victim(p->pc.proposals==prior,"store victim hit sector miss and other-set accesses cannot trigger early");p->access(lines[18],false);
  check_victim(p->pc.early_events==(v==14?2:1)&&p->pc.spacing_blocked==(v==15?1:0),"replacement admission retains spacing distinction");
  check_victim(p->pc.proposals==p->pc.pressure_rejected+p->pc.early_qualified&&p->pc.early_qualified==p->pc.early_events+p->pc.spacing_blocked,"replacement admission partition");
  p->assert_pressure_metadata();std::ostringstream o,z;p->write_ownership(p->policy,o,z);
 }
 for(auto*p:{early.get(),single.get(),partial.get(),paced.get(),both.get(),joint.get(),qualified.get()}){p->assert_pressure_metadata();check_victim(p->pc.early_sectors+p->pc.aged_sectors==p->c.read_clean_sectors,"phase reason exact");std::ostringstream o,v;p->write_ownership(p->policy,o,v);}
 std::cout<<"PASS phase closure, latest/partial/one-line exclusions, age15/16, spacing15/16, failed-closure consumption, cross-kernel state and one-clean attribution\n";
 selected_phase_variant=0;
}


void behavior_fixture(const char*config){
 auto fresh=[&](){auto p=std::make_unique<RetentionReplay>(config,"disabled",false);p->epoch=1;return p;};
 auto probe=fresh();std::vector<uint64_t> lines;uint64_t other=UINT64_MAX;
 for(uint64_t x=0;lines.size()<80&&x<(256ull<<20);x+=128){auto part=dram_partition_index(x,probe->opt);auto set=probe->caches[part].diagnostic_set_id(l2_cache_index_addr(x,probe->opt));if(part==0&&set==0)lines.push_back(x);else if(part==0&&other==UINT64_MAX)other=x;}
 check_victim(lines.size()==80&&other!=UINT64_MAX,"behavior fixture addresses");const BehaviorObserver::SetKey key={0,0};
 auto ages=fresh();ages->access(lines[0],true);ages->access(lines[1],false);ages->access(lines[2],false);ages->access(lines[0]+32,true);
 check_victim(ages->behavior.age(key,lines[0])==2&&ages->behavior.age(key,lines[0]+32)==0,"independent neighboring sector age");
 ages->access(lines[0]+32,true);check_victim(ages->behavior.overwrites==1&&ages->behavior.age(key,lines[0])==2,"pre-write overwrite and neighbor age");
 const auto c=ages->behavior.sets.at(key).allocations;ages->access(lines[0]+64,false);ages->access(other,false);ages->access(lines[3],true);
 check_victim(ages->behavior.sets.at(key).allocations==c,"sector miss other set and store miss do not age reads");
 ages->access(lines[4],false);check_victim(ages->behavior.age(key,lines[0])==3&&ages->behavior.age(key,lines[0]+32)==1,"one read allocation ages every dirty sector");ages->behavior.validate(ages->caches);
 auto window=fresh();std::vector<unsigned> oracle;
 for(unsigned i=0;i<140;++i){
   bool store=i==0||i==2||i==64||i==100;uint64_t addr=lines[0]+(i%3)*32;
   const bool was=window->behavior.stamps.count(addr);window->access(addr,store);
   unsigned flags=(!store&&window->last_access.result==CacheResult::LineMiss?1:0)|(store?2:0)|(store&&was?4:0);oracle.push_back(flags);if(oracle.size()>64)oracle.erase(oracle.begin());
   unsigned r=0,w=0,o=0;for(auto f:oracle){r+=bool(f&1);w+=bool(f&2);o+=bool(f&4);}const auto&s=window->behavior.sets.at(key);
   check_victim(s.n==oracle.size()&&s.reads==r&&s.writes==w&&s.overwrites==o,"fixed window independent oracle including wrap boundaries");
 }
 check_victim(window->behavior.sets.at(key).n==64,"window full population");
 auto clean=fresh();clean->access(lines[0],true);clean->access(lines[0]+32,true);clean->access(lines[1],false);
 auto payload=clean->caches[0].diagnostic_clean_pressure(l2_cache_index_addr(lines[0],clean->opt),lines[0],true);clean->emit(payload,"clean_retained");
 check_victim(!clean->behavior.stamps.count(lines[0])&&clean->behavior.age(key,lines[0]+32)==1&&clean->behavior.sets.at(key).tags==2,"single sector clean preserves neighbor and tag");
 clean->behavior.validate(clean->caches);const auto prior=clean->behavior.overwrites;clean->access(lines[0],true);check_victim(clean->behavior.overwrites==prior,"write after clean is new dirty");
 clean->access(lines[0]+32,true);check_victim(clean->behavior.overwrites==prior+1,"resident dirty overwrite after neighbor cleaned");
 auto evict=fresh();evict->access(lines[0],true);evict->access(lines[0]+32,true);for(unsigned i=1;i<=16;++i)evict->access(lines[i],false);
 check_victim(evict->last_access.evicted.addr==lines[0]&&evict->behavior.victim_samples==2&&evict->behavior.stamps.empty(),"tag eviction removes precisely dirty timestamps");evict->behavior.validate(evict->caches);
 for(const auto&[k,z]:evict->behavior.sectors)if(k[0]==3)check_victim(z.min==15&&z.max==15,"victim age uses pre-request allocation count");
 auto partial=fresh();partial->access(lines[0],true,255);partial->access(lines[1],false);partial->access(lines[0]+32,true);
 check_victim(partial->behavior.age(key,lines[0])==1&&partial->behavior.age(key,lines[0]+32)==0,"partial sector participates without invented full validity");
 const auto prior_age=partial->behavior.age(key,lines[0]);partial->access(lines[0],false);check_victim(partial->last_access.result==CacheResult::SectorMiss&&partial->behavior.age(key,lines[0])==prior_age,"preservation fill does not reset age");partial->behavior.validate(partial->caches);
 selected_write_quota=4;selected_read_age=16;selected_phase_variant=11;
 auto a=candidate_victim(config,2),b=candidate_victim(config,2);a->epoch=1;b->epoch=700;
 std::string prefix;
 for(unsigned i=0;i<120;++i){bool store=i%3==0;auto addr=lines[i%50]+(i%4)*32;
   if(i%7==0){b->epoch+=13;b->behavior.reset_counts();}a->access(addr,store);b->access(addr,store);
   check_victim(a->behavior.state_signature()==b->behavior.state_signature()&&a->c.emitted==b->c.emitted&&a->c.digest==b->c.digest&&a->c.tag_digest==b->c.tag_digest,"label and segmentation invariant on every prefix");
   for(unsigned p=0;p<a->caches.size();++p)check_victim(a->caches[p].diagnostic_same_read_state(b->caches[p]),"prefix tag valid data state");
   a->behavior.validate(a->caches);b->behavior.validate(b->caches);
 }
 prefix=b->behavior.state_signature();a->access(lines[79],false);check_victim(prefix==b->behavior.state_signature(),"future suffix cannot affect saved prefix state");
 std::ostringstream r,t,u;window->behavior.write("renamed",77,r,t,u);check_victim(!r.str().empty()&&!u.str().empty(),"behavior population output");
 selected_phase_variant=0;std::cout<<"PASS_BEHAVIOR_WINDOW_ORACLE_SECTOR_AGE_PARTIAL_CLEAN_EVICTION_LABEL_SEGMENT_PREFIX\n";
}


void read_pressure_fixture(const char*config){
 selected_phase_variant=11;selected_write_quota=4;selected_read_age=16;
 auto fresh=[&](unsigned arm){selected_behavior_arm=arm;auto p=candidate_victim(config,2);p->epoch=1;return p;};
 auto probe=fresh(0);std::vector<uint64_t> lines;
 for(uint64_t x=0;lines.size()<90&&x<(256ull<<20);x+=128)if(dram_partition_index(x,probe->opt)==0&&probe->caches[0].diagnostic_set_id(l2_cache_index_addr(x,probe->opt))==0)lines.push_back(x);
 check_victim(lines.size()==90,"window fixture mapped addresses");const BehaviorObserver::SetKey key={0,0};
 // Full cache request sequences produce exactly r new reads in the last64,
 // with two dirty resident lines and either a free way or a clean victim.
 for(unsigned arm:{0,1,2,3})for(unsigned r:{3,4,6,7})for(bool victim:{false,true}){
  auto p=fresh(arm);const unsigned initial=(victim?17:16)-r;
  for(unsigned i=0;i<initial;++i)p->access(lines[i],false);
  for(unsigned i=0;i<62-r;++i)p->access(lines[2],false);
  for(unsigned i=0;i<r-1;++i)p->access(lines[initial+i],false);
  p->access(lines[0],true);p->access(lines[1],true);p->epoch=71;p->access(lines[initial+r-1],false);
  const auto&s=p->behavior.sets.at(key);const bool accept=arm==0||(arm==3&&victim)||r>=(arm==1?4u:7u);
  check_victim(s.n==64&&s.reads==r&&p->last_access.evicted.present==victim&&!p->last_access.evicted.dirty,"exact window and clean victim fixture");
  check_victim(p->pc.proposals==1&&p->pc.early_events==unsigned(accept)&&p->pc.pressure_rejected==unsigned(!accept)&&p->c.emitted==(accept?32u:0u),"window read threshold exact boundary");
  p->assert_pressure_metadata();std::ostringstream o,z;p->write_ownership(p->policy,o,z);
  for(const auto&[k,z]:p->behavior.sectors)if(k[0]==1)check_victim(k[10]==victim,"proposal trigger is actual ordinary tag victim");
 }
 // Nonzero allocation event leaves the request window without changing C.
 BehaviorObserver::SetState s;std::vector<unsigned> oracle;
 for(unsigned i=0;i<150;++i){bool a=i%17==0,w=i%17!=0&&i%9==0,o=w&&i%2==0;s.push(a,w,o);oracle.push_back(a|unsigned(w)<<1|unsigned(o)<<2);if(oracle.size()>64)oracle.erase(oracle.begin());unsigned r=0,ws=0,ow=0;for(auto f:oracle){r+=bool(f&1);ws+=bool(f&2);ow+=bool(f&4);}check_victim(s.n==oracle.size()&&s.reads==r&&s.writes==ws&&s.overwrites==ow,"nonzero read allocation ring oracle");}
 for(unsigned arm:{1,2,3}){
  s.n=63;s.reads=63;check_victim(RetentionReplay::behavior_decision(s,arm,false)==2,"short population despite high density");
  check_victim(RetentionReplay::behavior_decision(s,arm,true)==(arm==3?1u:2u),"victim arm explicitly exempts full-window too");
  s.n=64;s.reads=arm==1?3:6;check_victim(RetentionReplay::behavior_decision(s,arm,false)==3,"one below threshold");++s.reads;check_victim(RetentionReplay::behavior_decision(s,arm,false)==4,"exact threshold");
 }
 // Rejected early must preserve the old age16 fallback and real dirty data.
 for(unsigned arm:{1,2}){auto p=fresh(arm);p->access(lines[0],true);for(unsigned i=1;i<=15;++i)p->access(lines[i],false);
  for(unsigned i=0;i<64;++i)p->access(lines[0],false);p->access(lines[1],true);p->access(lines[16],false);
  check_victim(p->pc.early_events==0&&p->pc.aged_events==1&&p->c.emitted==32&&p->pc.pressure_rejected>=1,"low density keeps age fallback");p->assert_pressure_metadata();}
 // Every actual candidate is invariant to epoch labels and statistics cuts.
 for(unsigned arm:{1,2,3}){auto a=fresh(arm),b=fresh(arm);for(unsigned i=0;i<400;++i){bool w=i%7==0;auto addr=lines[i%80]+(i%4)*32;if(i%11==0){b->epoch+=9;b->behavior.reset_counts();b->behavior_admission.clear();}a->access(addr,w);b->access(addr,w);check_victim(a->behavior.state_signature()==b->behavior.state_signature()&&a->c.emitted==b->c.emitted&&a->c.digest==b->c.digest&&a->c.tag_digest==b->c.tag_digest,"active window arm prefix label segmentation invariance");}a->assert_pressure_metadata();b->assert_pressure_metadata();auto saved=b->behavior.state_signature();a->access(lines[89],false);check_victim(saved==b->behavior.state_signature(),"future suffix independence");std::ostringstream o,z;a->write_ownership(a->policy,o,z);}
 selected_behavior_arm=0;selected_phase_variant=0;
 std::cout<<"PASS_READ_PRESSURE_THRESHOLD_TRIGGER_POPULATION_FALLBACK_CAUSAL_INVARIANCE\n";
}


void sector_age_fixture(const char*config){
 selected_phase_variant=11;selected_write_quota=4;selected_read_age=16;selected_behavior_arm=0;
 auto fresh=[&](bool sector){selected_sector_age=sector;auto p=candidate_victim(config,2);p->epoch=1;return p;};
 auto finder=fresh(false);std::vector<uint64_t> lines;
 for(uint64_t x=0;lines.size()<90&&x<(256ull<<20);x+=128)if(dram_partition_index(x,finder->opt)==0&&finder->caches[0].diagnostic_set_id(l2_cache_index_addr(x,finder->opt))==0)lines.push_back(x);
 check_victim(lines.size()==90,"sector fixture mapped addresses");const BehaviorObserver::SetKey key={0,0};
 auto old=fresh(false),now=fresh(true);
 for(auto*p:{old.get(),now.get()}){
  p->access(lines[0],true);p->access(lines[0]+64,true);
  for(unsigned i=1;i<=15;++i)p->access(lines[i],false);
  check_victim(p->c.emitted==0,"sector age15 retains");
  p->epoch=2;p->access(lines[0]+32,true);p->access(lines[16],false);
  p->assert_pressure_metadata();
 }
 check_victim(old->c.emitted==0&&now->c.emitted==64&&now->owners.size()==1&&now->owners.count(lines[0]+32),"independent ages emit noncontiguous mask5 only");
 check_victim(now->pc.aged_events==1&&now->pc.aged_sectors==2&&now->pc.early_events==0&&now->behavior.age(key,lines[0]+32)==1,"old sectors closed young remains");
 check_victim(now->caches[0].diagnostic_same_read_state(old->caches[0])&&now->written_at_read_count.count(lines[0]/128),"partial clean preserves tag valid and row metadata");
 now->epoch=3;now->access(lines[0],true);check_victim(now->behavior.age(key,lines[0]+32)==1&&now->behavior.age(key,lines[0])==0,"neighbor age survives redirty");
 now->access(lines[0]+32,true);check_victim(now->versions[2].superseded==1,"real retained version overwritten");
 for(auto*p:{old.get(),now.get()}){p->assert_pressure_metadata();std::ostringstream o,z;p->write_ownership(p->policy,o,z);}
 // Identical ages preserve the original whole-line decision and read state.
 auto a=fresh(false),b=fresh(true);
 for(auto*p:{a.get(),b.get()}){
  for(unsigned s=0;s<4;++s)p->access(lines[0]+s*32,true);
  for(unsigned i=1;i<=15;++i)p->access(lines[i],false);
  p->access(lines[0],false);p->access(lines[16],false);p->assert_pressure_metadata();
 }
 check_victim(a->c.emitted==128&&b->c.emitted==128&&a->behavior.state_signature()==b->behavior.state_signature(),"same-age original equivalence");
 // Unknown bytes still block the complete line, even when another sector is old.
 auto partial=fresh(true);partial->access(lines[0],true,255);partial->access(lines[0]+64,true);
 for(unsigned i=1;i<=15;++i)partial->access(lines[i],false);
 partial->access(lines[0]+32,true);partial->access(lines[16],false);check_victim(partial->c.emitted==0,"original partial-line gate retained");
 partial->access(lines[0],false);check_victim(partial->behavior.age(key,lines[0])==16,"preservation fill does not reset age");partial->access(lines[17],false);
 check_victim(partial->c.emitted==64&&partial->owners.count(lines[0]+32),"complete old sectors now eligible only");partial->assert_pressure_metadata();
 // Existing early path continues to clean a whole eligible old line, even at age1.
 for(bool enabled:{false,true}){
  auto p=fresh(enabled);for(unsigned i=2;i<=13;++i)p->access(lines[i],false);
  p->access(lines[0],true);p->access(lines[0]+64,true);p->access(lines[1],true);p->access(lines[14],false);
  check_victim(p->pc.early_events==1&&p->c.emitted==64,"early whole-line behavior unchanged");
  p->access(lines[15],true);p->access(lines[16],false);check_victim(p->pc.spacing_blocked==1&&p->pc.early_events==1,"early spacing unchanged");p->assert_pressure_metadata();
 }
 // Label, split, future suffix invariance for a common ordered prefix.
 auto u=fresh(true),v=fresh(true);u->policy="arbitrary-A";v->policy="arbitrary-B";
 for(unsigned i=0;i<400;++i){const auto addr=lines[(i*7)%40]+32*(i%4);const bool store=i%5<2;
  u->access(addr,store);v->epoch=1+i/73;v->access(addr,store);
  if(i%73==0)v->behavior.reset_counts();
  check_victim(u->c.emitted==v->c.emitted&&u->behavior.state_signature()==v->behavior.state_signature(),"sector age causal label and segmentation invariant");
 }
 for(auto*p:{u.get(),v.get()}){p->assert_pressure_metadata();std::ostringstream o,z;p->write_ownership(p->policy,o,z);}
 const auto prefix=u->behavior.state_signature();v->access(lines[80],false);check_victim(u->behavior.state_signature()==prefix,"future suffix cannot mutate common prefix");
 selected_sector_age=false;selected_phase_variant=0;
 std::cout<<"PASS_SECTOR_AGE_MASK_METADATA_PARTIAL_EARLY_VERSION_CAUSAL_FIXTURES\n";
}

#include "aged_ablation_fixture.inc"
#include "dirty_protection_fixture.inc"
#include "early_emission_fixture.inc"
int main(int argc,char**argv){try{
 if(const char*v=std::getenv("MEMGEN_DIAGNOSTIC_WRITE_QUOTA")){std::string q(v);check_victim(q=="2"||q=="3"||q=="4"||q=="16","unregistered orthogonal quota");selected_write_quota=std::stoul(q);}
 if(const char*v=std::getenv("MEMGEN_DIAGNOSTIC_READ_AGE")){std::string x(v);check_victim(x=="1"||x=="4"||x=="8"||x=="16","unregistered pressure age");selected_read_age=std::stoul(x);}
 if(argc==3&&std::string(argv[1])=="--fixture"){fixture_victim(argv[2]);census_fixture(argv[2]);phase_fixture(argv[2]);behavior_fixture(argv[2]);read_pressure_fixture(argv[2]);sector_age_fixture(argv[2]);aged_ablation_fixture(argv[2]);dirty_protection_fixture(argv[2]);early_emission_fixture(argv[2]);return 0;}
 if(const char*v=std::getenv("MEMGEN_PHASE_VARIANT")){std::string x(v);check_victim(x=="0"||x=="1"||x=="2"||x=="3"||x=="4"||x=="5"||x=="6"||x=="7"||x=="8"||x=="9"||x=="10"||x=="11"||x=="12"||x=="13"||x=="14"||x=="15","unregistered finite phase variant");selected_phase_variant=std::stoul(x);}
 if(const char*v=std::getenv("MEMGEN_BEHAVIOR_ARM")){std::string x(v);check_victim(x=="0"||x=="1"||x=="2"||x=="3","unregistered behavior arm");selected_behavior_arm=std::stoul(x);check_victim(selected_phase_variant==11,"window arms require fixed phase11");}
 if(const char*v=std::getenv("MEMGEN_SECTOR_AGE")){std::string x(v);check_victim(x=="0"||x=="1","unregistered sector age");selected_sector_age=x=="1";check_victim(selected_phase_variant==11&&!selected_behavior_arm,"sector age requires phase11 arm0");}
 if(const char*v=std::getenv("MEMGEN_DISABLE_AGED")){std::string x(v);check_victim(x=="0"||x=="1","unregistered aged ablation");selected_disable_aged=x=="1";check_victim(selected_phase_variant==11&&!selected_behavior_arm&&!selected_sector_age,"aged ablation requires line-age phase11 arm0");}
  using namespace hbserve_profile_stream;auto began=std::chrono::steady_clock::now();const Arguments args=parse_arguments(argc,argv);
 need(args.mode=="memgen"&&!fs::exists(args.output_dir)&&!fs::exists(args.stats),"fresh memgen outputs required");WorkloadSource source(args.profile_index,args.app_config,args.issue_config,true);
 std::vector<std::unique_ptr<RetentionReplay>> policies;for(unsigned i=0;i<3;++i)policies.push_back(candidate_victim(args.hw_config.c_str(),i));
 for(auto&p:policies)p->protection.enabled=true;
  std::array<NativeDirtyCensus,3> census;
 const auto parent=args.output_dir.parent_path();
  std::ofstream early_emission(parent/"early-emission.csv");early_emission<<"policy,kernel_id,tag_replacement,qualified,spacing_blocked,emitted_events,emitted_sectors";for(unsigned m=1;m<16;++m)early_emission<<",mask"<<m;early_emission<<'\n';
  std::ofstream protection_victims(parent/"protection-replacements.csv"),protection_sectors(parent/"protection-sectors.csv"),protection_totals(parent/"protection-totals.csv");
  protection_victims<<"policy,kernel_id,tags,dirty_lines,dirty_sectors,lru_dirty_sectors,actual_dirty_sectors,actual_rank,events\n";
  protection_sectors<<"policy,kernel_id,kind,rank,valid,line_dirty_sectors,redirty_cap32,read_age_cap256,clean_age_cap256,sectors,read_age_sum,read_age_min,read_age_max,clean_age_sum,clean_age_min,clean_age_max\n";
  protection_totals<<"policy,kernel_id,requests,reads,writes,hits,sector_misses,line_misses,free_allocations,replacements,clean_replacements,dirty_replacements,protected_events,protected_lines,protected_samples,resident_replacement_samples,clean_samples,victim_samples,resident_samples,created,overwrites,cleaned,evicted,entry_dirty,exit_dirty,residual\n";
 std::ofstream behavior_admit_out(parent/"behavior-admission.csv");
 behavior_admit_out<<"policy,kernel_id,tag_replacement,window_n,window_reads,window_writes,window_overwrites,decision,proposals\n";

 std::ofstream behavior_requests(parent/"behavior-requests.csv"),behavior_sectors(parent/"behavior-sectors.csv"),behavior_totals(parent/"behavior-totals.csv");
 behavior_requests<<"policy,kernel_id,event,tags,dirty_lines,dirty_sectors,window_n,window_reads,window_writes,window_overwrites,tag_eviction,requests\n";
 behavior_sectors<<"policy,kernel_id,kind,tags,dirty_lines,dirty_sectors,window_n,window_reads,window_writes,window_overwrites,age_capped32,valid,tag_replacement,sectors,age_sum,age_min,age_max\n";
 behavior_totals<<"policy,kernel_id,window,requests,writes,overwrites,read_allocations,resident_samples,proposal_samples,clean_samples,victim_samples,resident_dirty_sectors,residual\n";
std::ofstream admission_out(parent/"early-admission.csv"),admission_counts(parent/"early-admission-counts.csv");admission_out<<"policy,kernel_id,tag_replacement,selected_age_capped32,dirty_lines,tags,proposals\n";admission_counts<<"policy,kernel_id,proposals,no_victim_proposals,pressure_rejected,equivalence_checks\n";std::ofstream phase_out(parent/"phase-release.csv");phase_out<<"policy,kernel_id,closures,early_qualified,spacing_blocked,early_events,early_sectors,aged_events,aged_sectors\n";std::ofstream census_out(parent/"native-dirty-census.csv");census_out<<"policy,kernel_id,metric,a,b,c,d,e,events,sectors,covered_bytes\n";std::ofstream pressure(parent/"release-pressure.csv");pressure<<"policy,kernel_id,store_clean_sectors,read_clean_sectors,read_allocations,read_clean_events,dirty_seen,eligible_seen,partial_blocked,max_age\n";std::ofstream out(parent/"dirty-policies.csv"),snap(parent/"versions-at-kernel.csv");
 out<<"policy,kernel_id,requests,source_read_sectors,source_write_sectors,read_hits,read_misses,read_B,write_B,victim_lines,dirty_entry,dirty_created,redirty,dirty_evicted,cleaned_retained,dirty_exit,request_digest,tag_digest,read_state_equal,outcome_differences,victim_tag_differences,incomplete_victim_sectors,missing_victim_bytes,residual_B\n";
 snap<<"policy,boundary_kernel,producer,writes,superseded,emitted,resident,residual\n";
 int kernel=0;uint64_t requests=0;std::vector<uint64_t> entry(3),outcomes(3),victims(3),incomplete(3),missing(3);
 auto flush=[&](){if(!kernel)return;for(size_t i=0;i<policies.size();++i){auto&p=*policies[i];const auto&c=p.c;const auto after=p.dirty();const auto&ref=*policies[0];
  check_victim(entry[i]+c.created==after+c.evicted+c.cleaned&&c.created+c.redirty==c.writes&&c.emitted==32*(c.evicted+c.cleaned),"dirty/source conservation");
  check_victim(p.owners.size()==after&&c.reads+c.writes==requests&&c.read_hits+c.read_misses==c.reads&&c.fill==32*c.read_misses,"resident/request/read conservation");
  check_victim(ref.c.reads==c.reads&&ref.c.writes==c.writes&&ref.c.digest==c.digest,"candidate changed ordered input requests");p.assert_pressure_metadata();census[i].write(p,census_out);
  check_victim(p.behavior.accesses==c.reads+c.writes&&p.behavior.stores==c.writes&&p.behavior.overwrites==c.redirty&&p.behavior.clean_samples==c.cleaned&&p.behavior.victim_samples==c.evicted,"passive behavior census conservation");
  p.behavior.write(p.policy,kernel,behavior_requests,behavior_sectors,behavior_totals);
   check_victim(p.protection.requests==requests&&p.protection.reads==c.reads&&p.protection.writes==c.writes&&p.protection.created==c.created&&p.protection.overwrites==c.redirty&&p.protection.cleaned==c.cleaned&&p.protection.evicted==c.evicted,"protection independent core counts");
   p.protection.write(p.policy,kernel,protection_victims,protection_sectors,protection_totals,[&](uint64_t addr){return p.protection_info(addr);});
  if(p.disable_aged)check_victim(p.pc.aged_events==0&&p.pc.aged_sectors==0,"disabled aged emitted");
   check_victim(p.pc.early_sectors+p.pc.aged_sectors==(p.release_mode?c.read_clean_sectors:0),"phase release accounting");
  p.write_early_observation(early_emission);
  phase_out<<p.policy<<','<<kernel<<','<<p.pc.closures<<','<<p.pc.early_qualified<<','<<p.pc.spacing_blocked<<','<<p.pc.early_events<<','<<p.pc.early_sectors<<','<<p.pc.aged_events<<','<<p.pc.aged_sectors<<'\n';phase_out.flush();check_victim(bool(phase_out),"phase summary output");
  uint64_t proposal_sum=0,no_victim_sum=0;
  for(const auto &[key,n]:p.admission){proposal_sum+=n;if(!key[0])no_victim_sum+=n;admission_out<<p.policy<<','<<kernel<<','<<key[0]<<','<<key[1]<<','<<key[2]<<','<<key[3]<<','<<n<<'\n';}
  check_victim(proposal_sum==p.pc.proposals&&no_victim_sum==p.pc.no_victim_proposals&&p.pc.proposals==p.pc.pressure_rejected+p.pc.early_qualified,"early admission census conservation");
  admission_counts<<p.policy<<','<<kernel<<','<<p.pc.proposals<<','<<p.pc.no_victim_proposals<<','<<p.pc.pressure_rejected<<','<<p.pc.equivalence_checks<<'\n';admission_out.flush();admission_counts.flush();check_victim(bool(admission_out)&&bool(admission_counts),"admission outputs");
  uint64_t gate_total=0,gate_reject=0;
  for(const auto&[k,n]:p.behavior_admission){gate_total+=n;if(k[5]==2||k[5]==3)gate_reject+=n;behavior_admit_out<<p.policy<<','<<kernel;for(auto v:k)behavior_admit_out<<','<<v;behavior_admit_out<<','<<n<<'\n';}
  if(p.phase_variant==11)check_victim(gate_total==p.pc.proposals&&gate_reject==p.pc.pressure_rejected,"behavior gate independent population partition");
  behavior_admit_out.flush();check_victim(bool(behavior_admit_out),"behavior admission output");
  bool equal=true;for(size_t j=0;j<p.caches.size();++j)equal&=ref.caches[j].diagnostic_same_read_state(p.caches[j]);
  std::map<int,uint64_t> resident;for(const auto&x:p.owners)++resident[x.second];for(const auto&[id,v]:p.versions){check_victim(v.writes==v.superseded+v.emitted+resident[id],"version conservation");snap<<p.policy<<','<<kernel<<','<<id<<','<<v.writes<<','<<v.superseded<<','<<v.emitted<<','<<resident[id]<<",0\n";}
  pressure<<p.policy<<','<<kernel<<','<<c.store_clean_sectors<<','<<c.read_clean_sectors<<','<<c.pressure_reads<<','<<c.pressure_clean_events<<','<<c.pressure_dirty_seen<<','<<c.pressure_eligible_seen<<','<<c.pressure_partial_blocked<<','<<c.pressure_max_age<<'\n';
  check_victim(c.store_clean_sectors+c.read_clean_sectors==c.cleaned,"clean reason residual");
  out<<p.policy<<','<<kernel<<','<<requests<<','<<c.reads<<','<<c.writes<<','<<c.read_hits<<','<<c.read_misses<<','<<c.fill<<','<<c.emitted<<','<<c.victims<<','<<entry[i]<<','<<c.created<<','<<c.redirty<<','<<c.evicted<<','<<c.cleaned<<','<<after<<','<<c.digest<<','<<c.tag_digest<<','<<equal<<','<<outcomes[i]<<','<<victims[i]<<','<<incomplete[i]<<','<<missing[i]<<",0\n";
 }out.flush();snap.flush();pressure.flush();check_victim(bool(out)&&bool(snap)&&bool(pressure),"summary write failure");};
 int code=run_memgen(args,source,[&](const hyfiss_request_trace::L2AccessObservation&o){
  if(o.kernel_id!=kernel){flush();kernel=o.kernel_id;requests=0;for(size_t i=0;i<3;++i){policies[i]->epoch=kernel;policies[i]->c={};policies[i]->behavior.reset_counts();policies[i]->protection.reset_counts();policies[i]->behavior_admission.clear();policies[i]->pc={};policies[i]->early_observation={};policies[i]->admission.clear();policies[i]->joint.clear();census[i].reset_counts();entry[i]=policies[i]->dirty();outcomes[i]=victims[i]=incomplete[i]=missing[i]=0;}}
  check_victim((o.operation=='R'||o.operation=='W')&&o.addr%32==0&&o.outcome!=1,"unsupported L2 request domain");++requests;
  for(size_t i=0;i<3;++i){auto&p=*policies[i];check_victim(dram_partition_index(o.addr,p.opt)==o.partition&&l2_cache_index_addr(o.addr,p.opt)==o.index_addr,"same address mapping required");const auto census_before=p.c;const auto prior_owner=p.owners.find(o.addr);const int prior=prior_owner==p.owners.end()?0:prior_owner->second==p.epoch?1:2;p.access(o.addr,o.operation=='W',o.byte_mask);census[i].observe(p,o.addr,o.operation=='W',o.byte_mask,census_before,prior);const auto&a=p.last_access;if(!i)reference_victim(a,o);
   outcomes[i]+=result_victim(a)!=o.outcome;victims[i]+=a.evicted.present!=o.victim||(a.evicted.present&&a.evicted.addr!=o.victim_addr);incomplete[i]+=__builtin_popcount(a.evicted.incomplete_dirty_sectors);missing[i]+=a.evicted.missing_dirty_bytes;
  }
 });flush();need(code==0,"backend failure");
 std::ofstream ownership(parent/"ownership.csv"),versions(parent/"versions.csv");ownership<<"policy,producer,trigger,reason,sectors,bytes,queue,admission,completion\n";versions<<"policy,producer,writes,superseded,emitted,resident,residual\n";
 for(auto&p:policies)p->write_ownership(p->policy,ownership,versions);
 write_stats(args,source,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-began).count(),code);
 std::cout<<"PASS_SAME_L2_INPUT_DIFFERENT_VICTIM_READ_OUTCOMES_AND_VERSION_CONSERVATION\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
