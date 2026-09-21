// The state machine is the sealed phase11 core. This adapter adds true RMW
// accounting and explicit empty-kernel boundaries; no target traffic values.
#include "phase11_core.inc"

void frozen_policy_parameters(){
 selected_write_quota=4;selected_read_age=16;selected_phase_variant=11;
 selected_behavior_arm=0;selected_sector_age=false;selected_disable_aged=false;
}

struct PolicySpec {
 std::string id,family;
 unsigned write_budget=16,read_budget=16,release_age=0,phase_variant=0,behavior_arm=0;
 bool clean_first=false,release_mode=false,sector_age=false,disable_aged=false,native_age=false,one_sector=false;
 RetentionReplay::PendingMode pending_mode=RetentionReplay::PendingMode::None;
};

std::vector<PolicySpec> policy_registry(){
 std::vector<PolicySpec> r;
 r.push_back({"disabled","ordinary_lru"});
 for(unsigned q:{7u,8u})r.push_back({"fixed_q"+std::to_string(q),"fixed_quota",q,16});
 for(unsigned q:{1u,2u,3u,4u,5u,6u,7u,8u,16u}){
  PolicySpec s{"clean_q"+std::to_string(q),"clean_first_quota",q,16};s.clean_first=true;r.push_back(s);
 }
 {PolicySpec s{"store8_read7","legacy_asymmetric_quota",8,7};r.push_back(s);}
 {PolicySpec s{"store8to7_read7","legacy_hysteresis_quota",8,7};r.push_back(s);}
 for(unsigned h:{1u,4u,8u,16u}){
  PolicySpec s{"clean_q2_h"+std::to_string(h),"clean_first_read_pressure",2,16,h};
  s.clean_first=true;s.release_mode=true;r.push_back(s);
 }
 {PolicySpec s{"read_allocation_age8_clean_retained_v1","read_allocation_age",16,16,8};s.native_age=true;r.push_back(s);}
 {PolicySpec s{"read_allocation_age8_one_sector_v1","read_allocation_age_sector",16,16,8};s.native_age=true;s.one_sector=true;r.push_back(s);}
 {PolicySpec s{"select_at8_emit_next_distinct_line_v1","pending_selection"};s.pending_mode=RetentionReplay::PendingMode::SelectAt8;r.push_back(s);}
 {PolicySpec s{"overflow_armed_pending8_next_distinct_line_v1","pending_overflow"};s.pending_mode=RetentionReplay::PendingMode::OverflowArmed8;r.push_back(s);}
 {PolicySpec s{"selected8_cancel_on_victim_store_v1","pending_cancel"};s.pending_mode=RetentionReplay::PendingMode::CancelOnSelectedStore;r.push_back(s);}
 for(unsigned v=0;v<=15;++v){
  PolicySpec s{"phase"+std::to_string(v),"phase_release",4,16,16,v};
  s.clean_first=true;s.release_mode=true;r.push_back(s);
 }
 {PolicySpec s{"phase11_lru","phase_release_lru_victim",4,16,16,11};s.release_mode=true;r.push_back(s);}
 for(unsigned arm:{1u,2u,3u}){
  PolicySpec s{"phase11_arm"+std::to_string(arm),"phase_release_behavior_arm",4,16,16,11,arm};
  s.clean_first=true;s.release_mode=true;r.push_back(s);
 }
 {PolicySpec s{"phase11_sector_age","phase_release_sector_age",4,16,16,11};s.clean_first=true;s.release_mode=true;s.sector_age=true;r.push_back(s);}
 {PolicySpec s{"phase11_disable_aged","phase_release_early_only",4,16,16,11};s.clean_first=true;s.release_mode=true;s.disable_aged=true;r.push_back(s);}
 return r;
}

std::vector<std::string> split_policy_names(const std::string&text){
 std::vector<std::string> result;std::string item;
 for(char c:text){if(c==','){if(!item.empty())result.push_back(item);item.clear();}else item+=c;}
 if(!item.empty())result.push_back(item);return result;
}

std::vector<PolicySpec> selected_policy_specs(){
 const auto registry=policy_registry();std::map<std::string,PolicySpec> by_id;
 for(const auto&s:registry)check_victim(by_id.emplace(s.id,s).second,"duplicate policy id");
 const char*raw=std::getenv("FULL_POLICY_MATRIX");
 std::vector<std::string> names=raw&&*raw?split_policy_names(raw):std::vector<std::string>{"disabled","phase11","clean_q16"};
 if(names.size()==1&&names[0]=="all"){names.clear();for(const auto&s:registry)names.push_back(s.id);}
 if(std::find(names.begin(),names.end(),"disabled")==names.end())names.insert(names.begin(),"disabled");
 std::set<std::string> seen;std::vector<PolicySpec> result;
 for(const auto&name:names){auto it=by_id.find(name);check_victim(it!=by_id.end(),("unknown policy: "+name).c_str());if(seen.insert(name).second)result.push_back(it->second);}
 check_victim(!result.empty()&&result.front().id=="disabled","disabled reference must be first");return result;
}

std::unique_ptr<RetentionReplay> make_policy(const char*config,const PolicySpec&s){
 if(s.id=="disabled")return std::make_unique<RetentionReplay>(config,"disabled",false);
 if(s.pending_mode!=RetentionReplay::PendingMode::None){
  auto p=std::make_unique<RetentionReplay>(config,"fixed_budget8_clean_retained_after_store_v1",false);
  p->policy=s.id;p->write_budget=p->read_budget=16;p->pending_mode=s.pending_mode;return p;
 }
 if(s.native_age){
  auto p=std::make_unique<RetentionReplay>(config,s.one_sector?"read_allocation_age8_one_sector_v1":"read_allocation_age8_clean_retained_v1",false);
  p->release_age=s.release_age;return p;
 }
 auto p=std::make_unique<RetentionReplay>(config,"fixed_budget8_clean_retained_after_store_v1",false);
 p->policy=s.id;p->write_budget=s.write_budget;p->read_budget=s.read_budget;
 p->release_mode=s.release_mode;p->release_age=s.release_age;p->phase_variant=s.phase_variant;
 p->behavior_arm=s.behavior_arm;p->sector_age=s.sector_age;p->disable_aged=s.disable_aged;
 for(auto&c:p->caches)c.diagnostic_read_clean_first=s.clean_first;
 return p;
}

void write_policy_manifest(std::ostream&out,const std::vector<PolicySpec>&specs){
 out<<"policy,family,causal_online,write_budget,read_budget,release_mode,release_age,phase_variant,behavior_arm,clean_first,sector_age,disable_aged,native_age,one_sector,pending_mode,dirty_update_includes_atomic_rmw\n";
 for(const auto&s:specs)out<<s.id<<','<<s.family<<",1,"<<s.write_budget<<','<<s.read_budget<<','<<s.release_mode<<','<<s.release_age<<','<<s.phase_variant<<','<<s.behavior_arm<<','<<s.clean_first<<','<<s.sector_age<<','<<s.disable_aged<<','<<s.native_age<<','<<s.one_sector<<','<<unsigned(s.pending_mode)<<",1\n";
 out.flush();check_victim(bool(out),"policy manifest output failed");
}

struct PolicyCounters {uint64_t entry=0,outcomes=0,victims=0,incomplete=0,missing=0;};

struct FullPolicyReplay {
 std::vector<PolicySpec> specs;
 std::vector<std::unique_ptr<RetentionReplay>> policies;
 std::ostream &out;
 int kernel=0;uint64_t requests=0;
 std::vector<PolicyCounters> counters;
 std::vector<int> boundaries;
 explicit FullPolicyReplay(const char*config,std::ostream&o,std::vector<PolicySpec>s):specs(std::move(s)),out(o){
  for(const auto&spec:specs)policies.push_back(make_policy(config,spec));
  counters.resize(policies.size());
  out<<"policy,kernel_id,requests,source_read_sectors,source_write_sectors,source_atomic_sectors,dirty_update_sectors,read_hits,read_misses,atomic_hits,atomic_misses,read_B,write_B,victim_lines,dirty_entry,dirty_created,redirty,dirty_evicted,cleaned_retained,dirty_exit,request_digest,tag_digest,read_state_equal,outcome_differences,victim_tag_differences,incomplete_victim_sectors,missing_victim_bytes,residual_B\n";
 }
 void flush(){
  if(!kernel)return;
  for(size_t i=0;i<policies.size();++i){
   auto&p=*policies[i];const auto&c=p.c;const auto after=p.owners.size();const auto&ref=*policies[0];const auto&x=counters[i];
   check_victim(c.atomics<=c.writes,"atomic population exceeds dirty updates");
   check_victim(x.entry+c.created==after+c.evicted+c.cleaned&&c.created+c.redirty==c.writes&&c.emitted==32*(c.evicted+c.cleaned),"dirty/source conservation");
   check_victim(p.owners.size()==after&&c.reads+c.writes==requests&&c.read_hits+c.read_misses==c.reads&&c.atomic_hits+c.atomic_misses==c.atomics&&c.fill==32*(c.read_misses+c.atomic_misses),"resident/request/RMW conservation");
   check_victim(ref.c.reads==c.reads&&ref.c.writes==c.writes&&ref.c.atomics==c.atomics&&ref.c.digest==c.digest,"candidate changed ordered requests");
   p.validate_early_observation();
   check_victim(p.behavior.accesses==requests&&p.behavior.stores==c.writes&&p.behavior.overwrites==c.redirty&&p.behavior.clean_samples==c.cleaned&&p.behavior.victim_samples==c.evicted,"behavior update conservation");
   check_victim(c.store_clean_sectors+c.read_clean_sectors+c.pending_selected_clean_sectors+c.pending_overflow_clean_sectors==c.cleaned,"clean reason residual");
   check_victim(p.pc.early_sectors+p.pc.aged_sectors==(p.release_mode?c.read_clean_sectors:0),"phase release accounting");
   if(kernel%256==0){check_victim(p.dirty()==after,"periodic owner/cache dirty mismatch");p.assert_pressure_metadata();}
   out<<p.policy<<','<<kernel<<','<<requests<<','<<c.reads<<','<<(c.writes-c.atomics)<<','<<c.atomics<<','<<c.writes<<','<<c.read_hits<<','<<c.read_misses<<','<<c.atomic_hits<<','<<c.atomic_misses<<','<<c.fill<<','<<c.emitted<<','<<c.victims<<','<<x.entry<<','<<c.created<<','<<c.redirty<<','<<c.evicted<<','<<c.cleaned<<','<<after<<','<<c.digest<<','<<c.tag_digest<<",NOT_CHECKED,"<<x.outcomes<<','<<x.victims<<','<<x.incomplete<<','<<x.missing<<",0\n";
  }
  if(kernel%64==0)out.flush();check_victim(bool(out),"summary output failed");
 }
 void begin(int id){
  check_victim(id==kernel+1,"kernel population must be dense and ordered");flush();kernel=id;boundaries.push_back(id);requests=0;
  for(size_t i=0;i<policies.size();++i){
   auto&p=*policies[i];p.epoch=id;p.c={};p.behavior.reset_counts();p.protection.reset_counts();
   p.behavior_admission.clear();p.pc={};p.early_observation={};p.admission.clear();p.joint.clear();
   counters[i]={p.owners.size(),0,0,0,0};
  }
 }
 void observe(const hyfiss_request_trace::L2AccessObservation&o){
  check_victim(o.kernel_id==kernel,"request lacks explicit kernel boundary");
  check_victim((o.operation=='R'||o.operation=='W'||o.operation=='A')&&o.addr%32==0&&o.outcome!=1,"unsupported L2 request domain");++requests;
  for(size_t i=0;i<policies.size();++i){
   auto&p=*policies[i];check_victim(dram_partition_index(o.addr,p.opt)==o.partition&&l2_cache_index_addr(o.addr,p.opt)==o.index_addr,"same mapping required");
   p.access(o.addr,o.operation!='R',o.byte_mask,o.operation=='A');const auto&a=p.last_access;if(!i)reference_victim(a,o);
   counters[i].outcomes+=result_victim(a)!=o.outcome;counters[i].victims+=a.evicted.present!=o.victim||(a.evicted.present&&a.evicted.addr!=o.victim_addr);
   counters[i].incomplete+=__builtin_popcount(a.evicted.incomplete_dirty_sectors);counters[i].missing+=a.evicted.missing_dirty_bytes;
  }
 }
 void finalize(std::ostream&ownership,std::ostream&versions){
  out.flush();check_victim(bool(out),"summary output failed");
  for(auto&p:policies){
   check_victim(p->dirty()==p->owners.size(),"final owner/cache dirty mismatch");p->assert_pressure_metadata();p->validate_early_observation();
   p->write_ownership(p->policy,ownership,versions);
  }
 }
};

void full_domain_fixture(const char*config){
 frozen_policy_parameters();
 for(unsigned policy:{0u,2u}){
  auto p=candidate_victim(config,policy);p->epoch=1;p->access(0,true,15,true);
  check_victim(p->c.atomics==1&&p->c.writes==1&&p->c.reads==0&&p->c.atomic_misses==1&&p->c.fill==32&&p->dirty()==1&&p->c.emitted==0,"cold RMW must fill and retain dirty");
  p->epoch=3;p->access(0,true,15,true);
  check_victim(p->c.atomics==2&&p->c.atomic_hits==1&&p->c.fill==32&&p->versions[1].superseded==1&&p->owners.at(0)==3,"warm RMW version supersession");
  p->assert_pressure_metadata();
  auto partial=candidate_victim(config,policy);partial->epoch=1;partial->access(0,true,15);
  partial->epoch=3;partial->access(0,true,15,true);
  check_victim(partial->last_access.result==CacheResult::SectorMiss&&partial->c.fill==32&&partial->c.atomic_misses==1&&partial->last_access.valid_after==1&&partial->c.redirty==1,"RMW preserves unwritten partial bytes");
  partial->assert_pressure_metadata();
 }
 std::ostringstream out;FullPolicyReplay replay(config,out,selected_policy_specs());replay.begin(1);
 auto&ref=*replay.policies[0];std::vector<SectorLruCache> caches;
 for(unsigned p=0;p<ref.opt.num_partitions;++p)caches.emplace_back(ref.opt.l2_size_bytes/ref.opt.num_partitions,128,16,ref.opt.l2_set_index);
 auto access=[&](int kernel,uint64_t addr,char operation,uint32_t mask){
  const unsigned part=dram_partition_index(addr,ref.opt);const auto index=l2_cache_index_addr(addr,ref.opt);
  auto a=caches[part].access(addr,32,32,0,0,cache_operation(operation),operation!='R',index,true,false,mask);
  hyfiss_request_trace::L2AccessObservation o;o.kernel_id=kernel;o.partition=part;o.addr=addr;o.index_addr=index;o.operation=operation;o.byte_mask=mask;
  o.outcome=result_victim(a);o.valid_before=a.valid_before;o.valid_after=a.valid_after;o.dirty_before=a.dirty_before;o.dirty_after=a.dirty_after;
  o.victim=a.evicted.present;o.victim_addr=a.evicted.addr;o.victim_valid=a.evicted.valid_sectors;o.victim_dirty=a.evicted.dirty_sectors;replay.observe(o);
 };
 access(1,0,'W',15);replay.begin(2);replay.begin(3);access(3,0,'A',15);replay.begin(4);replay.flush();
 check_victim(replay.boundaries==std::vector<int>({1,2,3,4}),"empty middle/final kernel missing");
 for(auto&p:replay.policies){check_victim(p->owners.at(0)==3&&p->versions[1].superseded==1&&p->c.emitted==0,"empty boundary changed dirty state");std::ostringstream a,b;p->write_ownership(p->policy,a,b);}
 std::cout<<"PASS_FULL_DOMAIN_ATOMIC_MISS_HIT_PARTIAL_SUPERSESSION_AND_EMPTY_BOUNDARIES\n";
}


void retention_control_fixture(const char*config){
 frozen_policy_parameters();selected_write_quota=16;auto p=candidate_victim(config,1);
 check_victim(p->write_budget==16&&p->read_budget==16&&!p->release_mode,"retention parameters");
 std::vector<uint64_t> lines;
 for(uint64_t x=0;lines.size()<40&&x<(128ull<<20);x+=128)
  if(dram_partition_index(x,p->opt)==0&&p->caches[0].diagnostic_set_id(l2_cache_index_addr(x,p->opt))==0)lines.push_back(x);
 check_victim(lines.size()==40,"retention fixture population");
 p->epoch=1;p->access(lines[0],true);
 for(unsigned j=1;j<40;++j)p->access(lines[j],false);
 check_victim(p->dirty()==1&&p->c.emitted==0&&p->owners.at(lines[0])==1,"read pressure prematurely releases dirty");
 p->epoch=2;p->access(lines[0],true);
 check_victim(p->versions[1].superseded==1&&p->owners.at(lines[0])==2&&p->c.cleaned==0,"cross-boundary overwrite not retained");
 auto full=candidate_victim(config,1);full->epoch=1;
 for(unsigned j=0;j<16;++j)full->access(lines[j],true);
 check_victim(full->c.emitted==0&&full->dirty()==16,"unrequested quota cleaning");
 full->epoch=2;full->access(lines[16],false);
 check_victim(full->c.evicted==1&&full->c.emitted==32&&full->c.cleaned==0&&full->dirty()==15,"all-dirty fallback loses versions");
 for(auto* q:{p.get(),full.get()}){q->assert_pressure_metadata();std::ostringstream a,b;q->write_ownership(q->policy,a,b);}
 std::cout<<"PASS_RETENTION_READ_PRESSURE_OVERWRITE_AND_ALL_DIRTY_FALLBACK\n";
}

void pending_policy_fixture(const char*config){
 using Mode=RetentionReplay::PendingMode;
 auto model=[&](Mode mode,const char*name){
  auto p=std::make_unique<RetentionReplay>(config,"fixed_budget8_clean_retained_after_store_v1",false);
  p->policy=name;p->write_budget=p->read_budget=16;p->pending_mode=mode;return p;
 };
 auto probe=model(Mode::SelectAt8,"pending-address-probe");
 std::vector<uint64_t> lines;uint64_t other=UINT64_MAX;
 for(uint64_t x=0;lines.size()<40&&x<(256ull<<20);x+=128){
  const auto part=dram_partition_index(x,probe->opt);
  const auto set=probe->caches[part].diagnostic_set_id(l2_cache_index_addr(x,probe->opt));
  if(part==0&&set==0)lines.push_back(x);else if(part==0&&other==UINT64_MAX)other=x;
 }
 check_victim(lines.size()==40&&other!=UINT64_MAX,"pending fixture set population");

 // Selection is state preserving, crosses kernel boundaries, ignores other
 // sets, and admits only at the next distinct line in the selected set.
 auto selected=model(Mode::SelectAt8,"select_at8_emit_next_distinct_line_v1");
 auto control=std::make_unique<RetentionReplay>(config,"disabled",false);
 selected->epoch=control->epoch=1;
 for(unsigned i=0;i<8;++i)for(unsigned sec=0;sec<4;++sec){selected->access(lines[i]+32*sec,true);control->access(lines[i]+32*sec,true);}
 check_victim(selected->pending.size()==1&&selected->dirty()==32&&selected->c.emitted==0,"eight lines select without emission");
 const auto pending_address=selected->pending.begin()->second.address;
 selected->epoch=control->epoch=2;selected->access(other,false);control->access(other,false);
 check_victim(selected->pending.size()==1&&selected->pending.begin()->second.address==pending_address&&selected->c.emitted==0,"cross-set or boundary admitted pending selection");
 selected->access(lines[7],true);control->access(lines[7],true);
 check_victim(selected->pending.size()==1&&selected->c.emitted==0,"same trigger line admitted pending selection");
 selected->access(lines[6],true);control->access(lines[6],true);
 check_victim(selected->pending.empty()&&selected->c.emitted==128&&selected->dirty()==28,"next distinct line did not admit exact selected line");
 for(size_t i=0;i<selected->caches.size();++i)check_victim(selected->caches[i].diagnostic_same_read_state(control->caches[i]),"pending policy changed tag/valid/LRU state");
 selected->assert_pressure_metadata();std::ostringstream se,sv;selected->write_ownership(selected->policy,se,sv);

 // Partial dirty lines count toward the threshold but cannot themselves be
 // selected. If overflow consists only of incomplete lines, fail closed.
 auto partial=model(Mode::SelectAt8,"select_at8_partial");partial->epoch=1;
 partial->access(lines[0],true,15);for(unsigned i=1;i<8;++i)partial->access(lines[i],true);
 check_victim(partial->pending.size()==1&&partial->pending.begin()->second.address!=lines[0],"partial dirty line selected");
 partial->assert_pressure_metadata();
 bool rejected=false;try{
  auto blocked=model(Mode::OverflowArmed8,"overflow_partial_blocked");blocked->epoch=1;
  for(unsigned i=0;i<9;++i)blocked->access(lines[i],true,15);
 }catch(const std::exception&){rejected=true;}
 check_victim(rejected,"partial overflow must fail closed");

 auto overflow=model(Mode::OverflowArmed8,"overflow_armed_pending8_next_distinct_line_v1");overflow->epoch=1;
 for(unsigned i=0;i<8;++i)for(unsigned sec=0;sec<4;++sec)overflow->access(lines[i]+32*sec,true);
 check_victim(overflow->pending.empty()&&overflow->c.emitted==0,"eight lines armed prematurely");
 for(unsigned sec=0;sec<4;++sec)overflow->access(lines[8]+32*sec,true);
 check_victim(overflow->pending.size()==1&&overflow->c.pending_overflow_clean_sectors==4&&overflow->c.emitted==128,"ninth line overflow/selection mismatch");
 overflow->epoch=2;overflow->access(lines[6],true);
 check_victim(overflow->pending.empty()&&overflow->c.pending_selected_clean_sectors==4&&overflow->c.emitted==256,"armed pending admission mismatch");
 for(unsigned i=9;i<40;++i)overflow->access(lines[i],false);
 check_victim(overflow->dirty()==0&&overflow->pending.empty()&&overflow->pressure_armed.empty(),"ordinary replacement did not reset overflow arm");
 overflow->assert_pressure_metadata();std::ostringstream oe,ov;overflow->write_ownership(overflow->policy,oe,ov);

 auto cancel=model(Mode::CancelOnSelectedStore,"selected8_cancel_on_victim_store_v1");cancel->epoch=1;
 for(unsigned i=0;i<8;++i)for(unsigned sec=0;sec<4;++sec)cancel->access(lines[i]+32*sec,true);
 check_victim(cancel->pending.size()==1&&cancel->pending.begin()->second.address==lines[0],"cancel fixture selected wrong victim");
 cancel->epoch=2;cancel->access(lines[0],true,UINT32_MAX,true);
 check_victim(cancel->pending_cancelled_count==1&&cancel->c.atomics==1&&cancel->c.emitted==0,"atomic selected-store cancellation mismatch");
 cancel->access(lines[8],true);
 check_victim(cancel->pending_admitted_count==1&&cancel->c.emitted==128,"cancel reselection admission mismatch");
 cancel->assert_pressure_metadata();std::ostringstream ce,cv;cancel->write_ownership(cancel->policy,ce,cv);
 std::cout<<"PASS_PENDING_SELECT_OVERFLOW_CANCEL_BOUNDARY_PARTIAL_ATOMIC_AND_VERSION_CONSERVATION\n";
}

int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--list-policies"){
  write_policy_manifest(std::cout,policy_registry());return 0;
 }
 if(argc==3&&std::string(argv[1])=="--fixture"){
  fixture_victim(argv[2]);census_fixture(argv[2]);phase_fixture(argv[2]);behavior_fixture(argv[2]);read_pressure_fixture(argv[2]);sector_age_fixture(argv[2]);aged_ablation_fixture(argv[2]);dirty_protection_fixture(argv[2]);early_emission_fixture(argv[2]);full_domain_fixture(argv[2]);retention_control_fixture(argv[2]);pending_policy_fixture(argv[2]);return 0;
 }
 using namespace hbserve_profile_stream;auto began=std::chrono::steady_clock::now();const Arguments args=parse_arguments(argc,argv);
 need(args.mode=="memgen"&&!fs::exists(args.output_dir)&&!fs::exists(args.stats),"fresh memgen outputs required");WorkloadSource source(args.profile_index,args.app_config,args.issue_config,true);
 const auto parent=args.output_dir.parent_path();const auto specs=selected_policy_specs();
 std::ofstream manifest(parent/"policy-manifest.csv");write_policy_manifest(manifest,specs);
 std::ofstream out(parent/"dirty-policies.csv");FullPolicyReplay replay(args.hw_config.c_str(),out,specs);
 const int code=run_memgen(args,source,[&](const hyfiss_request_trace::L2AccessObservation&o){replay.observe(o);},{},[&](int id){replay.begin(id);});replay.flush();need(code==0,"backend failure");
 std::ofstream ownership(parent/"ownership.csv"),versions(parent/"versions.csv");ownership<<"policy,producer,trigger,reason,sectors,bytes,queue,admission,completion\n";versions<<"policy,producer,writes,superseded,emitted,resident,residual\n";
 replay.finalize(ownership,versions);
 write_stats(args,source,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-began).count(),code);
 std::cout<<"PASS_FULL_POLICY_CONTINUOUS_STREAM_FUNCTIONAL_ACCOUNTING\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
