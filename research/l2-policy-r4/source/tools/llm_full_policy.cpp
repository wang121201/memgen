// The state machine is the sealed phase11 core. This adapter adds true RMW
// accounting and explicit empty-kernel boundaries; no target traffic values.
#include "phase11_core.inc"

void frozen_policy_parameters(){
 selected_write_quota=4;selected_read_age=16;selected_phase_variant=11;
 selected_behavior_arm=0;selected_sector_age=false;selected_disable_aged=false;
}

struct FullPolicyReplay {
 std::vector<std::unique_ptr<RetentionReplay>> policies;
 std::ostream &out,&snap;
 int kernel=0;uint64_t requests=0;
 std::array<uint64_t,2> entry{},outcomes{},victims{},incomplete{},missing{};
 std::vector<int> boundaries;
 explicit FullPolicyReplay(const char*config,std::ostream&o,std::ostream&s):out(o),snap(s){
  frozen_policy_parameters();
  policies.push_back(candidate_victim(config,0));policies.push_back(candidate_victim(config,2));
  out<<"policy,kernel_id,requests,source_read_sectors,source_write_sectors,source_atomic_sectors,dirty_update_sectors,read_hits,read_misses,atomic_hits,atomic_misses,read_B,write_B,victim_lines,dirty_entry,dirty_created,redirty,dirty_evicted,cleaned_retained,dirty_exit,request_digest,tag_digest,read_state_equal,outcome_differences,victim_tag_differences,incomplete_victim_sectors,missing_victim_bytes,residual_B\n";
  snap<<"policy,boundary_kernel,producer,writes,superseded,emitted,resident,residual\n";
 }
 void flush(){
  if(!kernel)return;
  for(size_t i=0;i<policies.size();++i){
   auto&p=*policies[i];const auto&c=p.c;const auto after=p.dirty();const auto&ref=*policies[0];
   check_victim(c.atomics<=c.writes,"atomic population exceeds dirty updates");
   check_victim(entry[i]+c.created==after+c.evicted+c.cleaned&&c.created+c.redirty==c.writes&&c.emitted==32*(c.evicted+c.cleaned),"dirty/source conservation");
   check_victim(p.owners.size()==after&&c.reads+c.writes==requests&&c.read_hits+c.read_misses==c.reads&&c.atomic_hits+c.atomic_misses==c.atomics&&c.fill==32*(c.read_misses+c.atomic_misses),"resident/request/RMW conservation");
   check_victim(ref.c.reads==c.reads&&ref.c.writes==c.writes&&ref.c.atomics==c.atomics&&ref.c.digest==c.digest,"candidate changed ordered requests");
   p.assert_pressure_metadata();p.validate_early_observation();
   check_victim(p.behavior.accesses==requests&&p.behavior.stores==c.writes&&p.behavior.overwrites==c.redirty&&p.behavior.clean_samples==c.cleaned&&p.behavior.victim_samples==c.evicted,"behavior update conservation");
   check_victim(c.store_clean_sectors+c.read_clean_sectors==c.cleaned,"clean reason residual");
   check_victim(p.pc.early_sectors+p.pc.aged_sectors==(p.release_mode?c.read_clean_sectors:0),"phase release accounting");
   bool equal=true;for(size_t j=0;j<p.caches.size();++j)equal&=ref.caches[j].diagnostic_same_read_state(p.caches[j]);
   std::map<int,uint64_t> resident;for(const auto&x:p.owners)++resident[x.second];
   for(const auto&[id,v]:p.versions){
    check_victim(v.writes==v.superseded+v.emitted+resident[id],"version conservation");
    snap<<p.policy<<','<<kernel<<','<<id<<','<<v.writes<<','<<v.superseded<<','<<v.emitted<<','<<resident[id]<<",0\n";
   }
   out<<p.policy<<','<<kernel<<','<<requests<<','<<c.reads<<','<<(c.writes-c.atomics)<<','<<c.atomics<<','<<c.writes<<','<<c.read_hits<<','<<c.read_misses<<','<<c.atomic_hits<<','<<c.atomic_misses<<','<<c.fill<<','<<c.emitted<<','<<c.victims<<','<<entry[i]<<','<<c.created<<','<<c.redirty<<','<<c.evicted<<','<<c.cleaned<<','<<after<<','<<c.digest<<','<<c.tag_digest<<','<<equal<<','<<outcomes[i]<<','<<victims[i]<<','<<incomplete[i]<<','<<missing[i]<<",0\n";
  }
  out.flush();snap.flush();check_victim(bool(out)&&bool(snap),"summary output failed");
 }
 void begin(int id){
  check_victim(id==kernel+1,"kernel population must be dense and ordered");flush();kernel=id;boundaries.push_back(id);requests=0;
  for(size_t i=0;i<policies.size();++i){
   auto&p=*policies[i];p.epoch=id;p.c={};p.behavior.reset_counts();p.protection.reset_counts();
   p.behavior_admission.clear();p.pc={};p.early_observation={};p.admission.clear();p.joint.clear();
   entry[i]=p.dirty();outcomes[i]=victims[i]=incomplete[i]=missing[i]=0;
  }
 }
 void observe(const hyfiss_request_trace::L2AccessObservation&o){
  check_victim(o.kernel_id==kernel,"request lacks explicit kernel boundary");
  check_victim((o.operation=='R'||o.operation=='W'||o.operation=='A')&&o.addr%32==0&&o.outcome!=1,"unsupported L2 request domain");++requests;
  for(size_t i=0;i<policies.size();++i){
   auto&p=*policies[i];check_victim(dram_partition_index(o.addr,p.opt)==o.partition&&l2_cache_index_addr(o.addr,p.opt)==o.index_addr,"same mapping required");
   p.access(o.addr,o.operation!='R',o.byte_mask,o.operation=='A');const auto&a=p.last_access;if(!i)reference_victim(a,o);
   outcomes[i]+=result_victim(a)!=o.outcome;victims[i]+=a.evicted.present!=o.victim||(a.evicted.present&&a.evicted.addr!=o.victim_addr);
   incomplete[i]+=__builtin_popcount(a.evicted.incomplete_dirty_sectors);missing[i]+=a.evicted.missing_dirty_bytes;
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
 std::ostringstream out,snap;FullPolicyReplay replay(config,out,snap);replay.begin(1);
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

int main(int argc,char**argv){try{
 if(argc==3&&std::string(argv[1])=="--fixture"){
  fixture_victim(argv[2]);census_fixture(argv[2]);phase_fixture(argv[2]);behavior_fixture(argv[2]);read_pressure_fixture(argv[2]);sector_age_fixture(argv[2]);aged_ablation_fixture(argv[2]);dirty_protection_fixture(argv[2]);early_emission_fixture(argv[2]);full_domain_fixture(argv[2]);return 0;
 }
 using namespace hbserve_profile_stream;auto began=std::chrono::steady_clock::now();const Arguments args=parse_arguments(argc,argv);
 need(args.mode=="memgen"&&!fs::exists(args.output_dir)&&!fs::exists(args.stats),"fresh memgen outputs required");WorkloadSource source(args.profile_index,args.app_config,args.issue_config,true);
 const auto parent=args.output_dir.parent_path();std::ofstream out(parent/"dirty-policies.csv"),snap(parent/"versions-at-kernel.csv");FullPolicyReplay replay(args.hw_config.c_str(),out,snap);
 const int code=run_memgen(args,source,[&](const hyfiss_request_trace::L2AccessObservation&o){replay.observe(o);},{},[&](int id){replay.begin(id);});replay.flush();need(code==0,"backend failure");
 std::ofstream ownership(parent/"ownership.csv"),versions(parent/"versions.csv");ownership<<"policy,producer,trigger,reason,sectors,bytes,queue,admission,completion\n";versions<<"policy,producer,writes,superseded,emitted,resident,residual\n";
 for(auto&p:replay.policies)p->write_ownership(p->policy,ownership,versions);
 write_stats(args,source,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-began).count(),code);
 std::cout<<"PASS_FULL_POLICY_CONTINUOUS_STREAM_FUNCTIONAL_ACCOUNTING\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
