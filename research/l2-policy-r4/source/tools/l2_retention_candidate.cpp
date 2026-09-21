// Diagnostic mechanism test. No hardware topology or service-time claim.
// Streams ordinary self-buffer requests directly into the existing cache core.
#define HYFISS_REQUEST_TRACE_NO_MAIN
// The frozen core is already included by the live HBServe wrapper.

struct RetentionCounts {
  uint64_t reads=0,writes=0,read_hits=0,read_misses=0;
  uint64_t created=0,redirty=0,victims=0,evicted=0,cleaned=0;
  uint64_t pressure_reads=0,pressure_dirty_seen=0,pressure_eligible_seen=0,pressure_partial_blocked=0,pressure_max_age=0,pressure_clean_events=0,pressure_dirty_under8_events=0,pressure_dirty_read_hits=0,pressure_selected_revisited=0;
  std::array<uint64_t,5> pressure_selected_sectors{};
  uint64_t fill=0,emitted=0,digest=1469598103934665603ull,tag_digest=1469598103934665603ull;
};
struct RetentionReplay {
  Options opt;
  std::string policy;
  unsigned write_budget,read_budget;
  bool warp_major;
  std::vector<SectorLruCache> caches;
  RetentionCounts c;
  CacheAccess last_access;
  std::map<std::pair<unsigned,uint64_t>,uint64_t> set_read_allocations;
  std::unordered_map<uint64_t,uint64_t> written_at_read_count;
  std::unordered_set<uint64_t> dirty_read_since_store;
  std::array<uint64_t,20> partition_dirty{};
  using JointKey=std::tuple<unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned>;
  struct JointCount {uint64_t count=0,partition_sum=0,partition_min=UINT64_MAX,partition_max=0,selected_age_sum=0,selected_age_min=UINT64_MAX,selected_age_max=0;};
  std::map<JointKey,JointCount> joint;
  void observe_joint(const SectorLruCache::PressureSelection&s,const CacheAccess&a,unsigned partition){
    const auto pd=partition_dirty.at(partition);const unsigned n=__builtin_popcount(s.payload.dirty_sectors);
    JointKey key={s.tags,s.dirty,s.sectors,s.partial,s.eligible,s.rank,std::min<uint64_t>(s.selected_age,32),n,pd/4096,a.evicted.present,__builtin_popcount(a.evicted.dirty_sectors)};
    auto &j=joint[key];++j.count;j.partition_sum+=pd;j.partition_min=std::min(j.partition_min,pd);j.partition_max=std::max(j.partition_max,pd);
    if(s.payload.dirty){j.selected_age_sum+=s.selected_age;j.selected_age_min=std::min(j.selected_age_min,s.selected_age);j.selected_age_max=std::max(j.selected_age_max,s.selected_age);}
    if(joint.size()>262144)throw std::runtime_error("joint histogram bound");
  }
  void write_joint(std::ostream&out)const{
    uint64_t all=0,selected=0,eligible=0;
    for(const auto &[key,j]:joint){
      const auto [tags,dl,ds,partial,el,rank,age,n,pbin,victim,vd]=key;
      all+=j.count;selected+=j.count*(rank>0);eligible+=j.count*el;
      out<<policy<<','<<epoch<<','<<tags<<','<<dl<<','<<ds<<','<<partial<<','<<el<<','<<rank<<','<<age<<','<<n<<','<<pbin<<','<<victim<<','<<vd<<','<<j.count<<','<<j.partition_sum<<','<<j.partition_min<<','<<j.partition_max<<','<<j.selected_age_sum<<','<<(rank?j.selected_age_min:0)<<','<<j.selected_age_max<<'\n';
    }
    if(all!=c.pressure_reads || selected!=c.pressure_clean_events || eligible!=c.pressure_eligible_seen)throw std::runtime_error("joint population conservation");
    out.flush();if(!out)throw std::runtime_error("joint write");
  }
  bool pressure_policy()const{return policy=="read_allocation_age8_clean_retained_v1" || policy=="read_allocation_age8_one_sector_v1";}
  void assert_pressure_metadata()const{
    for(size_t p=0;p<caches.size();++p)if(partition_dirty.at(p)!=caches[p].dirty_sector_count())throw std::runtime_error("partition dirty counter mismatch");
    if(!pressure_policy())return;
    std::unordered_set<uint64_t> lines;
    for(const auto &o:owners)lines.insert(o.first/128);
    if(lines.size()!=written_at_read_count.size())throw std::runtime_error("pressure metadata population");
    for(uint64_t line:lines){
      const auto p=dram_partition_index(line*128,opt);const auto s=caches[p].diagnostic_set_id(l2_cache_index_addr(line*128,opt));
      auto w=written_at_read_count.find(line);auto n=set_read_allocations.find({p,s});
      if(w==written_at_read_count.end() || n==set_read_allocations.end() || w->second>n->second)throw std::runtime_error("pressure metadata age");
    }
    for(auto line:dirty_read_since_store)if(!lines.count(line))throw std::runtime_error("stale dirty read metadata");
  }

  // Aggregate ownership of the latest dirty version; not an arrival trace.
  int epoch=-1;
  struct Versions{uint64_t writes=0,superseded=0,emitted=0;};
  std::map<int,Versions> versions;
  std::unordered_map<uint64_t,int> owners;
  std::map<std::tuple<int,int,std::string>,uint64_t> lineage;
  void emit(const EvictedLine &payload,const char *reason){
    if(std::string(reason)=="clean_retained" && payload.dirty_sectors){
      auto &pd=partition_dirty.at(dram_partition_index(payload.addr,opt));const auto n=__builtin_popcount(payload.dirty_sectors);
      if(pd<n)throw std::runtime_error("partition counter clean underflow");pd-=n;
    }
    for_each_writeback_span(payload,32,128,[&](const WritebackSpan&s){
      if(s.addr%32||s.size%32)throw std::runtime_error("unaligned dirty emission");
      for(uint64_t addr=s.addr;addr<s.addr+s.size;addr+=32){
        auto it=owners.find(addr);if(it==owners.end())throw std::runtime_error("writeback has no dirty owner");
        ++versions[it->second].emitted;++lineage[{it->second,epoch,reason}];owners.erase(it);
      }
      c.emitted+=s.size;
    });
  }
  void write_ownership(const std::string&id,std::ostream&events,std::ostream&totals){
    std::map<int,uint64_t> resident;for(const auto &x:owners)++resident[x.second];
    if(owners.size()!=dirty())throw std::runtime_error("owner/resident mismatch");
    for(const auto &[key,n]:lineage){auto [producer,trigger,reason]=key;
      events<<id<<','<<producer<<','<<trigger<<','<<reason<<','<<n<<','<<n*32<<",NOT_MODELED,NOT_MODELED,NOT_MODELED\n";
    }
    for(const auto &[producer,v]:versions){
      if(v.writes!=v.superseded+v.emitted+resident[producer])throw std::runtime_error("dirty-version ownership residual");
      totals<<id<<','<<producer<<','<<v.writes<<','<<v.superseded<<','<<v.emitted<<','<<resident[producer]<<",0\n";
    }
    events.flush();totals.flush();if(!events||!totals)throw std::runtime_error("ownership summary write failed");
  }
  RetentionReplay(const char*config,const std::string&p,bool order):policy(p),write_budget(p=="disabled"?16:8),read_budget(p=="store8_read7"||p=="store8to7_read7"?7:16),warp_major(order) {
    opt.l1_fill_latency_set=opt.l2_fill_latency_set=true;
    apply_hw_options(opt,read_hw_params(config));
    opt.l2_dirty_drain=false;opt.l2_streaming_fill=false;
    const bool legacy=opt.num_partitions==10&&opt.l2_set_index==SetIndexFunction::Linear;
    const bool paper=opt.num_partitions==20&&opt.l2_set_index==SetIndexFunction::BitwiseXor;
    if(opt.l2_line_size!=128||opt.l2_assoc!=16||opt.l2_size_bytes!=41943040||
       !(legacy||paper)||!opt.mem_addr_mapping.empty()||opt.memory_partition_indexing)
      throw std::runtime_error("expected legacy diagnostic or frozen paper geometry");
    if(paper&&policy!="disabled"&&policy!="fixed_budget8_clean_retained_after_store_v1"&&!pressure_policy())throw std::runtime_error("paper geometry admits only frozen LRU and the predeclared fixed budget8 candidate");
    for(unsigned p=0;p<opt.num_partitions;++p)caches.emplace_back(opt.l2_size_bytes/opt.num_partitions,128,16,opt.l2_set_index);
  }
  void access(uint64_t addr,bool store,uint32_t byte_mask=UINT32_MAX) {
    for(uint64_t v:{addr,uint64_t(store)})for(unsigned b=0;b<8;++b)c.digest=(c.digest^((v>>(8*b))&255))*1099511628211ull;
    auto &cache=caches.at(dram_partition_index(addr,opt));
    const auto index=l2_cache_index_addr(addr,opt);
    auto a=cache.access(addr,32,32,0,0,store?CacheOperation::Store:CacheOperation::Read,store,index,true,false,byte_mask);
    last_access=a;
    auto &pd=partition_dirty.at(dram_partition_index(addr,opt));
    const auto before=__builtin_popcount(a.dirty_before),after=__builtin_popcount(a.dirty_after),victim=__builtin_popcount(a.evicted.dirty_sectors);
    if(after<before || pd+after-before<victim)throw std::runtime_error("partition counter access underflow");
    pd+=after-before;pd-=victim;
    if(store) {
      ++c.writes;
      const bool dirty=a.dirty_before&(1u<<((addr%128)/32));
      c.created+=!dirty;c.redirty+=dirty;
      auto prior=owners.find(addr);
      if(dirty!=(prior!=owners.end()))throw std::runtime_error("dirty bit/owner mismatch");
      if(prior!=owners.end())++versions[prior->second].superseded;
      owners[addr]=epoch;++versions[epoch].writes;
    }else {
      ++c.reads;c.read_hits+=a.result==CacheResult::Hit;c.read_misses+=a.result!=CacheResult::Hit;
      if(a.result!=CacheResult::Hit)c.fill+=32;
    }
    c.victims+=a.evicted.present;c.evicted+=__builtin_popcount(a.evicted.dirty_sectors);
    for(uint64_t v:{uint64_t(a.result),uint64_t(a.valid_before),uint64_t(a.valid_after),uint64_t(a.evicted.present),a.evicted.addr,uint64_t(a.evicted.valid_sectors)})
      for(unsigned b=0;b<8;++b)c.tag_digest=(c.tag_digest^((v>>(8*b))&255))*1099511628211ull;
    emit(a.evicted,"tag_eviction");

    if(pressure_policy()){
      const auto key=std::make_pair(dram_partition_index(addr,opt),cache.diagnostic_set_id(index));
      auto &count=set_read_allocations[key];
      if(a.evicted.present){written_at_read_count.erase(a.evicted.addr/128);dirty_read_since_store.erase(a.evicted.addr/128);}
      if(store){written_at_read_count[addr/128]=count;dirty_read_since_store.erase(addr/128);}
      else if(a.result==CacheResult::Hit && (a.dirty_before&(1u<<((addr%128)/32)))){
        ++c.pressure_dirty_read_hits;dirty_read_since_store.insert(addr/128);
      }
      else if(a.result==CacheResult::LineMiss){
        ++count;++c.pressure_reads;
        const auto selected=cache.diagnostic_select_pressure(index,count,written_at_read_count,opt.l2_assoc/2);
        observe_joint(selected,a,dram_partition_index(addr,opt));
        c.pressure_dirty_seen+=selected.dirty;c.pressure_eligible_seen+=selected.eligible;c.pressure_partial_blocked+=selected.partial;c.pressure_max_age=std::max(c.pressure_max_age,selected.max_age);
        if(selected.payload.dirty){
          ++c.pressure_selected_sectors[__builtin_popcount(selected.payload.dirty_sectors)];
          c.pressure_selected_revisited+=dirty_read_since_store.count(selected.payload.addr/128);
          const auto cleaned=cache.diagnostic_clean_pressure(index,selected.payload.addr,policy=="read_allocation_age8_one_sector_v1");
          c.cleaned+=__builtin_popcount(cleaned.dirty_sectors);++c.pressure_clean_events;c.pressure_dirty_under8_events+=selected.dirty<8;
          emit(cleaned,"clean_retained");
          if(cleaned.dirty_sectors==selected.payload.dirty_sectors){written_at_read_count.erase(cleaned.addr/128);dirty_read_since_store.erase(cleaned.addr/128);}
        }
      }
      return;
    }
    unsigned limit=store?write_budget:read_budget;
    if(store&&policy=="store8to7_read7") {
      // Predeclared hysteresis hypothesis: crossing eight dirty lines starts
      // a two-line clean down to seven. Merely reaching eight does nothing.
      // A full-associativity budget is a non-mutating census of this set.
      const auto state=cache.diagnostic_clean_set_dirty_overflow(index,16);
      if(!state.emitted.empty()||!state.satisfied||state.dirty_lines_before!=state.dirty_lines_after)
        throw std::runtime_error("dirty census unexpectedly mutated state");
      limit=state.dirty_lines_before>8?7:16;
    }
    if(limit<16) {
      auto cleaned=cache.diagnostic_clean_set_dirty_overflow(index,limit);
      if(!cleaned.satisfied)throw std::runtime_error("full-sector candidate left an unsatisfied budget");
      for(const auto &e:cleaned.emitted) {
        if(e.present||e.incomplete_dirty_sectors||e.missing_dirty_bytes)throw std::runtime_error("invalid retained payload");
        c.cleaned+=__builtin_popcount(e.dirty_sectors);
        emit(e,"clean_retained");
      }
    }
  }
  uint64_t dirty()const{uint64_t n=0;for(const auto &cache:caches)n+=cache.dirty_sector_count();return n;}
  template<class F>void ordered(uint64_t rounds,F visit,unsigned warps=384) {
    if(warp_major)for(unsigned w=0;w<warps;++w)for(uint64_t q=0;q<rounds;++q)visit(w,q);
    else for(uint64_t q=0;q<rounds;++q)for(unsigned w=0;w<warps;++w)visit(w,q);
  }
  void sum_warp(uint64_t sums,unsigned w) {
    // 32 lanes store uint64_t, hence eight full source sectors per warp.
    for(unsigned s=0;s<8;++s)access(sums+uint64_t(w)*256+s*32,true);
  }
  void stores(uint64_t base,uint64_t bytes,unsigned sectors,bool reverse,unsigned warps,uint64_t proof) {
    const auto lines=bytes/128;
    if(!sectors)return;
    ordered((lines+warps-1)/warps,[&](unsigned w,uint64_t q){
      auto line=q*warps+w;if(line>=lines)return;
      const bool retiring=line+warps>=lines;
      if(reverse)line=lines-1-line;
      for(unsigned s=0;s<sectors;++s)access(base+line*128+s*32,true);
      if(retiring)access(proof+uint64_t(w)*32,true);
    },warps);
  }
  void full_read(uint64_t base,uint64_t bytes,uint64_t sums) {
    const auto sectors=bytes/32;
    ordered((sectors+12287)/12288,[&](unsigned w,uint64_t q){
      const uint64_t first=q*12288+w*32;
      if(first>=sectors)return;
      for(unsigned lane=0;lane<32&&first+lane<sectors;++lane)access(base+(first+lane)*32,false);
      if(first+12288>=sectors)sum_warp(sums,w);
    });
  }
  void probe(uint64_t base,uint64_t bytes,unsigned sectors,unsigned mode,uint64_t sums) {
    if(mode==0)return; // Hardware still launches an empty kernel, with no requests.
    if(mode==5){for(unsigned w=0;w<384;++w)sum_warp(sums,w);return;}
    if(mode!=1)throw std::runtime_error("candidate supports empty, dense, and sums-only arms");
    const auto lines=bytes/128;const unsigned selected=sectors?sectors:1;
    ordered((lines+12287)/12288,[&](unsigned w,uint64_t q){
      const uint64_t first=q*12288+w*32;
      if(first>=lines)return;
      // This matches the hardware kernel's sector loop OUTSIDE the warp lanes.
      for(unsigned s=0;s<selected;++s)for(unsigned lane=0;lane<32&&first+lane<lines;++lane)
        access(base+(first+lane)*128+s*32,false);
      if(first+12288>=lines)sum_warp(sums,w);
    });
  }
};
#ifndef L2_RETENTION_CANDIDATE_NO_MAIN
int main(int argc,char**argv) {
 try {
  if(argc!=6)throw std::runtime_error("usage: candidate config cells.csv disabled|fixed_budget8_clean_retained_after_store_v1|store8|store8_read7|store8to7_read7 iteration|warp result.csv");
  const std::string policy=argv[3],order=argv[4];
  if((policy!="disabled"&&policy!="store8"&&policy!="store8_read7"&&policy!="store8to7_read7"&&policy!="fixed_budget8_clean_retained_after_store_v1")||(order!="iteration"&&order!="warp"))throw std::runtime_error("unregistered hypothesis");
  std::ifstream input(argv[2]);std::ofstream out(argv[5]);std::string row;std::getline(input,row);const bool timing=row.find(",probe_position")!=std::string::npos;
  out<<"case_id,span_mib,active_sectors,probe_mode,generations,store_order,policy,order,phase,ordinal,source_read_sectors,source_write_sectors,read_hits,read_misses,dirty_created,redirty,victims,dirty_evicted,cleaned_retained,resident_entry,resident_exit,read_fill_bytes,writeback_emitted_bytes,residual_bytes,ordered_digest,tag_outcome_digest\n";
  std::ofstream events(std::string(argv[5])+".ownership.csv"),totals(std::string(argv[5])+".versions.csv");
  if(!events||!totals)throw std::runtime_error("ownership output open failed");
  events<<"case_id,producer_ordinal,trigger_ordinal,reason,sectors,bytes,admission_epoch,service_epoch,completion_epoch\n";
  totals<<"case_id,producer_ordinal,write_versions,superseded_versions,emitted_versions,resident_versions,residual_sectors\n";
  while(std::getline(input,row)) {
   std::replace(row.begin(),row.end(),',',' ');std::istringstream in(row);
   std::string id,base_s,pressure_s,sums_s;unsigned mib,sectors,mode,generations,store_order,producer_warps;
   if(!(in>>id>>mib>>sectors>>mode>>generations>>store_order>>base_s>>pressure_s>>sums_s>>producer_warps))throw std::runtime_error("bad input cell");
   unsigned probe_position=0;if(timing&&(!(in>>probe_position)||probe_position>1))throw std::runtime_error("bad probe position");
   const uint64_t base=std::stoull(base_s,nullptr,0),pressure=std::stoull(pressure_s,nullptr,0),sums=std::stoull(sums_s,nullptr,0),bytes=uint64_t(mib)<<20;
   if(base%128||pressure%128||sums%128||mib<8||mib>(timing?64:40)||!sectors||sectors>4||store_order>1||generations!=4||(producer_warps!=1&&producer_warps!=384))throw std::runtime_error("unsupported input");
   RetentionReplay model(argv[1],policy,order=="warp");
   // Explicit modeled entry: empty -> ordinary128MiB pressure +96KiB sums.
   // DMA traffic and real entry state remain unobserved, never called cold HW.
   model.full_read(pressure,uint64_t(128)<<20,sums);
   unsigned ordinal=0;
   auto stage=[&](const char*name,uint64_t expected_read,uint64_t expected_write,auto action){
    const auto before=model.dirty();model.epoch=ordinal;model.c={};action();const auto after=model.dirty();const auto&c=model.c;
    if(model.owners.size()!=after)throw std::runtime_error("phase owner/resident mismatch");
    if(c.reads!=expected_read||c.writes!=expected_write||c.read_hits+c.read_misses!=c.reads||
       c.created+c.redirty!=c.writes||before+c.created!=after+c.evicted+c.cleaned||c.emitted!=32*(c.evicted+c.cleaned))
      throw std::runtime_error("candidate source or dirty conservation failed");
    out<<id<<','<<mib<<','<<sectors<<','<<mode<<','<<generations<<','<<store_order<<','<<policy<<','<<order<<','<<name<<','<<ordinal++<<','<<c.reads<<','<<c.writes<<','<<c.read_hits<<','<<c.read_misses<<','<<c.created<<','<<c.redirty<<','<<c.victims<<','<<c.evicted<<','<<c.cleaned<<','<<before<<','<<after<<','<<c.fill<<','<<c.emitted<<",0,"<<c.digest<<','<<c.tag_digest<<'\n';
    out.flush();if(!out)throw std::runtime_error("candidate output write failed");
   };
   const uint64_t proof=sums+6*98304;
   if(timing){
    auto probe_stage=[&](unsigned pass){stage("probe",mode==1?bytes/128*sectors:0,mode?3072:0,[&]{model.probe(base,bytes,sectors,mode,sums+uint64_t(pass)*98304);});};
    for(unsigned pass=0;pass<4;++pass){
     if(probe_position)probe_stage(pass);
     stage("store",0,bytes/128*sectors+producer_warps,[&]{model.stores(base,bytes,sectors,false,producer_warps,proof+uint64_t(pass)*12288);});
     stage("gap",0,0,[]{});
     if(!probe_position)probe_stage(pass);
    }
   }else{
   stage("store",0,bytes/128*sectors+producer_warps,[&]{model.stores(base,bytes,sectors,false,producer_warps,proof);});
   stage("gap",0,0,[]{});
   for(unsigned pass=0;pass<4;++pass) {
    if(generations==4&&pass)stage("store",0,bytes/128*sectors+producer_warps,[&]{model.stores(base,bytes,sectors,store_order&&(pass&1),producer_warps,proof+uint64_t(pass)*12288);});
    stage("probe",mode==1?bytes/128*(sectors?sectors:1):0,mode?3072:0,[&]{model.probe(base,bytes,sectors,mode,sums+uint64_t(pass)*98304);});
   }
   }
   stage("pressure",4194304,3072,[&]{model.full_read(pressure,uint64_t(128)<<20,sums+4*98304);});
   stage("verify",bytes/32,3072,[&]{model.full_read(base,bytes,sums+5*98304);});
   model.write_ownership(id,events,totals);
   std::cerr<<id<<" complete\n";
  }
  if(!input.eof())throw std::runtime_error("candidate input read failed");
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}

#endif
