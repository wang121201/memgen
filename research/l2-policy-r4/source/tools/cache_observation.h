// Included after the cache and semantic types. Summary-only, no event/address file.
// These counters observe the functional model; no physical write service is implied.
class CacheObservation {
public:
  using Key = std::tuple<int, std::string, unsigned, std::string, std::string, int>;
  enum Metric { Instructions, ActiveLanes, LaneAddresses, SourceSectors,
    SourceRead, SourceWrite, SourceAtomic, L1Requests, L1Hit, L1Reserved,
    L1LineMiss, L1SectorMiss, L1Bypass, L2Requests, L2Read, L2Write, L2Atomic,
    L2Hit, L2Reserved, L2LineMiss, L2SectorMiss, DirtyCreated, Redirty,
    Victim, CleanVictim, DirtyVictim, DirtyEvicted, ReadFill, Writeback,
    L1ActualLookups, L1ReadBypass, IncompleteWriteback, WritebackKnownBytes,
    WritebackMissingBytes, CleanedRetained, Count };
  using Counters = std::array<uint64_t, Count>;
  struct Boundary {
    uint64_t entry=0, exit=0, created=0, evicted=0, cleaned=0, cleaned_events=0;
    uint64_t digest=1469598103934665603ull, request_digest=1469598103934665603ull;
  };
  struct Owner { size_t key; int birth; };
  static std::string opcode_class(const std::string &s) {
    for (const auto *p : {"LDGSTS", "STG", "LDG", "STL", "LDL", "STS", "LDS",
                          "ATOM", "RED", "TEX", "TLD", "SULD", "SUST"})
      if (starts_with(s,p)) return p;
    return "OTHER";
  }
  static std::string space(const std::string &s) {
    auto c=opcode_class(s);
    if(c=="STL" || c=="LDL") return "local";
    if(c=="STS" || c=="LDS") return "shared";
    if(c=="TEX" || c=="TLD") return "texture";
    if(c=="SULD" || c=="SUST") return "surface";
    if(c=="OTHER") return "unknown";
    return "global";
  }
  size_t row(const KernelMeta &m, const MemoryInst &i, int semantic) {
    Key k{m.id,m.llm_phase,i.sm_id,opcode_class(i.opcode),space(i.opcode),semantic};
    return row_key(k);
  }
  size_t row_key(const Key &k) {
    auto f=index_.find(k);
    if(f!=index_.end()) return f->second;
    if(keys_.size()>=1000000) throw std::runtime_error("observation row limit exceeded");
    size_t n=keys_.size(); index_.emplace(k,n); keys_.push_back(k); rows_.push_back({});
    return n;
  }
  void instruction(const KernelMeta &m, const MemoryInst &i,
                   const std::vector<SectorRequest> &reqs, const SemanticDatabase &db) {
    int sem=reqs.empty()?0:static_cast<int>(db.lookup(reqs.front().addr).id);
    for(const auto &r:reqs) if(static_cast<int>(db.lookup(r.addr).id)!=sem) {sem=-1;break;}
    size_t n=row(m,i,sem); add(n,Instructions,1);
    add(n,ActiveLanes,__builtin_popcount(i.mask)); add(n,LaneAddresses,i.lanes.size());
    raw_opcodes_[{m.id,i.sm_id,i.opcode}]++;
  }
  void begin(int k,uint64_t resident) {
    if(active_) throw std::runtime_error("observation epoch still active");
    if(resident!=owners_.size()) throw std::runtime_error("observation entry dirty mismatch");
    if(boundaries_.count(k)) throw std::runtime_error("duplicate observation kernel epoch");
    boundaries_[k].entry=resident;
    active_=true;active_epoch_=k;
  }
  void occupancy(int k,const std::string &boundary,const std::vector<SectorLruCache> &caches) {
    for(size_t p=0;p<caches.size();++p)
      occupancy_.emplace(std::make_tuple(k,boundary,p),caches[p].occupancy());
  }
  void access(const KernelMeta &m,const MemoryInst &i,const SectorRequest &r,int sem,
              bool l1_lookup,const CacheAccess &l1,bool l2_lookup,const CacheAccess &l2) {
    require_epoch(m.id);
    if(r.size!=32) throw std::runtime_error("observation requires ordered 32B sectors");
    const size_t n=row(m,i,sem);
    add(n,SourceSectors,1); add(n,i.op=='R'?SourceRead:i.op=='W'?SourceWrite:SourceAtomic,1);
    add(n,l1_lookup?L1Requests:L1Bypass,1);
    if(l1_lookup) add(n,bypass_l1_read(i)?L1ReadBypass:L1ActualLookups,1);
    if(l1_lookup) outcome(n,l1.result,L1Hit,L1Reserved,L1LineMiss,L1SectorMiss);
    auto &b=boundaries_.at(m.id);
    for(uint64_t v:{r.addr,uint64_t(r.byte_mask),uint64_t(i.op),uint64_t(i.sm_id)}) hash(b.request_digest,v);
    if(!l2_lookup) return;
    add(n,L2Requests,1); add(n,i.op=='R'?L2Read:i.op=='W'?L2Write:L2Atomic,1);
    outcome(n,l2.result,L2Hit,L2Reserved,L2LineMiss,L2SectorMiss);
    for(uint64_t v:{r.addr,uint64_t(l2.result),uint64_t(l2.valid_before),uint64_t(l2.dirty_before),
       uint64_t(l2.valid_after),uint64_t(l2.dirty_after),uint64_t(l2.evicted.present),
       l2.evicted.addr,uint64_t(l2.evicted.valid_sectors),uint64_t(l2.evicted.dirty_sectors)}) hash(b.digest,v);
    if(i.op!='R') {
      const uint32_t bit=1u<<((r.addr%128)/32);
      const bool existed=owners_.count(r.addr);
      if(existed!=bool(l2.dirty_before&bit)) throw std::runtime_error("dirty before-mask owner mismatch");
      if(!(l2.dirty_after&bit)) throw std::runtime_error("write did not dirty its sector");
      add(n,existed?Redirty:DirtyCreated,1);
      if(!existed) { owners_.emplace(r.addr,Owner{n,m.id}); ++b.created; }
      else owners_.at(r.addr).key=n;
    }
    const auto &v=l2.evicted;
    if(v.present) {
      add(n,Victim,1); add(n,v.dirty?DirtyVictim:CleanVictim,1);
      add(n,IncompleteWriteback,__builtin_popcount(v.incomplete_dirty_sectors));
      add(n,WritebackKnownBytes,v.known_dirty_bytes);
      add(n,WritebackMissingBytes,v.missing_dirty_bytes);
      for(unsigned s=0;s<4;++s) if(v.dirty_sectors&(1u<<s)) {
        auto owner=owners_.find(v.addr+32*s);
        if(owner==owners_.end()) throw std::runtime_error("eviction missing dirty owner");
        auto pair=std::make_tuple(owner->second.key,n,owner->second.birth,m.id,false);
        if(writebacks_.size()>=1000000 && !writebacks_.count(pair))
          throw std::runtime_error("observation writeback row limit exceeded");
        writebacks_[pair]+=32; owners_.erase(owner);
        add(n,DirtyEvicted,1); add(n,Writeback,1); ++b.evicted;
      }
    }
    if(i.op!='W' && l2.result!=CacheResult::Hit && l2.result!=CacheResult::HitReserved)
      add(n,ReadFill,1);
  }
  // Explicit functional event only: selection and physical queue/service are
  // not modeled here. Classification attributes name the dirty producer, not
  // an inferred cleaner SM or a new memory instruction.
  void retained_clean(const KernelMeta &emission,const EvictedLine &v) {
    require_epoch(emission.id);
    const unsigned sectors=__builtin_popcount(v.dirty_sectors);
    if(v.present || !v.dirty || v.addr%128 || !sectors ||
       (v.valid_sectors|v.dirty_sectors)>15 || (v.dirty_sectors&~v.valid_sectors) ||
       v.incomplete_dirty_sectors || v.missing_dirty_bytes ||
       v.dirty_bytes!=32*sectors || v.known_dirty_bytes!=32*sectors)
      throw std::runtime_error("invalid retained-clean payload");
    uint32_t before=0;
    for(unsigned s=0;s<4;++s) {
      const bool owned=owners_.count(v.addr+32*s);
      if(owned) before|=1u<<s;
      if((v.dirty_sectors&(1u<<s))&&!owned)
        throw std::runtime_error("retained clean missing dirty owner");
    }
    // Validate the whole payload before releasing any owner.
    std::set<Key> new_keys;
    for(unsigned s=0;s<4;++s) if(v.dirty_sectors&(1u<<s)) {
      Key k=keys_[owners_.at(v.addr+32*s).key];
      std::get<0>(k)=emission.id;std::get<1>(k)=emission.llm_phase;
      if(!index_.count(k))new_keys.insert(k);
    }
    if(keys_.size()+new_keys.size()>1000000 || writebacks_.size()+sectors>1000000)
      throw std::runtime_error("retained clean observation row budget exceeded");
    auto &b=boundaries_.at(emission.id);
    for(uint64_t value:{uint64_t(0x434c45414e),v.addr,uint64_t(v.valid_sectors),
                       uint64_t(before),uint64_t(before&~v.dirty_sectors)})hash(b.digest,value);
    ++clean_transitions_[{emission.id,v.valid_sectors,before,before&~v.dirty_sectors}];
    for(unsigned s=0;s<4;++s) if(v.dirty_sectors&(1u<<s)) {
      auto owner=owners_.find(v.addr+32*s);
      Key classification=keys_[owner->second.key];
      std::get<0>(classification)=emission.id;std::get<1>(classification)=emission.llm_phase;
      size_t n=row_key(classification);
      writebacks_[{owner->second.key,n,owner->second.birth,emission.id,true}]+=32;
      add(n,CleanedRetained,1);add(n,Writeback,1);add(n,WritebackKnownBytes,32);
      owners_.erase(owner);++b.cleaned;
    }
    ++b.cleaned_events;
  }
  void end(int k,uint64_t resident) {
    require_epoch(k);
    auto &b=boundaries_.at(k); b.exit=resident;
    if(resident!=owners_.size() || b.entry+b.created!=b.exit+b.evicted+b.cleaned)
      throw std::runtime_error("observation resident dirty conservation failed");
    active_=false;
  }
  void write(const fs::path &dir,const std::map<int,KernelStats> &stats) const {
    static const char *names[]={"dynamic_instructions","active_lanes","lane_addresses","source_sectors",
      "source_read_sectors","source_write_sectors","source_atomic_sectors","l1_requests","l1_hit",
      "l1_hit_reserved","l1_line_miss","l1_sector_miss","l1_bypass","l2_requests","l2_read_requests",
      "l2_write_requests","l2_atomic_requests","l2_hit","l2_hit_reserved","l2_line_miss","l2_sector_miss",
      "dirty_created_sectors","redirty_sectors","victim_lines","clean_victim_lines","dirty_victim_lines",
      "dirty_evicted_sectors","dram_read_fill_sectors","dram_writeback_emitted_sectors",
      "l1_actual_lookups","l1_read_bypass_requests","incomplete_writeback_sectors",
      "writeback_known_bytes","writeback_missing_bytes","cleaned_retained_sectors"};
    Counters sum{}; std::map<int,Counters> per_kernel;
    for(size_t n=0;n<keys_.size();++n) for(size_t j=0;j<Count;++j) {
      sum[j]+=rows_[n][j]; per_kernel[std::get<0>(keys_[n])][j]+=rows_[n][j];
    }
    if(sum!=total_) throw std::runtime_error("observation classification sum mismatch");
    auto equal=[](uint64_t a,uint64_t b,const char *s) {
      if(a!=b) throw std::runtime_error(std::string("observation residual: ")+s);
    };
    for(const auto &kv:stats) {
      const auto &s=kv.second; const auto &c=per_kernel[kv.first];
      equal(c[Instructions],s.mem_insts,"instructions"); equal(c[LaneAddresses],s.lane_accesses,"lane addresses");
      equal(c[SourceSectors],s.sector_requests,"source");
      equal(c[SourceRead],s.read_sector_requests,"source read"); equal(c[SourceWrite],s.write_sector_requests,"source write");
      equal(c[SourceAtomic],s.atomic_sector_requests,"source atomic");
      equal(c[L1Requests],s.l1_requests,"l1 requests"); equal(c[L1Hit],s.l1_hits,"l1 hit");
      equal(c[L1Reserved],s.l1_pending_hits,"l1 reserved"); equal(c[L1LineMiss],s.l1_line_misses,"l1 line miss");
      equal(c[L1SectorMiss],s.l1_sector_misses,"l1 sector miss");
      equal(c[L1ActualLookups]+c[L1ReadBypass],s.l1_requests,"l1 lookup/bypass partition");
      equal(c[L1Requests]+c[L1Bypass],s.sector_requests,"l1 source partition");
      equal(c[L2Requests],s.l2_requests,"l2 requests"); equal(c[L2Hit],s.l2_hits,"l2 hit");
      equal(c[L2Reserved],s.l2_pending_hits,"l2 reserved"); equal(c[L2LineMiss],s.l2_line_misses,"l2 line miss");
      equal(c[L2SectorMiss],s.l2_sector_misses,"l2 sector miss");
      equal(c[L2Read],s.l2_by_direction[0].requests,"l2 read");
      equal(c[L2Write],s.l2_by_direction[1].requests,"l2 write");
      equal(c[L2Atomic],s.l2_by_direction[2].requests,"l2 atomic");
      equal(c[ReadFill]*32,s.dram_load_bytes,"read bytes"); equal(c[Writeback]*32,s.dram_store_bytes,"write bytes");
      equal(c[DirtyEvicted]+c[CleanedRetained],s.l2_writeback_dirty_sectors,"evicted plus retained clean");
      equal(c[DirtyVictim]+boundaries_.at(kv.first).cleaned_events,s.l2_writeback_events,"writeback event kinds");
      equal(s.l2_dirty_drain_sectors,0,"forbidden drain");
      equal(c[Writeback]*32,c[WritebackKnownBytes]+c[WritebackMissingBytes],"writeback byte coverage");
    }
    uint64_t wb=0; for(const auto &p:writebacks_) wb+=p.second;
    equal(wb,total_[Writeback]*32,"producer trigger emission bytes");
    std::ofstream occ(dir/"cache_observation_occupancy.csv");
    occ<<"kernel_id,boundary,partition,allocated_lines,clean_lines,dirty_lines,partial_dirty_lines,valid_sectors,reserved_sectors,dirty_sectors,incomplete_dirty_sectors,known_bytes,known_dirty_bytes,missing_dirty_bytes,line_residual,dirty_byte_residual\n";
    std::ofstream hist(dir/"cache_observation_set_histogram.csv");
    hist<<"kernel_id,boundary,partition,lines_per_set,sets_by_allocated_lines,sets_by_dirty_lines\n";
    std::ofstream masks(dir/"cache_observation_valid_dirty_masks.csv");
    masks<<"kernel_id,boundary,partition,valid_sector_mask,dirty_sector_mask,line_count\n";
    std::map<std::pair<int,std::string>,uint64_t> resident;
    for(const auto &[key,c]:occupancy_) {
      const auto &[k,b,p]=key;
      resident[{k,b}]+=c.dirty_sectors;
      occ<<k<<','<<b<<','<<p<<','<<c.allocated_lines<<','<<c.clean_lines<<','<<c.dirty_lines<<','
        <<c.partial_dirty_lines<<','<<c.valid_sectors<<','<<c.reserved_sectors<<','<<c.dirty_sectors<<','
        <<c.incomplete_dirty_sectors<<','<<c.known_bytes<<','<<c.known_dirty_bytes<<','<<c.missing_dirty_bytes<<",0,0\n";
      for(size_t n=0;n<c.sets_by_allocated_lines.size();++n)
        hist<<k<<','<<b<<','<<p<<','<<n<<','<<c.sets_by_allocated_lines[n]<<','<<c.sets_by_dirty_lines[n]<<'\n';
      for(const auto &[m,n]:c.lines_by_valid_dirty_masks)
        masks<<k<<','<<b<<','<<p<<','<<m.first<<','<<m.second<<','<<n<<'\n';
    }
    for(const auto &[k,b]:boundaries_) {
      equal(resident[{k,"entry"}],b.entry,"partition entry dirty census");
      equal(resident[{k,"exit"}],b.exit,"partition exit dirty census");
    }
    std::ofstream lookups(dir/"cache_observation_l1_lookups.csv");
    lookups<<"kernel_id,legacy_l1_requests,actual_l1_lookups,read_bypass_requests,actual_l1_fill_sectors,residual_bytes\n";
    for(const auto &[k,c]:per_kernel) lookups<<k<<','<<c[L1Requests]<<','<<c[L1ActualLookups]<<','<<c[L1ReadBypass]<<','
      <<c[L1LineMiss]+c[L1SectorMiss]-c[L1ReadBypass]<<",0\n";
    std::ofstream out(dir/"cache_observation.csv");
    out<<"row_id,kernel_id,phase,sm,opcode_class,memory_space,semantic_id";
    for(auto name:names) out<<','<<name; out<<'\n';
    for(size_t n=0;n<keys_.size();++n) {
      const auto &[k,p,sm,oc,sp,sem]=keys_[n];
      out<<n<<','<<k<<','<<csv_escape(p)<<','<<sm<<','<<oc<<','<<sp<<','<<sem;
      for(auto v:rows_[n]) out<<','<<v; out<<'\n';
    }
    std::ofstream state(dir/"cache_observation_state.csv");
    state<<"kernel_id,resident_dirty_entry_sectors,dirty_created_sectors,dirty_evicted_sectors,resident_dirty_exit_sectors,resident_residual_bytes,ordered_sector_digest,l2_transition_digest,queued_entry,queued_exit,inflight_entry,inflight_exit,writeback_admitted,writeback_completed,total_debt_check,cleaned_retained_sectors,retained_clean_events\n";
    for(const auto &[k,b]:boundaries_) state<<k<<','<<b.entry<<','<<b.created<<','<<b.evicted<<','<<b.exit<<",0,"<<b.request_digest<<','<<b.digest<<",null,null,null,null,null,null,NOT_EVALUABLE,"<<b.cleaned<<','<<b.cleaned_events<<'\n';
    std::ofstream w(dir/"cache_observation_writeback.csv");
    w<<"producer_row_id,eviction_trigger_row_id,dirty_birth_epoch,emission_epoch,admission_epoch,completion_epoch,emitted_bytes,event_kind,classification_row_id\n";
    for(const auto &[key,bytes]:writebacks_) {
      const auto &[producer,trigger,birth,epoch,retained]=key;
      w<<producer<<','<<(retained?"null":std::to_string(trigger))<<','<<birth<<','<<epoch<<",null,null,"<<bytes<<','<<(retained?"retained_clean":"eviction")<<','<<trigger<<'\n';
    }
    std::ofstream cleans(dir/"cache_observation_retained_clean.csv");
    cleans<<"emission_epoch,valid_before,dirty_before,valid_after,dirty_after,event_count\n";
    for(const auto &[key,count]:clean_transitions_){const auto &[k,v,b,a]=key;cleans<<k<<','<<v<<','<<b<<','<<v<<','<<a<<','<<count<<'\n';}
    std::ofstream raw(dir/"cache_observation_opcodes.csv");
    raw<<"kernel_id,sm,raw_opcode,dynamic_instructions\n";
    for(const auto &[key,count]:raw_opcodes_) raw<<std::get<0>(key)<<','<<std::get<1>(key)<<','<<csv_escape(std::get<2>(key))<<','<<count<<'\n';
    std::ofstream j(dir/"cache_observation.json");
    j<<"{\n\"status\":\"PASS_FUNCTIONAL_OBSERVATION\",\n\"hardware_acceptance\":\"DIAGNOSTIC_NOT_HARDWARE_ACCEPTANCE\",\n"
      <<"\"scope\":\"accepted ordered memory instructions only; unsupported/excluded clients require input census\",\n"
      <<"\"semantic_instruction_rule\":\"unique sector semantic id; -1=mixed; 0=unknown; no duplicated instructions\",\n"
      <<"\"epoch_definition\":\"kernel id in ordered replay; emission is a functional event, not service completion\",\n"
      <<"\"retained_clean_definition\":\"explicit internal event; classification SM/opcode/space/semantic are dirty producer provenance; eviction trigger null; no source instruction or lookup; valid mask retained; automatic policy disabled\",\n"
      <<"\"line_hit_definition\":\"l2_requests minus l2_line_miss (includes sector miss); sector hit excludes reserved\",\n"
      <<"\"l1_legacy_definition\":\"l1_requests includes read bypass as synthetic sector miss; actual_l1_lookups excludes all bypass\",\n"
      <<"\"sector_bytes\":32,\n\"classification_residual_bytes\":0,\n\"resident_residual_bytes\":0,\n"
      <<"\"producer_trigger_emission_residual_bytes\":0,\n\"full_raw_trace_bytes\":0,\n"
      <<"\"queued_dirty\":null,\"inflight_dirty\":null,\"dram_writeback_admitted\":null,\"dram_writeback_completed\":null,\n"
      <<"\"physical_service_status\":\"NOT_IMPLEMENTED\",\"total_dirty_debt_check\":\"NOT_EVALUABLE\",\n\"totals\":{";
    for(size_t n=0;n<Count;++n) j<<(n?",":"")<<'\"'<<names[n]<<"\":"<<total_[n];
    j<<"}}\n";
    if(!out || !state || !w || !raw || !j || !lookups || !occ || !hist || !masks || !cleans) throw std::runtime_error("observation output failed");
  }
private:
  void require_epoch(int k) const {if(!active_||active_epoch_!=k)throw std::runtime_error("inactive observation epoch");}
  bool active_=false;int active_epoch_=0;
  static void hash(uint64_t &h,uint64_t v) { for(unsigned i=0;i<8;++i) {h^=(v>>(8*i))&255;h*=1099511628211ull;} }
  void add(size_t n,Metric m,uint64_t v) {rows_[n][m]+=v;total_[m]+=v;}
  void outcome(size_t n,CacheResult r,Metric hit,Metric reserved,Metric line,Metric sector) {
    add(n,r==CacheResult::Hit?hit:r==CacheResult::HitReserved?reserved:r==CacheResult::LineMiss?line:sector,1);
  }
  std::map<Key,size_t> index_; std::vector<Key> keys_; std::vector<Counters> rows_; Counters total_{};
  std::unordered_map<uint64_t,Owner> owners_;
  std::map<int,Boundary> boundaries_;
  std::map<std::tuple<int,std::string,size_t>,CacheOccupancy> occupancy_;
  std::map<std::tuple<size_t,size_t,int,int,bool>,uint64_t> writebacks_;
  std::map<std::tuple<int,uint32_t,uint32_t,uint32_t>,uint64_t> clean_transitions_;
  std::map<std::tuple<int,unsigned,std::string>,uint64_t> raw_opcodes_;
};
