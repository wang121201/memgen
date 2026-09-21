// Addressless census before opcode filtering. It does not infer local or texture
// sectors from unqualified addresses and does not infer dynamic work from SASS.
class CacheInputCensus {
public:
  using Key=std::tuple<int,std::string,unsigned,std::string,std::string,std::string>;
  struct Counts { uint64_t records=0,instructions=0,active=0,addresses=0,zero_predicate=0; };
  void observe(const KernelMeta &m,const hyfiss_request_trace::OrderedMemoryInst &i,bool include_local) {
    auto cls=CacheObservation::opcode_class(i.opcode), sp=CacheObservation::space(i.opcode);
    if(starts_with(i.opcode,"ATOMS") || starts_with(i.opcode,"REDS")) sp="shared";
    const bool local=sp=="local";
    std::string status=classify_opcode(i.opcode,include_local)=='N'?"FILTERED":"ACCEPTED";
    if(local && !include_local) status="LOCAL_DISABLED";
    else if(local) status="REJECT_MISSING_LOCAL_IDENTITY";
    else if(sp=="shared") status="SHARED_NO_L2";
    else if(sp=="texture" || sp=="surface") status="UNSUPPORTED_CLIENT";
    else if(cls=="OTHER") status="UNSUPPORTED_OPCODE";
    unsigned active=__builtin_popcount(i.mask);
    auto &c=rows_[{m.id,m.llm_phase,i.sm_id,i.opcode,sp,status}];
    ++c.records; ++total_.records; c.instructions+=(active!=0); total_.instructions+=(active!=0);
    c.active+=active; total_.active+=active; c.addresses+=i.addr.size(); total_.addresses+=i.addr.size();
    c.zero_predicate+=(active==0); total_.zero_predicate+=(active==0);
    if(local && include_local) throw std::runtime_error("include_local=true rejected: ordered profile lacks warp owner and local address metadata");
    if(!active && !i.addr.empty()) throw std::runtime_error("zero predicate has addresses: refusing fallback lane inference");
    if(active && i.addr.empty()) throw std::runtime_error("active memory instruction lacks lane addresses");
    if(active && i.addr.size()%active) throw std::runtime_error("address count not divisible by active lane count");
    if(sp=="shared" && (starts_with(i.opcode,"ATOM") || starts_with(i.opcode,"RED")))
      throw std::runtime_error("shared atomic would enter legacy global RMW path; not admitted");
    if(cls=="LDGSTS" && active && i.addr.size()>active)
      throw std::runtime_error("LDGSTS multiple address refs lack global-source/shared-destination role metadata");
  }
  void write(const fs::path &dir,const std::map<int,KernelStats> &backend) const {
    Counts sum;
    std::map<int,Counts> accepted;
    std::ofstream out(dir/"cache_input_census.csv");
    out<<"kernel_id,phase,sm,raw_opcode,opcode_class,memory_space,admission,semantic_id,source_sector_status,records,dynamic_instructions,active_lanes,lane_reference_addresses,zero_predicate_records\n";
    for(const auto &[k,c]:rows_) {
      const auto &[id,phase,sm,op,space,status]=k;
      out<<id<<','<<csv_escape(phase)<<','<<sm<<','<<csv_escape(op)<<','<<CacheObservation::opcode_class(op)<<','<<space<<','<<status
         <<",unassigned,SEE_ACCEPTED_SECTOR_LEDGER,"<<c.records<<','<<c.instructions<<','<<c.active<<','<<c.addresses<<','<<c.zero_predicate<<'\n';
      sum.records+=c.records;sum.instructions+=c.instructions;sum.active+=c.active;sum.addresses+=c.addresses;sum.zero_predicate+=c.zero_predicate;
      if(status=="ACCEPTED") {auto &a=accepted[id];a.instructions+=c.instructions;a.addresses+=c.addresses;}
    }
    if(sum.records!=total_.records || sum.instructions!=total_.instructions || sum.active!=total_.active || sum.addresses!=total_.addresses || sum.zero_predicate!=total_.zero_predicate)
      throw std::runtime_error("input census residual");
    for(const auto &[id,c]:backend) {
      const auto &a=accepted[id];
      if(a.instructions!=c.mem_insts || a.addresses!=c.lane_accesses)
        throw std::runtime_error("input census to accepted backend bridge residual");
      accepted.erase(id);
    }
    if(!accepted.empty()) throw std::runtime_error("accepted census kernel missing from backend");
    std::ofstream j(dir/"cache_input_census.json");
    j<<"{\"scope\":\"ordered profile input; upstream capture completeness unverified\",\"records\":"<<total_.records
     <<",\"dynamic_instructions\":"<<total_.instructions<<",\"active_lanes\":"<<total_.active<<",\"lane_reference_addresses\":"<<total_.addresses
     <<",\"zero_predicate_records\":"<<total_.zero_predicate<<",\"classification_residual\":0,\"accepted_backend_bridge_residual\":0,\"excluded_source_sectors\":null,\"api_memcpy_memset\":null,\"api_audit_status\":\"UNOBSERVED\"}\n";
    if(!out || !j) throw std::runtime_error("census output failed");
  }
private:
  std::map<Key,Counts> rows_; Counts total_;
};
