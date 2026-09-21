// CPU-only replay of ordinary, self-owned write-coverage microbench requests.
// Hardware warp interleaving and entry cache state are unknown. Both orders are
// explicit hypotheses. No memory trace is collected, loaded, or persisted.
#define HYFISS_REQUEST_TRACE_NO_MAIN
#include "hyfiss_request_trace_generator_stream_r4_semantic.cc"

struct Counts {
  uint64_t instructions=0, active_lanes=0, ordered_digest=1469598103934665603ull, multiset_sum=0;
  uint64_t requests=0, hits=0, line_misses=0, sector_misses=0;
  uint64_t created=0, redirty=0, victims=0, dirty_victims=0, evicted=0;
  uint64_t read_bytes=0, write_bytes=0, incomplete_wb=0, missing_wb_bytes=0;
};

struct Replay {
  Options opt;
  std::vector<SectorLruCache> caches;
  Counts c;
  explicit Replay(const char *config) {
    opt.l1_fill_latency_set=opt.l2_fill_latency_set=true;
    apply_hw_options(opt,read_hw_params(config));
    if(opt.l2_line_size!=128 || opt.l2_set_index!=SetIndexFunction::Linear ||
       !opt.mem_addr_mapping.empty() || opt.memory_partition_indexing)
      throw std::runtime_error("driver currently supports the frozen linear diagnostic config only");
    for(unsigned p=0;p<opt.num_partitions;++p)
      caches.emplace_back(opt.l2_size_bytes/opt.num_partitions,128,opt.l2_assoc,opt.l2_set_index);
  }
  void access(uint64_t addr,bool store,uint32_t mask=UINT32_MAX) {
    uint64_t event=1469598103934665603ull;
    for(uint64_t value:{addr,uint64_t(store),uint64_t(mask)})for(unsigned b=0;b<8;++b) {
      const auto byte=(value>>(8*b))&255;
      event=(event^byte)*1099511628211ull;
      c.ordered_digest=(c.ordered_digest^byte)*1099511628211ull;
    }
    c.multiset_sum+=event;
    auto &cache=caches.at(dram_partition_index(addr,opt));
    auto a=cache.access(addr,32,32,0,0,store?CacheOperation::Store:CacheOperation::Read,
                        store,l2_cache_index_addr(addr,opt),true,false,mask);
    ++c.requests;
    c.hits+=a.result==CacheResult::Hit;
    c.line_misses+=a.result==CacheResult::LineMiss;
    c.sector_misses+=a.result==CacheResult::SectorMiss;
    if(store) {
      bool dirty=a.dirty_before&(1u<<((addr%128)/32));
      c.created+=!dirty; c.redirty+=dirty;
    } else if(a.result==CacheResult::LineMiss || a.result==CacheResult::SectorMiss) c.read_bytes+=32;
    c.victims+=a.evicted.present; c.dirty_victims+=a.evicted.dirty;
    c.evicted+=__builtin_popcount(a.evicted.dirty_sectors);
    for_each_writeback_span(a.evicted,32,128,[&](const WritebackSpan &s){c.write_bytes+=s.size;});
#ifndef L2_MODEL_BASELINE
    c.incomplete_wb+=__builtin_popcount(a.evicted.incomplete_dirty_sectors);
    c.missing_wb_bytes+=a.evicted.missing_dirty_bytes;
#endif
  }
  uint64_t dirty() const { uint64_t n=0;for(const auto &x:caches)n+=x.dirty_sector_count();return n; }
  void reads(uint64_t base,uint64_t bytes,bool warp_major) {
    // prepare_sector_read: 32 distinct sectors per warp instruction.
    const uint64_t sectors=bytes/32, rounds=(sectors+12287)/12288;
    auto visit=[&](uint64_t warp,uint64_t q) {
      for(unsigned lane=0;lane<32;++lane) {
        uint64_t s=q*12288+warp*32+lane;
        if(s<sectors) access(base+32*s,false);
      }
    };
    if(warp_major) for(unsigned w=0;w<384;++w) for(uint64_t q=0;q<rounds;++q) visit(w,q);
    else for(uint64_t q=0;q<rounds;++q) for(unsigned w=0;w<384;++w) visit(w,q);
  }
  void stores(uint64_t base,uint64_t bytes,unsigned coverage,unsigned sectors,bool warp_major) {
    if(!coverage)return;
    const uint64_t lines=bytes/128, rounds=(lines+383)/384;
    const uint32_t mask=coverage==32?UINT32_MAX:(uint32_t(1)<<coverage)-1;
    auto visit=[&](uint64_t warp,uint64_t q) {
      uint64_t line=q*384+warp;
      if(line<lines) {
        ++c.instructions;c.active_lanes+=sectors*(coverage/4);
        for(unsigned s=0;s<sectors;++s) access(base+128*line+32*s,true,mask);
      }
    };
    if(warp_major) for(unsigned w=0;w<384;++w) for(uint64_t q=0;q<rounds;++q) visit(w,q);
    else for(uint64_t q=0;q<rounds;++q) for(unsigned w=0;w<384;++w) visit(w,q);
  }
};

int main(int argc,char **argv) {
  try {
    if(argc!=7) throw std::runtime_error("usage: model config cells.csv iteration|warp cold|pressure result.csv occupancy.csv");
    std::string order=argv[3],entry=argv[4];
    if((order!="iteration"&&order!="warp") || (entry!="cold"&&entry!="pressure")) throw std::runtime_error("bad order/entry");
    std::ifstream input(argv[2]);std::ofstream result(argv[5]),states(argv[6]);
    result<<"matrix,mib,coverage,active_sectors,preread,order,entry,source_sectors,source_covered_bytes,requests,hits,line_misses,sector_misses,dirty_created,redirty,victims,dirty_victims,dirty_evicted,resident_dirty_entry,resident_dirty_exit,model_read_bytes,model_write_bytes,incomplete_writeback_sectors,missing_writeback_bytes,residual_bytes,active_store_warp_instructions,active_lanes,ordered_digest,request_multiset_sum\n";
    states<<"matrix,mib,coverage,active_sectors,preread,order,entry,boundary,partition,allocated_lines,clean_lines,dirty_lines,partial_dirty_lines,valid_sectors,dirty_sectors,incomplete_dirty_sectors,known_bytes,known_dirty_bytes,missing_dirty_bytes,line_residual,dirty_byte_residual\n";
    std::string row;std::getline(input,row);
    while(std::getline(input,row)) {
      if(row.empty())continue;
      std::replace(row.begin(),row.end(),',',' ');std::istringstream in(row);
      std::string matrix,base_s,pressure_s;unsigned mib,coverage,sectors,preread;
      if(!(in>>matrix>>mib>>coverage>>sectors>>preread>>base_s>>pressure_s)) throw std::runtime_error("bad cell");
      uint64_t base=std::stoull(base_s,nullptr,0),pressure=std::stoull(pressure_s,nullptr,0),bytes=uint64_t(mib)<<20;
      if(base%128||pressure%128||!mib||mib>64||coverage>32||coverage%4||!sectors||sectors>4||preread>1)
        throw std::runtime_error("invalid cell conditions");
      Replay model(argv[1]);
      // Cold means empty. Pressure means empty followed by a separate 128MiB
      // read buffer. Neither reconstructs runtime memcpy or checksum stores.
      if(entry=="pressure") model.reads(pressure,uint64_t(128)<<20,order=="warp");
      if(preread) model.reads(base,bytes,order=="warp");
      auto prefix=[&](std::ostream &out){out<<matrix<<','<<mib<<','<<coverage<<','<<sectors<<','<<preread<<','<<order<<','<<entry;};
      auto snapshot=[&](const char *boundary) {
#ifndef L2_MODEL_BASELINE
        for(size_t p=0;p<model.caches.size();++p) {
          auto x=model.caches[p].occupancy();prefix(states);
          states<<','<<boundary<<','<<p<<','<<x.allocated_lines<<','<<x.clean_lines<<','<<x.dirty_lines<<','<<x.partial_dirty_lines<<','<<x.valid_sectors<<','<<x.dirty_sectors<<','<<x.incomplete_dirty_sectors<<','<<x.known_bytes<<','<<x.known_dirty_bytes<<','<<x.missing_dirty_bytes<<",0,0\n";
        }
#endif
      };
      snapshot("entry");const uint64_t before=model.dirty(); model.c={};
      model.stores(base,bytes,coverage,sectors,order=="warp");
      snapshot("exit");const uint64_t after=model.dirty();const auto &c=model.c;
      const uint64_t expected=coverage?bytes/128*sectors:0;
      if(c.instructions!=(coverage?bytes/128:0) || c.active_lanes!=expected*(coverage/4) ||
         c.requests!=expected || c.requests!=c.hits+c.line_misses+c.sector_misses ||
         c.created+c.redirty!=expected || before+c.created!=after+c.evicted || c.write_bytes!=32*c.evicted)
        throw std::runtime_error("model stream/conservation mismatch");
      prefix(result);result<<','<<expected<<','<<expected*coverage<<','<<c.requests<<','<<c.hits<<','<<c.line_misses<<','<<c.sector_misses<<','<<c.created<<','<<c.redirty<<','<<c.victims<<','<<c.dirty_victims<<','<<c.evicted<<','<<before<<','<<after<<','<<c.read_bytes<<','<<c.write_bytes<<','<<c.incomplete_wb<<','<<c.missing_wb_bytes<<",0,"<<c.instructions<<','<<c.active_lanes<<','<<c.ordered_digest<<','<<c.multiset_sum<<'\n';
      result.flush();states.flush();if(!result||!states)throw std::runtime_error("output failed");
      std::cerr<<matrix<<' '<<mib<<' '<<coverage<<' '<<sectors<<' '<<preread<<" completed\n";
    }
    if(!input.eof())throw std::runtime_error("input read failed");
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
