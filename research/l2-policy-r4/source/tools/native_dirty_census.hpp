// Passive aggregates over the fixed ordered request stream. No raw addresses are written.
struct NativeDirtyCensus {
  using Key=std::tuple<unsigned,int,int,int,int,int>;
  struct Bin {uint64_t events=0,sectors=0,covered_bytes=0;};
  struct SetHistory {uint64_t read_allocations=0,run=0;int previous=2;};
  std::map<Key,Bin> bins;
  std::map<std::pair<unsigned,uint64_t>,SetHistory> sets;
  std::unordered_map<uint64_t,uint64_t> last_store;
  uint64_t requests=0,reads=0,writes=0,created=0,redirty=0,emitted=0,read_allocations=0,scanned_lines=0,scanned_sectors=0;
  static void require(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
  static unsigned age_bin(uint64_t n){return n==0?0:n<4?1:n<8?2:n<16?3:n<32?4:5;}
  void add(unsigned metric,int a,int b,int c,int d,int e,uint64_t sectors=0,uint64_t bytes=0){
    auto&v=bins[{metric,a,b,c,d,e}];++v.events;v.sectors+=sectors;v.covered_bytes+=bytes;
    require(bins.size()<=131072,"native census histogram bound");
  }
  void reset_counts(){bins.clear();requests=reads=writes=created=redirty=emitted=read_allocations=scanned_lines=scanned_sectors=0;}
  void observe(const RetentionReplay&p,uint64_t addr,bool store,uint32_t mask,const RetentionCounts&before,int prior){
    const auto&a=p.last_access;const auto partition=dram_partition_index(addr,p.opt);
    const auto set=p.caches[partition].diagnostic_set_id(l2_cache_index_addr(addr,p.opt));
    auto&history=sets[{partition,set}];const int op=store?1:0;
    add(0,op,history.previous,age_bin(history.run),0,0);
    if(history.previous==op)++history.run;else{history.previous=op;history.run=1;}
    if(a.evicted.present)last_store.erase(a.evicted.addr/128);
    if(!store&&a.result==CacheResult::LineMiss){++history.read_allocations;++read_allocations;}
    if(store)last_store[addr/128]=history.read_allocations;
    ++requests;reads+=!store;writes+=store;
    const bool dirty=a.dirty_before&(1u<<((addr%128)/32));
    require(!store || dirty==(prior!=0),"census prior producer/dirty mismatch");
    created+=store&&!dirty;redirty+=store&&dirty;
    const int outcome=a.result==CacheResult::Hit?0:a.result==CacheResult::HitReserved?1:a.result==CacheResult::LineMiss?2:3;
    add(1,op,outcome,store?prior:0,0,0,1,store?__builtin_popcount(mask):0);
    unsigned dl=0,ds=0,partial=0,eligible=0;
    for(const auto&line:p.last_raw_snapshot){
      if(!line.dirty)continue;++dl;ds+=__builtin_popcount(line.dirty);
      auto it=last_store.find(line.tag);require(it!=last_store.end()&&it->second<=history.read_allocations,"census missing dirty age");
      const auto age=history.read_allocations-it->second;const bool full=(line.dirty&~line.valid)==0;
      partial+=!full;eligible+=full&&age>=16;
      if(store||a.result==CacheResult::LineMiss){add(3,op,age_bin(age),full,0,0,__builtin_popcount(line.dirty));++scanned_lines;scanned_sectors+=__builtin_popcount(line.dirty);}
    }
    if(store)add(2,p.last_raw_snapshot.size(),dl,ds,partial,dl>p.write_budget);
    const auto store_clean=p.c.store_clean_sectors-before.store_clean_sectors;
    const auto read_clean=p.c.read_clean_sectors-before.read_clean_sectors;
    if(!store&&a.result==CacheResult::LineMiss){
      // Mutually exclusive raw-set eligibility, before the policy's optional cleanup.
      const int status=dl==0?0:partial==dl?1:eligible==0?2:3;
      add(4,status,dl,eligible,read_clean>0,0,read_clean);
    }
    const auto victim=__builtin_popcount(a.evicted.dirty_sectors);
    add(5,op,a.evicted.present?(victim?2:1):0,victim,0,0,victim);
    if(store_clean)add(6,1,op,0,0,0,store_clean,32*store_clean);
    if(read_clean)add(6,2,op,0,0,0,read_clean,32*read_clean);
    if(victim)add(6,0,op,0,0,0,victim,32*victim);
    emitted+=32*(victim+store_clean+read_clean);
    require(p.c.emitted-before.emitted==32*(victim+store_clean+read_clean),"census emitted reason residual");
    require(last_store.size()<=327680,"census resident history bound");
  }
  void write(const RetentionReplay&p,std::ostream&out){
    require(requests==p.c.reads+p.c.writes&&reads==p.c.reads&&writes==p.c.writes,"census request denominator");
    require(created==p.c.created&&redirty==p.c.redirty&&emitted==p.c.emitted,"census version/emission denominator");
    uint64_t resident=0;
    for(unsigned partition=0;partition<p.caches.size();++partition){
      uint64_t partition_total=0;
      p.caches[partition].diagnostic_visit_resident([&](uint64_t set,uint64_t tag,uint32_t valid,uint32_t dirty){
        if(!dirty)return;
        auto h=sets.find({partition,set});auto w=last_store.find(tag);
        require(h!=sets.end()&&w!=last_store.end()&&w->second<=h->second.read_allocations,"boundary age identity");
        const auto age=age_bin(h->second.read_allocations-w->second);
        for(unsigned s=0;s<4;++s)if(dirty&(1u<<s)){
          auto owner=p.owners.find(tag*128+s*32);require(owner!=p.owners.end()&&owner->second<=p.epoch,"boundary producer identity");
          add(7,partition,owner->second==p.epoch,age,(valid&(1u<<s))!=0,0,1,32);++resident;++partition_total;
        }
      });
      require(partition_total==p.partition_dirty.at(partition),"census partition resident residual");
    }
    require(resident==p.dirty(),"census boundary resident residual");
    uint64_t metric_count[8]={},metric_sectors[8]={},emission_reason[3]={},write_prior[3]={};
    for(const auto&[key,v]:bins){auto[metric,a,b,c,d,e]=key;metric_count[metric]+=v.events;metric_sectors[metric]+=v.sectors;
      if(metric==6)emission_reason[a]+=v.sectors;
      if(metric==1&&a==1)write_prior[c]+=v.events;
      out<<p.policy<<','<<p.epoch<<','<<metric<<','<<a<<','<<b<<','<<c<<','<<d<<','<<e<<','<<v.events<<','<<v.sectors<<','<<v.covered_bytes<<'\n';
    }
    require(metric_count[0]==requests&&metric_count[1]==requests&&metric_count[2]==writes&&metric_count[5]==requests,"census category counts");
    require(metric_count[4]==read_allocations&&metric_count[3]==scanned_lines&&metric_sectors[3]==scanned_sectors,"census read allocation/dirty scan partition");
    require(write_prior[0]==created&&write_prior[1]+write_prior[2]==redirty,"census producer overwrite partition");
    require(emission_reason[0]==p.c.evicted&&emission_reason[1]==p.c.store_clean_sectors&&emission_reason[2]==p.c.read_clean_sectors,"census reason partition");
    require(metric_sectors[6]*32==emitted&&metric_count[7]==resident,"census emitted/resident partition");
    out.flush();require(bool(out),"census summary write");
  }
};
