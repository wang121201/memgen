void dirty_protection_fixture(const char*config){
  selected_write_quota=16;selected_read_age=16;selected_phase_variant=11;selected_behavior_arm=0;selected_sector_age=false;selected_disable_aged=false;
  auto fresh=[&](unsigned policy=1,bool observe=true){auto p=candidate_victim(config,policy);p->protection.enabled=observe;p->epoch=1;return p;};
  auto finder=fresh();std::vector<uint64_t> line;
  for(uint64_t x=0;line.size()<40&&x<(128ull<<20);x+=128)if(dram_partition_index(x,finder->opt)==0&&finder->caches[0].diagnostic_set_id(l2_cache_index_addr(x,finder->opt))==0)line.push_back(x);
  check_victim(line.size()==40,"protection fixture address population");const DirtyProtectionObserver::Key key={0,0};
  auto p=fresh();p->access(line[0],true);p->access(line[0]+32,true,255);
  for(unsigned i=1;i<16;++i)p->access(line[i],false);
  check_victim(p->protection.free_alloc==15&&p->protection.replacements==0,"free allocations excluded");
  p->access(line[0]+64,false);p->access(line[0],false);
  check_victim(p->protection.sector_misses==1&&p->protection.hits==1&&p->protection.replacements==0,"same line sector miss/hit excluded");
  for(unsigned i=1;i<16;++i)p->access(line[i],false);
  p->access(line[16],false);
  check_victim(p->last_access.evicted.addr==line[1]&&!p->last_access.evicted.dirty&&p->protection.last_victim_rank==2,"oldest dirty skipped for oldest clean");
  check_victim(p->protection.protected_events==1&&p->protection.protected_lines==1&&p->protection.samples[0]==2,"one line/two sectors protected");
  uint64_t partial=0;for(const auto&[k,h]:p->protection.histogram)if(k[0]==0&&!k[2])partial+=h.n;
  check_victim(partial==1&&p->protection.clocks.at(key).clean==1,"partial dirty sector observed without validity fabrication");
  p->access(line[0],true);const auto&s0=p->protection.stamps.at(line[0]);const auto&s1=p->protection.stamps.at(line[0]+32);
  check_victim(s0.clean==1&&s0.overwrites==1&&s1.clean==0&&s1.overwrites==0,"sector-only write reset");
  p->epoch=7;p->protection.reset_counts();p->access(line[17],false);
  check_victim(p->protection.protected_events==0&&p->protection.clean_replacements==1,"ordinary LRU clean not protection");
  check_victim(p->protection.clocks.at(key).clean==2&&p->protection.stamps.at(line[0]+32).clean==0,"kernel/statistics cut keeps clocks");p->assert_pressure_metadata();
  // Explicit mask retirement exercises observer lifetime; the cache itself keeps the untouched dirty sector.
  auto maskclean=[&](uint32_t mask){const auto e=p->caches[0].diagnostic_clean_pressure_mask(l2_cache_index_addr(line[0],p->opt),line[0],mask);p->c.cleaned+=__builtin_popcount(mask);p->c.store_clean_sectors+=__builtin_popcount(mask);p->emit(e,"clean_retained");};
  maskclean(1);check_victim(!p->protection.stamps.count(line[0])&&p->protection.stamps.count(line[0]+32),"partial mask retirement only");
  uint64_t preclean_two=0;for(const auto&[k,h]:p->protection.histogram)if(k[0]==2&&k[3]==2)preclean_two+=h.n;check_victim(preclean_two==1,"partial payload reports full pre-clean line coverage");
  p->access(line[0]+32,false);maskclean(2);p->assert_pressure_metadata();check_victim(p->protection.stamps.empty(),"remaining dirty version retired");
  p->access(line[0],true);check_victim(p->protection.stamps.at(line[0]).clean==2&&p->protection.stamps.at(line[0]).overwrites==0,"new dirty version starts at current clock");
  for(unsigned policy:{0u,1u}){
    auto q=fresh(policy);for(unsigned i=0;i<16;++i)q->access(line[i],true);q->epoch=2;q->access(line[16],false);
    check_victim(q->last_access.evicted.addr==line[0]&&q->c.emitted==32&&q->protection.dirty_replacements==1&&q->protection.clean_replacements==0&&q->protection.protected_events==0,"all dirty normal LRU fallback");
    check_victim(!q->protection.stamps.count(line[0])&&q->protection.samples[3]==1,"victim destroys exact dirty stamp");q->assert_pressure_metadata();
  }
  {auto q=fresh(0);q->access(line[0],true);for(unsigned i=1;i<=16;++i)q->access(line[i],false);check_victim(q->c.emitted==32&&q->protection.protected_events==0,"disabled normal LRU victim");q->assert_pressure_metadata();}
  {DirtyProtectionObserver o;o.enabled=true;std::vector<SectorLruCache::ProtectionLine> b{{1,0,0,1}};CacheAccess a;a.result=CacheResult::LineMiss;bool rejected=false;try{o.access(key,256,false,a,b,true);}catch(const std::runtime_error&){rejected=true;}check_victim(rejected,"reserved prestate explicitly rejected");}
  auto signature=[](const DirtyProtectionObserver&o){std::ostringstream s;for(const auto&[k,c]:o.clocks)s<<k.first<<','<<k.second<<':'<<c.reads<<','<<c.clean<<';';std::map<uint64_t,DirtyProtectionObserver::Stamp> sorted(o.stamps.begin(),o.stamps.end());for(const auto&[a,v]:sorted)s<<a<<':'<<v.key.first<<','<<v.key.second<<','<<v.reads<<','<<v.clean<<','<<v.overwrites<<';';return s.str();};
  auto a=fresh(2),b=fresh(2),off=fresh(2,false);a->policy="renamed-A";b->policy="renamed-B";
  for(unsigned i=0;i<240;++i){const auto addr=line[(i*7)%40]+32*(i%4);const bool store=i%5<2;
    if(i%31==0){b->epoch+=9;b->protection.reset_counts();}
    for(auto*q:{a.get(),b.get(),off.get()})q->access(addr,store);
    check_victim(signature(a->protection)==signature(b->protection)&&a->c.emitted==b->c.emitted&&a->c.emitted==off->c.emitted&&a->c.fill==off->c.fill&&a->c.tag_digest==off->c.tag_digest&&a->dirty()==off->dirty(),"passive/label/segmentation invariance");
  }
  for(auto*q:{a.get(),b.get(),off.get()}){q->assert_pressure_metadata();std::ostringstream e,v;q->write_ownership(q->policy,e,v);}
  const auto saved=signature(a->protection);b->access(line[39],false);check_victim(saved==signature(a->protection),"future suffix isolation");
  std::ostringstream vo,so,to;a->protection.write(a->policy,a->epoch,vo,so,to,[&](uint64_t addr){return a->protection_info(addr);});check_victim(a->protection.samples[4]==a->owners.size(),"exact resident census");
  selected_write_quota=4;selected_phase_variant=0;
  std::cout<<"PASS_DIRTY_PROTECTION_PRESTATE_RANK_PARTIAL_CLOCK_MASK_RESERVED_CAUSAL_PASSIVE\n";
}
