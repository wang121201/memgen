void early_emission_fixture(const char*config){
 selected_phase_variant=11;selected_write_quota=4;selected_read_age=16;selected_behavior_arm=0;selected_sector_age=false;selected_disable_aged=false;
 auto fresh=[&](unsigned phase=11){selected_phase_variant=phase;auto p=candidate_victim(config,2);p->epoch=1;return p;};
 auto finder=fresh();std::vector<uint64_t> lines;
 for(uint64_t x=0;lines.size()<90&&x<(256ull<<20);x+=128)if(dram_partition_index(x,finder->opt)==0&&finder->caches[0].diagnostic_set_id(l2_cache_index_addr(x,finder->opt))==0)lines.push_back(x);
 check_victim(lines.size()==90,"early emission fixture addresses");
 auto close=[](RetentionReplay&p){p.validate_early_observation();p.assert_pressure_metadata();std::ostringstream out,owner,versions;p.write_early_observation(out);p.write_ownership(p.policy,owner,versions);};
 // No actual tag replacement, emitted mask is noncontiguous 0101.
 auto free=fresh();for(unsigned i=2;i<=13;++i)free->access(lines[i],false);free->access(lines[0],true);free->access(lines[0]+64,true);free->access(lines[1],true);free->access(lines[14],false);
 check_victim(free->early_observation.events[0]==1&&free->early_observation.sectors[0]==2&&free->early_observation.masks[0][5]==1&&free->early_observation.events[1]==0,"actual free-way early mask population");
 free->access(lines[15],true);free->access(lines[16],false);check_victim(free->early_observation.qualified[1]==1&&free->early_observation.blocked[1]==1&&free->early_observation.events[1]==0,"qualified replacement blocked is not emitted");close(*free);
 // Real tag replacement emits exactly the dirty sectors, with producer state across kernels.
 auto replaced=fresh();for(unsigned i=2;i<=15;++i)replaced->access(lines[i],false);replaced->access(lines[0],true);replaced->access(lines[0]+32,true);replaced->access(lines[0]+64,true);replaced->access(lines[0]+96,true);replaced->access(lines[1],true);replaced->epoch=2;replaced->access(lines[16],false);
 check_victim(replaced->early_observation.events[1]==1&&replaced->early_observation.sectors[1]==4&&replaced->early_observation.masks[1][15]==1&&replaced->versions[1].emitted==4,"replacement early full mask and old producer");close(*replaced);
 // A blocked second early falls back to aged; those sectors must not count as early.
 auto fallback=fresh();fallback->access(lines[0],true);fallback->access(lines[0]+64,true);for(unsigned i=1;i<=12;++i)fallback->access(lines[i],false);fallback->access(lines[1],true);fallback->access(lines[2],true);fallback->access(lines[13],false);fallback->access(lines[0],false);fallback->access(lines[2],true);fallback->access(lines[14],false);fallback->access(lines[15],false);fallback->access(lines[2],true);fallback->access(lines[16],false);
 check_victim(fallback->pc.aged_sectors==2&&fallback->early_observation.events[0]==1&&fallback->early_observation.sectors[0]==1&&fallback->early_observation.blocked[1]==1&&fallback->early_observation.events[1]==0,"aged fallback excluded from early emissions");close(*fallback);
 // In this zero-reserved 16-way domain, actual replacement implies tags=16.
 // This proves the gate predicate, not a second complete policy replay.
 auto p15=fresh(15);uint64_t checks=0;
 for(unsigned i=0;i<900;++i){auto addr=lines[(i*7)%50]+32*(i%4);bool store=i%7<2;
  p15->access(addr,store);const auto&a=p15->last_access;
  if(!store&&a.result==CacheResult::LineMiss){
   auto&cache=p15->caches[dram_partition_index(addr,p15->opt)];auto state=cache.diagnostic_protection_state(l2_cache_index_addr(addr,p15->opt));for(const auto&e:state)check_victim(e.reserved==0,"equivalence admitted zero-reserved domain");
   check_victim(!a.evicted.present||state.size()==16,"replacement always full in admitted domain");
   // Watermarks differ, but conjunction with actual replacement is equivalent for every occupancy.
   bool v11=state.size()>=15&&a.evicted.present;bool v15=a.evicted.present;check_victim(v11==v15,"phase11 watermark plus victim equals phase15 functional gate");++checks;
  }
  if(i%91==0){close(*p15);p15->early_observation={};p15->pc={};p15->behavior.reset_counts();p15->epoch++;}
 }
 check_victim(checks>100,"equivalence exercised");close(*p15);
 // Exhaust occupancy/decision booleans, without pretending impossible partial-set replacement is admissible.
 for(unsigned tags=0;tags<=16;++tags)for(bool victim:{false,true}){if(victim&&tags!=16)continue;for(bool closing:{false,true})for(bool full:{false,true})for(unsigned dirty=0;dirty<=tags;++dirty){bool a=closing&&full&&dirty>=2&&tags>=15&&victim;bool b=closing&&full&&dirty>=2&&victim;check_victim(a==b,"equivalent gates over admitted state domain");}}
 selected_phase_variant=0;
 std::cout<<"PASS_EARLY_ACTUAL_MASK_SPACING_FALLBACK_PRODUCER_AND_PHASE15_GATE_EQUIVALENCE\n";
}
