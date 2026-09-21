void aged_ablation_fixture(const char*config){
 selected_phase_variant=11;selected_write_quota=4;selected_read_age=16;selected_behavior_arm=0;selected_sector_age=false;
 auto fresh=[&](bool off){selected_disable_aged=off;auto p=candidate_victim(config,2);p->epoch=1;return p;};
 auto finder=fresh(false);std::vector<uint64_t> lines;
 for(uint64_t x=0;lines.size()<90&&x<(256ull<<20);x+=128)if(dram_partition_index(x,finder->opt)==0&&finder->caches[0].diagnostic_set_id(l2_cache_index_addr(x,finder->opt))==0)lines.push_back(x);
 check_victim(lines.size()==90,"aged ablation mapped population");
 auto close=[](RetentionReplay&p){p.assert_pressure_metadata();std::ostringstream o,v;p.write_ownership(p.policy,o,v);if(p.disable_aged)check_victim(p.pc.aged_events==0&&p.pc.aged_sectors==0,"aged off no emissions");};
 // Only aged eligible: all dirty sectors remain resident and readable in the off arm.
 auto old=fresh(false),off=fresh(true);
 for(auto*p:{old.get(),off.get()}){for(unsigned s=0;s<4;++s)p->access(lines[0]+32*s,true);for(unsigned i=1;i<=15;++i)p->access(lines[i],false);check_victim(p->c.emitted==0,"below age boundary");p->epoch=2;p->access(lines[16],false);close(*p);}
 check_victim(old->c.emitted==128&&old->pc.aged_sectors==4&&off->c.emitted==0&&off->dirty()==4&&off->owners.size()==4,"aged only action disabled");
 check_victim(off->partition_dirty[0]==4&&off->written_at_read_count.count(lines[0]/128)&&off->written_at_read_count.at(lines[0]/128)==0,"retained age ownership metadata");
 check_victim(old->caches[0].diagnostic_same_read_state(off->caches[0]),"aged off retains valid and tag state");
 off->access(lines[0],false);check_victim(off->last_access.result==CacheResult::Hit,"aged off dirty read hit");off->epoch=3;off->access(lines[0],true);check_victim(off->versions[1].superseded==1&&off->owners.at(lines[0])==3,"retained versions merge on real overwrite");close(*off);
 // The same early rule still fires, while a later early is blocked at spacing1.
 for(bool disabled:{false,true}){auto p=fresh(disabled);for(unsigned i=2;i<=13;++i)p->access(lines[i],false);p->access(lines[0],true);p->access(lines[0]+64,true);p->access(lines[1],true);p->access(lines[14],false);
  check_victim(p->pc.early_events==1&&p->pc.early_sectors==2&&p->c.emitted==64,"early untouched");p->access(lines[15],true);p->access(lines[16],false);check_victim(p->pc.spacing_blocked==1&&p->pc.early_events==1,"early spacing untouched");close(*p);}
 // Real early at allocation14 followed by a co-eligible aged/early at16.
 // The latter early is blocked: old falls back to aged A; off must retain A.
 auto a=fresh(false),b=fresh(true);
 for(auto*p:{a.get(),b.get()}){p->access(lines[0],true);p->access(lines[0]+64,true);for(unsigned i=1;i<=12;++i)p->access(lines[i],false);p->access(lines[1],true);p->access(lines[2],true);p->access(lines[13],false);check_victim(p->c.emitted==0,"below early watermark");p->access(lines[0],false);p->access(lines[2],true);p->access(lines[14],false);check_victim(p->c.emitted==32&&p->pc.early_events==1,"real prior early");p->access(lines[15],false);p->epoch=2;p->access(lines[2],true);p->access(lines[16],false);check_victim(p->pc.early_qualified==2&&p->pc.spacing_blocked==1&&p->pc.early_events==1,"real blocked early fallback");close(*p);}
 check_victim(a->c.emitted==96&&a->pc.aged_sectors==2&&b->c.emitted==32&&b->owners.count(lines[0])&&b->owners.count(lines[0]+64)&&b->owners.count(lines[2]),"blocked early cannot bypass aged off");
 check_victim(b->written_at_read_count.at(lines[0]/128)==0&&b->partition_dirty[0]==3,"fallback retains metadata and partition dirty");
 // Partial-sector preservation fill remains needed; aged-off does not invent valid bytes.
 auto partial_old=fresh(false),partial_off=fresh(true);
 for(auto*p:{partial_old.get(),partial_off.get()}){p->access(lines[0],true,255);for(unsigned i=1;i<=16;++i)p->access(lines[i],false);check_victim(p->c.emitted==0&&p->c.pressure_partial_blocked>0,"partial stays blocked");auto fill=p->c.fill;p->access(lines[0],false);check_victim(p->c.fill==fill+32,"partial preservation read retained");p->access(lines[17],false);close(*p);}
 check_victim(partial_old->c.emitted==32&&partial_off->c.emitted==0&&partial_off->owners.count(lines[0]),"only aged action changes after preservation fill");
 // Isolate the ordinary dirty tag eviction path using quota16 in both fixture arms.
 for(bool disabled:{false,true}){auto p=fresh(disabled);p->write_budget=16;for(unsigned i=0;i<16;++i)for(unsigned s=0;s<4;++s)p->access(lines[i]+32*s,true);p->epoch=2;p->access(lines[16],true);check_victim(p->c.evicted==4&&p->c.emitted==128&&p->c.cleaned==0,"ordinary dirty tag eviction unchanged");close(*p);}
 // Same causal stream under renamed labels and epoch boundaries.
 auto u=fresh(true),v=fresh(true);u->policy="renamed-A";v->policy="renamed-B";
 for(unsigned i=0;i<400;++i){auto addr=lines[(i*7)%40]+32*(i%4);bool write=i%5<2;u->access(addr,write);v->epoch=1+i/73;v->access(addr,write);if(i%73==0)v->behavior.reset_counts();check_victim(u->c.emitted==v->c.emitted&&u->behavior.state_signature()==v->behavior.state_signature(),"aged off label/segment invariant");}
 close(*u);close(*v);auto prefix=u->behavior.state_signature();v->access(lines[80],false);check_victim(prefix==u->behavior.state_signature(),"future suffix cannot alter prefix");
 selected_disable_aged=false;selected_phase_variant=0;
 std::cout<<"PASS_AGED_OFF_RETAIN_EARLY_BLOCKED_FALLBACK_PARTIAL_VICTIM_VERSION_CAUSAL\n";
}
