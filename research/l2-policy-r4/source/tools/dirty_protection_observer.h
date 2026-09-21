#pragma once
// Passive, addressless output. Rank 1 is the oldest pre-existing line.
struct DirtyProtectionObserver {
  using Key=std::pair<unsigned,uint64_t>;
  struct Clock {uint64_t reads=0,clean=0;};
  struct Stamp {Key key;uint64_t reads=0,clean=0,overwrites=0;};
  bool enabled=false;
  std::map<Key,Clock> clocks;
  std::unordered_map<uint64_t,Stamp> stamps;
  uint64_t requests=0,reads=0,writes=0,hits=0,sector_misses=0,line_misses=0,free_alloc=0;
  uint64_t replacements=0,clean_replacements=0,dirty_replacements=0,protected_events=0,protected_lines=0;
  uint64_t created=0,overwrites=0,cleaned=0,evicted=0,entry=0;
  std::array<uint64_t,5> samples{};
  unsigned last_victim_rank=0;
  using VictimKey=std::array<uint64_t,6>; // tags,dirty_lines,dirty_sectors,lru_dirty_sectors,actual_dirty_sectors,actual_rank
  std::map<VictimKey,uint64_t> victims;
  using SampleKey=std::array<uint64_t,7>; // kind,rank,valid,line_dirty_sectors,redirty_cap32,read_age_cap256,clean_age_cap256
  struct Sample {uint64_t n=0,read_sum=0,read_min=UINT64_MAX,read_max=0,clean_sum=0,clean_min=UINT64_MAX,clean_max=0;};
  std::map<SampleKey,Sample> histogram;
  static void need(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
  void reset_counts(){
    requests=reads=writes=hits=sector_misses=line_misses=free_alloc=0;
    replacements=clean_replacements=dirty_replacements=protected_events=protected_lines=0;
    created=overwrites=cleaned=evicted=0;entry=stamps.size();samples={};victims.clear();histogram.clear();
  }
  void sample(unsigned kind,uint64_t address,unsigned rank,bool valid,unsigned coverage){
    const auto &s=stamps.at(address);const auto &c=clocks.at(s.key);
    need(c.reads>=s.reads&&c.clean>=s.clean,"protection negative exposure");
    const auto ra=c.reads-s.reads,ca=c.clean-s.clean;
    need(ca<=ra,"clean replacement age exceeds read allocation age");
    auto &h=histogram[{kind,rank,uint64_t(valid),coverage,std::min<uint64_t>(s.overwrites,32),std::min<uint64_t>(ra,256),std::min<uint64_t>(ca,256)}];
    ++h.n;h.read_sum+=ra;h.read_min=std::min(h.read_min,ra);h.read_max=std::max(h.read_max,ra);
    h.clean_sum+=ca;h.clean_min=std::min(h.clean_min,ca);h.clean_max=std::max(h.clean_max,ca);++samples.at(kind);
    need(histogram.size()<=262144,"protection histogram bound");
  }
  template<class Lines>
  void access(const Key&key,uint64_t addr,bool store,const CacheAccess&a,const Lines&before,bool clean_first){
    if(!enabled)return;
    ++requests;auto &clock=clocks[key];bool found=false;
    for(const auto&l:before){need(!l.reserved,"protection observation requires zero reserved prestate");found|=l.tag==addr/128;}
    need((a.result==CacheResult::LineMiss)==!found,"protection prestate/tag outcome disagreement");
    last_victim_rank=0;
    for(size_t i=0;i<before.size();++i)if(a.evicted.present&&before[i].tag==a.evicted.addr/128)last_victim_rank=before.size()-i;
    if(a.evicted.present)need(last_victim_rank>0,"ordinary victim absent from prestate");
    if(store){
      ++writes;auto it=stamps.find(addr);const bool old=it!=stamps.end();
      need(old==bool(a.dirty_before&(1u<<((addr%128)/32))),"protection stamp/dirty before mismatch");
      uint64_t n=old?it->second.overwrites+1:0;stamps[addr]={key,clock.reads,clock.clean,n};
      if(old)++overwrites;else ++created;
      return;
    }
    ++reads;
    if(a.result==CacheResult::Hit){++hits;return;}
    if(a.result!=CacheResult::LineMiss){need(a.result!=CacheResult::HitReserved,"reserved outcome outside observer domain");++sector_misses;return;}
    ++line_misses;++clock.reads;
    if(!a.evicted.present){++free_alloc;return;}
    need(before.size()==16,"replacement requires full prestate");++replacements;
    const auto &lru=before.back();uint64_t expected=lru.tag;unsigned dl=0,ds=0;
    for(const auto&l:before){dl+=bool(l.dirty);ds+=__builtin_popcount(l.dirty);}
    if(clean_first)for(auto it=before.rbegin();it!=before.rend();++it)if(!it->dirty){expected=it->tag;break;}
    need(a.evicted.addr/128==expected,"actual victim differs from passive prestate prediction");
    const auto &actual=before[before.size()-last_victim_rank];
    need(actual.dirty==a.evicted.dirty_sectors&&actual.valid==a.evicted.valid_sectors,"victim masks changed across observation");
    const bool clean=!a.evicted.dirty_sectors;
    if(clean){++clean_replacements;++clock.clean;}else ++dirty_replacements;
    ++victims[{before.size(),dl,ds,uint64_t(__builtin_popcount(lru.dirty)),uint64_t(__builtin_popcount(actual.dirty)),last_victim_rank}];
    const bool protect=lru.dirty&&lru.tag!=actual.tag&&clean;
    need((last_victim_rank>1)==protect,"skipped eligible LRU must be dirty protection");
    protected_events+=protect;
    for(size_t i=0;i<before.size();++i){const auto &l=before[i];const unsigned rank=before.size()-i;
      if(l.tag==actual.tag)continue;
      if(protect&&rank<last_victim_rank){need(l.dirty!=0,"clean-first skipped clean line");++protected_lines;}
      for(unsigned sec=0;sec<4;++sec)if(l.dirty&(1u<<sec)){
        const auto address=l.tag*128+32*sec;need(stamps.at(address).key==key,"protection sector in wrong set");
        sample(1,address,rank,l.valid&(1u<<sec),__builtin_popcount(l.dirty));
        if(protect&&rank<last_victim_rank)sample(0,address,rank,l.valid&(1u<<sec),__builtin_popcount(l.dirty));
      }
    }
  }
  void retire(const EvictedLine&e,bool retained,unsigned rank,unsigned line_dirty_sectors){
    if(!enabled)return;
    need(line_dirty_sectors>=unsigned(__builtin_popcount(e.dirty_sectors))&&line_dirty_sectors<=4,"retirement pre-clean line coverage");
    for(unsigned sec=0;sec<4;++sec)if(e.dirty_sectors&(1u<<sec)){
      const auto address=e.addr+32*sec;
      sample(retained?2:3,address,rank,e.valid_sectors&(1u<<sec),line_dirty_sectors);
      need(stamps.erase(address)==1,"protection retire missing stamp");
      if(retained)++cleaned;else ++evicted;
    }
  }
  template<class Owners>void validate(const Owners&owners)const{
    if(!enabled)return;
    need(owners.size()==stamps.size(),"protection owner/stamp population");
    for(const auto&x:owners)need(stamps.count(x.first)==1,"protection missing resident owner");
    need(entry+created==stamps.size()+cleaned+evicted,"protection resident balance");
    need(created+overwrites==writes&&requests==reads+writes,"protection write/request balance");
    need(reads==hits+sector_misses+line_misses&&line_misses==free_alloc+replacements,"protection read partition");
    need(replacements==clean_replacements+dirty_replacements&&protected_events<=clean_replacements,"protection replacement partition");
    uint64_t count=0;for(const auto&x:victims)count+=x.second;need(count==replacements,"protection victim histogram partition");
    need(samples[2]==cleaned&&samples[3]==evicted,"protection retire samples");
    need(protected_events<=protected_lines&&protected_lines<=samples[0]&&samples[0]<=4*protected_lines,"protection skipped line/sector counts");
  }
  template<class Info>void write(const std::string&policy,int kernel,std::ostream&vo,std::ostream&so,std::ostream&to,Info info){
    if(!enabled)return;
    need(samples[4]==0,"duplicate resident snapshot");
    for(const auto&x:stamps){const auto r=info(x.first);sample(4,x.first,r[0],r[1],r[2]);}
    for(const auto&[k,n]:victims){vo<<policy<<','<<kernel;for(auto v:k)vo<<','<<v;vo<<','<<n<<'\n';}
    for(const auto&[k,h]:histogram){so<<policy<<','<<kernel;for(auto v:k)so<<','<<v;so<<','<<h.n<<','<<h.read_sum<<','<<h.read_min<<','<<h.read_max<<','<<h.clean_sum<<','<<h.clean_min<<','<<h.clean_max<<'\n';}
    to<<policy<<','<<kernel<<','<<requests<<','<<reads<<','<<writes<<','<<hits<<','<<sector_misses<<','<<line_misses<<','<<free_alloc<<','<<replacements<<','<<clean_replacements<<','<<dirty_replacements<<','<<protected_events<<','<<protected_lines;
    for(auto n:samples)to<<','<<n;
    to<<','<<created<<','<<overwrites<<','<<cleaned<<','<<evicted<<','<<entry<<','<<stamps.size()<<",0\n";
    vo.flush();so.flush();to.flush();need(bool(vo)&&bool(so)&&bool(to),"protection output failure");
  }
};
