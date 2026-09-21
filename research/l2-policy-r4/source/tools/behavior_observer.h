// Passive causal state. This type has no mutable reference to a cache or labels.
#pragma once
#include <array>
#include <map>
#include <unordered_map>
#include <sstream>

struct BehaviorObserver {
  static constexpr unsigned Window=64;
  using SetKey=std::pair<unsigned,uint64_t>;
  struct SetState {
    std::array<unsigned char,Window> ring{};
    unsigned pos=0,n=0,reads=0,writes=0,overwrites=0;
    uint64_t allocations=0,tags=0,dirty_lines=0,dirty_sectors=0;
    void push(bool allocation,bool store,bool overwrite) {
      auto adjust=[&](unsigned char f,bool add){
        if(add){reads+=bool(f&1);writes+=bool(f&2);overwrites+=bool(f&4);}
        else{reads-=bool(f&1);writes-=bool(f&2);overwrites-=bool(f&4);}
      };
      if(n==Window)adjust(ring[pos],false);else ++n;
      ring[pos]=(allocation?1:0)|(store?2:0)|(overwrite?4:0);
      adjust(ring[pos],true);pos=(pos+1)%Window;
      if(reads>n||writes>n||overwrites>writes)throw std::runtime_error("behavior window range");
    }
  };
  struct Stamp {SetKey key;uint64_t allocation;};
  std::map<SetKey,SetState> sets;
  std::unordered_map<uint64_t,Stamp> stamps;
  using RequestKey=std::array<uint64_t,9>;
  using SectorKey=std::array<uint64_t,11>;
  struct Ages {uint64_t count=0,sum=0,min=UINT64_MAX,max=0;};
  std::map<RequestKey,uint64_t> requests;
  std::map<SectorKey,Ages> sectors;
  uint64_t accesses=0,stores=0,overwrites=0,read_allocations=0;
  uint64_t resident_samples=0,proposal_samples=0,clean_samples=0,victim_samples=0;
  static void require(bool x,const char*msg){if(!x)throw std::runtime_error(msg);}
  uint64_t age(const SetKey&key,uint64_t addr)const {
    auto it=stamps.find(addr);require(it!=stamps.end()&&it->second.key==key,"behavior missing/misplaced timestamp");
    const auto c=sets.at(key).allocations;require(c>=it->second.allocation,"behavior timestamp from future");return c-it->second.allocation;
  }
  void sample(unsigned kind,const SetKey&key,uint64_t line,uint32_t mask,uint32_t valid,unsigned trigger=2) {
    require(kind==1?trigger<=1:trigger==2,"behavior proposal trigger scope");
    const auto &s=sets.at(key);
    for(unsigned i=0;i<4;++i)if(mask&(1u<<i)){
      const auto a=age(key,line+32*i);
      SectorKey k={kind,s.tags,s.dirty_lines,s.dirty_sectors,s.n,s.reads,s.writes,s.overwrites,std::min<uint64_t>(a,32),bool(valid&(1u<<i)),trigger};
      auto &z=sectors[k];++z.count;z.sum+=a;z.min=std::min(z.min,a);z.max=std::max(z.max,a);
      if(kind==0)++resident_samples;else if(kind==1)++proposal_samples;else if(kind==2)++clean_samples;else if(kind==3)++victim_samples;else require(false,"behavior sample kind");
    }
    require(sectors.size()<=262144,"behavior sector histogram bound");
  }
  void erase(const SetKey&key,uint64_t line,uint32_t mask) {
    for(unsigned i=0;i<4;++i)if(mask&(1u<<i)){
      age(key,line+32*i);require(stamps.erase(line+32*i)==1,"behavior duplicate emission");
    }
  }
  template<class Snapshot>
  void access(const SetKey&key,uint64_t addr,bool store,const CacheAccess&a,const Snapshot&snapshot) {
    auto &s=sets[key];const bool overwrite=store&&(a.dirty_before&(1u<<((addr%128)/32)));
    require(bool(stamps.count(addr))==bool(a.dirty_before&(1u<<((addr%128)/32))),"behavior pre-access dirty identity");
    // Victim samples observe the pre-request state, before this read allocation.
    if(a.evicted.present){
      sample(3,key,a.evicted.addr,a.evicted.dirty_sectors,a.evicted.valid_sectors);
      erase(key,a.evicted.addr,a.evicted.dirty_sectors);
    }
    const bool allocation=!store&&a.result==CacheResult::LineMiss;
    if(allocation){++s.allocations;++read_allocations;}
    if(store)stamps[addr]={key,s.allocations};
    s.push(allocation,store,overwrite);++accesses;stores+=store;overwrites+=overwrite;
    s.tags=snapshot.size();s.dirty_lines=s.dirty_sectors=0;
    for(const auto &line:snapshot){s.dirty_lines+=line.dirty!=0;s.dirty_sectors+=__builtin_popcount(line.dirty);}
    require(s.tags<=16&&s.dirty_lines<=s.tags&&s.dirty_sectors<=4*s.dirty_lines,"behavior occupancy range");
    unsigned kind=store?0:a.result==CacheResult::Hit?1:a.result==CacheResult::LineMiss?3:2;
    ++requests[{kind,s.tags,s.dirty_lines,s.dirty_sectors,s.n,s.reads,s.writes,s.overwrites,uint64_t(a.evicted.present)}];
    require(requests.size()<=262144,"behavior request histogram bound");
    // These samples observe normal allocation completed, before optional clean.
    if(allocation)for(const auto &line:snapshot)sample(0,key,line.tag*128,line.dirty,line.valid);
  }
  void proposal(const SetKey&key,const EvictedLine&payload,bool current_tag_eviction) {
    sample(1,key,payload.addr,payload.dirty_sectors,payload.valid_sectors,current_tag_eviction);
  }
  void clean(const SetKey&key,const EvictedLine&payload) {
    auto &s=sets.at(key);require(!payload.present&&!payload.incomplete_dirty_sectors&&!payload.missing_dirty_bytes,"behavior invalid retained emission");
    sample(2,key,payload.addr,payload.dirty_sectors,payload.valid_sectors);
    erase(key,payload.addr,payload.dirty_sectors);
    const auto n=__builtin_popcount(payload.dirty_sectors);require(s.dirty_sectors>=unsigned(n),"behavior dirty count underflow");s.dirty_sectors-=n;
    bool remains=false;for(unsigned i=0;i<4;++i)remains|=stamps.count(payload.addr+32*i)!=0;
    if(!remains&&n){require(s.dirty_lines>0,"behavior dirty lines underflow");--s.dirty_lines;}
  }
  template<class Caches>
  void validate(const Caches&caches)const {
    uint64_t count=0;
    for(unsigned p=0;p<caches.size();++p){
      std::map<uint64_t,std::array<uint64_t,3>> occupancy;
      caches[p].diagnostic_visit_resident([&](uint64_t set,uint64_t tag,uint32_t valid,uint32_t dirty){
        auto &n=occupancy[set];++n[0];n[1]+=dirty!=0;n[2]+=__builtin_popcount(dirty);
        for(unsigned i=0;i<4;++i)if(dirty&(1u<<i)){age({p,set},tag*128+32*i);++count;}
      });
      for(const auto &[id,n]:occupancy){const auto &s=sets.at({p,id});require(s.tags==n[0]&&s.dirty_lines==n[1]&&s.dirty_sectors==n[2],"behavior resident occupancy mismatch");}
    }
    require(count==stamps.size(),"behavior dangling dirty timestamps");
  }
  void reset_counts(){requests.clear();sectors.clear();accesses=stores=overwrites=read_allocations=0;resident_samples=proposal_samples=clean_samples=victim_samples=0;}
  void write(const std::string&policy,int kernel,std::ostream&r,std::ostream&t,std::ostream&totals)const {
    uint64_t n=0,w=0,alloc=0,resident=0;for(const auto &[k,c]:requests){n+=c;w+=(k[0]==0)*c;if(k[0]==3){alloc+=c;resident+=c*k[3];}r<<policy<<','<<kernel;for(auto x:k)r<<','<<x;r<<','<<c<<'\n';}
    require(n==accesses&&w==stores&&alloc==read_allocations&&resident==resident_samples,"behavior request population residual");
    std::array<uint64_t,4> pop{};for(const auto &[k,v]:sectors){pop.at(k[0])+=v.count;t<<policy<<','<<kernel;for(auto x:k)t<<','<<x;t<<','<<v.count<<','<<v.sum<<','<<v.min<<','<<v.max<<'\n';}
    require(pop==std::array<uint64_t,4>{resident_samples,proposal_samples,clean_samples,victim_samples},"behavior sector population residual");
    totals<<policy<<','<<kernel<<','<<Window<<','<<accesses<<','<<stores<<','<<overwrites<<','<<read_allocations<<','<<resident_samples<<','<<proposal_samples<<','<<clean_samples<<','<<victim_samples<<','<<stamps.size()<<",0\n";
    r.flush();t.flush();totals.flush();require(bool(r)&&bool(t)&&bool(totals),"behavior output write");
  }
  std::string state_signature()const {
    std::ostringstream o;for(const auto &[k,s]:sets){o<<k.first<<','<<k.second<<','<<s.pos<<','<<s.n<<','<<s.reads<<','<<s.writes<<','<<s.overwrites<<','<<s.allocations<<','<<s.tags<<','<<s.dirty_lines<<','<<s.dirty_sectors<<';';for(auto v:s.ring)o<<unsigned(v)<<',';}
    std::map<uint64_t,Stamp> sorted(stamps.begin(),stamps.end());for(const auto &[a,t]:sorted)o<<a<<','<<t.key.first<<','<<t.key.second<<','<<t.allocation<<';';return o.str();
  }
};
