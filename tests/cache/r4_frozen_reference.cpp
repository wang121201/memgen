// Serial read reference model. Misses fill synchronously; no timing or write claims.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
struct Config{std::string id,policy;unsigned unit,sets,hash,scale,seed;};
struct Node{uint64_t key=0;unsigned mask=0,rrpv=3,ref=0;int prev=-1,next=-1;};
struct Set{std::unordered_map<uint64_t,int> lookup;std::vector<Node> nodes;unsigned used=0,hand=0;int head=-1,tail=-1;explicit Set(unsigned ways):nodes(ways){lookup.reserve(ways*2);}};
struct Cache{
 Config cfg;unsigned ways,state;uint64_t hits=0,misses=0;std::vector<Set> sets;
 Cache(Config c,unsigned nominal):cfg(c),ways((uint64_t(nominal)*c.scale/1000)/(c.unit*c.sets)),state(c.seed?c.seed:1){assert(ways);for(unsigned i=0;i<c.sets;i++)sets.emplace_back(ways);}
 unsigned random(){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
 void unlink(Set& s,int i){auto& n=s.nodes[i];if(n.prev>=0)s.nodes[n.prev].next=n.next;else s.head=n.next;if(n.next>=0)s.nodes[n.next].prev=n.prev;else s.tail=n.prev;}
 void front(Set& s,int i){auto& n=s.nodes[i];n.prev=-1;n.next=s.head;if(s.head>=0)s.nodes[s.head].prev=i;else s.tail=i;s.head=i;}
 void access(uint32_t address){
  uint64_t key=address/cfg.unit;unsigned lg=__builtin_ctz(cfg.sets);uint32_t mix=uint32_t(key);if(cfg.hash==1)mix^=mix>>lg;
  if(cfg.hash==2){mix^=mix>>16;mix*=0x7feb352dU;mix^=mix>>15;mix*=0x846ca68bU;mix^=mix>>16;}
  unsigned index=mix&(cfg.sets-1);auto& s=sets[index];unsigned bit=1u<<((address%cfg.unit)/32);
  auto found=s.lookup.find(key);
  if(found!=s.lookup.end()){
   int i=found->second;auto& n=s.nodes[i];if(n.mask&bit)hits++;else{misses++;n.mask|=bit;}
   if(cfg.policy=="LRU"){unlink(s,i);front(s,i);}n.rrpv=0;n.ref=1;return;
  }
  misses++;int slot;
  if(s.used<ways){slot=s.used++;}
  else{slot=s.tail;
   if(cfg.policy=="RANDOM")slot=random()%ways;
   else if(cfg.policy=="CLOCK"){
    while(s.nodes[s.hand].ref){s.nodes[s.hand].ref=0;s.hand=(s.hand+1)%ways;}
    slot=s.hand;s.hand=(s.hand+1)%ways;
   }else if(cfg.policy=="SRRIP"||cfg.policy=="BRRIP"){
    for(;;){slot=-1;for(unsigned j=0;j<ways;j++)if(s.nodes[j].rrpv==3){slot=j;break;}
     if(slot>=0)break;for(auto& n:s.nodes)n.rrpv=std::min(3u,n.rrpv+1);
    }
   }
   s.lookup.erase(s.nodes[slot].key);unlink(s,slot);}
  s.nodes[slot].key=key;s.nodes[slot].mask=bit;s.nodes[slot].ref=1;
  s.nodes[slot].rrpv=(cfg.policy=="BRRIP"&&random()%32!=0)?3:2;s.lookup[key]=slot;front(s,slot);
 }
};
int main(int argc,char** argv){
 if(argc==2&&std::string(argv[1])=="--self-test"){
  Config c{"test","LRU",32,1,0,1000,11};Cache a(c,64);for(auto x:{0,32,0,64,0})a.access(x);assert(a.hits==2&&a.misses==3);
  c.policy="FIFO";Cache b(c,64);for(auto x:{0,32,0,64,0})b.access(x);assert(b.hits==1&&b.misses==4);
  c.policy="LRU";c.unit=128;Cache d(c,128);for(auto x:{0,32,0,128,32})d.access(x);assert(d.hits==1&&d.misses==4);
  Cache e(c,128);for(auto x:{0,4,8,28,32,36})e.access(x);assert(e.hits==4&&e.misses==2);
  c.policy="RANDOM";Cache f(c,256),g(c,256);for(unsigned i=0;i<10000;i++){unsigned addr=(i*73%41)*32;f.access(addr);g.access(addr);}assert(f.hits==g.hits&&f.misses==g.misses&&f.hits+f.misses==10000);
  std::cout<<"SELF_TEST_PASS 5\n";return 0;
 }
 if(argc!=3){std::cerr<<"usage: cache_policy_replay CASES_TSV CONFIGS_TSV\n";return 2;}
 std::ifstream cf(argv[2]);if(!cf)return 3;std::vector<Config> configs;Config cfg;
 while(cf>>cfg.id>>cfg.policy>>cfg.unit>>cfg.sets>>cfg.hash>>cfg.scale>>cfg.seed){assert(cfg.unit==32||cfg.unit==128);assert(cfg.sets&&!(cfg.sets&(cfg.sets-1)));assert(cfg.hash<=2);assert(cfg.policy=="LRU"||cfg.policy=="FIFO"||cfg.policy=="RANDOM"||cfg.policy=="CLOCK"||cfg.policy=="SRRIP"||cfg.policy=="BRRIP");configs.push_back(cfg);}
 assert(!configs.empty());std::ifstream cases(argv[1]);if(!cases)return 4;int id,cg;unsigned nominal,expected;std::string path;
 std::cout<<"config_id,seed,case_id,allocation_unit,sets,hash,capacity_bytes,hits,misses,requests\n";
 while(cases>>id>>nominal>>cg>>expected>>path){
  std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)return 5;size_t bytes=f.tellg();assert(bytes==uint64_t(expected)*4);std::vector<uint32_t> req(expected);f.seekg(0);f.read(reinterpret_cast<char*>(req.data()),bytes);assert(f.good());
  for(auto c:configs){Cache cache(c,nominal);if(!cg)for(auto addr:req)cache.access(addr);else cache.misses=req.size();assert(cache.hits+cache.misses==req.size());
   std::cout<<c.id<<','<<c.seed<<','<<id<<','<<c.unit<<','<<c.sets<<','<<c.hash<<','<<cache.ways*c.unit*c.sets<<','<<cache.hits<<','<<cache.misses<<','<<req.size()<<'\n';
  }
 }
}
