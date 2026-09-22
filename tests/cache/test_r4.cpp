#include "../../release/source/tools/r4_l1_read_filter.h"
#define main frozen_reference_main
#include "r4_frozen_reference.cpp"
#undef main
#include <functional>
using hyfiss_request_trace::R4L1ReadFilter;
using hyfiss_request_trace::R4Allocation;
static constexpr uint64_t base=0x123456780000ULL;
void need(bool b) {if(!b)throw std::runtime_error("r4 differential assertion");}
void check(const Cache &ref,const R4L1ReadFilter &model) {
  need(ref.ways==model.ways());
  for(unsigned i=0;i<16;++i) {
    const auto &a=ref.sets[i];const auto &b=model.inspect(0,i);
    need(a.hand==b.hand && a.used==b.nodes.size());
    for(unsigned j=0;j<a.used;++j) {
      need(a.nodes[j].key+base/128==b.nodes[j].tag);
      need(a.nodes[j].mask==b.nodes[j].mask && bool(a.nodes[j].ref)==b.nodes[j].ref);
    }
  }
}
void rejects(std::function<void()> f){bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}need(caught);}
int main(int argc,char **argv) {
  uint64_t compared=0,cases=0;
  for(auto shared:{8u,16u,32u,64u,100u}) {
    unsigned nominal=(shared==8?128:shared==16?112:128-shared)*1024;
    Cache ref(Config{"frozen","CLOCK",128,16,2,shared<32?1000u:1062u,0},nominal);
    R4L1ReadFilter model(1);model.begin_kernel(shared,{{1,base,uint64_t{1}<<32}});
    uint32_t x=11;
    for(unsigned i=0;i<8000;++i) {
      x^=x<<13;x^=x>>17;x^=x<<5;
      uint32_t off=i<8?std::array<uint32_t,8>{0,32,64,96,0,128,160,128}[i]:
          (i%3==0?(x&0xffffffe0u):((x%2400)*32));
      uint64_t misses=ref.misses;ref.access(off);auto r=model.access(0,base+off);
      need((r.outcome==0)==(ref.misses==misses));check(ref,model);++compared;
    }
  }
  R4L1ReadFilter m(2);
  rejects([&]{m.access(0,base);});
  rejects([&]{m.begin_kernel(0,{{1,base,128}});});
  rejects([&]{m.begin_kernel(32,{});});
  rejects([&]{m.begin_kernel(32,{{1,base+32,128}});});
  rejects([&]{m.begin_kernel(32,{{1,base,256},{2,base+128,256}});});
  rejects([&]{m.begin_kernel(32,{{1,base,128},{1,base+256,128}});});
  rejects([&]{m.begin_kernel(32,{{1,base,(uint64_t{1}<<32)+1}});});
  m.begin_kernel(32,{{1,base,128},{2,base+4096,128}});
  need(m.access(0,base).outcome==2 && m.access(0,base+4096).outcome==2);
  need(m.access(0,base).outcome==0 && m.access(0,base+4096).outcome==0);
  need(m.access(1,base).outcome==2);
  rejects([&]{m.access(0,base+128);});rejects([&]{m.access(0,base+4);});rejects([&]{m.access(2,base);});
  m.begin_kernel(32,{{1,base,128}});need(m.access(0,base).outcome==2);
  // One instance must reset tags, sector bits, CLOCK hands and span cache
  // when capacities or allocation generations change in either direction.
  R4L1ReadFilter reused(1);
  for(auto shared:{32u,100u,64u,32u,64u,100u}) {
    Cache ref(Config{"frozen","CLOCK",128,16,2,1062,0},(128-shared)*1024);
    reused.begin_kernel(shared,{{uint64_t(shared),base,uint64_t{1}<<32}});
    check(ref,reused);
    for(unsigned i=0;i<4096;++i) {
      const uint32_t off=((i*997)%2700)*32;
      auto misses=ref.misses;ref.access(off);auto r=reused.access(0,base+off);
      need((r.outcome==0)==(ref.misses==misses));check(ref,reused);++compared;
    }
  }
  reused.begin_kernel(64,{{101,base+4096,128},{102,base+8192,128}});
  need(reused.access(0,base+8192).outcome==2);
  rejects([&]{reused.access(0,base);});
  rejects([&]{reused.access(0,base+4096+128);});
  rejects([&]{reused.access(0,base+8192+128);});
  reused.begin_kernel(100,{{103,base,128}});
  need(reused.access(0,base).outcome==2);
  rejects([&]{reused.access(0,base+8192);});
  if(argc==2) {
    std::ifstream f(argv[1]);unsigned id,nominal,cg,count;std::string path;
    while(f>>id>>nominal>>cg>>count>>path) {
      unsigned shared=128-nominal/1024;
      Cache ref(Config{"frozen","CLOCK",128,16,2,1062,0},nominal);
      R4L1ReadFilter model(1);model.begin_kernel(shared,{{1,base,uint64_t{1}<<32}});
      std::ifstream addresses(path,std::ios::binary);need(bool(addresses));
      for(unsigned i=0;i<count;++i) {
        uint32_t off;addresses.read(reinterpret_cast<char*>(&off),4);need(bool(addresses));
        if(cg)continue;
        // The reference consumes scalar byte addresses; production receives
        // coalesced 32 B sectors. Keep the original address in the reference.
        auto misses=ref.misses;ref.access(off);auto r=model.access(0,base+(off&~31u));
        need((r.outcome==0)==(ref.misses==misses));if(i%1024==0)check(ref,model);++compared;
      }
      check(ref,model);++cases;
    }
    need(f.eof() && cases==384);
  }
  std::cout<<"{\"status\":\"PASS\",\"compared_requests\":"<<compared<<",\"frozen_cases\":"<<cases<<"}\n";
}
