// Compare explicit configuration with the prior built-in r4 parameters.
#include "../../release/source/tools/r4_l1_read_filter.h"
#include <iostream>
int main(int argc,char **argv) {try {
  if(argc!=2&&argc!=3)throw std::runtime_error("config [sensitivity-config] required");
  auto p=hyfiss_request_trace::HardwareProfile::load(argv[1]);
  if(!p)throw std::runtime_error("unified config required");
  uint64_t count=0,state=7;const uint64_t base=0x123456780000ULL;
  for(unsigned shared:{8,16,32,64,100}) {
    hyfiss_request_trace::R4L1ReadFilter old(3);
    hyfiss_request_trace::R4L1ReadFilter current(3,p);
    for(unsigned kernel=0;kernel<2;++kernel) {
      old.begin_kernel(shared,{{1,base,1<<24},{2,base+(1<<25),1<<24}});
      current.begin_kernel(shared,{{1,base,1<<24},{2,base+(1<<25),1<<24}});
      for(unsigned i=0;i<120000;++i) {
        state^=state<<13;state^=state>>7;state^=state<<17;
        const unsigned sm=i%3;
        uint64_t address=base+((state>>22)%2)*(1<<25)+((state%6000)*32);
        uint32_t mask=i%5?UINT32_MAX:0x11;
        auto a=old.access(sm,address,mask);auto b=current.access(sm,address,mask);++count;
        if(a.outcome!=b.outcome||a.before!=b.before||a.after!=b.after||a.victim!=b.victim||
           a.victim_addr!=b.victim_addr||a.victim_valid!=b.victim_valid)
          throw std::runtime_error("access state mismatch");
        if(i%997==0)for(unsigned s=0;s<16;++s) {
          const auto &x=old.inspect(sm,s);const auto &y=current.inspect(sm,s);
          if(x.hand!=y.hand||x.nodes.size()!=y.nodes.size()||x.lookup!=y.lookup)
            throw std::runtime_error("replacement state mismatch");
          for(size_t j=0;j<x.nodes.size();++j)
            if(x.nodes[j].tag!=y.nodes[j].tag||x.nodes[j].mask!=y.nodes[j].mask||x.nodes[j].ref!=y.nodes[j].ref)
              throw std::runtime_error("node state mismatch");
        }
      }
    }
  }
  if(argc==3) {
    auto changed=hyfiss_request_trace::HardwareProfile::load(argv[2]);
    hyfiss_request_trace::R4L1ReadFilter a(1,p),b(1,changed);
    a.begin_kernel(32,{{1,base,1<<24}});b.begin_kernel(32,{{1,base,1<<24}});
    unsigned misses_a=0,misses_b=0;uint64_t rng=23;
    for(unsigned i=0;i<100000;++i) {
      rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
      auto addr=base+(rng%5000)*32;
      misses_a+=a.access(0,addr).outcome!=0;misses_b+=b.access(0,addr).outcome!=0;
    }
    if(misses_a==misses_b)throw std::runtime_error("isolated L1 parameter change had no effect");
  }
  std::cout<<"{\"status\":\"PASS_EXPLICIT_AND_BUILTIN_R4_EQUIVALENCE\",\"requests\":"<<count<<"}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
