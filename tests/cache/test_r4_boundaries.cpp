#include "../../release/source/tools/r4_l1_read_filter.h"
#include <cassert>
#include <iostream>
using namespace hyfiss_request_trace;
template<class F> void rejects(F f) { bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}assert(rejected); }
int main(){
 R4L1ReadFilter c(2);rejects([&]{c.require_sector(4096,1);});
 c.begin_kernel(8,{{1,4096,36},{2,8192,128}});
 c.require_sector(4128,15);assert(c.access(0,4128,15).outcome==2);assert(c.access(0,4128,1).outcome==0);
 rejects([&]{c.require_sector(4128,16);});rejects([&]{c.require_sector(4128,UINT32_MAX);});
 rejects([&]{c.require_sector(4160,1);});rejects([&]{c.require_sector(4080,1);});
 rejects([&]{c.require_sector(4096,0);});rejects([&]{c.require_sector(UINT64_MAX-31,UINT32_MAX);});
 c.require_sector(8192,UINT32_MAX);rejects([&]{c.require_sector(4128,16);});
 c.begin_kernel(16,{{3,4096,36}});assert(c.access(0,4128,15).outcome==2);assert(c.access(1,4128,15).outcome==2);
 rejects([&]{c.access(2,4128,15);});
 std::cout<<"PASS_REQUESTED_BYTE_BOUNDARY_AND_GENERATION_RESET\n";
}
