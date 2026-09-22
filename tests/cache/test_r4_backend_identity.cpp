#define HYFISS_REQUEST_TRACE_NO_MAIN
#include "../../release/source/tools/hyfiss_request_trace_generator_stream_r4_semantic.cc"
using namespace hyfiss_request_trace;
int main(int argc,char **argv) {try {
 if(argc!=7)throw std::runtime_error("config out model shared opcode invalid");
 BackendOptions o;o.hw_config=argv[1];o.output_dir=argv[2];o.output_format="summary";o.order="timestamp";
 o.r4_l1_read_filter=true;if(std::string(argv[3])!="default")o.r4_model_id=argv[3];
 o.preserve_l1=false;o.preserve_l2=false;o.l2_dirty_drain=false;
 o.l1_fill_latency_set=o.l2_fill_latency_set=true;o.l1_fill_latency=o.l2_fill_latency=0;
 bool kernel=false,inst=false;unsigned shared=std::stoul(argv[4]);std::string opcode=argv[5];
 return run_from_sm_trace_source(o,[&](KernelTraceRef &ref){
  if(kernel)return false;kernel=true;ref=KernelTraceRef{};ref.kernel_id=1;ref.kernel_name="requested-byte-fixture";
  ref.r4_shared_kib=shared;ref.r4_allocations={{1,4096,36}};
  ref.next_ordered_inst=[&](OrderedMemoryInst &v){if(inst)return false;inst=true;
   v=OrderedMemoryInst{};v.opcode=opcode;v.mask=1;v.timestamp=1;v.sequence=1;v.pc=16;v.sm_id=0;
   v.addr={std::string(argv[6])=="invalid"?4132ull:4128ull};return true;};return true;
 });
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
