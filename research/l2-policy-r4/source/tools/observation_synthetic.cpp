#define HYFISS_REQUEST_TRACE_NO_MAIN
#include "hyfiss_request_trace_generator_stream_r4_semantic.cc"

int main(int argc,char **argv) {
  if(argc!=6) return 2;
  using namespace hyfiss_request_trace;
  BackendOptions opt; opt.hw_config=argv[1]; opt.output_dir=argv[2];
  opt.semantic_file=argv[3]; opt.semantic_summary=!opt.semantic_file.empty(); opt.observe_cache=true;
  opt.order="timestamp"; opt.l2_dirty_drain=false; opt.preserve_l2=true;
  opt.l1_fill_latency_set=true;opt.l2_fill_latency_set=true;
  opt.include_local=std::string(argv[5])=="true";
  std::string test=argv[4];
  std::vector<std::vector<OrderedMemoryInst>> kernels(2);
  auto add=[&](int k,const char *opcode,uint64_t addr,unsigned lanes=1) {
    OrderedMemoryInst i;i.opcode=opcode;i.pc=16*(kernels[k].size()+1);
    i.mask=lanes==32?UINT32_MAX:((1u<<lanes)-1);i.timestamp=kernels[k].size()+1;
    for(unsigned l=0;l<lanes;++l) i.addr.push_back(addr+4*l);
    kernels[k].push_back(i);
  };
  if(test=="dirty") {
    add(0,"STG.E.32",0x1000); add(0,"STG.E.32",0x1000);
    add(0,"LDG.E.32",0x1000); add(0,"STG.E.32",0x1040,8);
    add(0,"LDG.E.32",0x1080);
    add(1,"LDG.E.32",0x1100); add(1,"LDG.E.32",0x1180);
    add(1,"STG.E.32",0x1120); add(1,"ATOM.E.ADD.32",0x1140);
  } else if(test=="partial-eviction") {
    add(0,"STG.E.32",0x1000); add(0,"STG.E.32",0x1040,8);
    add(1,"LDG.E.32",0x1080,8);
  } else if(test=="local") {
    add(0,"STL.32",0x100);add(0,"LDL.32",0x100);
  } else if(test=="clients") {
    add(0,"STS.32",0x100);add(0,"LDS.32",0x100);add(0,"TEX.32",0x100);
    add(0,"TLD.32",0x100);add(0,"SULD.32",0x100);add(0,"SUST.32",0x100);
    add(0,"LDGSTS.E.32",0x1000);add(0,"RED.E.ADD.32",0x1000);
  } else if(test=="shared-atomic") add(0,"ATOMS.ADD.32",0x100);
  else if(test=="ldgsts-two-refs") {add(0,"LDGSTS.E.32",0x1000);kernels[0][0].addr.push_back(0x100);}
  else if(test=="zero-predicate") {add(0,"STG.E.32",0x1000);kernels[0][0].mask=0;}
  else return 2;
  size_t k=0,pos=0,current=0;
  return run_from_sm_trace_source(opt,[&](KernelTraceRef &ref) {
    if(k==kernels.size()) return false;
    current=k++;pos=0;ref={};ref.kernel_id=current+1;ref.kernel_name=test;
    ref.llm_phase=current?"decode_step_0":"prefill";
    ref.next_ordered_inst=[&](OrderedMemoryInst &out) {
      if(pos==kernels[current].size())return false;
      out=kernels[current][pos++];return true;
    };return true;
  });
}
