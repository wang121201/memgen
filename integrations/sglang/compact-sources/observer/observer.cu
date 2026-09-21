// Metadata-only derivative of the pinned HyFiSS NVBit callback/static-inspection
// implementation in reference/memory_tracer.original.cu. No dynamic insertion,
// channel, memory-address packets, allocation, synchronization, or custom kernel.
// NVBit's required unchanged nvbit_tool.h provides its own loader support code.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "nvbit.h"
#ifndef SG_OBSERVER_CPU_TEST
#include "nvbit_tool.h"
#endif

namespace sgobs {
using std::string;
const uint64_t DEFAULT_CAP = 1024ull << 20;
const uint64_t RESERVED_FINISH = 128ull << 10;
const uint64_t MAX_LAUNCHES = 262144, MAX_FUNCTIONS = 16384;
static thread_local unsigned internal_depth = 0;
static std::atomic<uint64_t> internal_callbacks{0}, internal_dispatches{0};

void need(bool yes, const char *message) { if (!yes) throw std::runtime_error(message); }
uint64_t now() { timespec t{}; need(clock_gettime(CLOCK_MONOTONIC,&t)==0,"clock_gettime"); return uint64_t(t.tv_sec)*1000000000ull+t.tv_nsec; }
uint64_t tid() { return uint64_t(syscall(SYS_gettid)); }
template<class T> uint64_t ptr(T p) { return uint64_t(reinterpret_cast<uintptr_t>(p)); }
string number(uint64_t n) { return std::to_string(n); }
string signed_number(int64_t n) { return std::to_string(n); }
string quote(const string &s) {
  std::ostringstream o; o << '"';
  for (unsigned char c:s) {
    if(c=='"'||c=='\\') o << '\\' << char(c);
    else if(c<32||c>=127) o << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c) << std::dec;
    else o << char(c);
  }
  o << '"'; return o.str();
}
string bounded(const char *p,size_t limit=4096) { need(p!=nullptr,"null metadata text"); size_t n=strnlen(p,limit+1);need(n<=limit,"metadata text cap");return string(p,n); }
string uarray(const std::vector<uint64_t> &v) { string o="[";for(size_t i=0;i<v.size();++i){if(i)o+=',';o+=number(v[i]);}return o+"]"; }
string dims(unsigned x,unsigned y,unsigned z) { return "["+number(x)+","+number(y)+","+number(z)+"]"; }
string sha(const string &s) {
  unsigned char result[EVP_MAX_MD_SIZE];unsigned n=0;
  EVP_MD_CTX *c=EVP_MD_CTX_new();need(c!=nullptr,"sha context");
  bool ok=EVP_DigestInit_ex(c,EVP_sha256(),nullptr)==1 && EVP_DigestUpdate(c,s.data(),s.size())==1 && EVP_DigestFinal_ex(c,result,&n)==1;
  EVP_MD_CTX_free(c);need(ok&&n==32,"sha calculation");
  std::ostringstream out;for(unsigned i=0;i<n;++i)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(result[i]);return out.str();
}
uint64_t start_ticks() {
#ifdef SG_OBSERVER_CPU_TEST
  return 42; // Synthetic fixture epoch; this define is forbidden by build.py.
#else
  std::ifstream f("/proc/self/stat"); string line;std::getline(f,line);
  size_t end=line.rfind(')');need(end!=string::npos,"process stat format");
  std::istringstream in(line.substr(end+2));string item;
  for(unsigned field=3;field<=22;++field){need(bool(in>>item),"process stat fields");if(field==22)return std::stoull(item);}
  throw std::runtime_error("process start time absent");
#endif
}
struct Internal { Internal(){++internal_depth;} ~Internal(){--internal_depth;} };
struct File {
  int fd=-1;uint64_t bytes=0;string name;EVP_MD_CTX *hash=nullptr;
  void open(int dir,const string &n) {
    name=n;fd=openat(dir,n.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0644);
    need(fd>=0,"exclusive output file open");hash=EVP_MD_CTX_new();need(hash&&EVP_DigestInit_ex(hash,EVP_sha256(),nullptr)==1,"file hash init");
  }
  void write(const string &b) {
    size_t offset=0;
    while(offset<b.size()) { ssize_t n=::write(fd,b.data()+offset,b.size()-offset);if(n<0&&errno==EINTR)continue;need(n>0,"metadata file write");
      need(EVP_DigestUpdate(hash,b.data()+offset,size_t(n))==1,"file hash update");offset+=size_t(n);bytes+=uint64_t(n); }
  }
  string digest() const {
    EVP_MD_CTX *c=EVP_MD_CTX_new();need(c&&EVP_MD_CTX_copy_ex(c,hash)==1,"file hash copy");
    unsigned char result[32];unsigned n=0;bool ok=EVP_DigestFinal_ex(c,result,&n)==1;EVP_MD_CTX_free(c);need(ok&&n==32,"file hash finalize");
    std::ostringstream out;for(unsigned i=0;i<n;++i)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(result[i]);return out.str();
  }
};
struct Scope { bool bound=false;uint64_t call=0;int64_t forward=-1;int32_t layer=-1;string phase="unbound",module="unbound",role="unbound"; };
static thread_local Scope scope;
struct Pending { uint64_t id=0;int cbid=0;string api,body; };
static thread_local std::vector<Pending> pending;
struct Count { uint64_t before=0,returned=0,errors=0; };
struct Context { uint64_t id=0;bool closed=false,info=false; };
struct Function { uint64_t id=0,module=0,module_epoch=0,address=0;int module_result=-1;string name,mangled,code_hash;std::vector<uint64_t> args; };
struct State {
  std::mutex lock;
  bool enabled=false,finished=false;
  pid_t owner=0;uint64_t ticks=0,event=0,launches=0,returned=0,failed_launches=0,unsupported=0,graph_nodes=0,unknown_attrs=0;
  uint64_t cap=DEFAULT_CAP,total=0,context_next=1,function_next=1,module_next=1,active_epoch=0,epoch_begun=0,epoch_ended=0,unbound_launches=0;
  uint64_t allocation_next=1,allocation_calls=0,unparsed_allocation_calls=0;
  int dir=-1;string path;
  File launch_file,scope_file,function_file,instruction_file,lifecycle_file,allocation_file;
  std::map<uint64_t,Context> contexts;
  std::map<std::pair<uint64_t,uint64_t>,uint64_t> modules;
  std::map<std::tuple<uint64_t,uint64_t,uint64_t>,Function> functions;
  std::map<string,Count> api_counts;
  std::map<std::pair<uint64_t,uint64_t>,uint64_t> live_allocations;
  std::set<uint64_t> launch_threads,streams,epoch_threads,epoch_streams;
  std::vector<string> errors;
  void error(const string &s) { if(errors.size()<32)errors.push_back(s.substr(0,1024)); }
  string prefix(const string &type) { return "{\"schema\":\"sg_nvbit_observer_event_v1\",\"type\":"+quote(type)+",\"event_ordinal\":"+number(event++)+",\"monotonic_ns\":"+number(now())+",\"pid\":"+number(owner)+",\"start_ticks\":"+number(ticks)+",\"host_thread_id\":"+number(tid())+",\"epoch_id\":"+number(active_epoch); }
  string scope_fields() { return ",\"scope_bound\":"+string(scope.bound?"true":"false")+",\"call_id\":"+number(scope.call)+",\"forward_id\":"+signed_number(scope.forward)+",\"layer_id\":"+signed_number(scope.layer)+",\"phase\":"+quote(scope.phase)+",\"module_scope\":"+quote(scope.module)+",\"role\":"+quote(scope.role); }
  void emit(File &f,const string &b) { need(b.size()<=(1ull<<20),"one metadata row cap");need(total<=cap-RESERVED_FINISH&&b.size()<=cap-RESERVED_FINISH-total,"observer total metadata quota");f.write(b);total+=b.size(); }
  uint64_t context(CUcontext ctx) {
    need(ctx!=nullptr,"launch missing CUDA context");auto &c=contexts[ptr(ctx)];
    if(!c.id){c.id=context_next++;c.closed=false;}
    need(!c.closed,"CUDA context pointer reused without ctx_init epoch");return c.id;
  }
  void initialize() {
    owner=getpid();ticks=start_ticks();const char *root=getenv("SG_NVBIT_OUTPUT_ROOT");if(!root||!*root)return;
    const char *budget=getenv("SG_NVBIT_MAX_BYTES");if(budget&&*budget){string b=bounded(budget,20);need(b.find_first_not_of("0123456789")==string::npos,"decimal metadata quota");cap=std::stoull(b);}
    need(cap>=(2ull<<20)&&cap<=DEFAULT_CAP,"metadata quota outside 2..1024 MiB task envelope");
    char real[PATH_MAX];need(realpath(root,real)!=nullptr&&string(root)==real,"canonical existing output root");
    int parent=::open(real,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);need(parent>=0,"output root open");
    struct stat st{};need(fstat(parent,&st)==0&&S_ISDIR(st.st_mode)&&st.st_uid==getuid(),"output root owner/type");
    string sub="process-"+number(owner)+"-"+number(ticks);need(mkdirat(parent,sub.c_str(),0755)==0,"fresh process output directory");
    dir=openat(parent,sub.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);close(parent);need(dir>=0,"process output open");path=string(real)+"/"+sub;
    launch_file.open(dir,"launch-journal.jsonl");scope_file.open(dir,"scope-journal.jsonl");function_file.open(dir,"functions.jsonl");instruction_file.open(dir,"static-instructions.jsonl");lifecycle_file.open(dir,"lifecycle.jsonl");allocation_file.open(dir,"allocation-journal.jsonl");enabled=true;
    emit(lifecycle_file,prefix("observer_init")+",\"nvbit_version\":"+quote(NVBIT_VERSION)+",\"nvbit_cuda_header_version\":"+number(CUDA_VERSION)+",\"dynamic_instrumentation\":false,\"code_hash_kind\":\"sha256_nvbit_decoded_instruction_rows_v1\",\"module_binary_hash_available\":false,\"static_instruction_rows_emitted\":false,\"max_total_bytes\":"+number(cap)+"}\n");
  }
  void finish(const string &why);
};
State &state() { static State *s=new State;return *s; }
void fatal(const string &message) {
  // No CUDA calls, no remote process signals, no cleanup outside this process.
  State &s=state();s.error(message);
  try{s.finish("observer_error");}catch(...){}
  string line="sg_nvbit_observer fatal: "+message.substr(0,1024)+"\n";::write(STDERR_FILENO,line.data(),line.size());_exit(74);
}
bool dispatch(const string &n) { return (n.rfind("cuLaunch",0)==0&&n.rfind("cuLaunchHostFunc",0)!=0)||n.rfind("cuGraphLaunch",0)==0; }
bool direct(nvbit_api_cuda_t c) { return c==API_CUDA_cuLaunchKernel||c==API_CUDA_cuLaunchKernel_ptsz; }
bool extended(nvbit_api_cuda_t c) { return c==API_CUDA_cuLaunchKernelEx||c==API_CUDA_cuLaunchKernelEx_ptsz; }

string attributes(const CUlaunchConfig &c) {
  State &s=state();need(c.numAttrs<=64&&(!c.numAttrs||c.attrs),"bounded Ex attribute array");string out="[";
  for(unsigned i=0;i<c.numAttrs;++i) {
    const auto &a=c.attrs[i];if(i)out+=',';out+="{\"id\":"+signed_number(int(a.id));bool known=true;
    // Only read the active union member selected by the ID. Never expose pad
    // bytes, which may be uninitialized and are not attribute semantics.
    switch(int(a.id)) {
      case 0:out+=",\"name\":\"IGNORE\"";break;
      case 1:{uint32_t ratio=0;static_assert(sizeof(ratio)==sizeof(a.value.accessPolicyWindow.hitRatio),"float width");std::memcpy(&ratio,&a.value.accessPolicyWindow.hitRatio,sizeof(ratio));out+=",\"name\":\"ACCESS_POLICY_WINDOW\",\"base_u64\":"+number(ptr(a.value.accessPolicyWindow.base_ptr))+",\"bytes\":"+number(a.value.accessPolicyWindow.num_bytes)+",\"hit_ratio_float_bits_u32\":"+number(ratio)+",\"hit_prop\":"+signed_number(int(a.value.accessPolicyWindow.hitProp))+",\"miss_prop\":"+signed_number(int(a.value.accessPolicyWindow.missProp));break;}
      case 2:out+=",\"name\":\"COOPERATIVE\",\"value\":"+signed_number(a.value.cooperative);break;
      case 3:out+=",\"name\":\"SYNCHRONIZATION_POLICY\",\"value\":"+signed_number(int(a.value.syncPolicy));break;
      case 4:out+=",\"name\":\"CLUSTER_DIMENSION\",\"value\":"+dims(a.value.clusterDim.x,a.value.clusterDim.y,a.value.clusterDim.z);break;
      case 5:out+=",\"name\":\"CLUSTER_SCHEDULING_POLICY_PREFERENCE\",\"value\":"+signed_number(int(a.value.clusterSchedulingPolicyPreference));break;
      case 6:out+=",\"name\":\"PROGRAMMATIC_STREAM_SERIALIZATION\",\"value\":"+signed_number(a.value.programmaticStreamSerializationAllowed);break;
      case 7:out+=",\"name\":\"PROGRAMMATIC_EVENT\",\"event_u64\":"+number(ptr(a.value.programmaticEvent.event))+",\"flags\":"+signed_number(a.value.programmaticEvent.flags)+",\"trigger_at_block_start\":"+signed_number(a.value.programmaticEvent.triggerAtBlockStart);break;
      case 8:out+=",\"name\":\"PRIORITY\",\"value\":"+signed_number(a.value.priority);break;
      case 9:out+=",\"name\":\"MEM_SYNC_DOMAIN_MAP\",\"default\":"+number(a.value.memSyncDomainMap.default_)+",\"remote\":"+number(a.value.memSyncDomainMap.remote);break;
      case 10:out+=",\"name\":\"MEM_SYNC_DOMAIN\",\"value\":"+signed_number(int(a.value.memSyncDomain));break;
      case 11:out+=",\"name\":\"PREFERRED_CLUSTER_DIMENSION\",\"value\":"+dims(a.value.preferredClusterDim.x,a.value.preferredClusterDim.y,a.value.preferredClusterDim.z);break;
      case 12:out+=",\"name\":\"LAUNCH_COMPLETION_EVENT\",\"event_u64\":"+number(ptr(a.value.launchCompletionEvent.event))+",\"flags\":"+signed_number(a.value.launchCompletionEvent.flags);break;
      case 13:out+=",\"name\":\"DEVICE_UPDATABLE_KERNEL_NODE\",\"device_updatable\":"+signed_number(a.value.deviceUpdatableKernelNode.deviceUpdatable)+",\"node_handle_u64\":"+number(ptr(a.value.deviceUpdatableKernelNode.devNode));break;
      case 14:out+=",\"name\":\"PREFERRED_SHARED_MEMORY_CARVEOUT\",\"value\":"+number(a.value.sharedMemCarveout);break;
      case 16:out+=",\"name\":\"NVLINK_UTIL_CENTRIC_SCHEDULING\",\"value\":"+number(a.value.nvlinkUtilCentricScheduling);break;
      default:known=false;++s.unknown_attrs;out+=",\"name\":\"UNKNOWN_ATTRIBUTE\",\"value\":null";
    }
    out+=",\"metadata_decoded\":"+string(known?"true":"false")+",\"dynamic_scheduling_qualified\":false}";
  }
  return out+"]";
}

Function describe(CUcontext ctx,CUfunction f) {
  State &s=state();Internal internal;uint64_t cid=s.context(ctx);CUmodule module=nullptr;
  CUresult mr=cuFuncGetModule(&module,f);uint64_t mh=mr==CUDA_SUCCESS?ptr(module):0,epoch=0;
  if(mh){auto &v=s.modules[{cid,mh}];if(!v)v=s.module_next++;epoch=v;}
  const auto key=std::make_tuple(cid,ptr(f),epoch);auto existing=s.functions.find(key);if(existing!=s.functions.end())return existing->second;
  need(s.functions.size()<MAX_FUNCTIONS,"function metadata count cap");
  Function d;d.id=s.function_next++;d.module=mh;d.module_epoch=epoch;d.module_result=int(mr);d.address=nvbit_get_func_addr(ctx,f);
  d.name=bounded(nvbit_get_func_name(ctx,f));d.mangled=bounded(nvbit_get_func_name(ctx,f,true));
  const auto &instructions=nvbit_get_instrs(ctx,f);need(instructions.size()<=100000,"static function instruction cap");
  EVP_MD_CTX *h=EVP_MD_CTX_new();need(h&&EVP_DigestInit_ex(h,EVP_sha256(),nullptr)==1,"static hash init");
  std::map<string,uint64_t> space_counts;uint64_t omitted_shared=0;
  for(auto ins:instructions) {
    string op=bounded(ins->getOpcode(),1024),sass=bounded(ins->getSass(),16384);int space=int(ins->getMemorySpace());
    string space_name=space>=0&&space<12?InstrType::MemorySpaceStr[space]:"UNRECOGNIZED";
    ++space_counts[space_name];if(ins->getMemorySpace()==InstrType::MemorySpace::SHARED)++omitted_shared;
    int nops=ins->getNumOperands();need(nops>=0&&nops<=64,"static operand bound");string operands="[";
    for(int i=0;i<nops;++i){if(i)operands+=',';operands+=signed_number(int(ins->getOperand(i)->type));}operands+=']';
    string canonical="{\"index\":"+number(ins->getIdx())+",\"offset\":"+number(ins->getOffset())+",\"opcode\":"+quote(op)+",\"sass\":"+quote(sass)+",\"memory_space\":"+quote(space_name)+",\"nvbit_size\":"+signed_number(ins->getSize())+",\"operand_types\":"+operands+",\"has_predicate\":"+(ins->hasPred()?"true":"false");
    if(ins->hasPred())canonical+=",\"predicate_number\":"+signed_number(ins->getPredNum())+",\"predicate_negated\":"+(ins->isPredNeg()?"true":"false")+",\"predicate_uniform\":"+(ins->isPredUniform()?"true":"false");
    canonical+="}\n";need(EVP_DigestUpdate(h,canonical.data(),canonical.size())==1,"static hash update");
    // Task-private compact journal: decoded-SASS digest above is retained; static rows omitted.
  }
  unsigned char result[32];unsigned count=0;bool ok=EVP_DigestFinal_ex(h,result,&count)==1;EVP_MD_CTX_free(h);need(ok&&count==32,"static hash finalize");
  std::ostringstream hex;for(unsigned i=0;i<count;++i)hex<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(result[i]);d.code_hash=hex.str();
  bool kernel=nvbit_is_func_kernel(ctx,f);if(kernel){auto sizes=nvbit_get_kernel_argument_sizes(ctx,f);need(sizes.size()<=256,"kernel argument count cap");for(int n:sizes){need(n>=0&&n<=65536,"kernel argument size bound");d.args.push_back(uint64_t(n));}}
  string spaces="{";for(auto i=space_counts.begin();i!=space_counts.end();++i){if(i!=space_counts.begin())spaces+=',';spaces+=quote(i->first)+":"+number(i->second);}spaces+='}';
  s.emit(s.function_file,"{\"schema\":\"sg_nvbit_function_v1\",\"function_id\":"+number(d.id)+",\"context_id\":"+number(cid)+",\"function_handle_u64\":"+number(ptr(f))+",\"function_address_u64\":"+number(d.address)+",\"function_name\":"+quote(d.name)+",\"mangled_name\":"+quote(d.mangled)+",\"module_handle_u64\":"+number(mh)+",\"module_epoch\":"+number(epoch)+",\"cuFuncGetModule_result\":"+signed_number(mr)+",\"module_binary_sha256\":null,\"module_binary_identity_reason\":\"not_dumped_by_metadata_only_observer\",\"code_sha256_kind\":\"sha256_nvbit_decoded_instruction_rows_v1\",\"code_sha256\":"+quote(d.code_hash)+",\"instruction_count\":"+number(instructions.size())+",\"memory_space_instruction_counts\":"+spaces+",\"pure_shared_instructions_outside_old_dynamic_domain\":"+number(omitted_shared)+",\"is_kernel\":"+(kernel?"true":"false")+",\"argument_sizes\":"+uarray(d.args)+",\"parameter_values_captured\":false}\n");
  s.functions[key]=d;return d;
}

string launch_body(CUcontext ctx,nvbit_api_cuda_t cbid,void *params) {
  State &s=state();need(params,"launch params null");CUfunction f=nullptr;CUstream stream=nullptr;
  unsigned gx=0,gy=0,gz=0,bx=0,by=0,bz=0,shared=0;string attrs="[]";
  if(direct(cbid)) {
    const auto &p=*static_cast<const cuLaunchKernel_params*>(params);f=p.f;stream=p.hStream;
    gx=p.gridDimX;gy=p.gridDimY;gz=p.gridDimZ;bx=p.blockDimX;by=p.blockDimY;bz=p.blockDimZ;shared=p.sharedMemBytes;
  } else {
    const CUlaunchConfig *c=nullptr;
    if(cbid==API_CUDA_cuLaunchKernelEx){const auto &p=*static_cast<const cuLaunchKernelEx_params*>(params);f=p.f;c=p.config;}
    else {const auto &p=*static_cast<const cuLaunchKernelEx_ptsz_params*>(params);f=p.f;c=p.config;}
    need(c,"Ex config null");stream=c->hStream;gx=c->gridDimX;gy=c->gridDimY;gz=c->gridDimZ;bx=c->blockDimX;by=c->blockDimY;bz=c->blockDimZ;shared=c->sharedMemBytes;attrs=attributes(*c);
  }
  need(f,"launch function null");Internal internal;Function d;uint64_t cid=s.context(ctx);
  std::vector<uint64_t> ids;std::set<uint64_t> seen;
  const bool inspected=s.active_epoch!=0;
  if(inspected) {
    d=describe(ctx,f);
    auto related=nvbit_get_related_functions(ctx,f);need(related.size()<=MAX_FUNCTIONS,"related function bound");
    for(auto r:related){Function other=describe(ctx,r);if(seen.insert(other.id).second)ids.push_back(other.id);}
    if(seen.insert(d.id).second)ids.push_back(d.id);
  } else {
    // Initialization/warmup remain in the full launch census. Only the marked
    // inference epochs materialize static SASS/function/parameter descriptions.
    // function_id=0 and null code/layout hashes explicitly mean not inspected.
    d.name=bounded(nvbit_get_func_name(ctx,f));
  }
  const string code_json=inspected?quote(d.code_hash):"null";
  const string layout_json=inspected?quote(sha(uarray(d.args))):"null";
  // No nvbit_insert_call appears in this source. Explicitly choose the original
  // function even if a stale external setting requested instrumentation.
  nvbit_enable_instrumented(ctx,f,false);
  int nr=0,ss=0,ls=0,bv=0,pv=0;CUresult rr=cuFuncGetAttribute(&nr,CU_FUNC_ATTRIBUTE_NUM_REGS,f);
  CUresult sr=cuFuncGetAttribute(&ss,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f),lr=cuFuncGetAttribute(&ls,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);
  CUresult br=cuFuncGetAttribute(&bv,CU_FUNC_ATTRIBUTE_BINARY_VERSION,f),pr=cuFuncGetAttribute(&pv,CU_FUNC_ATTRIBUTE_PTX_VERSION,f);
  s.launch_threads.insert(tid());s.streams.insert(ptr(stream));if(s.active_epoch){s.epoch_threads.insert(tid());s.epoch_streams.insert(ptr(stream));}
  return ",\"metadata_supported\":true,\"context_id\":"+number(cid)+",\"context_handle_u64\":"+number(ptr(ctx))+",\"function_id\":"+number(d.id)+",\"related_function_ids\":"+uarray(ids)+",\"function_name\":"+quote(d.name)+",\"code_sha256\":"+code_json+",\"static_inspection_performed\":"+(inspected?"true":"false")+",\"code_sha256_kind\":\"sha256_nvbit_decoded_instruction_rows_v1\",\"module_handle_u64\":"+number(d.module)+",\"module_epoch\":"+number(d.module_epoch)+",\"module_binary_sha256\":null,\"stream_u64\":"+number(ptr(stream))+",\"grid\":"+dims(gx,gy,gz)+",\"block\":"+dims(bx,by,bz)+",\"dynamic_shared_bytes\":"+number(shared)+",\"static_shared_bytes\":"+(sr==CUDA_SUCCESS?signed_number(ss):"null")+",\"registers\":"+(rr==CUDA_SUCCESS?signed_number(nr):"null")+",\"local_bytes_per_thread\":"+(lr==CUDA_SUCCESS?signed_number(ls):"null")+",\"binary_version\":"+(br==CUDA_SUCCESS?signed_number(bv):"null")+",\"ptx_version\":"+(pr==CUDA_SUCCESS?signed_number(pv):"null")+",\"attribute_query_results\":["+signed_number(rr)+","+signed_number(sr)+","+signed_number(lr)+","+signed_number(br)+","+signed_number(pr)+"],\"launch_attributes\":"+attrs+",\"argument_sizes\":"+uarray(d.args)+",\"parameter_layout_sha256\":"+layout_json+",\"parameter_values_captured\":false";
}

bool allocation_api(const string &api) {
  const char *prefixes[]={"cuMemAlloc","cuMemFree","cuMemPool","cuMemCreate","cuMemMap","cuMemUnmap",
    "cuMemAddressReserve","cuMemAddressFree","cuMemRelease","cuMemImportFromShareableHandle","cuMemExportToShareableHandle","cuMemRetainAllocationHandle"};
  for(auto prefix:prefixes)if(api.rfind(prefix,0)==0)return true;
  return false;
}
void allocation_event(CUcontext ctx,bool exit,nvbit_api_cuda_t cbid,const string &api,void *params,CUresult *status) {
  State&s=state();need(params,"allocation callback params");bool known=true,alloc=false,free=false,async=false;
  uint64_t base=0,bytes=0,stream=0,pool=0;CUdeviceptr *result=nullptr;unsigned int *result_v1=nullptr;
  switch(cbid) {
    case API_CUDA_cuMemAlloc_v2:{auto p=static_cast<cuMemAlloc_v2_params*>(params);alloc=true;result=p->dptr;bytes=p->bytesize;break;}
    case API_CUDA_cuMemAlloc:{auto p=static_cast<cuMemAlloc_params*>(params);alloc=true;result_v1=p->dptr;bytes=p->bytesize;break;}
    case API_CUDA_cuMemAllocManaged:{auto p=static_cast<cuMemAllocManaged_params*>(params);alloc=true;result=p->dptr;bytes=p->bytesize;break;}
    case API_CUDA_cuMemFree_v2:{auto p=static_cast<cuMemFree_v2_params*>(params);free=true;base=p->dptr;break;}
    case API_CUDA_cuMemFree:{auto p=static_cast<cuMemFree_params*>(params);free=true;base=p->dptr;break;}
    case API_CUDA_cuMemAllocAsync:{auto p=static_cast<cuMemAllocAsync_params*>(params);alloc=async=true;result=p->dptr;bytes=p->bytesize;stream=ptr(p->hStream);break;}
    case API_CUDA_cuMemAllocAsync_ptsz:{auto p=static_cast<cuMemAllocAsync_ptsz_params*>(params);alloc=async=true;result=p->dptr;bytes=p->bytesize;stream=ptr(p->hStream);break;}
    case API_CUDA_cuMemFreeAsync:{auto p=static_cast<cuMemFreeAsync_params*>(params);free=async=true;base=p->dptr;stream=ptr(p->hStream);break;}
    case API_CUDA_cuMemFreeAsync_ptsz:{auto p=static_cast<cuMemFreeAsync_ptsz_params*>(params);free=async=true;base=p->dptr;stream=ptr(p->hStream);break;}
    case API_CUDA_cuMemAllocFromPoolAsync:{auto p=static_cast<cuMemAllocFromPoolAsync_params*>(params);alloc=async=true;result=p->dptr;bytes=p->bytesize;stream=ptr(p->hStream);pool=ptr(p->pool);break;}
    case API_CUDA_cuMemAllocFromPoolAsync_ptsz:{auto p=static_cast<cuMemAllocFromPoolAsync_ptsz_params*>(params);alloc=async=true;result=p->dptr;bytes=p->bytesize;stream=ptr(p->hStream);pool=ptr(p->pool);break;}
    default:known=false;
  }
  if(!exit){++s.allocation_calls;if(!known)++s.unparsed_allocation_calls;}
  uint64_t generation=0;bool matched=false;
  if(exit&&status&&*status==CUDA_SUCCESS&&known) {
    if(alloc){need(result||result_v1,"successful allocation missing output pointer");base=result?uint64_t(*result):uint64_t(*result_v1);
      generation=s.allocation_next++;s.live_allocations[{ptr(ctx),base}]=generation;}
    else if(free){auto key=std::make_pair(ptr(ctx),base);auto i=s.live_allocations.find(key);if(i!=s.live_allocations.end()){generation=i->second;matched=true;s.live_allocations.erase(i);}}
  }
  string out=s.prefix("allocation_api")+",\"edge\":"+quote(exit?"return":"before")+",\"cuda_api\":"+quote(api)+",\"cbid\":"+signed_number(int(cbid))+",\"context_handle_u64\":"+number(ptr(ctx))+s.scope_fields()+",\"parsed\":"+(known?"true":"false")+",\"action\":"+quote(alloc?"allocate":free?"free":"unparsed_vmm_pool_or_other_memory_API")+",\"base_u64\":"+((free||(alloc&&exit&&status&&*status==CUDA_SUCCESS))?number(base):"null")+",\"bytes\":"+(alloc?number(bytes):"null")+",\"async\":"+(async?"true":"false")+",\"stream_u64\":"+(async?number(stream):"null")+",\"pool_handle_u64\":"+(pool?number(pool):"null")+",\"cuda_status\":"+(exit&&status?signed_number(int(*status)):"null")+",\"allocation_generation\":"+(generation?number(generation):"null")+",\"free_matched_observed_generation\":"+(matched?"true":"false")+",\"generation_semantics\":\"host_API_return_not_device_completion\",\"complete_allocator_or_device_lifetime_qualified\":false}\n";
  s.emit(s.allocation_file,out);
}

void State::finish(const string &why) {
  if(!enabled||finished||getpid()!=owner)return;finished=true;
  uint64_t open_contexts=0;for(const auto &c:contexts)if(!c.second.closed)++open_contexts;
  bool closed=launches==returned&&open_contexts==0&&active_epoch==0&&epoch_begun==epoch_ended;
  bool pass=closed&&errors.empty()&&failed_launches==0&&unsupported==0&&graph_nodes==0&&unknown_attrs==0&&internal_dispatches.load()==0;
  string inventory="[";File *files[]={&launch_file,&scope_file,&function_file,&instruction_file,&lifecycle_file,&allocation_file};
  for(unsigned i=0;i<6;++i){if(i)inventory+=',';File &f=*files[i];inventory+="{\"name\":"+quote(f.name)+",\"bytes\":"+number(f.bytes)+",\"sha256\":"+quote(f.digest())+"}";}inventory+=']';
  string apis="{";for(auto it=api_counts.begin();it!=api_counts.end();++it){if(it!=api_counts.begin())apis+=',';apis+=quote(it->first)+":{\"before\":"+number(it->second.before)+",\"return\":"+number(it->second.returned)+",\"errors\":"+number(it->second.errors)+"}";}apis+='}';
  string es="[";for(size_t i=0;i<errors.size();++i){if(i)es+=',';es+=quote(errors[i]);}es+=']';
  string out="{\"schema\":\"sg_nvbit_observer_finish_v1\",\"status\":"+quote(pass?"PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE":"FAIL_OR_UNSUPPORTED_METADATA_OBSERVER")+",\"reason\":"+quote(why)+",\"pid\":"+number(owner)+",\"start_ticks\":"+number(ticks)+",\"output\":"+quote(path)+",\"launch_before_count\":"+number(launches)+",\"launch_return_count\":"+number(returned)+",\"launch_error_count\":"+number(failed_launches)+",\"unsupported_dispatch_count\":"+number(unsupported)+",\"graph_node_callback_count\":"+number(graph_nodes)+",\"unknown_launch_attribute_count\":"+number(unknown_attrs)+",\"unbound_launch_count\":"+number(unbound_launches)+",\"context_count\":"+number(contexts.size())+",\"open_context_count\":"+number(open_contexts)+",\"epoch_begin_count\":"+number(epoch_begun)+",\"epoch_end_count\":"+number(epoch_ended)+",\"active_epoch\":"+number(active_epoch)+",\"function_count\":"+number(functions.size())+",\"observed_launch_thread_count\":"+number(launch_threads.size())+",\"observed_stream_count\":"+number(streams.size())+",\"marked_epoch_thread_count\":"+number(epoch_threads.size())+",\"marked_epoch_stream_count\":"+number(epoch_streams.size())+",\"internal_inspection_callback_count\":"+number(internal_callbacks.load())+",\"internal_inspection_dispatch_count\":"+number(internal_dispatches.load())+",\"metadata_bytes_before_finish\":"+number(total)+",\"max_total_bytes\":"+number(cap)+",\"api_counts\":"+apis+",\"errors\":"+es+",\"files\":"+inventory+",\"dynamic_instrumentation\":false,\"memory_addresses_captured\":false,\"actual_sm_placement_captured\":false,\"kernel_argument_values_captured\":false,\"module_binary_hash_available\":false,\"static_instruction_rows_emitted\":false,\"static_code_hash_is_decoded_sass_not_cubin\":true,\"device_synchronization_inserted\":false,\"gpu_memory_allocated_by_observer\":false,\"native_callback_completeness_independently_proven\":false,\"template_or_gtsim_admission\":false}\n";
  out.pop_back();out.pop_back();out+=",\"allocation_API_calls\":"+number(allocation_calls)+",\"unparsed_allocation_API_calls\":"+number(unparsed_allocation_calls)+",\"complete_allocator_coverage\":false}\n";
  need(out.size()<=RESERVED_FINISH&&total+out.size()<=cap,"finish reserve cap");File finish_file;finish_file.open(dir,"finish.json");finish_file.write(out);::close(finish_file.fd);
  for(File *f:files){need(::close(f->fd)==0,"metadata close failure");f->fd=-1;}::close(dir);dir=-1;
}
} // namespace sgobs

extern "C" int sg_nvbit_observer_set_scope(uint64_t call_id,int64_t forward_id,int32_t layer_id,const char *phase,const char *module_scope,const char *role) {
  using namespace sgobs;State &s=state();std::lock_guard<std::mutex> guard(s.lock);
  try{if(!s.enabled||s.finished||getpid()!=s.owner)return 0;need(pending.empty(),"scope update inside pending CUDA launch");need(forward_id>=-1&&layer_id>=-1&&layer_id<1024,"scope ordinal domain");
    Scope next;next.bound=true;next.call=call_id;next.forward=forward_id;next.layer=layer_id;next.phase=bounded(phase,128);next.module=bounded(module_scope,2048);next.role=bounded(role,64);
    need(!next.phase.empty()&&!next.module.empty()&&(next.role=="warmup"||next.role=="measurement"||next.role=="initialization"),"scope strings");
    scope=next;s.emit(s.scope_file,s.prefix("scope_set")+s.scope_fields()+"}\n");return 1;
  }catch(const std::exception &e){s.error(e.what());return 0;}
}
extern "C" int sg_nvbit_observer_clear_scope() {
  using namespace sgobs;State &s=state();std::lock_guard<std::mutex> guard(s.lock);
  try{if(!s.enabled||s.finished||getpid()!=s.owner)return 0;need(pending.empty(),"scope clear inside pending launch");scope=Scope{};s.emit(s.scope_file,s.prefix("scope_clear")+s.scope_fields()+"}\n");return 1;}catch(const std::exception&e){s.error(e.what());return 0;}
}
extern "C" int sg_nvbit_observer_begin_epoch(uint64_t epoch) {
  using namespace sgobs;State &s=state();std::lock_guard<std::mutex> guard(s.lock);
  try{if(!s.enabled||s.finished||getpid()!=s.owner)return 0;need(epoch>0&&!s.active_epoch&&pending.empty(),"epoch begin state");s.active_epoch=epoch;++s.epoch_begun;s.emit(s.scope_file,s.prefix("epoch_begin")+s.scope_fields()+"}\n");return 1;}catch(const std::exception&e){s.error(e.what());return 0;}
}
extern "C" int sg_nvbit_observer_end_epoch(uint64_t epoch) {
  using namespace sgobs;State &s=state();std::lock_guard<std::mutex> guard(s.lock);
  try{if(!s.enabled||s.finished||getpid()!=s.owner)return 0;need(epoch>0&&s.active_epoch==epoch&&pending.empty(),"epoch end state");s.emit(s.scope_file,s.prefix("epoch_end")+s.scope_fields()+"}\n");s.active_epoch=0;++s.epoch_ended;return 1;}catch(const std::exception&e){s.error(e.what());return 0;}
}
extern "C" int sg_nvbit_observer_get_status() { using namespace sgobs;State&s=state();std::lock_guard<std::mutex> guard(s.lock);return s.enabled&&!s.finished&&s.errors.empty()&&getpid()==s.owner?1:0; }

void nvbit_at_init() { try{sgobs::state().initialize();}catch(const std::exception&e){sgobs::fatal(e.what());} }
void nvbit_at_ctx_init(CUcontext ctx) {
  using namespace sgobs;State&s=state();if(!s.enabled||getpid()!=s.owner)return;std::lock_guard<std::mutex> lock(s.lock);
  try{auto &c=s.contexts[ptr(ctx)];if(c.id&&!c.closed){s.error("duplicate context init");return;}c=Context{};c.id=s.context_next++;s.emit(s.lifecycle_file,s.prefix("context_begin")+",\"context_id\":"+number(c.id)+",\"context_handle_u64\":"+number(ptr(ctx))+"}\n");}catch(const std::exception&e){fatal(e.what());}
}
void nvbit_at_ctx_term(CUcontext ctx) {
  using namespace sgobs;State&s=state();if(!s.enabled||s.finished||getpid()!=s.owner)return;std::lock_guard<std::mutex> lock(s.lock);
  try{auto it=s.contexts.find(ptr(ctx));need(it!=s.contexts.end()&&!it->second.closed,"unmatched context termination");it->second.closed=true;s.emit(s.lifecycle_file,s.prefix("context_end")+",\"context_id\":"+number(it->second.id)+",\"context_handle_u64\":"+number(ptr(ctx))+"}\n");}catch(const std::exception&e){fatal(e.what());}
}
void nvbit_at_term() { using namespace sgobs;State&s=state();if(!s.enabled||getpid()!=s.owner)return;std::lock_guard<std::mutex> lock(s.lock);try{s.finish("nvbit_at_term");}catch(const std::exception&e){fatal(e.what());} }

void nvbit_at_cuda_event(CUcontext ctx,int is_exit,nvbit_api_cuda_t cbid,const char *name,void *params,CUresult *status) {
  using namespace sgobs;State &s=state();if(!s.enabled||s.finished)return;
  if(getpid()!=s.owner)fatal("forked child attempted CUDA with inherited observer state");
  string api;
  try{api=bounded(name,256);}catch(const std::exception&e){fatal(e.what());}
  if(internal_depth){++internal_callbacks;if(!is_exit&&dispatch(api))++internal_dispatches;return;}
  std::lock_guard<std::mutex> lock(s.lock);
  try {
    need(s.api_counts.count(api)||s.api_counts.size()<1024,"CUDA API name census cap");auto &count=s.api_counts[api];
    if(!is_exit)++count.before;else {++count.returned;if(!status||*status!=CUDA_SUCCESS)++count.errors;}
    if(allocation_api(api))allocation_event(ctx,bool(is_exit),cbid,api,params,status);
    // Observe module unload without reading module images or library filenames.
    if(is_exit&&status&&*status==CUDA_SUCCESS&&cbid==API_CUDA_cuModuleUnload&&params) {
      auto p=static_cast<const cuModuleUnload_params*>(params);auto ci=s.contexts.find(ptr(ctx));
      if(ci!=s.contexts.end())s.modules.erase({ci->second.id,ptr(p->hmod)});
      s.emit(s.lifecycle_file,s.prefix("module_unload")+",\"module_handle_u64\":"+number(ptr(p->hmod))+",\"context_handle_u64\":"+number(ptr(ctx))+"}\n");
    }
    if(!dispatch(api))return;
    if(!is_exit) {
      need(s.launches<MAX_LAUNCHES&&pending.size()<8,"launch census or nesting cap");Pending p;p.id=s.launches++;p.cbid=int(cbid);p.api=api;
      string specific;
      if(direct(cbid)||extended(cbid))specific=launch_body(ctx,cbid,params);
      else {++s.unsupported;specific=",\"metadata_supported\":false,\"unsupported_reason\":\"unparsed_kernel_dispatch_API_not_silently_omitted\",\"context_handle_u64\":"+number(ptr(ctx))+",\"geometry\":null";}
      if(!scope.bound)++s.unbound_launches;
      p.body=",\"launch_id\":"+number(p.id)+",\"cuda_api\":"+quote(api)+",\"cbid\":"+signed_number(int(cbid))+s.scope_fields()+specific;
      s.emit(s.launch_file,s.prefix("launch")+",\"edge\":\"before\""+p.body+"}\n");pending.push_back(p);
    } else {
      need(!pending.empty()&&pending.back().cbid==int(cbid)&&pending.back().api==api,"launch return lacks exact same-thread entry");Pending p=pending.back();pending.pop_back();++s.returned;
      if(!status||*status!=CUDA_SUCCESS)++s.failed_launches;
      s.emit(s.launch_file,s.prefix("launch")+",\"edge\":\"return\""+p.body+",\"cuda_status\":"+(status?signed_number(int(*status)):"null")+"}\n");
    }
  }catch(const std::exception&e){fatal(e.what());}
}

void nvbit_at_graph_node_launch(CUcontext ctx,CUfunction function,CUstream stream,uint64_t handle) {
  using namespace sgobs;State&s=state();if(!s.enabled||s.finished||getpid()!=s.owner)return;std::lock_guard<std::mutex> lock(s.lock);
  try{++s.graph_nodes;s.emit(s.lifecycle_file,s.prefix("unsupported_graph_node")+s.scope_fields()+",\"context_handle_u64\":"+number(ptr(ctx))+",\"function_handle_u64\":"+number(ptr(function))+",\"stream_u64\":"+number(ptr(stream))+",\"launch_handle\":"+number(handle)+",\"metadata_supported\":false}\n");}catch(const std::exception&e){fatal(e.what());}
}
