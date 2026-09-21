#pragma once
// Included in the derived observer TU after sgobs definitions. Only this task's
// derivative installs dynamic calls. The frozen metadata observer is untouched.
#include <array>
#include <thread>
#include <condition_variable>
#include <unordered_set>
#include <cuda_runtime.h>
#include "utils/channel.hpp"
#include "utils/utils.h"
#include "packet_pipe.h"
#include "flush_ledger.h"

__device__ __managed__ uint32_t sg_sample_cta_count=0;
static constexpr uint32_t SG_MAX_SELECTED_CTAS=8192;
__device__ __managed__ uint64_t sg_sample_cta_ids[SG_MAX_SELECTED_CTAS];
__device__ __managed__ uint32_t sg_sample_entry_function=0;
__device__ __managed__ uint32_t sg_entry_calls[SG_MAX_SELECTED_CTAS*32],sg_entry_masks[SG_MAX_SELECTED_CTAS*32];
static __managed__ ChannelDev channel_dev;
static __managed__ unsigned long long pushed_record_count=0;
static ChannelHost channel_host;
static bool transfer_sm89=false;
static CUcontext capture_context=nullptr;

namespace sgsample {
using std::string;
struct Plan {
  uint64_t epoch,ordinal;int layer;string phase,module,api,name,code,role;
  std::array<uint64_t,3> grid,block;uint64_t shared;string attrs;
  std::vector<uint64_t> fit,hold;
};
}
#include "sample_plan.h"

namespace sgsample {
static sgpipe::Writer pipe;
static const uint64_t CHANNEL_BYTES=4ull<<20;
static std::atomic<bool> receiving{false},running{false};
static std::atomic<unsigned> sentinels_here{0};
static std::thread receiver;
static std::mutex receiver_wake_mutex;static std::condition_variable receiver_wake;
static bool initialized=false,finalized=false;
static size_t plan_cursor=0;
static uint64_t total_received=0,total_selected=0,selected_kernels=0,flushes=0;
static std::vector<sgflush::Row> flush_ledger;
static std::map<uint64_t,uint64_t> epoch_ordinals;
static const Plan *active=nullptr;
static uint64_t pushed_before=0,received_here=0,entry_function=0;
static unsigned sms_count=0;
static uint64_t measured_thread=0,measured_stream=0;static bool measured_bound=false;
static std::set<uint64_t> selected_ctas,seen_ctas;
static std::map<uint64_t,int> cta_sms;
static std::map<std::pair<uint64_t,uint32_t>,uint64_t> clocks;
static string active_key,final_stream_sha;
static std::map<uint32_t,std::map<string,uint64_t>> omitted_spaces;
static std::map<std::pair<uint32_t,int>,string> static_records;
static std::map<std::pair<uint32_t,int>,std::tuple<int,int,unsigned,unsigned>> static_shapes;
static std::set<uint32_t> current_functions;
void require(bool b,const char*s){sgobs::need(b,s);}
string n(uint64_t v){return sgobs::number(v);}
string q(const string&v){return sgobs::quote(v);}
string list(const std::vector<uint64_t>&v){return sgobs::uarray(v);}
string setlist(const std::set<uint64_t>&v){return list(std::vector<uint64_t>(v.begin(),v.end()));}
string omissions(){std::map<string,uint64_t> counts;for(auto fid:current_functions)for(auto x:omitted_spaces[fid])counts[x.first]+=x.second;
  string out="{";for(auto i=counts.begin();i!=counts.end();++i){if(i!=counts.begin())out+=',';out+=q(i->first)+":"+n(i->second);}return out+"}";}
void cuda_ok(cudaError_t e,const char*why){require(e==cudaSuccess,why);}
void packet(const fast_mem_access_t&p){
  require(active&&p.capture_seq==0,"sample packet outside selected kernel");
  require(total_received<MAX_RECEIVED_RECORDS,"sample selected-CTA packet quota");
  const auto&g=active->grid;uint64_t threads=active->block[0]*active->block[1]*active->block[2];
  uint32_t mask=p.active_mask&p.predicate_mask;
  require(mask&&p.mref_id>=1&&p.mref_id<=2&&p.sm_id>=0&&unsigned(p.sm_id)<sms_count&&p.cta_warp_id<(threads+31)/32,"packet mask/ref/SM/warp");
  require(p.cta_id_x>=0&&uint64_t(p.cta_id_x)<g[0]&&p.cta_id_y>=0&&uint64_t(p.cta_id_y)<g[1]&&p.cta_id_z>=0&&uint64_t(p.cta_id_z)<g[2],"packet CTA out of source grid");
  uint64_t cta=p.cta_id_x+g[0]*(p.cta_id_y+g[1]*p.cta_id_z);require(selected_ctas.count(cta),"device gate leaked unselected CTA");
  require(current_functions.count(p.function_id),"packet function outside selected launch");
  auto it=static_shapes.find({p.function_id,p.pc});require(it!=static_shapes.end(),"packet unknown static PC");
  auto shape=it->second;require(p.opcode_id==std::get<0>(shape)&&p.mref_id==std::get<1>(shape)&&p.transfer_width==std::get<2>(shape)&&p.transfer_policy==std::get<3>(shape),"packet differs from static form");
  for(int r=0;r<2;++r){uint32_t a=p.global_mask[r],b=p.local_mask[r],c=p.shared_mask[r];
    require(r<p.mref_id?((a|b|c)==mask&&!(a&b)&&!(a&c)&&!(b&c)):!(a|b|c),"unknown/overlapping packet space");
    const uint64_t*addresses=r?p.mem_addrs2:p.mem_addrs1;
    for(unsigned lane=0;lane<32;++lane)if(r>=p.mref_id||!(mask&(1u<<lane)))require(addresses[lane]==0,"unused lane/reference address");}
  if(p.transfer_policy==2)require(p.mref_id==1&&(p.transfer_width==8||p.transfer_width==16)&&p.global_mask[0]==mask&&!(p.source_read_mask&~mask),"LDG source control");
  else if(p.transfer_width)require(p.mref_id==2&&(p.transfer_width==4||p.transfer_width==8||p.transfer_width==16)&&p.transfer_policy<=1&&(!p.transfer_policy||p.transfer_width==16)&&p.shared_mask[0]==mask&&p.global_mask[1]==mask&&!(p.source_read_mask&~mask),"LDGSTS reference roles/source mask");
  else require(!p.transfer_policy&&!p.source_read_mask,"unknown source control");
  auto sm=cta_sms.find(cta);require(sm==cta_sms.end()||sm->second==p.sm_id,"CTA actual SM conflict");cta_sms[cta]=p.sm_id;seen_ctas.insert(cta);
  auto key=std::make_pair(cta,p.cta_warp_id);auto clock=clocks.find(key);require(clock==clocks.end()||clock->second<=p.curr_clk,"same-warp clock regression");clocks[key]=p.curr_clk;
  pipe.packet(received_here,received_here,p);++received_here;++total_received;++total_selected;
}
void recv_loop(){
  sgobs::Internal internal;std::vector<char> buffer(CHANNEL_BYTES);
  try{while(running.load(std::memory_order_acquire)){
    if(!receiving.load(std::memory_order_acquire)){std::unique_lock<std::mutex> wait(receiver_wake_mutex);receiver_wake.wait(wait,[]{return !running.load(std::memory_order_acquire)||receiving.load(std::memory_order_acquire);});continue;}
    uint32_t bytes=channel_host.recv(buffer.data(),buffer.size());
    if(!bytes){std::this_thread::yield();continue;}
    require(bytes%sizeof(fast_mem_access_t)==0,"misaligned device channel packet batch");
    for(uint32_t offset=0;offset<bytes;offset+=sizeof(fast_mem_access_t)){
      fast_mem_access_t p;std::memcpy(&p,buffer.data()+offset,sizeof(p));
      if(p.cta_id_x==-1){require(active&&p.capture_seq==UINT64_MAX&&offset+sizeof(p)==bytes&&sentinels_here.fetch_add(1)==0,"invalid/nonterminal/duplicate flush sentinel");receiving.store(false,std::memory_order_release);break;}
      packet(p);
    }
  }}catch(const std::exception&e){string message="sg_native_sampler receiver FAIL: "+string(e.what())+"\n";::write(2,message.data(),message.size());_exit(74);}
}
void initialize(CUcontext ctx){
  require(!initialized,"dynamic sampler supports one CUDA context");capture_context=ctx;
  const char*f=getenv("SG_SAMPLE_PIPE_FD");require(f&&*f,"explicit inherited sample pipe required");
  string value(f);require(value.find_first_not_of("0123456789")==string::npos&&value.size()<10,"sample pipe fd integer");
  pipe.open(std::stoi(value),MAX_WIRE_BYTES);
  pipe.json(sgpipe::HELLO,"{\"schema\":\"SG_PACKET_STREAM_HELLO_V1\",\"packet_bytes\":616,\"device_packet_abi_unchanged\":true,\"device_cta_filter\":true,\"max_wire_bytes\":"+n(MAX_WIRE_BYTES)+",\"plan_sha256\":"+q(PLAN_SHA256)+",\"source_observer_receipt_sha256\":"+q(OBSERVER_RECEIPT_SHA256)+",\"whole_kernel_dynamic_census\":false}");
  CUcontext current=nullptr;CUdevice device;int major=0,minor=0,count=0;
  require(cuCtxGetCurrent(&current)==CUDA_SUCCESS&&current==ctx&&cuCtxGetDevice(&device)==CUDA_SUCCESS,"sample actual context");
  require(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device)==CUDA_SUCCESS&&cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device)==CUDA_SUCCESS&&cuDeviceGetAttribute(&count,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,device)==CUDA_SUCCESS,"sample device attributes");
  require(major==8&&minor==9&&count==48,"candidate requires measured sm89/48SM device");transfer_sm89=true;sms_count=count;
  channel_host.init(0,CHANNEL_BYTES,&channel_dev,nullptr);initialized=true;running.store(true);receiver=std::thread(recv_loop);
}
void register_instruction(uint32_t fid,Instr*ins,int opcode,int refs,unsigned tw,unsigned policy,unsigned load_width){
  auto key=std::make_pair(fid,int(ins->getOffset()));require(!static_records.count(key),"duplicate instrumented PC");
  unsigned packet_width=tw?tw:load_width,packet_policy=load_width?2u:policy;
  string kind=tw?"validated_sm89_ldgsts":load_width?"ldg_predicate_candidate":"none";
  require(ins->getSize()>0&&ins->getSize()<=128&&(ins->isLoad()||ins->isStore()||tw),"missing NVBit memory width/direction");
  string row="{\"schema\":\"SG_SAMPLE_STATIC_MEMORY_V1\",\"function_id\":"+n(fid)+",\"pc\":"+n(ins->getOffset())+",\"opcode_id\":"+n(opcode)+",\"opcode\":"+q(ins->getOpcode())+",\"sass\":"+q(ins->getSass())+",\"width\":"+n(ins->getSize())+",\"is_load\":"+(ins->isLoad()?"true":"false")+",\"is_store\":"+(ins->isStore()?"true":"false")+",\"ref_count\":"+n(refs)+",\"transfer_width\":"+n(packet_width)+",\"transfer_policy\":"+n(packet_policy)+",\"source_control_kind\":"+q(kind)+",\"nvbit_version\":"+q(NVBIT_VERSION)+",\"target_sm\":89}";
  static_shapes[key]=std::make_tuple(opcode,refs,packet_width,packet_policy);static_records[key]=row;pipe.json(sgpipe::STATIC,row);
}
} // namespace sgsample

// This is the frozen legacy classifier/insertion sequence with only host
// identity/output adaptation. No new opcode forms are admitted here.
#include "stable_instrumentation.inc"

__global__ void sg_flush_channel(){fast_mem_access_t p{};p.capture_seq=UINT64_MAX;p.cta_id_x=-1;channel_dev.push(&p,sizeof(p));channel_dev.flush();}
namespace sgsample {
bool prepare(CUcontext ctx,CUfunction f,nvbit_api_cuda_t cbid,const string&attrs,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,uint64_t stream_u64,const string&code,const string&name,uint64_t fid,const std::vector<uint64_t>&ids){
  auto&s=sgobs::state();if(!s.active_epoch)return false;
  require(sgobs::scope.bound&&(sgobs::scope.role=="measurement"||sgobs::scope.role=="warmup"),"sample epoch needs declared capture scope");
  if(!measured_bound){measured_thread=sgobs::tid();measured_stream=stream_u64;measured_bound=true;}
  require(measured_thread==sgobs::tid()&&measured_stream==stream_u64,"sampling contract requires one measured host thread/stream");
  uint64_t ordinal=epoch_ordinals[s.active_epoch]++;require(plan_cursor<plans.size(),"measurement has extra native launch");
  const Plan&p=plans[plan_cursor++];string api=cbid==API_CUDA_cuLaunchKernel?"cuLaunchKernel":cbid==API_CUDA_cuLaunchKernel_ptsz?"cuLaunchKernel_ptsz":cbid==API_CUDA_cuLaunchKernelEx?"cuLaunchKernelEx":"cuLaunchKernelEx_ptsz";
  require(p.role==sgobs::scope.role&&p.epoch==s.active_epoch&&p.ordinal==ordinal&&p.layer==sgobs::scope.layer&&p.phase==sgobs::scope.phase&&p.module==sgobs::scope.module&&p.api==api&&p.code==code&&p.name==name,"actual native launch/scope/code differs from sealed plan");
  require(p.grid==std::array<uint64_t,3>{{gx,gy,gz}}&&p.block==std::array<uint64_t,3>{{bx,by,bz}}&&p.shared==shared&&p.attrs==attrs,"actual geometry/shared/Ex attrs differ from sealed plan");
  if(p.fit.empty())return false;
  require(initialized&&ctx==capture_context&&!active&&!receiving.load()&&selected_kernels<MAX_SELECTED_KERNELS,"selected kernel state/context/cap");
  // Synchronization is an explicit sampling intervention. We do not claim the
  // resulting inter-kernel timing/PDL overlap equals uninstrumented inference.
  cuda_ok(cudaDeviceSynchronize(),"sync before selected native kernel");
  selected_ctas.clear();selected_ctas.insert(p.fit.begin(),p.fit.end());selected_ctas.insert(p.hold.begin(),p.hold.end());
  require(!selected_ctas.empty()&&selected_ctas.size()<=SG_MAX_SELECTED_CTAS,"selected CTA device array capacity");
  sg_sample_cta_count=selected_ctas.size();size_t i=0;for(auto c:selected_ctas)sg_sample_cta_ids[i++]=c;
  require(fid<=UINT32_MAX,"entry function id overflow");sg_sample_entry_function=uint32_t(fid);
  std::fill(sg_entry_calls,sg_entry_calls+selected_ctas.size()*32,0);
  std::fill(sg_entry_masks,sg_entry_masks+selected_ctas.size()*32,0);
  current_functions.clear();for(auto id:ids){require(id<=UINT32_MAX,"function ID width");current_functions.insert(uint32_t(id));}
  instrument_function_if_needed(ctx,f);
  active=&p;entry_function=fid;received_here=0;sentinels_here.store(0);seen_ctas.clear();cta_sms.clear();clocks.clear();pushed_before=pushed_record_count;active_key="epoch-"+n(p.epoch)+"-launch-"+n(p.ordinal);
  string row="{\"schema\":\"SG_KERNEL_SAMPLE_BEGIN_V1\",\"source_launch_key\":"+q(active_key)+",\"epoch_id\":"+n(p.epoch)+",\"epoch_launch_ordinal\":"+n(p.ordinal)+",\"phase\":"+q(p.phase)+",\"layer_id\":"+sgobs::signed_number(p.layer)+",\"call_id\":"+n(sgobs::scope.call)+",\"module_scope\":"+q(p.module)+",\"function_name\":"+q(name)+",\"kernel_name\":"+q(name)+",\"function_id\":"+n(fid)+",\"related_function_ids\":"+list(ids)+",\"code_sha256\":"+q(code)+",\"grid\":"+sgobs::dims(gx,gy,gz)+",\"block\":"+sgobs::dims(bx,by,bz)+",\"sm_count\":"+n(sms_count)+",\"stream_u64\":";
  pipe.json(sgpipe::BEGIN,row+n(stream_u64)+",\"role\":"+q(p.role)+",\"entry_proof_schema\":\"SG_CTA_WARP_ENTRY_V1\",\"fit_ctas\":"+list(p.fit)+",\"holdout_ctas\":"+list(p.hold)+",\"launch_attributes\":"+attrs+",\"omitted_static_memory_classes\":"+omissions()+",\"cross_warp_dependencies_captured\":false}");
  {std::lock_guard<std::mutex> wake(receiver_wake_mutex);receiving.store(true,std::memory_order_release);}receiver_wake.notify_one();return true;
}
string entry_receipt(){
  require(active,"entry receipt outside kernel");
  uint64_t threads=active->block[0]*active->block[1]*active->block[2],warps=(threads+31)/32;
  require(threads&&threads<=1024,"entry block geometry");
  string rows="[";size_t slot=0;
  for(auto c:selected_ctas){
    uint64_t calls=0,seen=0,duplicates=0,bad_lanes=0;
    for(unsigned w=0;w<32;++w){
      uint32_t count=sg_entry_calls[slot*32+w],mask=sg_entry_masks[slot*32+w];
      if(w>=warps){require(!count&&!mask,"entry outside logical block warps");continue;}
      calls+=count;if(count)seen|=1ull<<w;if(count>1)duplicates|=1ull<<w;
      uint64_t lanes=std::min<uint64_t>(32,threads-w*32),expected=(1ull<<lanes)-1;
      if(mask!=expected)bad_lanes|=1ull<<w;
    }
    require(calls==warps&&seen==((1ull<<warps)-1)&&!duplicates&&!bad_lanes,"missing/duplicate/partial CTA warp entry");
    if(slot)rows+=",";
    rows+="["+n(c)+","+n(calls)+","+n(seen)+","+n(duplicates)+","+n(bad_lanes)+"]";++slot;
  }
  rows+="]";std::set<uint64_t> empty;
  for(auto c:seen_ctas)require(selected_ctas.count(c),"packet outside executed CTAs");
  for(auto c:selected_ctas)if(!seen_ctas.count(c))empty.insert(c);
  return "{\"schema\":\"SG_CTA_WARP_ENTRY_V1\",\"source_launch_key\":"+q(active_key)+
    ",\"entry_function_id\":"+n(entry_function)+",\"scope\":\"selected_CTA_entry_and_instrumented_memory_packets_only\",\"completion_sync_passed\":true,\"warps_per_cta\":"+n(warps)+
    ",\"columns\":[\"cta\",\"calls\",\"seen_warps\",\"duplicate_warps\",\"bad_lane_warps\"],\"rows\":"+rows+",\"packetless_ctas\":"+setlist(empty)+"}";
}
void complete(CUresult*status){
  if(!active)return;require(status&&*status==CUDA_SUCCESS,"selected launch failed");
  sgobs::Internal internal;cuda_ok(cudaDeviceSynchronize(),"selected kernel completion sync");
  require(flush_ledger.size()==selected_kernels&&flush_ledger.size()<MAX_SELECTED_KERNELS,"flush ledger cardinality before submit");
  sgflush::Row row;row.key=active_key;row.ordinal=flush_ledger.size();row.planned_ctas=selected_ctas.size();
  flush_ledger.push_back(row);auto &flush=flush_ledger.back();++flushes;
  auto &state=sgobs::state();
  auto journal=[&](const char *edge,int error){state.emit(state.lifecycle_file,state.prefix("sampling_flush")+",\"source_launch_key\":"+q(active_key)+",\"flush_ordinal\":"+n(flush.ordinal)+",\"edge\":"+q(edge)+",\"cuda_error\":"+sgobs::signed_number(error)+"}\n");};
  journal("submit",-1);
  sg_flush_channel<<<1,1>>>();
  flush.launch_error=int(cudaGetLastError());journal("launch_error_check",flush.launch_error);cuda_ok(cudaError_t(flush.launch_error),"flush dispatch");
  flush.sync_error=int(cudaDeviceSynchronize());journal("completion_sync",flush.sync_error);cuda_ok(cudaError_t(flush.sync_error),"flush completion");
  uint64_t deadline=sgobs::now()+15000000000ull;
  while(receiving.load(std::memory_order_acquire)){require(sgobs::now()<deadline,"selected packet receiver did not close");std::this_thread::yield();}
  flush.sentinel=sentinels_here.load()==1;require(flush.sentinel,"missing/duplicate selected flush sentinel");
  uint64_t pushed=pushed_record_count-pushed_before;
  const string entry=entry_receipt();
  flush.pushed=pushed;flush.received=received_here;flush.selected=received_here;flush.packet_ctas=seen_ctas.size();flush.executed_ctas=selected_ctas.size();
  flush.conservation=pushed==received_here;
  journal("receiver_and_cta_closed",flush.conservation?0:-1);
  require(flush.conservation,"selected CTA conservation/coverage");
  pipe.json(sgpipe::END,"{\"schema\":\"SG_KERNEL_SAMPLE_END_V1\",\"source_launch_key\":"+q(active_key)+",\"pushed_records\":"+n(pushed)+",\"received_records\":"+n(received_here)+",\"selected_records\":"+n(received_here)+",\"omitted_records\":null,\"whole_kernel_dynamic_census\":false,\"selected_ctas_seen\":"+setlist(seen_ctas)+",\"entry_proof\":"+entry+",\"all_memory_active_ctas_seen_count\":null,\"unknown_space_lane_references\":0,\"overflow\":false,\"source_closed\":true,\"omitted_static_memory_classes\":"+omissions()+"}");
  ++selected_kernels;active=nullptr;sg_sample_cta_count=0;sg_sample_entry_function=0;
}
void context_end(){require(!active&&!receiving.load(),"context closed with active sample");{std::lock_guard<std::mutex> wake(receiver_wake_mutex);running.store(false,std::memory_order_release);}receiver_wake.notify_one();if(receiver.joinable())receiver.join();channel_host.destroy(false);}
bool flush_ledger_closed(){
  std::vector<string> keys;for(const auto&p:plans)if(!p.fit.empty())keys.push_back("epoch-"+n(p.epoch)+"-launch-"+n(p.ordinal));
  return sgflush::closed(flush_ledger,keys,selected_kernels,flushes,sgobs::internal_dispatches.load(),sgobs::internal_dispatch_returns.load());
}
void finish(){
  if(finalized)return;require(initialized&&!running.load()&&!active&&plan_cursor==plans.size(),"sampler source/plan/context not closed");
  auto&s=sgobs::state();for(auto c:s.contexts)require(c.second.closed,"sampler open context");require(s.errors.empty(),"observer error prevents sample closure");
  require(flush_ledger_closed(),"flush submission/completion/receiver/plan ledger not closed or internal dispatch callbacks observed");
  pipe.json(sgpipe::CLOSED,"{\"schema\":\"SG_PACKET_STREAM_CLOSED_V1\",\"wire_sha256_before_close\":"+q(pipe.digest())+",\"kernels\":"+n(selected_kernels)+",\"selected_records\":"+n(total_selected)+",\"contexts_closed\":true,\"errors\":[],\"selected_plan_complete\":true}");
  final_stream_sha=pipe.digest();pipe.close();finalized=true;
}
bool is_closed(){return finalized;}
uint64_t flush_count(){return flushes;}
string flush_receipt_fields(){
  uint64_t accepted=0,completed=0,sentinels=0,conserved=0;string rows="[";
  for(const auto&r:flush_ledger){if(rows.size()>1)rows+=",";accepted+=r.launch_error==0;completed+=r.sync_error==0;sentinels+=r.sentinel;conserved+=r.conservation;
    rows+="["+q(r.key)+","+n(r.ordinal)+","+sgobs::signed_number(r.launch_error)+","+sgobs::signed_number(r.sync_error)+","+(r.sentinel?"true":"false")+","+(r.conservation?"true":"false")+","+n(r.pushed)+","+n(r.received)+","+n(r.selected)+","+n(r.executed_ctas)+","+n(r.packet_ctas)+","+n(r.planned_ctas)+"]";}
  rows+="]";
  return ",\"sampling_flush_protocol\":\"EXPLICIT_HOST_LEDGER_ENTRY_V2\",\"internal_dispatch_visibility\":\"NOT_OBSERVED_REQUIRED_R4\",\"sampling_flush_ledger_closed\":"+string(flush_ledger_closed()?"true":"false")+
    ",\"sampling_flush_submit_attempts\":"+n(flushes)+",\"sampling_flush_submitted\":"+n(accepted)+",\"sampling_flush_launch_checks_passed\":"+n(accepted)+",\"sampling_flush_completed\":"+n(completed)+",\"sampling_flush_sentinels_received\":"+n(sentinels)+",\"sampling_flush_conservation_closed\":"+n(conserved)+
    ",\"sampling_flush_ledger_columns\":[\"source_launch_key\",\"flush_ordinal\",\"cuda_get_last_error\",\"cuda_sync_error\",\"sentinel_received\",\"cta_conservation_closed\",\"pushed_records\",\"received_records\",\"selected_records\",\"executed_cta_count\",\"packet_cta_count\",\"planned_cta_count\"],\"sampling_flush_ledger\":"+rows;
}
string receipt_fields(){return flush_receipt_fields()+",\"sampler_finalized\":"+string(finalized?"true":"false")+",\"sample_plan_sha256\":"+q(PLAN_SHA256)+",\"sampled_kernels\":"+n(selected_kernels)+",\"selected_cta_records\":"+n(total_received)+",\"sample_wire_bytes\":"+n(pipe.bytes)+",\"sample_wire_sha256\":"+q(final_stream_sha)+",\"sampling_flush_kernels\":"+n(flushes)+",\"whole_kernel_dynamic_census\":false,\"device_cta_filter\":true,\"device_packet_abi_unchanged\":true,\"consumer_admission_pending\":true";}
} // namespace sgsample

void nvbit_tool_init(CUcontext ctx){if(!sgobs::state().enabled)return;try{sgobs::Internal internal;sgsample::initialize(ctx);}catch(const std::exception&e){sgobs::fatal(e.what());}}
