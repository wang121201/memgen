#ifndef MEMGEN_HARDWARE_CONFIG_H
#define MEMGEN_HARDWARE_CONFIG_H
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <boost/property_tree/json_parser.hpp>

namespace hyfiss_request_trace {
// Version 1 describes the implemented traffic model, not undocumented silicon.
// Registration, validation and serialization live here; hot loops never parse.
struct HardwareProfile {
  std::string source_text;
  std::map<std::string,std::string> values;
  unsigned sms=0, channels=0, subpartitions=0, banks=0, partition_bit=0;
  unsigned l1_sets=0, l2_sets=0, l2_ways=0;
  std::map<unsigned,unsigned> shared_ways;
  uint64_t l2_bytes=0;
  unsigned issue_interval=0, kernel_gap=0;
  bool preserve_l2=true;
  std::string write_sector_policy, dram_store_policy, l2_index;

  static void require(bool ok,const std::string &why) {
    if(!ok)throw std::runtime_error("hardware config: "+why);
  }
  static unsigned number(const std::string &s,unsigned lo,unsigned hi) {
    require(!s.empty(),"empty unsigned integer");
    uint64_t n=0;
    for(char c:s) {
      require(c>='0'&&c<='9',"invalid unsigned decimal: "+s);
      n=n*10+unsigned(c-'0');require(n<=hi,"integer out of range: "+s);
    }
    require(n>=lo,"integer below minimum: "+s);return unsigned(n);
  }
  unsigned ways_for(unsigned shared) const {
    auto it=shared_ways.find(shared);
    require(it!=shared_ways.end(),"unconfigured shared partition: "+std::to_string(shared));
    return it->second;
  }
  static std::shared_ptr<const HardwareProfile> load(const std::string &path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    require(bool(in),"cannot open "+path);
    const auto bytes=in.tellg();require(bytes>=0&&bytes<=65536,"file exceeds 64 KiB or size is unavailable");
    in.seekg(0);
    std::string raw((std::istreambuf_iterator<char>(in)),{});
    require(raw.size()<=65536,"file exceeds 64 KiB");
    bool unified=false;std::istringstream probe(raw);std::string line;
    while(std::getline(probe,line)) {
      line=line.substr(0,line.find('#'));std::istringstream row(line);std::string key;row>>key;
      if(key.rfind("-memgen_",0)==0)unified=true;
    }
    if(!unified)return {}; // Legacy reader remains behavior-compatible.
    auto p=std::make_shared<HardwareProfile>();p->source_text=raw;
    // Every registered field is required: no inherited timing/pipeline defaults.
    const std::map<std::string,std::string> registry={
      {"schema","1"},{"profile_id",""},{"calibration_status","experimental"},
      {"num_sms",""},{"warp_size","32"},{"memory_channels",""},
      {"subpartitions_per_channel",""},{"dram_banks",""},{"partition_index_bit",""},
      {"address_mapping","fallback_quotient_v2"},
      {"l1_model","allocation_clock_v1"},{"context_model_id","r4-small-shared-20260922"},
      {"l1_sets",""},{"l1_line_bytes","128"},{"sector_bytes","32"},
      {"l1_shared_kib_to_ways",""},{"l1_replacement","CLOCK"},
      {"l1_index","allocation_relative_hash2_u32"},{"l1_tag","absolute"},
      {"l1_read_policy","ldg_strong_gpu_bypass_v1"},{"l1_store_policy","bypass"},
      {"l1_preserve_across_kernels","0"},{"l1_fill_latency","0"},
      {"l2_sets_per_partition",""},{"l2_ways",""},{"l2_line_bytes","128"},
      {"l2_index",""},{"l2_replacement","LRU"},{"l2_clean_first_k","0"},
      {"l2_data_validity","known_bytes_union_v1"},{"write_sector_policy",""},
      {"dram_store_policy",""},{"l2_preserve_across_kernels",""},
      {"l2_dirty_drain","0"},{"l2_streaming_fill","0"},{"l2_fill_latency","0"},
      {"mshr_model","disabled"},{"timing_model","disabled"},
      {"cta_placement","round_robin"},{"instruction_order","timestamp"},
      {"monotonic_sm","1"},{"issue_interval",""},{"kernel_gap",""}
    };
    std::istringstream lines(raw);unsigned ordinal=0;
    while(std::getline(lines,line)) {
      ++ordinal;line=line.substr(0,line.find('#'));std::istringstream row(line);
      std::string key,value,extra;if(!(row>>key))continue;
      require(key.rfind("-memgen_",0)==0,"unexpected key at line "+std::to_string(ordinal)+": "+key);
      key=key.substr(8);
      require(registry.count(key),"unknown key: "+key);
      require(bool(row>>value)&&!(row>>extra),"expected one value for "+key);
      require(p->values.emplace(key,value).second,"duplicate key: "+key);
    }
    for(const auto &entry:registry) {
      auto it=p->values.find(entry.first);require(it!=p->values.end(),"missing key: "+entry.first);
      if(!entry.second.empty())require(it->second==entry.second,"unsupported "+entry.first+"="+it->second);
    }
    const auto &v=p->values;
    const auto &id=v.at("profile_id");
    require(id.size()<=128&&id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")==std::string::npos,"invalid profile_id");
    p->sms=number(v.at("num_sms"),1,256); // HBServe compact placement stores uint8_t SM ids.
    p->channels=number(v.at("memory_channels"),1,128);
    p->subpartitions=number(v.at("subpartitions_per_channel"),1,32);
    require((p->subpartitions&(p->subpartitions-1))==0,"subpartitions must be power of two");
    p->banks=number(v.at("dram_banks"),1,128);
    require((p->banks&(p->banks-1))==0,"banks must be power of two");
    p->partition_bit=number(v.at("partition_index_bit"),7,31);
    p->l1_sets=number(v.at("l1_sets"),1,4096);
    require((p->l1_sets&(p->l1_sets-1))==0,"L1 sets must be power of two");
    std::istringstream pairs(v.at("l1_shared_kib_to_ways"));std::string pair;
    const auto &mapping=v.at("l1_shared_kib_to_ways");
    require(!mapping.empty()&&mapping.back()!=',',"invalid shared partition table");
    while(std::getline(pairs,pair,',')) {
      auto sep=pair.find(':');require(sep!=std::string::npos,"expected shared:ways pair");
      unsigned shared=number(pair.substr(0,sep),0,1024),ways=number(pair.substr(sep+1),1,1024);
      require(p->shared_ways.emplace(shared,ways).second,"duplicate shared partition");
    }
    require(!p->shared_ways.empty(),"empty shared partition table");
    p->l2_sets=number(v.at("l2_sets_per_partition"),1,1048576);
    require((p->l2_sets&(p->l2_sets-1))==0,"L2 sets must be power of two");
    p->l2_ways=number(v.at("l2_ways"),1,1024);
    p->l2_bytes=uint64_t(p->channels)*p->subpartitions*p->l2_sets*p->l2_ways*128;
    require(p->l2_bytes<=UINT32_MAX,"L2 exceeds current backend byte-count representation");
    p->l2_index=v.at("l2_index");require(p->l2_index=="L"||p->l2_index=="X","L2 index must be L or X");
    p->write_sector_policy=v.at("write_sector_policy");
    require(p->write_sector_policy=="line-miss-only"||p->write_sector_policy=="all","unsupported write sector policy");
    p->dram_store_policy=v.at("dram_store_policy");
    require(p->dram_store_policy=="writeback"||p->dram_store_policy=="request","unsupported DRAM store policy");
    p->preserve_l2=number(v.at("l2_preserve_across_kernels"),0,1)!=0;
    p->issue_interval=number(v.at("issue_interval"),1,1000000000);
    p->kernel_gap=number(v.at("kernel_gap"),0,1000000000);
    return p;
  }
  boost::property_tree::ptree resolved() const {
    boost::property_tree::ptree root,parameters,capacities;
    root.put("schema","MEMGEN_EFFECTIVE_HARDWARE_V1");
    // property_tree serializes all scalar leaves as strings. Use a status enum,
    // never the string "false" under a boolean-sounding acceptance field.
    root.put("hardware_accuracy_status","not_accepted");
    root.put("hardware_parameter_precedence","unified_config_authoritative");
    root.put("cta_placement_scope","frontend_contract;generic_backend_consumes_supplied_mapping");
    root.put("num_sms",sms);root.put("warp_size",32);
    root.put("l2_partitions",channels*subpartitions);root.put("l2_total_bytes",l2_bytes);
    for(const auto &entry:values)parameters.put(entry.first,entry.second);
    for(const auto &entry:shared_ways) {
      boost::property_tree::ptree row;row.put("shared_kib",entry.first);row.put("ways",entry.second);
      row.put("effective_bytes",uint64_t(l1_sets)*128*entry.second);capacities.push_back({"",row});
    }
    root.add_child("parameters",parameters);root.add_child("l1_capacity_table",capacities);
    return root;
  }
};
}
#endif
