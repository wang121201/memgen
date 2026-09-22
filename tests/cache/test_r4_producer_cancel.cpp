// Exercise backend initialization failure after the HBServe producer starts.
#define main hbserve_existing_main
#include "../../release/source/tools/hbserve_profile_stream_cache_semantic_r17.cpp"
#undef main
int main(int argc,char**argv) {try {
  if(argc!=5)throw std::runtime_error("usage: test FIXTURE CONFIG CONTEXT FRESH_OUTPUT");
  using namespace hbserve_profile_stream;
  Arguments a;a.mode="memgen";a.profile_index=fs::path(argv[1])/"profiles.index.jsonl";
  a.app_config=fs::path(argv[1])/"app.config";a.issue_config=fs::path(argv[1])/"issue.config";
  a.hw_config=argv[2];a.r4_context=argv[3];a.output_dir=argv[4];
  need(!fs::exists(a.output_dir),"fresh output required");fs::create_directory(a.output_dir);
  {std::ofstream f(a.output_dir/"r4_l1_profiles.csv");f<<"existing evidence\n";}
  WorkloadSource source(a.profile_index,a.app_config,a.issue_config,true);
  need(run_memgen(a,source)!=0,"backend failure was swallowed");
  std::ifstream f(a.output_dir/"r4_l1_profiles.csv");std::string line;std::getline(f,line);
  need(line=="existing evidence","existing ledger overwritten");
  need(!fs::exists(a.output_dir/"r4_context_identity.json"),"failure reported success");
  std::cout<<"PASS_BACKEND_INITIALIZATION_CANCEL_AFTER_PRODUCER_START\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
