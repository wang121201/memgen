#define HYFISS_REQUEST_TRACE_NO_MAIN
#include "hyfiss_request_trace_generator_stream_r4_semantic.cc"

unsigned checks = 0;
void check(bool ok, const char *why) {
  if (!ok) throw std::runtime_error(why);
  ++checks;
}
std::string state(const SectorLruCache &cache) {
  std::ostringstream out(std::ios::binary); cache.save(out); return out.str();
}
SectorLruCache restore(const SectorLruCache &cache, unsigned size=128) {
  SectorLruCache out(size,128,1);
  std::istringstream in(state(cache),std::ios::binary); out.load(in);
  check(state(cache)==state(out),"partial state roundtrip differs"); return out;
}
CacheAccess store(SectorLruCache &cache,uint32_t mask,uint64_t now=0,bool dirty=true) {
  return cache.access(0,32,32,now,300,CacheOperation::Store,dirty,0,false,false,mask);
}
uint32_t field(const SectorLruCache &cache,size_t offset) {
  const auto bytes=state(cache); uint32_t value=0;
  check(offset+sizeof(value)<=bytes.size(),"serialized state too short");
  std::copy_n(bytes.data()+offset,sizeof(value),reinterpret_cast<char *>(&value)); return value;
}

int main() {
  try {
    // Enumerate all combinations of eight disjoint words with an independent
    // byte-address set; count of writes alone cannot establish completeness.
    for(unsigned subset=0;subset<256;++subset) {
      SectorLruCache cache(128,128,1); std::set<unsigned> written;
      for(unsigned word=0;word<8;++word) if(subset&(1u<<word)) {
        store(cache,15u<<(word*4),word);
        for(unsigned byte=0;byte<4;++byte) written.insert(word*4+byte);
      }
      auto full=restore(cache), small=restore(cache);
      const auto expected=written.size()==32 ? CacheResult::Hit :
                          written.empty() ? CacheResult::LineMiss : CacheResult::SectorMiss;
      check(full.access(0,32,32,1000,0).result==expected,"full-sector union classification");
      check(small.access(0,4,32,1000,0).result==expected,"partial read falsely forwarded known bytes");
      if(!written.empty()) check(cache.dirty_sector_count()==1,"duplicate writes multiplied dirty sectors");
    }
    SectorLruCache repeated(128,128,1);
    for(unsigned i=0;i<8;++i) store(repeated,15u,i);
    check(repeated.access(0,32,32,1000,300).result==CacheResult::SectorMiss,"repeated partial stores became valid");
    auto pending=restore(repeated);
    store(pending,240u,1001);
    check(pending.access(0,32,32,1100,300).result==CacheResult::HitReserved,"partial store cancelled pending read");
    check(pending.access(0,32,32,1300,300).result==CacheResult::Hit,"read completion was delayed by partial store");

    SectorLruCache overwrite(128,128,1);
    overwrite.access(0,32,32,0,300);
    store(overwrite,UINT32_MAX,1);
    check(field(overwrite,68)==1,"full overwrite cancelled previously issued read");
    auto complete=restore(overwrite);
    check(complete.access(0,32,32,2,300).result==CacheResult::Hit,"full overwrite failed to supply data");
    complete.access(0,32,32,300,300);
    check(field(complete,68)==0,"pending read failed to retire");

    SectorLruCache through(128,128,1);
    store(through,15u,0,false);
    check(through.dirty_sector_count()==0,"write-through marked dirty");
    check(through.access(0,32,32,1000,0).result==CacheResult::SectorMiss,"write-through partial store fabricated read fill");
    SectorLruCache atomic(128,128,1);
    check(atomic.access(0,32,32,0,300,CacheOperation::Atomic,true).result==CacheResult::LineMiss,"cold atomic did not miss");
    auto atomic_pending=restore(atomic);
    check(atomic_pending.drain_eligible_dirty(32,100,0,1).empty(),"atomic drained before old data arrived");
    check(atomic_pending.dirty_sector_count()==1,"deferred atomic drain cleared dirty state");
    check(atomic_pending.access(0,32,32,1,300).result==CacheResult::HitReserved,"atomic overwrite skipped read dependency");
    check(atomic_pending.access(0,32,32,300,300).result==CacheResult::Hit,"atomic read did not complete");
    check(atomic_pending.dirty_sector_count()==1,"atomic dirty state lost");
    check(atomic_pending.drain_eligible_dirty(32,300,0,1).size()==1,"ready atomic cannot drain");
    auto reject_pending_eviction=[&](bool flush) {
      SectorLruCache cache(128,128,1);
      cache.access(0,32,32,0,300,CacheOperation::Atomic,true);
      bool rejected=false;
      try {
        if(flush) cache.drain_dirty(32,100);
        else cache.access(128,32,32,100,300);
      } catch(const std::runtime_error &) { rejected=true; }
      check(rejected,"incomplete pending dirty eviction fabricated writeback");
    };
    reject_pending_eviction(false);reject_pending_eviction(true);
    SectorLruCache supplied(128,128,1);
    supplied.access(0,32,32,0,300);store(supplied,UINT32_MAX,1);
    check(supplied.drain_eligible_dirty(32,2,0,1).size()==1,"complete overwrite unnecessarily waits for pending read");

    SectorLruCache drained(128,128,1); store(drained,15u);
    auto ev=drained.drain_eligible_dirty(32,1000,0,1);
    check(ev.size()==1 && ev[0].dirty_bytes==32 && drained.dirty_sector_count()==0,"partial dirty drain changed sectors");
    auto clean_partial=restore(drained);
    check(clean_partial.access(0,32,32,1001,0).result==CacheResult::SectorMiss,"dirty drain fabricated complete data");
    for(unsigned i=1;i<8;++i) store(drained,15u<<(i*4),1001+i);
    check(drained.access(0,32,32,1010,0).result==CacheResult::Hit,"dirty drain lost known bytes");

    SectorLruCache line(128,128,1);
    line.access(0,128,32,0,0,CacheOperation::Store,true);
    check(line.access(0,128,32,1,0).result==CacheResult::Hit,"whole-line store semantics changed");
    SectorLruCache mixed(128,128,1);
    mixed.access(0,32,32,0,0); mixed.access(32,32,32,0,300);
    check(mixed.access(0,64,32,1,300).result==CacheResult::HitReserved,"mixed valid/pending issued duplicate fill");

    MemoryInst inst; inst.op='R';
    for(const auto &opcode : {"LDG.E.STRONG.GPU","LDG.E.64.STRONG.GPU","LDG.E.128.STRONG.GPU"}) {
      inst.opcode=opcode; check(bypass_l1_read(inst),"measured LDG cache policy not matched");
    }
    for(const auto &opcode : {"LDG.E","LD.E.STRONG.GPU","LDL.STRONG.GPU","LDG.E.STRONG.SYS","LDG.E.STRONG.GPU.EXTRA"}) {
      inst.opcode=opcode; check(!bypass_l1_read(inst),"unverified opcode cache policy inferred");
    }
    std::cout<<"PASS "<<checks<<" byte-valid cache assertions\n";
  } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
