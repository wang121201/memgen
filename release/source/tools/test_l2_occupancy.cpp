#define HYFISS_REQUEST_TRACE_NO_MAIN
#include "hyfiss_request_trace_generator_stream_r4_semantic.cc"

static unsigned checks=0;
static void check(bool condition,const char *message) {
  ++checks;if(!condition)throw std::runtime_error(message);
}
static std::string serialized(const SectorLruCache &cache) {
  std::ostringstream out(std::ios::binary);cache.save(out);return out.str();
}
int main() {
  try {
    SectorLruCache c(256,128,2);
    for(unsigned i=0;i<9;++i)c.access(0,32,32,i,0,CacheOperation::Store,true,0,false,false,15);
    const auto before=serialized(c);auto x=c.occupancy();
    check(serialized(c)==before,"census mutated state");
    check(x.allocated_lines==1&&x.dirty_lines==1&&x.valid_sectors==0,"partial dirty tag is allocated without valid sector");
    check(x.dirty_sectors==1&&x.known_dirty_bytes==4&&x.missing_dirty_bytes==28,"repeated words must not complete a sector");
    check(x.sets_by_allocated_lines[1]==1&&x.sets_by_dirty_lines[1]==1,"set histogram lost partial dirty line");
    c.access(64,32,32,10,0,CacheOperation::Store,true);
    c.access(128,32,32,11,0);
    x=c.occupancy();
    check(x.allocated_lines==2&&x.clean_lines==1&&x.dirty_lines==1,"line census conflated sector count");
    check(x.dirty_sectors==2&&x.incomplete_dirty_sectors==1&&x.partial_dirty_lines==1,"sparse mixed-coverage dirty masks");
    auto ev=c.access(256,32,32,12,0).evicted;
    check(ev.present&&ev.addr==0&&ev.dirty_sectors==5&&ev.dirty_bytes==64,"sparse victim selection changed");
    check(ev.incomplete_dirty_sectors==1&&ev.known_dirty_bytes==36&&ev.missing_dirty_bytes==28,"victim coverage metadata lost");
    std::vector<WritebackSpan> spans;
    for_each_writeback_span(ev,32,128,[&](const WritebackSpan&s){spans.push_back(s);});
    check(spans.size()==2&&spans[0].addr==0&&spans[0].size==32&&spans[1].addr==64&&spans[1].size==32,"coverage counters changed sparse writeback");
    x=c.occupancy();check(x.clean_lines==2&&x.dirty_sectors==0,"evicted dirty state still resident");

    // Independent word-set oracle for all partial/full sector combinations.
    for(unsigned subset=0;subset<256;++subset) {
      SectorLruCache p(128,128,1);std::set<unsigned> bytes;
      for(unsigned word=0;word<8;++word)if(subset&(1u<<word)) {
        p.access(0,32,32,word,0,CacheOperation::Store,true,0,false,false,15u<<(4*word));
        for(unsigned b=0;b<4;++b)bytes.insert(4*word+b);
      }
      const auto saved=serialized(p);auto o=p.occupancy();
      check(serialized(p)==saved,"census changed checkpoint bytes");
      check(o.allocated_lines==!bytes.empty()&&o.known_dirty_bytes==bytes.size(),"byte oracle mismatch");
      check(o.missing_dirty_bytes==(bytes.empty()?0:32-bytes.size()),"missing-byte partition mismatch");
      check(o.valid_sectors==(bytes.size()==32)&&o.partial_dirty_lines==(!bytes.empty()&&bytes.size()<32),"partial/full classification mismatch");
      auto v=p.access(128,32,32,100,0).evicted;
      check(v.known_dirty_bytes==bytes.size()&&v.missing_dirty_bytes==(bytes.empty()?0:32-bytes.size()),"eviction coverage mismatch");
    }
    SectorLruCache pending(128,128,1);
    pending.access(0,32,32,0,300,CacheOperation::Atomic,true);
    const auto saved=serialized(pending);x=pending.occupancy();
    check(x.reserved_sectors==1&&x.missing_dirty_bytes==32&&serialized(pending)==saved,"observation fabricated read completion");
    pending.access(0,32,32,300,0);x=pending.occupancy();
    check(x.valid_sectors==1&&x.known_dirty_bytes==32&&x.missing_dirty_bytes==0,"completed fill coverage missing");
    // Independent masks: a dirty sector need not yet have all readable bytes.
    // Construct every V/D pair using full reads and one-byte stores. The zero
    // pair is represented by a live clean reservation, not a missing tag.
    for(unsigned valid=0;valid<16;++valid)for(unsigned dirty=0;dirty<16;++dirty){
      SectorLruCache joint(128,128,1);
      for(unsigned s=0;s<4;++s){
        if(valid&(1u<<s))joint.access(s*32,32,32,0,0);
        if(dirty&(1u<<s))joint.access(s*32,32,32,0,0,CacheOperation::Store,true,0,false,false,1);
      }
      if(!valid&&!dirty)joint.access(0,32,32,0,100);
      auto before_joint=serialized(joint);auto o=joint.occupancy();
      check(o.lines_by_valid_dirty_masks.size()==1&&o.lines_by_valid_dirty_masks.at({valid,dirty})==1,"joint mask census differs from independent construction");
      check(o.allocated_lines==1&&o.valid_sectors==static_cast<unsigned>(__builtin_popcount(valid))&&o.dirty_sectors==static_cast<unsigned>(__builtin_popcount(dirty)),"joint count weighted totals differ");
      check(o.partial_dirty_lines==bool(dirty&~valid),"joint incomplete dirty classification differs");
      check(serialized(joint)==before_joint,"joint snapshot changed reserved or replacement state");
    }
    // Explicit retained-clean primitive: no trigger policy is enabled by this
    // test. Unknown old bytes may not silently lose their dirty obligation.
    SectorLruCache retained(256,128,2);
    retained.access(0,32,32,0,0,CacheOperation::Store,true,0,false,false,15);
    const auto partial_saved=serialized(retained);
    check(retained.drain_eligible_dirty(32,10,0,0).empty(),"incomplete unreserved dirty sector was cleaned without preserving old bytes");
    check(serialized(retained)==partial_saved,"skipped partial clean changed state");
    retained.access(64,32,32,11,0,CacheOperation::Store,true);
    auto cleaned=retained.drain_eligible_dirty(32,12,0,0);auto kept=retained.occupancy();
    check(cleaned.size()==1&&!cleaned[0].present&&cleaned[0].dirty_sectors==4&&cleaned[0].valid_sectors==4&&cleaned[0].dirty_bytes==32,"retained writeback was mistaken for invalidation or included partial data");
    check(kept.allocated_lines==1&&kept.valid_sectors==1&&kept.dirty_sectors==1&&kept.known_dirty_bytes==4&&kept.missing_dirty_bytes==28,"clean-retained lost tag, valid bytes, or partial dirty debt");
    check(retained.access(64,32,32,13,0).result==CacheResult::Hit,"retained clean invalidated readable sector");
    check(retained.drain_eligible_dirty(32,14,0,0).empty(),"retained clean emitted duplicate dirty bytes");
    check(retained.access(0,32,32,15,0).result==CacheResult::SectorMiss,"partial bytes incorrectly claimed readable");
    cleaned=retained.drain_eligible_dirty(32,16,0,0);kept=retained.occupancy();
    check(cleaned.size()==1&&cleaned[0].dirty_sectors==1&&cleaned[0].known_dirty_bytes==32&&cleaned[0].missing_dirty_bytes==0,"completed old-data fill was not eligible for complete-sector cleaning");
    check(kept.allocated_lines==1&&kept.valid_sectors==2&&kept.dirty_sectors==0,"clean-retained final state invalid");
    retained.access(128,32,32,17,0,CacheOperation::Store,true);
    cleaned=retained.drain_eligible_dirty(32,18,0,0);
    check(retained.access(256,32,32,19,0).evicted.addr==0,"retained clean changed replacement order");
    // A line budget is independent of tag capacity and sector population.
    for(unsigned sectors:{1u,4u})for(bool touch:{false,true}) {
      SectorLruCache quota(512,128,4);
      for(unsigned line=0;line<3;++line)for(unsigned s=0;s<sectors;++s)
        quota.access(line*128+s*32,32,32,0,0,CacheOperation::Store,true);
      auto unchanged=serialized(quota);
      check(quota.diagnostic_clean_set_dirty_overflow(0,4).emitted.empty()&&serialized(quota)==unchanged,"inactive budget altered state");
      if(touch)quota.access(0,32,32,0,0);
      auto q=quota.diagnostic_clean_set_dirty_overflow(0,2);
      check(q.satisfied&&q.dirty_lines_before==3&&q.dirty_lines_after==2&&q.emitted.size()==1,"line quota counted sectors or lost count");
      check(!q.emitted[0].present&&q.emitted[0].addr==(touch?128u:0u)&&q.emitted[0].dirty_bytes==sectors*32,"read recency or retained payload differs");
      auto state=quota.occupancy();check(state.allocated_lines==3&&state.valid_sectors==3*sectors&&state.dirty_sectors==2*sectors,"quota erased valid tags");
      check(quota.diagnostic_clean_set_dirty_overflow(0,2).emitted.empty(),"quota emitted duplicate clean");
      for(unsigned line=0;line<3;++line)check(quota.access(line*128,32,32,0,0).result==CacheResult::Hit,"quota made a readable tag miss");
    }
    SectorLruCache partial_quota(256,128,2);
    partial_quota.access(0,32,32,0,0,CacheOperation::Store,true,0,false,false,1);
    partial_quota.access(128,32,32,0,0,CacheOperation::Store,true);
    auto q=partial_quota.diagnostic_clean_set_dirty_overflow(0,0);
    check(!q.satisfied&&q.dirty_lines_after==1&&q.emitted.size()==1&&q.emitted[0].addr==128,"partial line obligation lost or failed to skip safely");
    check(partial_quota.occupancy().known_dirty_bytes==1&&partial_quota.occupancy().missing_dirty_bytes==31,"quota manufactured old bytes");
    const auto partial_state=serialized(partial_quota);
    check(partial_quota.diagnostic_clean_set_dirty_overflow(0,0).emitted.empty()&&serialized(partial_quota)==partial_state,"unsatisfied quota changed state or reemitted");
    bool bad=false;try{partial_quota.diagnostic_clean_set_dirty_overflow(0,3);}catch(const std::runtime_error&){bad=true;}
    check(bad&&serialized(partial_quota)==partial_state,"invalid quota accepted or mutated cache");
    for(unsigned coverage:{0u,4u,8u,16u,32u})for(unsigned sectors:{1u,2u,4u}) {
      MemoryInst i;i.op='W';i.mem_width=4;
      for(unsigned lane=0;lane<32;++lane)if(lane%8<coverage/4&&lane/8<sectors) {
        i.mask|=1u<<lane;i.lanes.push_back(LaneAddress{lane,0,0x8000+lane*4});
      }
      auto requests=coalesce_to_sectors(i,32);
      check(requests.size()==(coverage?sectors:0),"analytic producer disagrees with Memgen coalescer");
      for(unsigned s=0;s<requests.size();++s) {
        const uint32_t mask=coverage==32?UINT32_MAX:(1u<<coverage)-1;
        check(requests[s].addr==0x8000+s*32&&requests[s].byte_mask==mask&&requests[s].lane_count==coverage/4,
              "producer byte mask/lane coalescing differs");
      }
    }
    std::cout<<"PASS "<<checks<<" occupancy/coverage assertions\n";
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
