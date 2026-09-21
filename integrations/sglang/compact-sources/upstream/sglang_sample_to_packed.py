"""Bounded native CTA samples -> existing affine fitter -> frozen Memgen.

Only final templates and aggregate receipts persist. Each accepted kernel is a
cold isolated diagnostic, never an incomplete prefix presented as whole-model.
"""
from pathlib import Path
import argparse,collections,csv,functools,hashlib,heapq,importlib.util,json,math,os,stat,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
F=Path('/home/xmu/nvidiagds/codex-runs/memgen-paper-ada-v1-20260916-01a08d87-r1')
sys.path.insert(0,str(HERE/'template_adapter_r4'))
if (HERE/'nvbit_sampler_r4/cta_entry_contract.py').is_file():sys.path.insert(0,str(HERE/'nvbit_sampler_r4'))
spec=importlib.util.spec_from_file_location('frozen_profile_rules',F/'workflow/hyfiss_sampled_sass_trace_profile_rules_r15.py')
rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
EXPECTED_MANIFEST='a97b93c93247acf42c542453f3b936cf726ef71db5212dd04f82f8cba678eeda'
# Conservative deep_size charges repeated dictionary keys once per record;
# 400k complete-warp MemoryInst records require >8 GiB under this accounting.
# The controller independently retains its 48 GiB actual RSS ceiling.
MAX_DECODED_BYTES=12 << 30

def capture_limits(target=None):
 # The capture contract is frozen by the controller. Total records and wire
 # grow for a multi-kernel window; each decoded kernel retains the old ceiling.
 contract=HERE/'target-contract.json'
 if target is None:target=json.loads(contract.read_text()) if contract.exists() else {}
 records=target.get('max_records',400000);wire=target.get('max_wire_bytes',256<<20)
 encoded=target.get('max_encoded_bytes',256<<20)
 assert type(records) is int and 1<=records<=12000000
 assert type(wire) is int and records*640<wire<=8<<30
 assert type(encoded) is int and 0<encoded<=512<<20
 kernel_records=target.get('max_kernel_records',400000)
 decoded=target.get('max_decoded_bytes',MAX_DECODED_BYTES)
 assert type(kernel_records) is int and 1<=kernel_records<=1000000
 assert type(decoded) is int and decoded in [12<<30,24<<30]
 return dict(max_records=records,max_encoded_bytes=encoded,max_kernel_records=kernel_records,max_decoded_bytes=decoded)
def frozen_gate():
 manifest=F/'release-manifest.json'
 assert hashlib.sha256(manifest.read_bytes()).hexdigest()==EXPECTED_MANIFEST
 for row in json.loads(manifest.read_text())['files']:
  assert hashlib.sha256((F/row['path']).read_bytes()).hexdigest()==row['sha256'],'frozen payload drift: '+row['path']
@functools.lru_cache(maxsize=128)
def direction_width(opcode):
 direction='R' if opcode.startswith('LDG') else 'W' if opcode.startswith('STG') else None
 assert direction is not None,'opcode outside explicit global load/store projection'
 width=next((bits//8 for bits in [128,64,32,16,8] if any('.'+prefix+str(bits) in opcode for prefix in ['U','S','B',''])),4)
 return direction,width

def global_opcode(opcode):
 # Called only after the dynamic projection has proved every retained lane
 # global. Preserve the original opcode in a separate observed census.
 stem,sep,suffix=opcode.partition('.')
 if stem in ['LD','ST']:return stem+'G'+sep+suffix
 return opcode

@functools.lru_cache(maxsize=4096)
def lane_offsets(mask,pairs):
 # Cache only immutable relative offsets. Every dynamic base/address is still
 # checked independently against the captured record; no records are retained.
 return rules.active_lane_offsets(mask,list(pairs))

def expand_lane_addresses(record,bases=None):
 base_values=list(bases) if bases is not None else [g['base'] for g in record['groups']]
 rules.require(len(base_values)==len(record['groups']),'base/group count mismatch')
 return [base+offset for base,g in zip(base_values,record['groups'])
         for _,offset in lane_offsets(record['mask'],tuple(g['pairs']))]
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def projection_record_census(collector):
 # Count while the bounded decoded sample is still in RAM. An active warp
 # with no global source lanes is distinct from an inactive instruction.
 keys=['decoded_records','active_global_records','no_global_lane_records',
       'no_global_effective_zero_records','no_global_effective_nonzero_records']
 result=dict.fromkeys(keys,0)
 for frames in collector.samples.values():
  for m in frames:
   assert type(m['mask']) is int and 0<=m['mask']<1<<32
   result['decoded_records']+=1
   if m['lanes']:
    assert m['mask']
    result['active_global_records']+=1
   else:
    result['no_global_lane_records']+=1
    result['no_global_effective_nonzero_records' if m['mask'] else 'no_global_effective_zero_records']+=1
 assert result['decoded_records']==collector.count
 shared=dict(getattr(collector,'shared_only_census',{}))
 result['observed_shared_only_records']=shared.get('records',0)
 assert type(result['observed_shared_only_records']) is int and 0<=result['observed_shared_only_records']<=result['no_global_effective_nonzero_records']
 # Shared-only records were independently counted from dynamic reference
 # masks by SampleCollector; this remainder does not identify their cause.
 result['other_no_global_effective_nonzero_records']=result['no_global_effective_nonzero_records']-result['observed_shared_only_records']
 result['record_residual']=result['decoded_records']-result['active_global_records']-result['no_global_lane_records']
 assert result['record_residual']==0
 assert result['no_global_lane_records']==result['no_global_effective_zero_records']+result['no_global_effective_nonzero_records']
 result['raw_records_saved']=False
 return result

def normalize(samples):
 out=collections.defaultdict(list)
 for (cta,warp),frames in sorted(samples.items()):
  if frames:out[cta] # Observed zero-global activity is not an absent CTA.
  for m in frames:
   if not m['lanes']:continue
   assert m['op'] in [ord('R'),ord('W')]
   opcode=global_opcode(m['opcode']);direction,width=direction_width(opcode)
   assert ord(direction)==m['op'] and width==m['mem_width'],'native direction/width differs from backend opcode semantics'
   assert all(x['is_local']==0 for x in m['lanes'])
   by_lane={x['lane']:x['addr'] for x in m['lanes']}
   assert len(by_lane)==len(m['lanes']) # exactly one projected global reference
   mask=sum(1<<i for i in by_lane)
   assert mask==m['mask'],'source mask differs from effective guard; not silently lowered'
   base=by_lane[min(by_lane)];addresses=[by_lane.get(i,base) for i in range(32)]
   pairs=[str(addresses[i]-addresses[i-1])+':1' for i in range(1,32)]
   seq=len(out[cta])
   out[cta].append(dict(block=cta,pc=hex(m['pc']),opcode=opcode,mask=hex(mask),timestamp=seq,source_sequence=seq,groups=[dict(base=addresses[0],pairs=pairs)]))
 return dict(out)
def template_by_cta(profile):
 size=profile['kernel']['grid_size']
 if 'structural_classes' not in profile:return [profile['template']]*size
 classes={x['class_id']:x for x in profile['structural_classes']}
 assert len(classes)==len(profile['structural_classes'])
 if 'structural_class_selector' in profile:
  selector=profile['structural_class_selector'];gx,gy,gz=profile['kernel']['grid_dims']
  assert selector['kind']=='categorical_y' and len(selector['class_by_y'])==gy
  assert set(selector['class_by_y'])==set(classes)
  assert all(selector['class_by_y'][(c//gx)%gy]==cid for cid,x in classes.items() for c in x['ctas'])
  return [classes[selector['class_by_y'][(c//gx)%gy]]['template'] for c in range(size)]
 mapping=profile['cta_class_by_id'];assert len(mapping)==size
 members=[c for x in classes.values() for c in x['ctas']]
 assert len(members)==size and set(members)==set(range(size))
 for cid,x in classes.items():
  assert x['ctas'] and all(mapping[c]==cid for c in x['ctas'])
 return [classes[cid]['template'] for cid in mapping]

def validate_samples(profile,sampled):
 checked=0;templates=template_by_cta(profile);grid=profile['kernel']['grid_dims']
 for c,actual in sorted(sampled.items()):
  ordered=sorted(templates[c],key=lambda e:(int(e.get('sampled_timestamp_delta_by_cta',{}).get(str(c),e['sampled_timestamp_delta'])),e['ordinal']))
  assert len(ordered)==len(actual),'sample instruction count mismatch'
  for entry,record in zip(ordered,actual):
   assert all(entry[k]==record[k] for k in ['pc','opcode','mask']),'sample ordered opcode/PC/mask mismatch'
   assert expand_lane_addresses(entry,rules.predict_bases(entry,c,grid))==expand_lane_addresses(record),'sample ordered lane mismatch'
   checked+=1
 return checked

def captured_digest(sampled):
 # Independent reference from native records, not the generated template.
 # Placement is the explicitly modeled round-robin policy used by run_model.
 mask=(1<<64)-1;a=1469598103934665603;b=0x243f6a8885a308d3
 heap=[((c//48)*1000000,c%48,c,0) for c in sampled if sampled[c]];heapq.heapify(heap)
 while heap:
  timestamp,sm,c,pos=heapq.heappop(heap);r=sampled[c][pos];addresses=expand_lane_addresses(r)
  values=[1,c,sm,timestamp,int(r['pc'],16),int(r['mask'],16),*r['opcode'].encode('ascii'),len(addresses),*addresses]
  for value in values:
   a=((a^value)*1099511628211)&mask
   b=(b+value+0x9e3779b97f4a7c15+(b<<6)+(b>>2))&mask
  if pos+1<len(sampled[c]):heapq.heappush(heap,((c//48)*1000000+pos+1,sm,c,pos+1))
 sa=1469598103934665603;sb=0x243f6a8885a308d3
 sa=(sa^(a+0x9e3779b97f4a7c15+(sa<<6)+(sa>>2)))&mask
 sb=(sb^(b+0x517cc1b727220a95+(sb<<7)+(sb>>3)))&mask
 return dict(semantic_digest_a=f'{sa:016x}',semantic_digest_b=f'{sb:016x}')

def fit(collector,begin,end=None,transport=None,timings=None):
 timings={} if timings is None else timings;tick=time.perf_counter()
 def mark(name):
  nonlocal tick
  now=time.perf_counter();timings[name]=now-tick;tick=now
 assert not collector.estimated_sites,'estimated predicate template excluded'
 record_census=projection_record_census(collector)
 sampled=normalize(collector.samples);train=sorted(begin['fit_ctas']);hold=sorted(begin['holdout_ctas']);packetless=set()
 if begin.get('entry_proof_schema'):
  from cta_entry_contract import validate_entry_proof
  assert transport is not None and transport['status']=='PASS_SAMPLED_TRANSPORT_ONLY'
  proof=validate_entry_proof(begin,end,sorted({c for (c,w),frames in collector.samples.items() if frames}))
  source=next(x for x in transport['kernels'] if x['source_launch_key']==begin['source_launch_key'])
  assert source['entry_proof']==end['entry_proof'],'qualified transport entry proof differs'
  packetless=set(proof['packetless_ctas'])
  if packetless:
   assert all(k in ['CONSTANT','SHARED'] or not v for k,v in end['omitted_static_memory_classes'].items()),'uncovered potential L2 client prevents packetless profile admission'
  for c in packetless:assert c not in sampled;sampled[c]=[]
 grid=tuple(begin['grid']);size=math.prod(grid);complete=set(sampled)==set(range(size))
 assert set(sampled)==set(train+hold) and set(train).isdisjoint(hold)
 if complete:train=sorted(sampled);hold=[]
 else:assert hold
 indexed={c:rules.indexed_records(v) for c,v in sampled.items()}
 mark('normalize_and_index_seconds')
 zero_global={c for c,v in sampled.items() if not v}
 groups=collections.defaultdict(list)
 for c,(_,slots) in indexed.items():
  assert slots or c in zero_global,'empty/unobserved CTA cannot be invented'
  groups[frozenset(slots)].append(c)
 profile=dict(schema={'name':'hbserve.hyfiss_sampled_sass_profile','version':5 if complete else 4},kernel={'id':1,'name':begin['kernel_name'],'phase':begin['phase'],'grid_size':size,'grid_dims':list(grid),'block_size':math.prod(begin['block'])})
 if len(groups)==1 and next(iter(groups)):
  profile['template']=rules.build_template(sampled,indexed,train,grid,coordinate_rules=True,allow_exact_table=complete)
 elif complete:
  profile['schema']['version']=9;classes=[];mapping=[None]*size
  for i,ctas in enumerate(sorted(groups.values(),key=min)):
   ctas=sorted(ctas);cid=f'class-{i:02d}'
   template=rules.build_template(sampled,indexed,ctas,grid,coordinate_rules=True,allow_exact_table=True)
   classes.append(dict(class_id=cid,ctas=ctas,cta_membership_scope='complete_grid',template=template))
   for c in ctas:assert mapping[c] is None;mapping[c]=cid
  profile.update(structural_classes=classes,cta_class_by_id=mapping)
 else:
  # Use the frozen backend's existing categorical-y schema. Every category
  # must have training evidence and an independently held-out exact witness.
  assert grid[1]>1,'sparse heterogeneous CTA structure has no categorical-y witness'
  structures={c:frozenset(slots) for c,(_,slots) in indexed.items()}
  class_by_y=[];training_groups=collections.defaultdict(list)
  for c in train:training_groups[structures[c]].append(c)
  for y in range(grid[1]):
   candidates={structures[c] for c in train if (c//grid[0])%grid[1]==y}
   held=[c for c in hold if (c//grid[0])%grid[1]==y]
   assert len(candidates)==1 and held,'categorical-y training or independent holdout missing'
   key=next(iter(candidates));assert all(structures[c]==key for c in held),'heldout categorical-y structure differs'
   class_by_y.append(key)
  assert set(class_by_y)==set(training_groups)
  ids={};classes=[]
  for i,(key,ctas) in enumerate(sorted(training_groups.items(),key=lambda x:min(x[1]))):
   cid=f'class-{i:02d}';ids[key]=cid;ctas=sorted(ctas)
   try:template=rules.build_template(sampled,indexed,ctas,grid,coordinate_rules=True,allow_exact_table=False,rule_ctas=train)
   except ValueError:
    # A slot confined to one categorical row has no y-axis observations.
    # The existing linear-CTA affine rule is usable only when training fits;
    # the independent held-out lane/address check below remains mandatory.
    template=rules.build_template(sampled,indexed,ctas,grid,coordinate_rules=False,allow_exact_table=False)
   classes.append(dict(class_id=cid,ctas=ctas,cta_membership_scope='training_samples_only',template=template))
  profile['schema']['version']=11
  profile.update(structural_classes=classes,structural_class_selector=dict(kind='categorical_y',class_by_y=[ids[k] for k in class_by_y],training_y_coverage=list(range(grid[1])),holdout_ctas_exact_structure_match=len(hold)))
 mark('fit_rules_seconds')
 checked=validate_samples(profile,sampled)
 mark('validate_all_sample_addresses_seconds')
 expected={'mem_insts':0,'lane_accesses':0,'read_sector_requests':0,'write_sector_requests':0,'native_read_lane_bytes':0,'native_write_lane_bytes':0}
 caches={}
 for c,template in enumerate(template_by_cta(profile)):
  for entry in template:
   key=id(entry)
   if key not in caches:
    assert len(entry['groups'])==1
    direction,width=direction_width(entry['opcode'])
    offsets=[offset for lane,offset in lane_offsets(entry['mask'],tuple(entry['groups'][0]['pairs']))]
    counts={mod:len({sector for off in offsets for sector in range((mod+off)//32,(mod+off+width-1)//32+1)}) for mod in range(32)}
    caches[key]=(direction,width,len(offsets),counts)
   direction,width,lanes,counts=caches[key]
   base=rules.predict_bases(entry,c,grid)[0]
   expected['mem_insts']+=1;expected['lane_accesses']+=lanes
   expected['read_sector_requests' if direction=='R' else 'write_sector_requests']+=counts[base%32]
   expected['native_read_lane_bytes' if direction=='R' else 'native_write_lane_bytes']+=lanes*width
 profile.update(status='PASS_NATIVE_CTA_PACKED_EXACT_SAMPLES',independent_source_census=expected,
  source={'kind':'bounded_native_dynamic_CTA_pipe','launch':begin,'raw_trace_saved':False},sampling={'training_ctas':train,'holdout_ctas':hold,'complete_grid_no_extrapolation':complete,'checked_instructions':checked,'structural_class_count':len(groups),'certified_packetless_ctas':sorted(packetless),'packetless_scope':'instrumented memory classes only; not all hardware L2 clients'},model={'ordering':'CTA round robin across 48 SMs; canonical per-warp order; physical cross-warp arrival NOT captured','cache_entry':'COLD_ISOLATED_KERNEL_DIAGNOSTIC','prefix_replayed':False,'complete_model':False})
 profile['shared_only_projection_census']=dict(getattr(collector,'shared_only_census',{}))
 profile['shared_only_projection_census']['l2_requests']=0
 profile['sampling']['observed_zero_global_ctas']=sorted(zero_global)
 lowerings=collections.Counter((m['opcode'],global_opcode(m['opcode'])) for frames in collector.samples.values() for m in frames if m['lanes'] and global_opcode(m['opcode'])!=m['opcode'])
 profile['observed_global_opcode_lowering']=[dict(raw_opcode=raw,lowered_opcode=lowered,observed_records=n) for (raw,lowered),n in sorted(lowerings.items())]
 assert checked==record_census['active_global_records']
 if complete:assert checked==expected['mem_insts']
 profile['source_census_scope']='full_grid_observed' if complete else 'full_grid_predicted_from_exact_samples'
 profile['projection_record_census']=record_census
 if begin.get('entry_proof_schema'):profile['source']['entry_proof']=end['entry_proof']
 mark('census_seconds')
 if complete:profile['native_reference_digest']=captured_digest(sampled)
 mark('native_digest_seconds')
 return profile
def run_model(out,profile,placements=None):
 out.mkdir();pack=out/'profiles.pack';raw=rules.canonical_json(profile);pack.write_bytes(raw)
 index=out/'profiles.index.jsonl'
 index.write_text(json.dumps(dict(kernel_id=1,path=str(pack),offset=0,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),status=profile['status']))+'\n')
 k=profile['kernel'];g=k['grid_dims'];fields={'kernel_name':k['name'],'llama_phase':k['phase'],'grid_dim_x':g[0],'grid_dim_y':g[1],'grid_dim_z':g[2],'grid_size':k['grid_size'],'block_size':k['block_size']}
 app=out/'app.config';app.write_text(''.join('-kernel_1_%s %s\n'%(a,b) for a,b in fields.items()))
 if placements is None:placements={c:(c%48,(c//48)*1000000) for c in range(k['grid_size'])}
 assert set(placements)==set(range(k['grid_size']))
 assert all(type(sm) is int and 0<=sm<48 and type(t) is int and 0<=t<(1<<63) for sm,t in placements.values())
 by_sm=collections.defaultdict(list)
 for c,(sm,t) in placements.items():by_sm[sm].append((t,c))
 issue=out/'issue.config';issue.write_text(''.join('-trace_issued_sm_id_%d '%sm+' '.join('(1,%d,%x)'%(c,t) for t,c in sorted(by_sm[sm]))+'\n' for sm in sorted(by_sm)))
 argv=[str(F/'bin/hbserve'),'--mode','memgen','--profile-index',str(index),'--app-config',str(app),'--issue-config',str(issue),'--hw-config',str(F/'config/RTX4000Ada.paper-v1.config'),'--stats',str(out/'source-stats.json'),'--output-dir',str(out/'model'),'--include-local','false','--observe-cache','true']
 save(out/'command.json',argv)
 with (out/'stdout.log').open('x') as o,(out/'stderr.log').open('x') as e:p=subprocess.run(argv,stdout=o,stderr=e,timeout=180)
 assert p.returncode==0,'frozen Memgen execution failed'
 source=json.loads((out/'source-stats.json').read_text());assert source['status']=='PASS' and source['materialized_raw_sass_bytes']==0
 expected=profile['independent_source_census']['mem_insts'];assert source['generated_memory_instructions']==expected
 for field,value in profile.get('native_reference_digest',{}).items():assert source[field]==value,'C++ native-reference order/address digest mismatch: '+field
 with (out/'model/kernel_summary.csv').open() as f:rows=list(csv.DictReader(f));assert len(rows)==1
 census=profile['independent_source_census']
 for field in ['mem_insts','lane_accesses','read_sector_requests','write_sector_requests']:assert int(rows[0][field])==census[field],field+' independent source census mismatch'
 assert int(rows[0]['sector_requests'])==census['read_sector_requests']+census['write_sector_requests']
 observation=json.loads((out/'model/cache_observation.json').read_text())
 for field in ['classification_residual_bytes','resident_residual_bytes','producer_trigger_emission_residual_bytes']:assert observation[field]==0
 return {'source':source,'kernel_summary':rows[0],'expected_generated_instructions':expected,'independent_source_census':census,'source_sector_residual_B':0}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--transport-receipt',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);frozen_gate()
 reports=[]
 def decoded(collector,begin,end,transport):
  row={'source_launch_key':begin['source_launch_key'],'kernel_name':begin['kernel_name'],'code_sha256':begin['code_sha256'],'phase':begin['phase'],'grid':begin['grid'],'block':begin['block'],'status':'REJECTED','hardware_accuracy_accepted':False}
  try:
   # Source qualification is genuine, but normalized replay is not yet closed.
   # Files remain provisional; only the final analyze return permits acceptance.
   assert transport['status']=='PASS_SAMPLED_TRANSPORT_ONLY'
   assert end['source_closed'] and not end['overflow'] and end['unknown_space_lane_references']==0
   timings={};profile=fit(collector,begin,end,transport,timings=timings)
   model_start=time.perf_counter()
   # Preserve the existing artifact path contract. Qualification is granted
   # only by this postprocessor's final root receipt, never by file existence.
   artifact_root=a.output/begin['source_launch_key']
   result=run_model(artifact_root,profile)
   timings['memgen_seconds']=time.perf_counter()-model_start
   row.update(status='PROVISIONAL_COLD_ISOLATED_PACKED_MODEL_DIAGNOSTIC',replay_admitted=False,artifact_root=str(artifact_root),model=result,training=profile['sampling'],stage_timings=timings,lane_offset_cache=lane_offsets.cache_info()._asdict())
  except Exception:row['error']=traceback.format_exc()
  reports.append(row)
 try:
  assert stat.S_ISFIFO(os.fstat(0).st_mode)
  from postprocess_samples import analyze
  result=analyze(sys.stdin.buffer,a.transport_receipt,**capture_limits(),model_policy='strict',on_decoded_capture=decoded)
  assert result['replay_closed'] and result['transport_qualified']
  for row in reports:
   if row['status']=='PROVISIONAL_COLD_ISOLATED_PACKED_MODEL_DIAGNOSTIC':
    row.update(status='PASS_COLD_ISOLATED_PACKED_MODEL_DIAGNOSTIC',replay_admitted=True)
  result['packed_cache_models']=reports;result['packed_models_accepted']=sum(x['status'].startswith('PASS_') for x in reports)
  save(a.output/'receipt.json',result);return 0
 except Exception:
  for row in reports:
   if row['status'].startswith(('PROVISIONAL_','PASS_')):row.update(status='REJECTED_GLOBAL_REPLAY_CLOSURE',replay_admitted=False)
  save(a.output/'receipt.json',{'status':'FAIL','error':traceback.format_exc(),'packed_models_accepted':0,'packed_cache_models':reports});return 1
if __name__=='__main__':raise SystemExit(main())
