"""Persist compact fitted profiles, never raw records; commit only at replay EOF."""
from pathlib import Path
import argparse,hashlib,json,os,stat,sys,time,traceback
import sglang_sample_to_packed as p
# The adapter owns the policy list; this driver only chooses from it.
from memory_projection import MODEL_POLICIES

def capture_limits(plan):
 # The plan is the single source of truth for how many records the sampler may
 # emit: a sparse single-layer plan states 12M, a native-full plan states 200M,
 # because it selects every launch rather than one layer's launches.  Large
 # grids remain coordinate-stratified; this is not a full-CTA raw trace.
 # Deriving the replay limits from it keeps the two ceilings from drifting; the
 # historical hard-coded 12M silently truncated a native-full stream into an
 # empty replay. A missing/legacy plan keeps the historical sparse defaults.
 records=int(plan.get('max_received_records',12000000)) if plan else 12000000
 assert 1<=records<=1_000_000_000,'plan record ceiling'
 if plan is not None:
  assert 1024<=int(plan.get('max_wire_bytes',16<<30))<=16<<30,'plan wire ceiling'
 if records<=12000000:
  # The sparse single-layer contract is unchanged: 12M records, 512MiB encoded,
  # 1M per kernel, 24GiB decoded. Only a native-full plan raises these.
  return dict(max_records=12000000,max_kernel_records=1000000,
              max_encoded_bytes=512<<20,max_decoded_bytes=24<<30)
 # A native-full plan selects every launch, but each large grid is coordinate-
 # stratified, so no single kernel holds the whole record bound. The per-kernel
 # ceiling stays at the historical 1M (the largest stratified kernel is ~97K
 # records: cutlass gemm 17 CTAs x 5707 records/CTA); only the total record bound
 # and the encoded ceiling scale with the plan. Setting max_kernel_records to the
 # total bound is wrong: it trips SampleCollector's own 1M per-kernel budget.
 return dict(max_records=records,max_kernel_records=1000000,
             max_encoded_bytes=max(512<<20,min(64<<30,records*512)),
             max_decoded_bytes=24<<30)


def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--transport-receipt',type=Path,required=True);a.add_argument('--plan',type=Path);a.add_argument('--model-policy',choices=MODEL_POLICIES,default='strict');a=a.parse_args()
 a.output.mkdir(parents=True);p.frozen_gate();rows=[];offset=0;profile_bytes=0
 plan=json.loads(a.plan.read_text()) if a.plan else None;limits=capture_limits(plan)
 pack=a.output/'profiles.pack';index=a.output/'profiles.index.jsonl'
 # Every row retains an actual launch identity. No copy from another layer,
 # zero filling rejected launches, or source-count-based traffic injection.
 def decoded(collector,begin,end,transport):
  nonlocal offset,profile_bytes
  started=time.monotonic();key=begin['source_launch_key']
  row=dict(source_launch_key=key,phase=begin['phase'],role=begin.get('role'),layer_id=begin['layer_id'],code_sha256=begin['code_sha256'],kernel_name=begin['kernel_name'],status='REJECTED',hardware_accuracy_accepted=False)
  try:
   timings={};v=p.fit(collector,begin,end,transport,timings=timings)
   kid=next(i for i,x in enumerate(transport['kernels'],1) if x['source_launch_key']==key)
   v['kernel']['id']=kid
   if 'native_reference_digest' in v:v['original_isolated_reference_digest']=v.pop('native_reference_digest')
   v['model'].update(cache_entry='WARM_PREFIX_CONTINUOUS_REPLAY_PENDING',complete_model=False)
   raw=p.rules.canonical_json(v);assert profile_bytes+len(raw)<=8<<30,'bounded packed profile bytes'
   with pack.open('ab') as f:f.write(raw)
   r=dict(kernel_id=kid,source_launch_key=key,path=str(pack),offset=offset,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),status=v['status'])
   with index.open('a') as f:f.write(json.dumps(r)+'\n')
   offset+=len(raw);profile_bytes+=len(raw)
   row.update(status='PROVISIONAL_PROFILE_EXACT_SAMPLES',profile_index=r,census=v['independent_source_census'],sampling=v['sampling'],stage_timings=timings)
  except Exception as e:row.update(error=traceback.format_exc(),rejection_reason=str(e))
  row['seconds']=time.monotonic()-started;rows.append(row)
  print(json.dumps(dict(progress=len(rows),key=key,status=row['status'],seconds=row['seconds'],reason=row.get('rejection_reason'))),flush=True)
 result={}
 try:
  assert stat.S_ISFIFO(os.fstat(0).st_mode)
  from postprocess_samples import analyze
  result=analyze(sys.stdin.buffer,a.transport_receipt,model_policy=a.model_policy,on_decoded_capture=decoded,compile_native_templates=False,**limits)
  assert result['replay_closed'] and result['transport_qualified']
  for row in rows:
   if row['status'].startswith('PROVISIONAL_'):row.update(status='PASS_PROFILE_EXACT_SAMPLES',replay_admitted=True)
  allkeys=[x['source_launch_key'] for x in result['kernels']];mapped={x['source_launch_key'] for x in rows}
  # Rejection before decoding is still a missing source launch, never zero.
  for r in result['kernels']:
   if r['source_launch_key'] not in mapped:rows.append(dict(source_launch_key=r['source_launch_key'],status='REJECTED_BEFORE_PACKING',reason=r['candidate_rejections'],phase=r['phase'],layer_id=r['layer_id'],kernel_name=r['kernel_name']))
  accepted=sum(r['status']=='PASS_PROFILE_EXACT_SAMPLES' for r in rows)
  assert len(rows)==len(allkeys) and {r['source_launch_key'] for r in rows}==set(allkeys)
  result.update(packed_profiles=rows,packed_models_accepted=accepted,profile_bytes=profile_bytes,
     packed_models_missing=len(allkeys)-accepted,
     full_launch_population_complete=accepted==len(allkeys),complete_inference_model_comparison=False,
     full_source_hardware_qualification=False,raw_memory_trace_bytes=0)
  code=0
 except Exception:
  for row in rows:
   if row['status'].startswith(('PROVISIONAL_','PASS_')):row.update(status='REJECTED_GLOBAL_REPLAY_CLOSURE',replay_admitted=False)
  result=dict(status='FAIL',error=traceback.format_exc(),packed_models_accepted=0,packed_profiles=rows,complete_inference_model_comparison=False);code=1
 raw=(json.dumps(result,indent=2)+'\n').encode();assert len(raw)<=128<<20,'full source receipt cap'
 with (a.output/'receipt.json').open('xb') as f:f.write(raw)
 return code

if __name__=='__main__':raise SystemExit(main())
