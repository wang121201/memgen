"""Exact observed code/site admission and an addressless guard/source ledger."""
from pathlib import Path
import collections,hashlib,json,math
POLICY='validated_ldg_source_predicate'
def need(ok,why):
 if not ok:raise ValueError(why)
class Qualification:
 def __init__(self,scope):
  need(scope['status']=='PASS_NATIVE_SOURCE_PREDICATE_SCOPED_CENSUS','unqualified source scope')
  need(scope['hardware_repeats']==3 and scope['predicate_interfaces']==2,'missing independent source evidence')
  self.scope=scope
  self.sites={(x['function_id'],x['pc'],x['opcode'],x['width']) for x in scope['sites']}
  need(len(self.sites)==len(scope['sites'])==24,'duplicate/missing scoped sites')
 def begin(self,b):
  for k in ['code_sha256','kernel_name','grid','block']:need(b[k]==self.scope[k],'unqualified launch '+k)
  need(b['epoch_id']==2 and b['epoch_launch_ordinal']==0 and b['role']=='measurement','unqualified epoch/role')
  need(b['phase']=='NativeMLP' and b['layer_id']==1 and b['module_scope']=='model.layers.1.mlp.gate_up_proj','unqualified native scope')
  need(b['fit_ctas']==list(range(math.prod(b['grid']))) and b['holdout_ctas']==[],'complete observed grid required')
 def record(self,r):
  need(r['code_sha256']==self.scope['code_sha256'],'unqualified record code')
  need((r['function_id'],r['pc'],r['opcode'],r['width']) in self.sites,'unqualified source site')
  need(r['source_control_kind']=='ldg_predicate_candidate' and r['projection_kind']=='predicated_global_read_candidate','unqualified source kind')
  need(r['is_load'] is True and r['is_store'] is False and r['transfer_width']==r['width']==16 and r['transfer_policy']==2,'unqualified source direction/width')
_qualification=None
def measurement_qualification():
 global _qualification
 if _qualification is None:
  root=Path(__file__).resolve().parent.parent
  manifest=json.loads((root/'source-qualification.json').read_text());p=Path(manifest['scope_path']);raw=p.read_bytes()
  need(hashlib.sha256(raw).hexdigest()==manifest['scope_sha256'],'qualification identity changed')
  audit=Path(manifest['audit_finish_path']);need(hashlib.sha256(audit.read_bytes()).hexdigest()==manifest['audit_finish_sha256'],'qualification audit changed')
  a=json.loads(audit.read_text());need(a['status']=='PASS_NATIVE_PARALLEL_SOURCE_AND_MECHANISM_AUDIT' and not a.get('error'),'qualification audit not accepted')
  _qualification=Qualification(json.loads(raw))
 return _qualification

class WarmQualification(Qualification):
 def __init__(self,scope):
  need(scope['status']=='PASS_WARM_SOURCE_PREDICATE_SCOPED_CENSUS','unqualified warm source scope')
  need(scope['hardware_repeats']==1 and scope['predicate_interfaces']==2,'warm evidence population changed')
  need(scope['epoch_id']==1 and scope['epoch_launch_ordinal']==0 and scope['role']=='warmup','warm scope identity')
  self.scope=scope
  self.sites={(x['function_id'],x['pc'],x['opcode'],x['width']) for x in scope['sites']}
  need(len(self.sites)==len(scope['sites'])==24 and all(x['transfer_policy']==2 for x in scope['sites']),'warm site population')
 def begin(self,b):
  for k in ['epoch_id','epoch_launch_ordinal','role','phase','module_scope','layer_id']:need(b[k]==self.scope[k],'unqualified warm '+k)
  # Reuse geometry/full-grid checks. Measurement evidence is never relabeled.
  super().begin(dict(b,epoch_id=2,role='measurement'))
_active_qualification=None
_warm_qualification=None
def qualification(begin=None):
 global _active_qualification,_warm_qualification
 if begin is None:return _active_qualification or measurement_qualification()
 if begin['epoch_id']==1:
  if _warm_qualification is None:
   root=Path(__file__).resolve().parent.parent
   manifest=json.loads((root/'warm-source-qualification.json').read_text());p=Path(manifest['scope_path']);raw=p.read_bytes()
   need(hashlib.sha256(raw).hexdigest()==manifest['scope_sha256'],'warm scope changed')
   audit=Path(manifest['audit_finish_path']);need(hashlib.sha256(audit.read_bytes()).hexdigest()==manifest['audit_finish_sha256'],'warm audit changed')
   a=json.loads(audit.read_text());need(a['status']=='PASS_N1_WARM_SOURCE_INDEPENDENT_AUDIT' and not a.get('error'),'warm audit not accepted')
   _warm_qualification=WarmQualification(json.loads(raw))
  q=_warm_qualification
 else:q=measurement_qualification()
 q.begin(begin);_active_qualification=q;return q

class GuardSourceLedger:
 def __init__(self,begin):
  self.q=qualification(begin);self.q.begin(begin);self.counts=collections.Counter();self.groups=collections.Counter();self.warps={}
 def record(self,r,direction,width,mask,ranges):
  guard=r['effective_mask'];need(mask&~guard==0,'source access outside guard')
  sectors=len({s for x in ranges for s in range(x['offset_bytes']//32,(x['offset_bytes']+x['byte_count']-1)//32+1)})
  candidate=r.get('projection_kind')=='predicated_global_read_candidate'
  key=(r['cta_linear_id'],r['cta_warp_id'],r['function_id'],r['pc'],r['opcode'],r['active_mask'],guard,mask,direction,width,candidate)
  self.groups[key]+=1;need(len(self.groups)<=400000,'provenance group cap')
  self.counts.update(raw_events=1,nonempty_events=int(bool(mask)),zero_source_events=int(not mask),guard_lanes=guard.bit_count(),source_lanes=mask.bit_count(),active_lanes=r['active_mask'].bit_count(),candidate_events=int(candidate),read_sectors=sectors if direction=='load' else 0,write_sectors=sectors if direction=='store' else 0)
  wk=(key[0],key[1])
  if wk not in self.warps:self.warps[wk]=dict(access=hashlib.sha256(),provenance=hashlib.sha256(),records=0)
  signature=[r['function_id'],r['pc'],r['opcode'],mask,direction,width,[x['offset_bytes'] for x in ranges]]
  self.warps[wk]['access'].update(json.dumps(signature,separators=(',',':')).encode()+b'\n')
  self.warps[wk]['provenance'].update(json.dumps([guard,*signature],separators=(',',':')).encode()+b'\n');self.warps[wk]['records']+=1
 def verify_decoded(self,samples):
  need(set(samples)==set(self.warps),'decoded warp population changed');zero=set();nonzero=set()
  for wk,frames in samples.items():
   h=hashlib.sha256();need(len(frames)==self.warps[wk]['records'],'codec dropped dynamic event')
   for m in frames:
    mask=sum(1<<x['lane'] for x in m['lanes']);need(mask==m['mask'] and len({x['lane'] for x in m['lanes']})==len(m['lanes']),'decoded lane/mask mismatch')
    signature=[m['function_id'],m['pc'],m['opcode'],mask,'load' if m['op']==ord('R') else 'store',m['mem_width'],[x['addr'] for x in m['lanes']]]
    h.update(json.dumps(signature,separators=(',',':')).encode()+b'\n');(nonzero if mask else zero).add(wk[0])
   need(h.hexdigest()==self.warps[wk]['access'].hexdigest(),'codec changed actual memory semantics')
  self.zero_access_ctas=sorted(zero-nonzero);return self.zero_access_ctas
 def receipt(self,raw_count):
  c=self.counts;q=self.q.scope
  need(c['raw_events']==raw_count==q['raw_records'],'raw event population changed')
  need(c['nonempty_events']==q['nonempty_records'] and c['zero_source_events']==q['zero_source_records'],'zero-source population changed')
  need(c['raw_events']==c['nonempty_events']+c['zero_source_events'],'event residual')
  need(c['read_sectors']==q['source_read_sectors'] and c['write_sectors']==q['source_write_sectors'],'hardware source census mismatch')
  rows=[dict(cta=k[0],warp=k[1],function_id=k[2],pc=k[3],opcode=k[4],active_mask=k[5],guard_mask=k[6],source_access_mask=k[7],direction=k[8],width=k[9],qualified_candidate=k[10],events=v) for k,v in sorted(self.groups.items())]
  need(sum(x['events'] for x in rows)==raw_count,'provenance classification residual')
  for field,maskfield in [('guard_lanes','guard_mask'),('source_lanes','source_access_mask'),('active_lanes','active_mask')]:need(sum(x['events']*x[maskfield].bit_count() for x in rows)==c[field],'lane provenance residual')
  need(hasattr(self,'zero_access_ctas'),'decoded projection not verified')
  warps=[dict(cta=k[0],warp=k[1],raw_events=v['records'],access_sha256=v['access'].hexdigest(),guard_source_sha256=v['provenance'].hexdigest()) for k,v in sorted(self.warps.items())]
  return dict(status='PASS_SCOPED_SOURCE_PROJECTION_AND_GUARD_PROVENANCE',counts=dict(c),groups=rows,warp_digests=warps,zero_access_ctas=self.zero_access_ctas,classification_residual=0,source_read_sector_residual=0,source_write_sector_residual=0,scope=q,guard_not_used_as_access_mask=True,raw_records_saved=False)
