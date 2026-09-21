#!/usr/bin/env python3
"""Select one decoder layer plus global/boundary exceptions from a real census.

Layer reuse is a declared profile model. This creates a structural binding plan;
it does not assert that target addresses have already been rebound or validated.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


FIELDS=('function_name','code_sha256','grid','block','dynamic_shared_bytes','launch_attributes',
        'epoch_id','phase','layer_id','module_scope','cuda_api','role')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
    if not ok: raise ValueError(msg)


def read_census(journal):
    finish=json.loads((journal/'finish.json').read_text())
    need(finish['status']=='PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE','Metadata observer did not close')
    need(finish['epoch_begin_count']==finish['epoch_end_count'] and finish['active_epoch']==0,'Epoch closure')
    for k in ('launch_error_count','unsupported_dispatch_count','graph_node_callback_count','unknown_launch_attribute_count','open_context_count'):
        need(finish[k]==0,'Observer error '+k)
    epochs=Counter();rows=[];after={}
    for line in (journal/'launch-journal.jsonl').open():
        x=json.loads(line)
        if x.get('type')!='launch' or not x.get('epoch_id'): continue
        if x['edge']!='before':
            need(x['launch_id'] not in after,'Duplicate launch return');after[x['launch_id']]=x;continue
        need(x['scope_bound'] and x['metadata_supported'] and x['static_inspection_performed'],'Unqualified static launch')
        r={k:x[k] for k in FIELDS};r['source_launch_id']=x['launch_id']
        r['epoch_launch_ordinal']=epochs[r['epoch_id']];epochs[r['epoch_id']]+=1;rows.append(r)
    need(rows and set(after)=={r['source_launch_id'] for r in rows},'Launch begin/return coverage')
    return rows,finish


def signature(row):
    # Kernel identity, geometry and per-module occurrence must all match.
    # Canonical shared-module names (e.g. RoPE) remain unchanged.
    return (row['role'],row['phase'],re.sub(r'\.layers\.\d+(?=\.|$)', '.layers.<L>',row['module_scope']),
            row['function_name'],row['code_sha256'],tuple(row['grid']),tuple(row['block']),
            row['dynamic_shared_bytes'],json.dumps(row['launch_attributes'],sort_keys=True),row['cuda_api'])


def ctas(grid):
    n=math.prod(grid)
    if n<=5: return list(range(n)),[]
    train={0};stride=1
    for dim in grid:
        train.update(k*stride for k in (1,2,4) if k<dim);stride*=dim
    hold={c for c in (3,5,n//2,n//3,n-1,n-2) if 0<=c<n}-train
    need(hold,'Independent CTA holdout required')
    return sorted(train),sorted(hold)


def select(rows,layer=0):
    occurrences=Counter();keys=[]
    for r in rows:
        sig=signature(r);key=(r['layer_id'],sig)
        keys.append((sig,occurrences[key]));occurrences[key]+=1
    sources={}
    for i,r in enumerate(rows):
        if r['layer_id']==layer: sources.setdefault(keys[i],i)
    plan=[];bindings=[];counts=Counter()
    for i,r in enumerate(rows):
        if r['layer_id']<0:
            source=i;why='global_boundary_own_sample'
        elif keys[i] in sources:
            source=sources[keys[i]];why='primary_layer_sample' if source==i else 'same_signature_layer_profile_expansion'
        else:
            source=i;sources[keys[i]]=i;why='boundary_or_kernel_shape_exception_own_sample'
        counts[why]+=1
        fit,hold=ctas(r['grid']) if source==i else ([],[])
        plan.append({k:v for k,v in r.items() if k!='source_launch_id'}|dict(fit_ctas=fit,holdout_ctas=hold))
        bindings.append(dict(target_launch_id=r['source_launch_id'],template_launch_id=rows[source]['source_launch_id'],
            target_layer=r['layer_id'],template_layer=rows[source]['layer_id'],role=r['role'],phase=r['phase'],
            target_key=[r['epoch_id'],r['epoch_launch_ordinal']],template_key=[rows[source]['epoch_id'],rows[source]['epoch_launch_ordinal']],
            policy=why,address_rebinding_status='REQUIRES_SAME_PROCESS_TENSOR_ROOT_BINDING',profile_admitted=False))
    return plan,bindings,dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--host-finish',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',type=int,default=0)
    a=p.parse_args()
    host=json.loads(a.host_finish.read_text());need(host['status']=='PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE','Host incomplete')
    contract=host['input_contract'];need(0<=a.layer<contract['layers'],'Selected layer out of range')
    rows,finish=read_census(a.journal)
    seen={(r['role'],r['phase']) for r in rows};expected={(role,phase) for role in ('warmup','measurement') for phase in contract['phases']}
    need(seen==expected,'Complete warmup/measurement phase population required')
    for role,phase in expected:
        need({r['layer_id'] for r in rows if r['role']==role and r['phase']==phase and r['layer_id']>=0}==set(range(contract['layers'])),'Original layer census incomplete')
    plan,bindings,counts=select(rows,a.layer);selected=sum(bool(r['fit_ctas']) for r in plan)
    need(selected<=32768,'Task-private sampler supports at most 32768 selected kernels per process')
    result=dict(schema='SG_NATIVE_PACKET_SAMPLE_PLAN_V1',source_observer_receipt_sha256=sha(a.journal/'finish.json'),
        max_wire_bytes=8<<30,max_received_records=12_000_000,max_selected_kernels=selected,launches=plan)
    a.output.mkdir(parents=True,exist_ok=False)
    for n,v in [('sample-plan.json',result),('layer-bindings.json',dict(schema='SGLANG_LAYER_PROFILE_BINDINGS_V1',input_contract=contract,
        primary_layer=a.layer,bindings=bindings,source_census_launches=len(rows),selected_launches=selected,counts=counts,
        full_raw_capture=False,cache_traffic_multiplication=False,profile_expansion_executed=False,
        full_model_accuracy_accepted=False)),('census.json',dict(status='PASS_NATIVE_CENSUS_AND_ONE_LAYER_SAMPLE_SELECTION',
        source_journal=str(a.journal),observer_finish_sha256=sha(a.journal/'finish.json'),host_finish_sha256=sha(a.host_finish),
        input_contract=contract,launches=len(rows),selected_launches=selected,counts=counts,
        selected_ctas=sum(len(r['fit_ctas'])+len(r['holdout_ctas']) for r in plan),warm_prefix_preserved=True))]:
        (a.output/n).write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps(dict(status='PASS_PLAN_ONLY_NOT_SAMPLED',launches=len(rows),selected_launches=selected,counts=counts)))


if __name__=='__main__': main()
