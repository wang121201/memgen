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


def native_full_ctas(row):
    """Native-full CTA selection: every launch gets native address coverage, but
    large grids are coordinate-stratified rather than fully enumerated, so the
    16 GiB wire ceiling holds. This is the historical native-full strategy
    (small-domain enumeration + large-domain stratification), not the every-CTA
    enumeration that exhausts the wire. A stratified grid still carries a fresh
    independent holdout: the fitter requires one for any non-complete grid
    (sglang_sample_to_packed.fit asserts `hold` when the sampled set is not the
    whole grid), and the holdout is what validates the affine witness rather
    than a coverage claim.
    """
    gx, gy, gz = row['grid']
    n = gx * gy * gz
    name = row['function_name']
    # Small heterogeneous grids and data-dependent gather addresses are
    # enumerated rather than inferred from an inadequate affine witness. The
    # threshold is 32, not the historical 64: cutlass::Kernel2 grids at n=48/64
    # (8x6 / 8x8) are affine (their STG/LDG rules carry cta_x_stride/cta_y_stride
    # and the sparse sampler already admits them exactly), so enumerating them
    # full-grid exhausts the wire (6272 CTAs x ~5707 records/CTA ~= 21 GiB) for
    # no accuracy gain. n<=32 still enumerates genuinely small grids.
    small_gemm = n <= 32 and (
        'cutlass::Kernel2' in name or
        name == 'ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn')
    singleton_rope_rows = 'BatchQKApplyRotary' in name and gx == 1
    observed_irregular_copy = 'direct_copy_kernel_cuda' in name and n <= 8192
    variable_merge = 'PersistentVariableLengthMergeStatesKernel' in name and n <= 8192
    if n <= 32 or small_gemm or singleton_rope_rows or observed_irregular_copy or variable_merge or ('indexSelectLargeIndex' in name and n <= 4096):
        return list(range(n)), []
    train = set()
    if 'BatchQKApplyRotary' in name and gy > 1 and gz == 1:
        # Every y category needs native coverage; this does not assert
        # within-row homogeneity. When gx is so small that the {0,1,gx//2,gx-1}
        # witness covers every x, there is no room for a fresh holdout, so the
        # grid is enumerated instead (the x axis is too short to stratify).
        if gx <= 4:
            return list(range(n)), []
        for y in range(gy):
            train.update(y * gx + x for x in {0, 1, gx // 2, gx - 1} if x < gx)
        categories = [[y * gx + x for x in range(gx)] for y in range(gy)]
        wanted = 2
    else:
        stride = 1
        for dim in row['grid']:
            train.update(k * stride for k in {0, 1, 2, 4, 8, dim // 4, dim // 2, dim - 1} if k < dim)
            stride *= dim
        # Cross-axis witnesses prevent a purely local affine fit from being
        # admitted before mid/late x positions have constrained the model.
        for y, z in {(gy - 1, 0), (0, gz - 1), (gy - 1, gz - 1)}:
            train.update(x + gx * (y + gy * z) for x in {gx // 2, gx - 1})
        categories = [list(range(n))]
        wanted = 2
    hold = []
    for category in categories:
        remaining = [c for c in category if c not in train]
        # Deterministic, address-independent ordering frozen before capture.
        remaining.sort(key=lambda c: hashlib.sha256(
            (row['code_sha256'] + repr(row['grid']) + ':fresh-holdout-v1:' + str(c)).encode()).digest())
        need(len(remaining) >= wanted, 'insufficient fresh holdout for %s grid %s' % (name, row['grid']))
        hold.extend(remaining[:wanted])
    need(train, 'Native-full stratification produced no CTA')
    need(set(hold).isdisjoint(train), 'native-full holdout overlaps training')
    return sorted(train), sorted(hold)


def select(rows,layer=0,capture_mode='sparse'):
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
        if capture_mode=='native-full':
            source=i;why='native_full_launch_capture'
            # Every launch gets native address coverage, but each large grid is
            # coordinate-stratified (not fully enumerated) so the 16 GiB wire
            # ceiling holds. The every-CTA enumeration exhausts the wire; the
            # historical stratified strategy does not.
            fit,hold=native_full_ctas(r)
        else:
            fit,hold=ctas(r['grid']) if source==i else ([],[])
        counts[why]+=1
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
    p.add_argument('--capture-mode',choices=('sparse','native-full'),default='sparse')
    a=p.parse_args()
    host=json.loads(a.host_finish.read_text());need(host['status']=='PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE','Host incomplete')
    contract=host['input_contract'];need(0<=a.layer<contract['layers'],'Selected layer out of range')
    rows,finish=read_census(a.journal)
    seen={(r['role'],r['phase']) for r in rows};expected={(role,phase) for role in ('warmup','measurement') for phase in contract['phases']}
    need(seen==expected,'Complete warmup/measurement phase population required')
    for role,phase in expected:
        need({r['layer_id'] for r in rows if r['role']==role and r['phase']==phase and r['layer_id']>=0}==set(range(contract['layers'])),'Original layer census incomplete')
    plan,bindings,counts=select(rows,a.layer,a.capture_mode);selected=sum(bool(r['fit_ctas']) for r in plan)
    need(selected<=32768,'Task-private sampler supports at most 32768 selected kernels per process')
    # A native-full plan covers every launch, so its record stream is larger than
    # the sparse single-layer plan's, but each large grid is coordinate-stratified
    # (not fully enumerated), so the bound is the wire ceiling, not the 35.8x
    # every-CTA enumeration that exhausted it. The sampler's own cap is 1e9
    # records (compile_plan.py), and the wire ceiling (16 GiB / 616 B packet) is
    # the real constraint; the plan states a record bound well above the sparse
    # 12M ceiling without pretending the every-CTA enumeration fits.
    max_records=200_000_000 if a.capture_mode=='native-full' else 12_000_000
    result=dict(schema='SG_NATIVE_PACKET_SAMPLE_PLAN_V1',capture_mode=a.capture_mode,
        native_full_model=a.capture_mode=='native-full',source_observer_receipt_sha256=sha(a.journal/'finish.json'),
        max_wire_bytes=16<<30,max_received_records=max_records,max_selected_kernels=selected,launches=plan)
    a.output.mkdir(parents=True,exist_ok=False)
    for n,v in [('sample-plan.json',result),('layer-bindings.json',dict(schema='SGLANG_LAYER_PROFILE_BINDINGS_V1',input_contract=contract,
        capture_mode=a.capture_mode,native_full_model=a.capture_mode=='native-full',
        primary_layer=a.layer,bindings=bindings,source_census_launches=len(rows),selected_launches=selected,counts=counts,
        full_raw_capture=False,cache_traffic_multiplication=False,profile_expansion_executed=False,
        full_model_accuracy_accepted=False)),('census.json',dict(status='PASS_NATIVE_CENSUS_AND_NATIVE_FULL_CAPTURE_PLAN' if a.capture_mode=='native-full' else 'PASS_NATIVE_CENSUS_AND_ONE_LAYER_SAMPLE_SELECTION',
        source_journal=str(a.journal),observer_finish_sha256=sha(a.journal/'finish.json'),host_finish_sha256=sha(a.host_finish),
        input_contract=contract,launches=len(rows),selected_launches=selected,counts=counts,
        selected_ctas=sum(len(r['fit_ctas'])+len(r['holdout_ctas']) for r in plan),warm_prefix_preserved=True))]:
        (a.output/n).write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps(dict(status='PASS_PLAN_ONLY_NOT_SAMPLED',launches=len(rows),selected_launches=selected,counts=counts)))


if __name__=='__main__': main()
