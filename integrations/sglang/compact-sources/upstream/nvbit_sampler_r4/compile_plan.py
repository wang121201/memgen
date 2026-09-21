"""Compile a root-reviewed actual NVBit launch plan into a small C++ header.
No CUDA calls, no guessed layer schedules and no module-binary identity fields.
"""
import argparse
import hashlib
import json
from pathlib import Path
from packet_stream import canonical, integer, need, parse_json, MAX_WIRE, MAX_SELECTED_CTAS

APIS={'cuLaunchKernel','cuLaunchKernel_ptsz','cuLaunchKernelEx','cuLaunchKernelEx_ptsz'}

def validate(plan):
    need(plan['schema']=='SG_NATIVE_PACKET_SAMPLE_PLAN_V1','plan schema')
    need(type(plan['source_observer_receipt_sha256']) is str and len(plan['source_observer_receipt_sha256'])==64,
         'source observer receipt SHA')
    integer(plan['max_wire_bytes'],1024,MAX_WIRE)
    integer(plan['max_received_records'],1,1_000_000_000)
    integer(plan['max_selected_kernels'],1,32768)
    rows=plan['launches'];need(0<len(rows)<=262_144,'measurement census bound')
    selected=0;expected={};seen_epochs=[]
    for row in rows:
        epoch=integer(row['epoch_id'],1);ordinal=integer(row['epoch_launch_ordinal'])
        if epoch not in expected:
            expected[epoch]=0;seen_epochs.append(epoch)
        need(epoch==seen_epochs[-1] and ordinal==expected[epoch],'epoch launch chronology')
        expected[epoch]+=1
        need(row['cuda_api'] in APIS,'unsupported dispatch cannot be sampled')
        integer(row['layer_id'],-1,1023)
        need(row.get('role') in ('warmup','measurement'),'explicit capture role')
        for field,cap in [('phase',128),('module_scope',2048),('function_name',4096)]:
            need(type(row[field]) is str and 0<len(row[field])<=cap,'plan text '+field)
        code=row['code_sha256'];need(type(code) is str and len(code)==64 and all(c in '0123456789abcdef' for c in code),'decoded SASS SHA')
        g=b=1
        for field,cap in [('grid',65536),('block',1024)]:
            d=row[field];need(type(d) is list and len(d)==3,'xyz')
            product=1
            for n in d:product*=integer(n,1,cap)
            need(product<=cap,'geometry product cap')
            if field=='grid':g=product
            else:b=product
        integer(row['dynamic_shared_bytes'],0,1<<20)
        fit,hold=row['fit_ctas'],row['holdout_ctas'];need(type(fit) is list and type(hold) is list,'CTA list')
        need(len(set(fit+hold))==len(fit)+len(hold) and len(fit)+len(hold)<=MAX_SELECTED_CTAS,'CTA overlap/cap')
        for c in fit+hold:integer(c,0,g-1)
        need(not hold or fit,'holdout without fit')
        if fit:selected+=1
        attrs=row['launch_attributes'];need(type(attrs) is list and len(attrs)<=64,'Ex attributes')
        if 'Ex' not in row['cuda_api']:need(attrs==[],'direct attributes')
        for a in attrs:
            need(a.get('metadata_decoded') is True,'unknown Ex attribute')
            aid=integer(a['id'],0,16)
            # Values with cross-process pointer/event identity, clustered grids,
            # or cooperative residency require a separate observed contract.
            need(aid in (0,3,6,8,9,10,14,16) or
                 (aid in (4,11) and a.get('value')==[1,1,1]),
                 'Ex attribute needs explicit dynamic sampling qualification')
        # is_load/is_store/operand roles come from NVBit instructions at runtime,
        # never from function names or Python module annotations.
    need(0<selected<=plan['max_selected_kernels'],'selected kernel cap')
    return dict(launches=len(rows),selected_kernels=selected,epochs=len(expected),
                selected_ctas=sum(len(x['fit_ctas'])+len(x['holdout_ctas']) for x in rows))


def cpp_string(x):
    # JSON string escaping is valid for our C++ ASCII strings; escape UTF-8 as
    # fixed octal bytes to avoid universal character ambiguity in compiler input.
    b=x.encode();out='"'
    for n in b:
        if 32<=n<127 and n not in (34,92):out+=chr(n)
        elif n==34:out+='\\"'
        elif n==92:out+='\\\\'
        else:out+='\\%03o'%n
    return out+'"'


def compile_header(plan):
    stats=validate(plan);digest=hashlib.sha256(canonical(plan)).hexdigest()
    rows=[];ctas=[]
    for r in plan['launches']:
        attrs=json.dumps(r['launch_attributes'],separators=(',',':'),ensure_ascii=True)
        values=[str(r['epoch_id']),str(r['epoch_launch_ordinal']),str(r['layer_id'])]
        values += [cpp_string(r[k]) for k in ('phase','module_scope','cuda_api','function_name','code_sha256','role')]
        values += ['{'+','.join(map(str,r[k]))+'}' for k in ('grid','block')]
        values += [str(r['dynamic_shared_bytes']),cpp_string(attrs)]
        for name in ['fit_ctas','holdout_ctas']:
            values.extend([str(len(ctas)),str(len(r[name]))]);ctas.extend(r[name])
        rows.append('  {'+','.join(values)+'}')
    text='''#pragma once
namespace sgsample {
static const char *PLAN_SHA256=%s;
static const char *OBSERVER_RECEIPT_SHA256=%s;
static const uint64_t MAX_WIRE_BYTES=%d, MAX_RECEIVED_RECORDS=%d;
static const uint64_t MAX_SELECTED_KERNELS=%d;
struct PlanLiteral {
 uint64_t epoch,ordinal;int layer;
 const char *phase,*module,*api,*name,*code,*role;
 std::array<uint64_t,3> grid,block;uint64_t shared;const char *attrs;
 uint64_t fit_offset,fit_size,hold_offset,hold_size;
};
static const uint64_t plan_ctas[]={%s};
static const PlanLiteral plan_literals[]={
%s
};
static std::vector<Plan> make_plans() {
 std::vector<Plan> out;out.reserve(sizeof(plan_literals)/sizeof(plan_literals[0]));
 for(const auto &r:plan_literals) {
  out.push_back(Plan{r.epoch,r.ordinal,r.layer,r.phase,r.module,r.api,r.name,r.code,r.role,
   r.grid,r.block,r.shared,r.attrs,
   std::vector<uint64_t>(plan_ctas+r.fit_offset,plan_ctas+r.fit_offset+r.fit_size),
   std::vector<uint64_t>(plan_ctas+r.hold_offset,plan_ctas+r.hold_offset+r.hold_size)});
 }
 return out;
}
static const std::vector<Plan> plans=make_plans();
}
'''%(cpp_string(digest),cpp_string(plan['source_observer_receipt_sha256']),plan['max_wire_bytes'],plan['max_received_records'],plan['max_selected_kernels'],','.join(map(str,ctas)) or '0',',\n'.join(rows))
    return text,dict(schema='SG_COMPILED_SAMPLE_PLAN_V1',plan_sha256=digest,**stats,
        header_sha256=hashlib.sha256(text.encode()).hexdigest(),GPU_execution=False,
        inherited_scope='eager TP1 single application stream; sampler inserts per-selected-launch completion')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();need(a.plan.stat().st_size<=256<<20,'bounded plan file');plan=parse_json(a.plan.read_bytes())
    header,receipt=compile_header(plan)
    with a.output.open('x') as f:f.write(header)
    print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
