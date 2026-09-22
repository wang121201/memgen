#!/usr/bin/env python3
"""Portable CPU cache regression; fresh output, no wall-clock cutoff or GPU."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def make_fixture(out):
    source = ROOT / 'release/fixtures/smoke'
    out.mkdir()
    rows = [json.loads(line) for line in (source/'profiles.index.jsonl').read_text().splitlines()]
    for row in rows:
        row['path'] = str(source/'profiles.pack.jsonl')
    (out/'profiles.index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'app.config').write_bytes((source/'configs/app.config').read_bytes())
    (out/'issue.config').write_text('-trace_issued_sm_id_0 (1,0,100) (2,0,200)\n'
                                  '-trace_issued_sm_id_1 (1,1,200) (2,1,180)\n'
                                  '-trace_issued_sm_id_2 (1,2,80)\n')
    context = dict(schema='MEMGEN_R4_CONTEXT_V1', model_id='r4-small-shared-20260922',
        profile_index_sha256=sha(out/'profiles.index.jsonl'),app_config_sha256=sha(out/'app.config'),
        claim_boundary='synthetic test allocations, never hardware evidence',
        kernels=[dict(kernel_id=i,shared_kib=s,shared_evidence='synthetic fixture, no hardware measurement',allocations=[dict(id=i,base=b,bytes=512)])
                 for i,s,b in [(1,8,4096),(2,16,8192)]])
    (out/'context.json').write_text(json.dumps(context))
    return context

def variant_fixture(out,change,index_change=None):
    context=make_fixture(out)
    original=ROOT/'release/fixtures/smoke/profiles.pack.jsonl'
    profiles=[json.loads(line) for line in original.read_text().splitlines()]
    change(profiles[0]);rows=[];offset=0
    with (out/'variant.pack').open('xb') as f:
        for i,profile in enumerate(profiles,1):
            raw=(json.dumps(profile)+'\n').encode();f.write(raw)
            rows.append(dict(kernel_id=i,path=str(out/'variant.pack'),offset=offset,bytes=len(raw),
                             sha256=hashlib.sha256(raw).hexdigest(),status=profile['status']))
            offset+=len(raw)
    if index_change:index_change(rows[0])
    (out/'profiles.index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    context['profile_index_sha256']=sha(out/'profiles.index.jsonl')
    (out/'context.json').write_text(json.dumps(context))

def intervals(profile):
    profile['schema']['version']=12
    profile.pop('cta_class_by_id')
    profile['sampling']=dict(training_ctas=[0,1,2],holdout_ctas=[])
    profile['source']=dict(launch=dict(fit_ctas=[0,1,2],holdout_ctas=[]))
    profile['structural_class_selector']=dict(kind='linear_cta_intervals',intervals=[
        dict(start=0,stop=1,class_id='active'),dict(start=1,stop=2,class_id='empty'),dict(start=2,stop=3,class_id='active')])
    for cls in profile['structural_classes']:
        ctas=[0,2] if cls['class_id']=='active' else [1]
        cls.update(domain_cta_count=len(ctas),ctas=ctas,independent_holdout_ctas=[],observed_domain_complete=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',gpu_used=False,wallclock_cutoff=False,hardware_accuracy_accepted=False,steps=[])
    def save(): (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(name,cmd,success=True):
        start=time.monotonic();row=dict(name=name,command=list(map(str,cmd)));result['steps'].append(row);save()
        with (out/(name+'.log')).open('x') as f:
            p=subprocess.run(row['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                             env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
        row.update(returncode=p.returncode,seconds=time.monotonic()-start);save()
        assert (p.returncode==0)==success,(name,p.returncode)
        return (out/(name+'.log')).read_text()
    def build(name,src):
        run('build-'+name,['mpic++','-std=c++17','-O2','-ffunction-sections','-fdata-sections',
            '-Wl,--gc-sections',src,'-l:libzstd.so.1','-lz','-lboost_mpi','-lboost_serialization',
            '-lcrypto','-pthread','-o',out/name])
    try:
        run('legacy-smoke',['bash',ROOT/'scripts/run_cpu_smoke.sh',out/'legacy'])
        binary=out/'legacy/build/hbserve';config=ROOT/'release/config/RTX4000Ada.r4.config'
        result.update(binary_sha256=sha(binary),config_sha256=sha(config))
        resolved=json.loads(run('describe',[binary,'--describe-hardware-config',config]))
        assert resolved['hardware_accuracy_status']=='not_accepted'
        assert int(resolved['num_sms'])==48 and int(resolved['l2_total_bytes'])==40*1024*1024
        assert [int(x['effective_bytes']) for x in resolved['l1_capacity_table']]==[131072,114688,102400,67584,28672]
        assert resolved['source_sha256']==sha(config)
        raw=config.read_text();bad={
            'unknown':raw+'\n-memgen_mshr_entries 192\n',
            'duplicate':raw+'\n-memgen_num_sms 48\n',
            'missing':raw.replace('-memgen_l2_index X',''),
            'negative':raw.replace('-memgen_num_sms 48','-memgen_num_sms -1'),
            'overflow':raw.replace('-memgen_num_sms 48','-memgen_num_sms 999999999999999999999999999'),
            'trailing':raw.replace('-memgen_num_sms 48','-memgen_num_sms 48garbage'),
            'tokens':raw.replace('-memgen_num_sms 48','-memgen_num_sms 48 2'),
            'geometry':raw.replace('-memgen_l1_sets 16','-memgen_l1_sets 17'),
            'capacity':raw.replace('-memgen_l2_sets_per_partition 1024','-memgen_l2_sets_per_partition 1048576'),
            'replacement':raw.replace('-memgen_l1_replacement CLOCK','-memgen_l1_replacement LRU'),
            'index':raw.replace('-memgen_l2_index X','-memgen_l2_index invalid'),
            'mshr':raw.replace('-memgen_mshr_model disabled','-memgen_mshr_model finite'),
            'timing':raw.replace('-memgen_l2_fill_latency 0','-memgen_l2_fill_latency 238'),
            'shared':raw.replace('8:64,16:56','8:64,8:56'),
            'shared-trailing':raw.replace('100:14','100:14,'),
            'sm-representation':raw.replace('-memgen_num_sms 48','-memgen_num_sms 257'),
            'file-size':raw+'#'+'x'*65536,
        }
        for name,text in bad.items():
            path=out/(name+'.config');path.write_text(text)
            assert 'hardware config:' in run('reject-'+name,[binary,'--describe-hardware-config',path],False)
        tests=ROOT/'tests/cache'
        for name in ['test_r4','test_r4_boundaries','test_hardware_config','test_r4_backend_identity','test_r4_producer_cancel']:
            build(name,tests/(name+'.cpp'))
        run('reference',[out/'test_r4'])
        run('boundaries',[out/'test_r4_boundaries'])
        run('explicit-config',[out/'test_hardware_config',config])
        for name,text in [('l1-sets',raw.replace('-memgen_l1_sets 16','-memgen_l1_sets 8')),
                          ('l1-ways',raw.replace('32:50','32:2'))]:
            path=out/(name+'.config');path.write_text(text)
            run(name,[out/'test_hardware_config',config,path])
        for opcode in ['LDG.E.32','STG.E.32','ATOMG.E.ADD.STRONG.GPU','LDG.E.STRONG.GPU']:
            name=opcode.replace('.','-')
            for valid in ['valid','invalid']:
                run(name+'-'+valid,[out/'test_r4_backend_identity',config,out/(name+'-'+valid),'r4-small-shared-20260922',8,opcode,valid],valid=='valid')
        run('model-mismatch',[out/'test_r4_backend_identity',config,out/'bad-model','wrong-model',8,'LDG.E.32','valid'],False)
        fixture=out/'fixture';context=make_fixture(fixture)
        def front(name,ctx,issue=None):
            dest=out/name;dest.mkdir()
            cmd=[binary,'--mode','memgen','--profile-index',fixture/'profiles.index.jsonl',
                '--app-config',fixture/'app.config','--issue-config',issue or fixture/'issue.config',
                '--hw-config',config,'--stats',dest/'source.json','--output-dir',dest/'model']
            if ctx:cmd+=['--r4-context',ctx]
            return cmd
        run('r4-smoke',front('r4-smoke',fixture/'context.json'))
        source=json.loads((out/'r4-smoke/source.json').read_text())
        assert source['status']=='PASS' and source['materialized_raw_sass_bytes']==0
        assert source['generated_memory_instructions']==6 and source['generated_lane_addresses']==136
        summary=list(csv.DictReader((out/'r4-smoke/model/kernel_summary.csv').open()))
        keys=['l1_requests','l1_hits','l1_misses','l2_read_requests','l2_write_requests','dram_load_bytes','dram_store_bytes']
        assert [[int(row[k]) for k in keys] for row in summary]==[[8,0,8,8,8,128,0],[2,0,2,2,0,64,0]]
        ledger=list(csv.DictReader((out/'r4-smoke/model/r4_l1_profiles.csv').open()))
        assert [(int(r['shared_kib']),int(r['ways']),int(r['effective_bytes'])) for r in ledger]==[(8,64,131072),(16,56,114688)]
        assert all(r['model']=='r4-small-shared-20260922' for r in ledger)
        identity=json.loads((out/'r4-smoke/model/hardware.identity.json').read_text())
        assert identity['source_sha256']==sha(config) and identity['context_sha256']==sha(fixture/'context.json')
        assert identity['parameters']['l1_replacement']=='CLOCK' and int(identity['parameters']['sector_bytes'])==32
        run('missing-context',front('missing-context',None),False)
        context['model_id']='wrong-model';(fixture/'wrong-context.json').write_text(json.dumps(context))
        run('context-mismatch',front('context-mismatch',fixture/'wrong-context.json'),False)
        run('placement-mismatch',front('placement-mismatch',fixture/'context.json',ROOT/'release/fixtures/smoke/configs/issue.config'),False)
        run('producer-cancel',[out/'test_r4_producer_cancel',fixture,config,fixture/'context.json',out/'cancel'])
        base_fixture=fixture
        mutations={
            'interval-valid':lambda p:None,
            'interval-hole':lambda p:p['structural_class_selector']['intervals'][1].update(start=2),
            'interval-overlap':lambda p:p['structural_class_selector']['intervals'][1].update(start=0),
            'interval-train-hold-overlap':lambda p:(p['sampling'].update(holdout_ctas=[0]),p['source']['launch'].update(holdout_ctas=[0])),
        }
        for name,mutate in mutations.items():
            fixture=out/(name+'-fixture')
            variant_fixture(fixture,lambda p:(intervals(p),mutate(p)))
            valid=name=='interval-valid'
            run(name,front(name,fixture/'context.json'),valid)
            if valid:
                assert (out/name/'model/kernel_summary.csv').read_bytes()==(out/'r4-smoke/model/kernel_summary.csv').read_bytes()
            else:assert not (out/name/'source.json').exists()
        for name,change,index_change in [
            ('rebound-status',lambda p:p.update(status='PASS_MODELED_LAYER_PROFILE_BINDING'),None),
            ('rebound-model',lambda p:p.update(model=dict(layer_rebinding='synthetic translation')),None),
            ('rebound-unobserved',lambda p:p.update(model=dict(target_addresses_hardware_observed=False)),None),
            ('rebound-index',lambda p:None,lambda row:row.update(status='PASS_MODELED_LAYER_PROFILE_BINDING')),
        ]:
            fixture=out/(name+'-fixture');variant_fixture(fixture,change,index_change)
            log=run(name,front(name,fixture/'context.json'),False)
            assert 'modeled layer rebinding' in log and not (out/name/'source.json').exists()
        result.update(status='PASS_CACHE_CORE_SOFTWARE_REGRESSION',negative_parser_cases=len(bad),
                      independent_reference_requests=64576,config_equivalence_requests=1200000)
    except BaseException:
        result.update(status='FAIL',error=traceback.format_exc())
    finally: save()
    print(json.dumps(result),flush=True)
    return 0 if result['status'].startswith('PASS_') else 1

if __name__=='__main__':raise SystemExit(main())
