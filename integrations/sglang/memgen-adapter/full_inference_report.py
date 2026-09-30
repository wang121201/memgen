#!/usr/bin/env python3
"""Build a self-contained full-inference execution and error-proxy report.

This report distinguishes native sampled profiles from modeled launches.  It
does not claim hardware accuracy or invent a ground-truth error when a full
raw SASS trace was not retained.  ``profile_rejection_fraction`` and
``modeled_launch_fraction`` are measurable coverage/error proxies; the cache
traffic totals are the replay result for the declared case.
"""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def one(root, pattern):
    rows=sorted(Path(root).glob(pattern))
    return rows[0] if len(rows)==1 else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args(); work=a.work.resolve(); out=(a.output or work/'full-inference-report.json').resolve()
    follows=sorted((work/'runs').glob('*-collect/followthrough'))
    if len(follows)!=1: raise SystemExit('expected exactly one collect/followthrough under --work')
    follow=follows[0]; sample=follow/'sample'; profiles=sample/'profiles'; cache=follow/'cache'
    sample_finish=read(sample/'finish.json') or {}
    consumer=read(sample/'consumer.json') or {}
    profile_receipt=read(profiles/'receipt.json') or {}
    expansion=read(follow/'expanded/manifest.json') or {}
    finish=read(follow/'finish.json') or {}
    rows=[]
    summary=cache/'model/kernel_summary.csv'
    if summary.is_file():
        with summary.open(newline='') as f: rows=list(csv.DictReader(f))
    numeric_fields=('mem_insts','lane_accesses','read_sector_requests','write_sector_requests',
                    'atomic_sector_requests','l1_requests','l1_hits','l1_misses',
                    'l2_read_requests','l2_write_requests','dram_load_bytes','dram_store_bytes')
    totals={k:sum(int(r.get(k,0) or 0) for r in rows if str(r.get(k,'')).lstrip('-').isdigit()) for k in numeric_fields}
    selected=int(sample_finish.get('selected_kernels',0) or 0)
    accepted=int(sample_finish.get('profiles_accepted',0) or 0)
    rejected=int(sample_finish.get('profiles_rejected',0) or 0)
    target=int(expansion.get('target_launches',selected) or 0)
    modeled=int(expansion.get('modeled_launches',0) or 0)
    exact=int(expansion.get('exact_launches',0) or 0)
    report={
        'schema':'MEMGEN_FULL_INFERENCE_REPORT_V1',
        'work':str(work),
        'case_id':(finish.get('input_contract') or {}).get('case_id'),
        'execution':{
            'status':finish.get('status'),
            'sample_stage':sample_finish.get('status'),
            'expansion_complete_full_model':expansion.get('complete_full_model'),
            'cache_replay_executed':bool((cache/'model/kernel_summary.csv').is_file()),
            'hardware_accuracy_accepted':False,
            'raw_sass_trace_persisted':False,
        },
        'native_sampling':{
            'selected_kernels':selected,
            'selected_records':sample_finish.get('selected_records'),
            'wire_bytes':consumer.get('wire_bytes'),
            'profiles_accepted':accepted,
            'profiles_rejected':rejected,
            'profile_rejection_fraction':(rejected/selected if selected else None),
            'profile_bytes':profile_receipt.get('profile_bytes'),
            'full_native_address_coverage':sample_finish.get('full_native_address_coverage',False),
        },
        'full_model_expansion':{
            'target_launches':target,
            'packed_launches':expansion.get('packed_launches'),
            'exact_launches':exact,
            'modeled_launches':modeled,
            'modeled_launch_fraction':(modeled/target if target else None),
            'modeled_by_cause':expansion.get('modeled_by_cause'),
            'fully_exact':expansion.get('fully_exact'),
            'modeled_completion':expansion.get('modeled_completion'),
        },
        'cache_replay_totals':totals,
        'error_proxy_definition':{
            'profile_rejection_fraction':'rejected sampled launch profiles / selected launches',
            'modeled_launch_fraction':'modeled full-inference launches / target launches',
            'traffic_error':'not assessed here; no hardware oracle is silently substituted',
        },
        'artifacts':{
            'sample_finish':str(sample/'finish.json'),
            'profile_receipt':str(profiles/'receipt.json'),
            'expansion_manifest':str(follow/'expanded/manifest.json'),
            'cache_kernel_summary':str(summary) if summary.is_file() else None,
            # Current cache replay names this conservation receipt
            # semantic_conservation.json; retain the older observation name as
            # a compatibility fallback for earlier outputs.
            'cache_observation':str(cache/'model/cache_observation.json') if (cache/'model/cache_observation.json').is_file() else (str(cache/'model/semantic_conservation.json') if (cache/'model/semantic_conservation.json').is_file() else None),
        },
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': raise SystemExit(main())
