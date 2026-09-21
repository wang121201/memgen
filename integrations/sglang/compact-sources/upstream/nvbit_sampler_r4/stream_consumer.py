"""CPU streaming sample consumer. Frames are never persisted to disk.

Default: consume/validate immediately and retain only a bounded small receipt.
--emit-projection-records: normalized begin/record/end JSONL to another pipe.
This mode does not invoke a simulator and cannot qualify its downstream consumer.
"""
import argparse
import json
import os
from pathlib import Path
import stat
import sys
import time
import hashlib
from packet_stream import read_stream, need, MAX_WIRE, canonical, parse_json, integer
from ram_capture import SampleRAM
from compile_plan import validate


FLUSH_COLUMNS=['source_launch_key','flush_ordinal','cuda_get_last_error','cuda_sync_error',
    'sentinel_received','cta_conservation_closed','pushed_records','received_records',
    'selected_records','executed_cta_count','packet_cta_count','planned_cta_count']

PROJECTION_HEADER = 'SG_QUALIFIED_PROJECTION_REPLAY_V1'


def normalized_bytes(row):
    return (json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()


def replay_projection(ram, emit):
    """Publish actual source qualification; replay acceptance still awaits EOF.

    Producer exit and host identity gates have already passed. The final
    receipt separately seals this normalized replay after RAM validation.
    """
    need(ram.qualified is not None and ram.qualified['status']=='PASS_SAMPLED_TRANSPORT_ONLY',
         'qualified source required before projection replay')
    # Normalize integer map keys before hashing their JSON representation.
    transport=json.loads(canonical(ram.qualified))
    emit(dict(schema=PROJECTION_HEADER,transport=transport,
              transport_sha256=hashlib.sha256(canonical(transport)).hexdigest()))
    digest=hashlib.sha256();byte_count=0;row_count=0
    def deliver(row):
        nonlocal byte_count,row_count
        raw=normalized_bytes(row);digest.update(raw);byte_count+=len(raw);row_count+=1
        emit(row)
    ram.replay(on_begin=deliver,on_record=deliver,on_end=deliver)
    return dict(schema=PROJECTION_HEADER,sha256=digest.hexdigest(),bytes=byte_count,rows=row_count)


def validate_flush_ledger(finish,result,plan):
    """Independently bind the host ledger to the parsed stream and sealed plan."""
    need(finish.get('sampling_flush_protocol')=='EXPLICIT_HOST_LEDGER_ENTRY_V2' and
         finish.get('internal_dispatch_visibility')=='NOT_OBSERVED_REQUIRED_R4','flush protocol')
    need(integer(finish['internal_inspection_dispatch_count'])==0 and
         integer(finish['internal_inspection_dispatch_return_count'])==0,'tool dispatch callback unexpectedly observed')
    expected=[r for r in plan['launches'] if r['fit_ctas']]
    count=len(expected)
    need(count>0 and len(result['kernels'])==count,'flush selected plan coverage')
    for name in ('sampled_kernels','sampling_flush_kernels','sampling_flush_submit_attempts',
                 'sampling_flush_submitted','sampling_flush_launch_checks_passed',
                 'sampling_flush_completed','sampling_flush_sentinels_received','sampling_flush_conservation_closed'):
        need(integer(finish[name])==count,'flush count '+name)
    need(finish['sampling_flush_ledger_closed'] is True and finish['sampling_flush_ledger_columns']==FLUSH_COLUMNS,'flush ledger schema/closure')
    rows=finish['sampling_flush_ledger'];need(type(rows) is list and len(rows)==count,'flush ledger cardinality')
    for ordinal,(raw,kernel,planned) in enumerate(zip(rows,result['kernels'],expected)):
        need(type(raw) is list and len(raw)==len(FLUSH_COLUMNS),'flush row shape')
        row=dict(zip(FLUSH_COLUMNS,raw));key='epoch-%d-launch-%d'%(planned['epoch_id'],planned['epoch_launch_ordinal'])
        need(row['source_launch_key']==key==kernel['source_launch_key'] and integer(row['flush_ordinal'])==ordinal,'flush source/order')
        need(integer(row['cuda_get_last_error'],-1)==0 and integer(row['cuda_sync_error'],-1)==0 and
             row['sentinel_received'] is True and row['cta_conservation_closed'] is True,'flush success stages')
        for name in ('pushed_records','received_records','selected_records'):
            need(integer(row[name])==kernel['selected_records'],'flush/stream record count')
        selected=planned['fit_ctas']+planned['holdout_ctas']
        need(kernel['fit_ctas']==planned['fit_ctas'] and kernel['holdout_ctas']==planned['holdout_ctas'],'flush planned CTA coordinates')
        need(integer(row['executed_cta_count'])==integer(row['planned_cta_count'])==len(selected)==len(kernel['cta_entry']['executed_ctas']),'flush executed CTA cardinality')
        need(integer(row['packet_cta_count'])==len(kernel['actual_cta_sm'])==len(kernel['cta_entry']['packet_ctas']),'flush packet CTA cardinality')
    return dict(protocol='EXPLICIT_HOST_LEDGER_ENTRY_V2',status='PASS_FLUSH_LEDGER_ONLY',kernels=count,
                internal_dispatch_visibility='NOT_OBSERVED_REQUIRED_R4')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--read-fd',type=int,required=True)
    p.add_argument('--max-wire-bytes',type=int,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--producer-exit',type=Path,required=True)
    p.add_argument('--emit-projection-records',action='store_true')
    a=p.parse_args();need(stat.S_ISFIFO(os.fstat(a.read_fd).st_mode),'input must be sample pipe')
    if a.emit_projection_records:need(stat.S_ISFIFO(os.fstat(1).st_mode),'record output must be pipe, never durable file')
    def emit(row):
        b=normalized_bytes(row)
        limit=64<<20 if row.get('schema')==PROJECTION_HEADER else 1<<20
        need(len(b)<=limit,'projection JSONL scratch')
        view=memoryview(b)
        while view:
            n=os.write(1,view);need(n>0,'short/closed projection pipe');view=view[n:]
    ram=SampleRAM(a.max_wire_bytes)
    try:
        need(a.plan.is_file() and not a.plan.is_symlink() and a.plan.stat().st_size<=256<<20,'bounded plan')
        plan_raw=a.plan.read_bytes();plan=parse_json(plan_raw);validate(plan)
        need(plan['max_wire_bytes']==a.max_wire_bytes,'plan/CLI wire cap')
        plan_sha=hashlib.sha256(canonical(plan)).hexdigest()
        with os.fdopen(a.read_fd,'rb',buffering=65536) as f:
            result=ram.capture(f)
        # GPU owns no downstream simulator wait: record processing after this
        # point starts only once the controller observes actual worker exit.
        deadline=time.monotonic()+30
        while not a.producer_exit.exists():
            need(time.monotonic()<deadline,'producer exit marker absent after EOF')
            time.sleep(.05)
        need(a.producer_exit.is_file() and not a.producer_exit.is_symlink() and a.producer_exit.stat().st_size<=1<<20,'bounded producer marker')
        marker_raw=a.producer_exit.read_bytes();marker=parse_json(marker_raw)
        need(marker['schema']=='SG_SAMPLE_PRODUCER_EXIT_V1','producer marker schema')
        integer(marker['returncode'],-255,255);integer(marker['pid'],1);integer(marker['start_ticks'],1)
        path=Path(marker['observer_finish'])
        need(path.is_absolute() and path.is_file() and not path.is_symlink() and path.stat().st_size<=4<<20,'bounded observer finish')
        finish_raw=path.read_bytes();finish=parse_json(finish_raw)
        need(finish['status']=='PASS_SAMPLER_HOST_CLOSED_AWAITING_CONSUMER','sampler host did not close')
        need(finish['pid']==marker['pid'] and finish['start_ticks']==marker['start_ticks'],'producer/host process epoch')
        need(finish['sampler_finalized'] is True and finish['sample_plan_sha256']==plan_sha,'source plan/finalize')
        need(finish['sample_wire_sha256']==result['stream_sha256'] and finish['sample_wire_bytes']==result['wire_bytes'], 'source/consumer wire identity')
        need(finish['selected_cta_records']==result['selected_records'] and finish['sampled_kernels']==len(result['kernels']),'source/consumer census')
        need(finish['launch_before_count']==finish['launch_return_count'] and finish['launch_error_count']==0 and
             finish['open_context_count']==0 and finish['epoch_begin_count']==finish['epoch_end_count'] and finish['active_epoch']==0 and
             finish['unsupported_dispatch_count']==0 and finish['graph_node_callback_count']==0 and finish['unknown_launch_attribute_count']==0 and finish['errors']==[], 'host launch/epoch/error closure')
        flush_receipt=validate_flush_ledger(finish,result,plan)
        need(marker['source_pins_match'] is True,'controller source rehash failed')
        ram.qualify(producer_exit_code=marker['returncode'],observer_closed=True,source_pins_match=True,consumer_callbacks_closed=True)
        replay_seal=replay_projection(ram,emit) if a.emit_projection_records else None
        result=dict(ram.qualified,flush_ledger=flush_receipt,allocated_sample_ram_bytes=sum(map(len,ram.chunks)),
            retained_selected_wire_bytes=ram.size,producer_exit_marker_sha256=hashlib.sha256(marker_raw).hexdigest(),
            observer_finish_sha256=hashlib.sha256(finish_raw).hexdigest(),sample_plan_sha256=plan_sha,
            source_plan_file_sha256=hashlib.sha256(plan_raw).hexdigest(),downstream_projection_emitted=a.emit_projection_records,normalized_replay=replay_seal,
            downstream_template_or_gtsim_admission=False)
        code=0
    except BaseException as e:
        result=dict(schema='SG_PACKET_CONSUMER_RECEIPT_V1',status='FAIL_SAMPLE_PIPE',
                    error=type(e).__name__+': '+str(e),memory_projection_admitted=False,template_or_gtsim_admitted=False)
        code=1
    finally:ram.clear()
    data=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    need(len(data)<=64<<20,'bounded consumer receipt cap')
    fd=os.open(a.receipt,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as f:f.write(data)
    return code
if __name__=='__main__':raise SystemExit(main())
