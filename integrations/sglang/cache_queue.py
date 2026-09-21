#!/usr/bin/env python3
"""Static eight-worker CPU queue for the 22 remaining ready tilegraphs.

prepare only writes pinned worker plans. execute consumes one plan, reusing the
reviewed batch helper's run_job wrapper and the unchanged shared run_job.py.
No GPU, NVBit, graph extrapolation, rebuild, or automatic retry of failed tests.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

CPUS = (2, 3, 4, 5, 10, 11, 14, 15)
MODELS = ('qwen25_1p5b', 'llama3_8b')
PREFILLS = (128, 256, 512, 1024)
DECODES = (32, 64, 128)
PILOTS = {model+'-p128-d32' for model in MODELS}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_helper(path):
    spec = importlib.util.spec_from_file_location('tilegraph_matrix_batch_helper', Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_cases():
    return [f'{model}-p{P}-d{D}' for P in PREFILLS for D in DECODES if (P,D)!=(128,32) for model in MODELS]


def prepare(args):
    h = load_helper(args.batch_helper)
    root = args.root.resolve(strict=True); out = args.out.resolve()
    need(not out.exists(), 'Fresh worker plan directory required')
    need(len(args.metadata_queue)==2, 'Exactly two prepared model metadata queues required')
    queues={}; rows={}
    for path in args.metadata_queue:
        q=json.loads(path.read_text());need(q['schema']=='SGLANG_METADATA_GRAPH_QUEUE_V1' and q['model'] in MODELS,'Metadata queue schema/model')
        need(q['model'] not in queues and Path(q['root']).resolve()==root,'Unique model queue on same root required')
        for key in ('controller','lifecycle','converter','runner'):
            h.verify(q[key])
        for r in q['cases']:
            need(r['case_id'] not in rows and r['case_id'] not in PILOTS,'Duplicate or pilot case forbidden')
            rows[r['case_id']]=dict(r,metadata_queue=h.pin(path),converter=q['converter'])
        queues[q['model']]=q
    need(set(queues)==set(MODELS) and set(rows)==set(expected_cases()),'Exactly requested 24 minus two running pilots required')
    first=queues[MODELS[0]]
    need(all(q['controller']==first['controller'] and q['lifecycle']==first['lifecycle'] for q in queues.values()),'Shared controller identity differs')
    pilot=json.loads(args.pilot_template.read_text())
    need(pilot['case_id'] in PILOTS and pilot['tool']=='tilegen' and pilot['input_kind']=='tilegraph' and pilot.get('gpu') is None,
         'Use an existing CPU TileGen pilot spec')
    need('--execute' in pilot['argv'] and '--graph' in pilot['argv'] and '--progress' in pilot['argv'],'Complete pilot CLI required')
    for row in pilot['sources']:
        h.verify(row)
    build_path=args.build_receipt.resolve(strict=True);build=json.loads(build_path.read_text())
    need(build['status']=='PASS_BUILD_ONLY_NOT_TRAFFIC' and build['source_unchanged'] is True,'Successful immutable existing build required')
    binary=h.pin(pilot['argv'][0]);h.verify(build['executable'])
    need(binary==build['executable'],'Pilot binary differs from build receipt')
    build_pin=h.pin(build_path);pilot_sources={str(Path(p['path']).resolve()):p for p in pilot['sources']}
    need(str(build_path) in pilot_sources and pilot_sources[str(build_path)]['sha256']==build_pin['sha256'],'Pilot must pin this build')
    for row in build['dependencies']:
        h.verify(row)
    fixed={p['path']:p for p in [binary,build_pin,first['controller'],first['lifecycle'],*build['dependencies']]}
    ordered=[rows[case] for case in expected_cases()]
    out.mkdir(parents=True,exist_ok=False)
    workers=[]
    for index,cpu in enumerate(CPUS):
        cases=ordered[index::len(CPUS)]
        for row in cases:
            row['cache_output']=str(root/'runs'/(row['case_id']+'-tilegen-matrix-r1'))
            need(not Path(row['cache_output']).exists(),'Existing traffic output must not be overwritten: '+row['case_id'])
        plan=dict(schema='SGLANG_TILEGRAPH_CACHE_WORKER_V1',status='PREPARED_NOT_STARTED',worker_index=index,cpu=cpu,gpu=None,
            maximum_shared_cpu_pool=16,worker_cpu_pool=list(CPUS),maximum_worker_processes=8,parallel_jobs_per_worker=1,
            root=str(root),controller=first['controller'],lifecycle=first['lifecycle'],
            controller_python='/usr/bin/python3',batch_helper=h.pin(args.batch_helper),runner=h.pin(__file__),
            binary=binary,build_receipt=build_pin,fixed_source_pins=list(fixed.values()),
            pilot_template=h.pin(args.pilot_template),pilot_argv=pilot['argv'],
            queue_wall_limit_seconds=24*3600,maximum_resource_wait_seconds=3600,graph_wait_limit_seconds=6*3600,
            case_wall_limit_seconds=21600,never_restart_cases=sorted(PILOTS),cases=cases,
            does_not_collect_trace=True,does_not_use_GPU=True)
        path=out/(f'worker-cpu{cpu}.json');h.fresh_json(path,plan)
        workers.append(dict(cpu=cpu,plan=h.pin(path),cases=[r['case_id']for r in cases]))
    manifest=dict(status='PREPARED_EIGHT_CPU_WORKERS_NOT_STARTED',workers=workers,cases=22,cpu_pool=list(CPUS),
                  excluded_running_pilots=sorted(PILOTS),maximum_worker_processes=8)
    h.fresh_json(out/'manifest.json',manifest);print(json.dumps(manifest,indent=2))


def graph_readiness(h,row):
    """Return WAITING or READY; closed failed/malformed source raises, never runs."""
    metadata=Path(row['metadata_output']);converter_output=Path(row['graph_job_output'])
    metadata_job=metadata/'job-finish.json';converter_job=converter_output/'job-finish.json'
    if metadata_job.exists():
        closed=json.loads(metadata_job.read_text())
        need(closed['status']=='PASS_PROCESS_ONLY','Metadata job failed for '+row['case_id'])
    if not converter_job.exists():
        return dict(status='WAITING_GRAPH',reason='Converter job has not closed',converter_job=str(converter_job))
    job=json.loads(converter_job.read_text())
    need(job['status']=='PASS_PROCESS_ONLY' and job['process']['cleanup']['owned_descendants_empty'],
         'Converter job failed or owned cleanup not closed')
    need(job['case_id']==row['case_id'],'Converter case differs')
    graph=Path(row['graph_output']);receipt_path=graph.with_suffix('.receipt.json')
    need(graph.is_file() and receipt_path.is_file(),'Closed converter is missing graph or receipt')
    receipt=json.loads(receipt_path.read_text());contract=receipt['input_contract']
    need(receipt['status']=='PASS_FULL_DECLARED_TILEGRAPH_INPUT' and contract['case_id']==row['case_id'] and
         contract['model_key']==row['model'] and contract['prefill_length']==row['prefill'] and contract['decode_steps']==row['decode'],
         'Graph receipt workload mismatch')
    L=28 if row['model']==MODELS[0] else 32
    need(receipt['operators']==2*(row['decode']+1)*(9*L+3) and receipt['phase_count']==2*(row['decode']+1),
         'Incomplete full real-layer/stage graph')
    need(receipt.get('native_instruction_trace_used') is False,'Wrong instruction replay route')
    argv=job['argv'];need(argv[argv.index('--out')+1]==str(graph),'Converter wrote another graph path')
    source_map={p['path']:p for p in job['sources']}
    need(row['converter']['path'] in source_map and source_map[row['converter']['path']]['sha256']==row['converter']['sha256'],
         'Converter source identity differs from prepared metadata queue')
    h.verify(row['converter'])
    finishes=[p for p in job['sources'] if Path(p['path']).name=='finish.json' and Path(p['path']).is_relative_to(metadata)]
    need(len(finishes)==1 and finishes[0]['sha256']==receipt['metadata_finish_sha256'],'Graph metadata source receipt differs')
    h.verify(finishes[0]);graph_pin=h.pin(graph)
    need(graph_pin['sha256']==receipt['graph_sha256'],'Graph content does not match complete receipt')
    return dict(status='READY',graph=graph_pin,receipt=h.pin(receipt_path),converter_job=h.pin(converter_job),
                metadata_finish=finishes[0],contract=contract,operators=receipt['operators'],phase_count=receipt['phase_count'])


def result_summary(h,path,ready):
    result=json.loads(path.read_text())
    need(result['status']=='PASS_COMPLETE_DECLARED_SUPPORTED_GRAPH_NOT_NATIVE_EQUIVALENCE' and
         result['full_measured_declared_operator_graph_executed'] is True,'Traffic result is not complete declared graph')
    need(result['input_contract']==ready['contract'] and result['plan']['operator_count']==ready['operators'] and
         len(result['operators'])==ready['operators'],'Traffic input/operator closure differs')
    need(len(result['phases'])==ready['phase_count'] and sum(k.startswith('Measured/')for k in result['phases'])==ready['contract']['decode_steps']+1,
         'Traffic stage coverage differs')
    cache=result['cache_snapshot'];need(cache['dirty_sector_ledger_closed'] and cache['writeback_byte_ledger_closed'],'Cache ledger did not close')
    measured=result['measured_traffic']
    need(measured['DRAM_read_bytes']>=0 and measured['DRAM_write_bytes']>=0,'Measured DRAM counters absent')
    return dict(traffic=h.pin(path),measured_DRAM_read_bytes=measured['DRAM_read_bytes'],
        measured_DRAM_write_bytes=measured['DRAM_write_bytes'],native_instruction_equivalence=False,
        full_measured_declared_operator_graph_executed=True,full_native_kernel_memory_program_executed=False)


def execute(args):
    need(sys.platform=='linux','Actual execution requires XMU Linux')
    worker=json.loads(args.worker.read_text());need(worker['schema']=='SGLANG_TILEGRAPH_CACHE_WORKER_V1','Worker schema')
    need(worker['cpu'] in CPUS and worker['worker_cpu_pool']==list(CPUS) and worker['gpu'] is None and
         worker['maximum_shared_cpu_pool']==16 and worker['case_wall_limit_seconds']==21600,'CPU-only resource bounds changed')
    os.sched_setaffinity(0,{worker['cpu']})
    h=load_helper(worker['batch_helper']['path'])
    for pin in [worker['runner'],worker['batch_helper'],worker['pilot_template'],*worker['fixed_source_pins']]:
        h.verify(pin)
    need(h.pin(__file__)==worker['runner'],'Worker pins another queue implementation')
    need(all(r['case_id']not in PILOTS for r in worker['cases']),'Running pilots must never enter queue')
    need(len({r['case_id']for r in worker['cases']})==len(worker['cases']),'Duplicate static cases')
    for r in worker['cases']:
        h.verify(r['metadata_queue'])
    args.state_dir.mkdir(parents=True,exist_ok=False)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,h.stop_handler)
    start=time.monotonic()
    state=dict(schema='SGLANG_TILEGRAPH_CACHE_WORKER_STATE_V1',status='RUNNING_WORKER',worker=h.pin(args.worker),
        cpu=worker['cpu'],gpu=None,maximum_shared_cpu_pool=16,controller_pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cases=[dict(case_id=r['case_id'],status='PENDING_GRAPH')for r in worker['cases']],active=None)
    state_path=args.state_dir/'state.json'
    try:
        for index,row in enumerate(worker['cases']):
            entry=state['cases'][index];case=row['case_id'];wait_start=time.monotonic();output=Path(row['cache_output'])
            try:
                need(not output.exists(),'Existing case output retained; will not start duplicate '+case)
                while True:
                    need(h.STOP is None,'Worker stopped by signal')
                    need(time.monotonic()-start<worker['queue_wall_limit_seconds'],'Worker wall bound reached')
                    ready=graph_readiness(h,row)
                    if ready['status']=='READY':break
                    need(time.monotonic()-wait_start<worker['graph_wait_limit_seconds'],'Graph wait bound reached')
                    entry.update(status='WAITING_GRAPH',reason=ready['reason'],wait_seconds=time.monotonic()-wait_start)
                    state['active']=dict(case_id=case,status='WAITING_GRAPH');h.save_state(state_path,state)
                    time.sleep(10)
                sources=[*worker['fixed_source_pins'],ready['graph'],ready['receipt'],ready['converter_job'],ready['metadata_finish'],row['converter']]
                argv=copy.deepcopy(worker['pilot_argv'])
                h.replace_arg(argv,'--graph',row['graph_output']);h.replace_arg(argv,'--out',output/'traffic.json')
                h.replace_arg(argv,'--progress',output/'traffic-progress.json')
                spec=dict(case_id=case,tool='tilegen',input_kind='tilegraph',cpu=worker['cpu'],gpu=None,
                    seconds=21600,rss_limit_bytes=64<<30,argv=argv,sources=sources)
                spec_path=args.state_dir/(case+'-cache.json');h.fresh_json(spec_path,spec)
                entry.update(status='WAITING_CPU',graph=ready['graph'],job_output=str(output),spec=h.pin(spec_path))
                state['active']=dict(case_id=case,status=entry['status']);h.save_state(state_path,state)
                # One sleeping observer thread shares this worker's single-CPU
                # affinity. It only changes state once the owned controller has
                # acquired its lease and written job-start; no new test process.
                stop_observer=threading.Event()
                def observe_start():
                    while not stop_observer.wait(1):
                        if (output/'job-start.json').is_file():
                            entry['status']='RUNNING_CACHE'
                            state['active']=dict(case_id=case,status='RUNNING_CACHE')
                            h.save_state(state_path,state)
                            return
                observer=threading.Thread(target=observe_start,daemon=True);observer.start()
                try:
                    job=h.run_job(worker,spec_path,output,args.state_dir,case+'-cache',start)
                finally:
                    stop_observer.set();observer.join()
                summary=result_summary(h,output/'traffic.json',ready)
                entry.update(status='PASS_COMPLETE_DECLARED_TILEGRAPH',job_finish=h.pin(output/'job-finish.json'),
                    CPU_minutes=job['CPU_minutes'],wall_minutes=job['wall_minutes'],**summary)
            except Exception as error:
                if h.STOP is not None:
                    raise
                # Preserve every failed/partial artifact, never substitute zero
                # or claim full result. Other independent static cases may run.
                entry.update(status='FAILED_CASE_RETAINED',error=type(error).__name__+': '+str(error),
                             full_measured_declared_operator_graph_executed=False)
                if (output/'job-finish.json').exists():entry['job_finish']=h.pin(output/'job-finish.json')
            state['active']=None;h.save_state(state_path,state)
        state['status']='COMPLETE_WORKER_WITH_FAILURES' if any(r['status']=='FAILED_CASE_RETAINED'for r in state['cases']) else 'PASS_ALL_ASSIGNED_CASES'
    except BaseException as error:
        if h.CHILD is not None and h.CHILD.poll() is None:
            h.CHILD.send_signal(signal.SIGTERM);h.CHILD.wait(timeout=120)
        state.update(status='STOPPED_WORKER',error=type(error).__name__+': '+str(error));raise
    finally:
        state.update(elapsed_seconds=time.monotonic()-start,stopped_by_signal=h.STOP,
            finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        h.save_state(state_path,state)
    print(json.dumps(state,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--root',type=Path,required=True)
    prep.add_argument('--metadata-queue',type=Path,action='append',required=True)
    prep.add_argument('--pilot-template',type=Path,required=True);prep.add_argument('--build-receipt',type=Path,required=True)
    prep.add_argument('--batch-helper',type=Path,required=True);prep.add_argument('--out',type=Path,required=True)
    run=commands.add_parser('execute');run.add_argument('--worker',type=Path,required=True);run.add_argument('--state-dir',type=Path,required=True)
    args=p.parse_args();return prepare(args)if args.command=='prepare'else execute(args)


if __name__=='__main__':main()
