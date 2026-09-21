#!/usr/bin/env python3
"""Prepare two explicit metadata->tilegraph queues; execute only on request.

prepare writes 22 metadata specs and two reviewed queues, never starts a test.
execute runs ONE model's eleven cases serially via the existing run_job.py.
Start the two queue processes separately to use two GPUs. No cache executors,
NVBit, sampling, remote commands, or automatic retries of failed tests here.
"""
import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MODELS = ('qwen25_1p5b', 'llama3_8b')
PREFILLS = (128, 256, 512, 1024)
DECODES = (32, 64, 128)
HERE = Path(__file__).resolve().parent
STOP = None
CHILD = None


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(row):
    got = pin(row['path'])
    need(got['bytes'] == row['bytes'] and got['sha256'] == row['sha256'], 'Frozen source changed: ' + row['path'])


def fresh_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def save_state(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def replace_arg(argv, name, value):
    need(argv.count(name) == 1, 'One explicit ' + name + ' argument required')
    index = argv.index(name)
    need(index + 1 < len(argv), 'Missing argument ' + name)
    argv[index + 1] = str(value)


def prepare(args):
    root, out = args.root.resolve(strict=True), args.out.resolve()
    need(not out.exists(), 'Fresh queue output directory required')
    cpus = dict(zip(MODELS, (args.qwen_cpu, args.llama_cpu)))
    need(all(type(n) is int and 0 <= n < 16 for n in cpus.values()) and len(set(cpus.values())) == 2,
         'Choose two distinct CPU slots in the shared 0..15 pool')
    controller = root / 'run_job.py'
    lifecycle = root / 'vendor/parent_controller.py'
    converter = args.converter.resolve(strict=True)
    controller_pin, lifecycle_pin, converter_pin = pin(controller), pin(lifecycle), pin(converter)
    templates = dict(zip(MODELS, (args.qwen_template, args.llama_template)))
    loaded = {}
    for model, path in templates.items():
        template = json.loads(path.read_text())
        need(template['case_id'] == model + '-p128-d32' and template['tool'] == 'tilegen' and
             template['input_kind'] == 'tilegraph' and template.get('gpu'), 'Use each verified initial metadata job spec')
        need(not any(k in template.get('environment', {}) for k in ('LD_PRELOAD', 'SG_NVBIT_SCOPE_ABI')),
             'Metadata queue must not activate an instruction tracer')
        for source in template['sources']:
            verify(source)
        indexed = {str(Path(p['path']).resolve(strict=True)): pin(p['path']) for p in template['sources']}
        need(indexed.get(str(controller)) == controller_pin and indexed.get(str(lifecycle)) == lifecycle_pin,
             'Initial template must pin this exact shared controller/lifecycle')
        host = Path(template['argv'][2]).resolve(strict=True)
        need(template['argv'][1] == '-B' and str(host) in indexed and host.name == 'metadata_host.py',
             'Expected frozen metadata host argv/source pin')
        need(template['argv'][template['argv'].index('--model') + 1] == model, 'Model template mismatch')
        loaded[model] = template
    need(loaded[MODELS[0]]['gpu'] != loaded[MODELS[1]]['gpu'], 'Two independent GPU queues required')
    out.mkdir(parents=True, exist_ok=False)
    queues = []
    for model in MODELS:
        template = loaded[model]
        rows = []
        for P in PREFILLS:
            for D in DECODES:
                if (P, D) == (128, 32):
                    continue
                case = f'{model}-p{P}-d{D}'
                metadata_output = root / 'runs' / (case + '-metadata-matrix-r1')
                graph_output = root / 'graphs' / (case + '.json')
                graph_job_output = root / 'runs' / (case + '-graph-matrix-r1')
                need(not metadata_output.exists() and not graph_output.exists() and not graph_job_output.exists(),
                     'Refuse to replace existing output for ' + case)
                spec = copy.deepcopy(template)
                spec.update(case_id=case, cpu=cpus[model], seconds=1800)
                replace_arg(spec['argv'], '--prefill-length', P)
                replace_arg(spec['argv'], '--decode-steps', D)
                replace_arg(spec['argv'], '--output', metadata_output / 'host')
                spec_path = out / model / (case + '-metadata.json')
                fresh_json(spec_path, spec)
                rows.append(dict(case_id=case, model=model, prefill=P, decode=D,
                    metadata_spec=pin(spec_path), metadata_output=str(metadata_output),
                    graph_output=str(graph_output), graph_job_output=str(graph_job_output)))
        queue = dict(schema='SGLANG_METADATA_GRAPH_QUEUE_V1', status='PREPARED_NOT_STARTED', model=model,
            cpu=cpus[model], gpu=template['gpu'], maximum_shared_cpu_pool=16, parallel_jobs_per_queue=1,
            root=str(root), controller=controller_pin, lifecycle=lifecycle_pin, converter=converter_pin,
            metadata_source_pins=template['sources'], template=pin(templates[model]), runner=pin(__file__),
            controller_python='/usr/bin/python3', conversion_python='/usr/bin/python3',
            queue_wall_limit_seconds=24*3600, maximum_resource_wait_seconds=3600,
            skipped_completed_initial_case=model+'-p128-d32', cases=rows,
            does_not_launch_cache_executor=True, no_instruction_trace=True)
        path = out / (model + '-queue.json')
        fresh_json(path, queue)
        queues.append(dict(path=str(path), cpu=cpus[model], gpu=template['gpu'], cases=len(rows)))
    result = dict(status='PREPARED_22_CASES_NOT_STARTED', queues=queues,
                  next_step='Review and separately invoke execute --queue QUEUE --state-dir FRESH_STATE for each model.')
    fresh_json(out / 'manifest.json', result)
    print(json.dumps(result, indent=2))


def stop_handler(signum, frame):
    global STOP
    STOP = signum
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def child_parent_guard():
    parent = os.getppid()
    if parent <= 1 or ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent:
        os._exit(125)


def run_job(queue, spec_path, output, state_dir, task_name, queue_started):
    """Only resource admission races retry; any started/failed test stops queue."""
    global CHILD
    argv = [queue['controller_python'], '-B', queue['controller']['path'], '--spec', str(spec_path),
            '--output', str(output), '--execute']
    start = time.monotonic()
    attempt = 0
    while True:
        need(STOP is None, 'Queue stopped by signal')
        need(time.monotonic()-queue_started < queue['queue_wall_limit_seconds'], 'Queue wall bound reached')
        attempt += 1
        verify(queue['controller']); verify(queue['lifecycle'])
        stdout = state_dir / (task_name + f'.attempt-{attempt}.stdout.log')
        stderr = state_dir / (task_name + f'.attempt-{attempt}.stderr.log')
        with stdout.open('xb') as so, stderr.open('xb') as se:
            CHILD = subprocess.Popen(argv, stdout=so, stderr=se, start_new_session=True, preexec_fn=child_parent_guard)
            while CHILD.poll() is None:
                if STOP is not None:
                    CHILD.send_signal(signal.SIGTERM)
                    CHILD.wait(timeout=120)
                    raise RuntimeError('Queue stopped; active owned controller was asked to clean up')
                if time.monotonic()-queue_started >= queue['queue_wall_limit_seconds']:
                    CHILD.send_signal(signal.SIGTERM)
                    CHILD.wait(timeout=120)
                    raise RuntimeError('Queue wall bound reached; owned controller cleaned up')
                time.sleep(.25)
            returncode = CHILD.returncode
            CHILD = None
        if returncode == 0:
            finish = json.loads((output / 'job-finish.json').read_text())
            need(finish['status'] == 'PASS_PROCESS_ONLY' and finish['process']['cleanup']['owned_descendants_empty'],
                 'Job closure/owned cleanup failed')
            return finish
        error = stderr.read_text()
        if 'ResourceBusy:' in error and not output.exists() and time.monotonic()-start < queue['maximum_resource_wait_seconds']:
            time.sleep(10)
            continue
        raise RuntimeError('Job failed or resource wait expired; inspect ' + str(stderr))


def execute(args):
    need(sys.platform == 'linux', 'Actual queue execution is for XMU Linux')
    queue = json.loads(args.queue.read_text())
    need(queue['schema'] == 'SGLANG_METADATA_GRAPH_QUEUE_V1' and queue['model'] in MODELS and len(queue['cases']) == 11,
         'Explicit eleven-case queue required')
    need(0 <= queue['cpu'] < 16 and queue['maximum_shared_cpu_pool'] == 16, 'Shared CPU pool differs')
    os.sched_setaffinity(0, {queue['cpu']})
    for row in [queue['controller'], queue['lifecycle'], queue['converter'], queue['template'], queue['runner'], *queue['metadata_source_pins']]:
        verify(row)
    need(pin(__file__) == queue['runner'], 'Prepared queue pins another runner')
    args.state_dir.mkdir(parents=True, exist_ok=False)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop_handler)
    started = time.monotonic()
    state = dict(status='RUNNING_METADATA_AND_GRAPH_QUEUE', queue=pin(args.queue), model=queue['model'],
                 cpu=queue['cpu'], gpu=queue['gpu'], completed=[], active=None,
                 controller_pid=os.getpid(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        for row in queue['cases']:
            need(STOP is None, 'Queue stopped')
            verify(row['metadata_spec'])
            case = row['case_id']; metadata_out = Path(row['metadata_output'])
            graph_out = Path(row['graph_output']); graph_job_out = Path(row['graph_job_output'])
            need(not metadata_out.exists() and not graph_out.exists() and not graph_job_out.exists(), 'Fresh case outputs required: ' + case)
            state['active'] = dict(case_id=case, step='metadata')
            save_state(args.state_dir/'state.json', state)
            metadata_job = run_job(queue, Path(row['metadata_spec']['path']), metadata_out, args.state_dir, case+'-metadata', started)
            finishes = list((metadata_out/'host').glob('process-*/finish.json'))
            need(len(finishes) == 1, 'Exactly one metadata process finish required')
            metadata_finish = finishes[0]
            record = json.loads(metadata_finish.read_text())
            need(record['status'] == 'PASS_REAL_VIEW_METADATA_NO_MEMORY_TRACE' and record['input_contract']['case_id'] == case,
                 'Metadata schema/case did not close')
            sources = [queue['controller'], queue['lifecycle'], queue['converter'], pin(metadata_finish)]
            for source in record['files']:
                path = (metadata_finish.parent/source['path']).resolve(strict=True)
                need(path.is_relative_to(metadata_finish.parent), 'Metadata artifact escapes owned process directory')
                expected = dict(path=str(path), bytes=source['bytes'], sha256=source['sha256'])
                verify(expected); sources.append(expected)
            graph_spec = dict(case_id=case,tool='tilegen',input_kind='tilegraph',cpu=queue['cpu'],gpu=None,seconds=600,
                argv=[queue['conversion_python'],'-B',queue['converter']['path'],'--metadata',str(metadata_finish.parent),'--out',str(graph_out)],
                sources=sources)
            graph_spec_path = args.state_dir/(case+'-graph.json')
            fresh_json(graph_spec_path, graph_spec)
            state['active'] = dict(case_id=case, step='graph_conversion')
            save_state(args.state_dir/'state.json', state)
            graph_job = run_job(queue, graph_spec_path, graph_job_out, args.state_dir, case+'-graph', started)
            graph_receipt = json.loads(graph_out.with_suffix('.receipt.json').read_text())
            need(graph_receipt['status'] == 'PASS_FULL_DECLARED_TILEGRAPH_INPUT' and
                 graph_receipt['input_contract']['case_id'] == case and
                 graph_receipt['graph_sha256'] == pin(graph_out)['sha256'], 'Graph did not close with expected input/pin')
            state['completed'].append(dict(case_id=case,graph=pin(graph_out),metadata_finish=pin(metadata_finish),
                phase_count=graph_receipt['phase_count'],operators=graph_receipt['operators'],
                metadata_CPU_minutes=metadata_job['CPU_minutes'],metadata_wall_minutes=metadata_job['wall_minutes'],
                conversion_CPU_minutes=graph_job['CPU_minutes'],conversion_wall_minutes=graph_job['wall_minutes']))
            state['active'] = None
            save_state(args.state_dir/'state.json', state)
        state['status'] = 'PASS_ELEVEN_METADATA_AND_GRAPH_CASES_NO_CACHE_EXECUTION'
    except BaseException as error:
        if CHILD is not None and CHILD.poll() is None:
            CHILD.send_signal(signal.SIGTERM)
            CHILD.wait(timeout=120)
        state.update(status='STOPPED_OR_FAILED_QUEUE', error=type(error).__name__+': '+str(error))
        raise
    finally:
        state.update(elapsed_seconds=time.monotonic()-started, stopped_by_signal=STOP,
                     finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        save_state(args.state_dir/'state.json', state)
    print(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Write explicit specs/queues only; no test is started')
    prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--qwen-template', type=Path, required=True)
    prep.add_argument('--llama-template', type=Path, required=True)
    prep.add_argument('--converter', type=Path, required=True)
    prep.add_argument('--qwen-cpu', type=int, required=True)
    prep.add_argument('--llama-cpu', type=int, required=True)
    prep.add_argument('--out', type=Path, required=True)
    run = commands.add_parser('execute', help='Run one reviewed queue serially through shared run_job.py')
    run.add_argument('--queue', type=Path, required=True)
    run.add_argument('--state-dir', type=Path, required=True)
    args = parser.parse_args()
    return prepare(args) if args.command == 'prepare' else execute(args)


if __name__ == '__main__':
    main()
