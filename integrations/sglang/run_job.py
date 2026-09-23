#!/usr/bin/env python3
"""One explicit matrix job, using the existing shared 16-CPU leases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
import parent_controller as life

LABEL = 'sglang-tilegraph-sampled-profile-controller-v0.1.0-dev.1'
GPU_POOL = {
    'GPU-18ace299-5348-e6e4-d48c-1ee5a602859b',
    'GPU-7a22d253-7921-4e49-992e-0199eebd6f86',
    'GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13',
}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    cpu = spec['cpu']
    life.need(type(cpu) is int and 0 <= cpu < 16, 'Shared CPU pool is 0..15')
    gpu = spec.get('gpu')
    life.need(gpu is None or gpu in GPU_POOL, 'GPU2 and unknown GPU are excluded')
    argv = spec['argv']
    life.need(isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv), 'Explicit argument vector required')
    life.need(Path(argv[0]).is_absolute(), 'Absolute executable path required')
    seconds = spec.get('seconds', 7200)
    life.need(type(seconds) is int and (seconds == 0 or 1 <= seconds <= 86400),
              'Bounded runtime required, or 0 for no wall-clock deadline')
    # 0 means run to completion. The replay entry documents that it has no
    # wall-clock deadline, so the controller must not silently impose one; a
    # finite sentinel keeps every receipt strict JSON instead of "Infinity".
    deadline = (1 << 53) if seconds == 0 else seconds
    pins = []
    for row in spec['sources']:
        got = life.pin(row['path'])
        life.need(got['bytes'] == row['bytes'] and got['sha256'] == row['sha256'], 'Job source changed: ' + row['path'])
        pins.append(got)
    life.need(pins, 'Pin the job implementation and its explicit inputs')
    locks = [life.cpu_lock_path(cpu)] + ([life.gpu_lock_path(gpu)] if gpu else [])
    receipt = dict(schema='SGLANG_SAMPLED_MATRIX_JOB_V1', label=LABEL, status='PREPARED',
                   spec=life.pin(args.spec), controller=life.pin(__file__),
                   lifecycle=life.pin(life.__file__), sources=pins, case_id=spec['case_id'],
                   tool=spec['tool'], input_kind=spec['input_kind'], argv=argv,
                   cpu=cpu, maximum_shared_cpu_pool=16, gpu=gpu,
                   locks=[str(x) for x in locks], seconds=seconds,
                   wall_clock_deadline='unbounded' if seconds == 0 else seconds)
    if not args.execute:
        print(json.dumps(receipt, indent=2)); return 0
    life.need(sys.platform == 'linux', 'Actual tests run on XMU Linux')
    life.cpu_admission(cpu)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, life.stop_handler)
    life.enable_subreaper()
    with life.leases(locks) as held:
        if gpu: receipt['gpu_admission'] = life.gpu_admission(gpu)
        os.sched_setaffinity(0, {cpu})
        args.output.mkdir(parents=True, exist_ok=False)
        cache = Path(spec.get('cache_directory', str(args.output/'cache')))
        cache.mkdir(parents=True, exist_ok=True)
        env = life.environment(args.output, gpu or '')
        env['NVIDIA_VISIBLE_DEVICES'] = gpu or 'void'
        for key, name in [('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),
                          ('XDG_CACHE_HOME','xdg'),('FLASHINFER_WORKSPACE_BASE','flashinfer'),('TMPDIR','tmp')]:
            p = cache/name; p.mkdir(exist_ok=True); env[key] = str(p)
        # Sampler options must be explicit, recorded in spec and reviewed before execution.
        env.update(spec.get('environment', {}))
        for key in life.THREAD_KEYS: env[key] = '1'
        env['CUDA_VISIBLE_DEVICES'] = gpu or ''
        start = time.monotonic(); usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.update(status='RUNNING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                       controller_pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)))
        life.save(args.output/'job-start.json', receipt)
        process = life.run_process(argv,args.output,env,deadline,held_fds=tuple(fd for _,fd in held),
                                   rss_limit=spec.get('rss_limit_bytes',64<<30))
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.update(status=process['status'],process=process,
                       finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                       wall_minutes=(time.monotonic()-start)/60,
                       CPU_minutes=((after.ru_utime-usage.ru_utime)+(after.ru_stime-usage.ru_stime))/60)
        life.save(args.output/'job-finish.json',receipt)
        print(json.dumps(receipt,indent=2))
        return 0 if process['status']=='PASS_PROCESS_ONLY' else 1

if __name__ == '__main__':
    raise SystemExit(main())
