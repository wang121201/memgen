#!/usr/bin/env python3
"""Replay admitted compact profiles without storing expanded memory traces.

Legacy --expanded input retains its cross-layer synthetic-address boundary.
Direct --profile-index input accepts caller-supplied app/issue files and, for
r4, the original allocation/shared-memory context. It does not construct or
certify that context. No wall-clock deadline is imposed by this replay entry.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
# Retained only for old callers which do not supply --binary.
F=Path('/home/xmu/nvidiagds/codex-runs/memgen-paper-ada-v1-20260916-01a08d87-r1')

def child_guard(parent):
    # The Linux replay is one process with worker threads. An unexpected
    # controller death must not leave that process running without an owner.
    if os.getppid()!=parent or ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:
        os._exit(125)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def resolve_hardware(binary,config,env):
    lines=[line.split('#',1)[0].split() for line in config.read_text().splitlines()]
    if any(row and row[0].startswith('-memgen_') for row in lines):
        p=subprocess.run([str(binary),'--describe-hardware-config',str(config)],
                         env=env,text=True,capture_output=True,check=True)
        resolved=json.loads(p.stdout)
        if resolved['source_sha256']!=sha(config):raise ValueError('hardware description identity mismatch')
        if resolved['hardware_accuracy_status']!='not_accepted':raise ValueError('unexpected calibration status')
        return resolved
    # Compatibility reader only; all unified configuration goes through C++.
    values={row[0]:row[1] for row in lines if len(row)>=2 and row[0].startswith('-')}
    l1=int(values['-gpgpu_l1d_cache_sets'])*int(values['-gpgpu_l1d_cache_associative'])*int(values['-gpgpu_l1d_cache_block_size'])
    partitions=int(values['-gpgpu_num_memory_controllers'])*int(values['-gpgpu_num_sub_partition_per_memory_channel'])
    l2=partitions*int(values['-gpgpu_l2d_cache_sets'])*int(values['-gpgpu_l2d_cache_associative'])*int(values['-gpgpu_l2d_cache_block_size'])
    return dict(schema='legacy',source_sha256=sha(config),l1_bytes_per_sm=l1,l2_total_bytes=l2,
                hardware_accuracy_status='not_accepted')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--expanded',type=Path)
    source.add_argument('--profile-index',type=Path)
    p.add_argument('--app-config',type=Path)
    p.add_argument('--issue-config',type=Path)
    p.add_argument('--r4-context',type=Path)
    p.add_argument('--semantic-file',type=Path)
    p.add_argument('--binary',type=Path,default=F/'bin/hbserve')
    p.add_argument('--hw-config',type=Path,default=ROOT/'release/config/RTX4000Ada.paper-v1.config')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,help='deprecated compatibility argument; ignored (no wall-clock deadline)')
    p.add_argument('--allow-partial-diagnostic',action='store_true')
    a=p.parse_args()
    if a.seconds is not None:print('Note: --seconds is ignored; replay has no wall-clock deadline.',flush=True)
    manifest=None
    if a.expanded:
        if a.app_config or a.issue_config or a.r4_context or a.semantic_file:
            p.error('--expanded owns its input paths; use --profile-index for explicit native inputs')
        a.expanded=a.expanded.resolve();manifest=json.loads((a.expanded/'manifest.json').read_text())
        if not manifest['complete_full_model'] and not a.allow_partial_diagnostic:
            raise ValueError('Unsupported profiles remain; explicit partial diagnostic required')
        a.profile_index=a.expanded/'profiles.index.jsonl'
        a.app_config=a.expanded/'app.config';a.issue_config=a.expanded/'issue.config'
        semantic=a.expanded/'semantic.ranges'
        if semantic.is_file() and semantic.stat().st_size:a.semantic_file=semantic
    elif not a.app_config or not a.issue_config:
        p.error('--profile-index requires --app-config and --issue-config')
    elif a.allow_partial_diagnostic:
        p.error('--allow-partial-diagnostic only applies to an expanded manifest')
    for name in ['binary','hw_config','profile_index','app_config','issue_config','r4_context','semantic_file']:
        path=getattr(a,name)
        if path is not None:
            path=path.resolve(strict=True);setattr(a,name,path)
            if not path.is_file():raise ValueError('file required: '+str(path))
    a.output=a.output.resolve()
    if a.output.exists():raise ValueError('fresh output required: '+str(a.output))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('LD_PRELOAD',None)
    inputs=[a.binary,a.hw_config,a.profile_index,a.app_config,a.issue_config]
    inputs += [x for x in [a.r4_context,a.semantic_file,a.expanded/'manifest.json' if a.expanded else None] if x is not None]
    pins={str(path):sha(path) for path in inputs}
    resolved=resolve_hardware(a.binary,a.hw_config,env)
    unified=resolved.get('schema')!='legacy'
    if unified and manifest is not None:
        raise ValueError('r4 requires original allocation context; legacy cross-layer expanded input is synthetic')
    if unified != bool(a.r4_context):
        raise ValueError('unified r4 config and --r4-context must be supplied together')
    a.output.mkdir(parents=True,exist_ok=False)
    command=[str(a.binary),'--mode','memgen','--profile-index',str(a.profile_index),
        '--app-config',str(a.app_config),'--issue-config',str(a.issue_config),'--hw-config',str(a.hw_config),
        '--stats',str(a.output/'source-stats.json'),'--output-dir',str(a.output/'model'),
        '--include-local','false','--observe-cache','false']
    if a.r4_context:command+=['--r4-context',str(a.r4_context)]
    if a.semantic_file:command+=['--semantic-file',str(a.semantic_file)]
    (a.output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    (a.output/'input-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    (a.output/'hardware-description.json').write_text(json.dumps(resolved,indent=2)+'\n')
    before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic();child=None
    result=dict(status='FAIL',hardware_accuracy_accepted=False,allow_full_NCU_accuracy_comparison=False,
                full_native_address_coverage=False,wallclock_cutoff=False,
                input_scope='legacy_cross_layer_expansion' if manifest else 'caller_supplied_profile_stream',
                binary_sha256=pins[str(a.binary)],config_sha256=pins[str(a.hw_config)],
                profile_index_sha256=pins[str(a.profile_index)],L2_bytes=int(resolved['l2_total_bytes']),
                raw_trace_materialized=False)
    if unified:
        result.update(l1_capacity_table=resolved['l1_capacity_table'],cache_policy='explicit_unified_config',
                      r4_context_sha256=pins[str(a.r4_context)])
    else:result.update(L1_bytes_per_SM=resolved['l1_bytes_per_sm'],cache_policy='legacy_paper_v1',
                       dirty_writeback_bytes=32,preserve_l1=False,preserve_l2=True,final_dirty_drain=False)
    if manifest:
        result['expanded_manifest_sha256']=pins[str(a.expanded/'manifest.json')]
        for key in ['complete_declared_profile_stream','complete_full_model','unsupported_launches',
                    'unknown_private_allocations','unknown_private_bytes']:result[key]=manifest[key]
    def cancelled(signum,frame):raise InterruptedError('cancel signal '+str(signum))
    handlers={sig:signal.signal(sig,cancelled) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        with (a.output/'stdout.log').open('xb') as out,(a.output/'stderr.log').open('xb') as err:
            parent=os.getpid()
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                   preexec_fn=lambda:child_guard(parent))
            (a.output/'process.json').write_text(json.dumps(dict(controller_pid=parent,replay_pid=child.pid))+'\n')
            child.wait()
        if child.returncode!=0:raise RuntimeError('MemGen returncode '+str(child.returncode))
        source=json.loads((a.output/'source-stats.json').read_text())
        if source['status']!='PASS' or source['materialized_raw_sass_bytes']!=0:
            raise RuntimeError('Source replay closure failed')
        if any(sha(path)!=digest for path,digest in pins.items()):raise RuntimeError('Replay input identity changed')
        if unified:
            identity=json.loads((a.output/'model/hardware.identity.json').read_text())
            if identity['source_sha256']!=pins[str(a.hw_config)] or identity['context_sha256']!=pins[str(a.r4_context)]:
                raise RuntimeError('Effective hardware/context identity changed')
        status='PASS_PROFILE_STREAM_CACHE_REPLAY'
        if manifest:status='PASS_COMPLETE_SAMPLED_MODEL_CACHE' if manifest['complete_full_model'] else 'PASS_PARTIAL_MODEL_CACHE_DIAGNOSTIC'
        result.update(status=status,generated_memory_instructions=source['generated_memory_instructions'],
                      generated_lane_addresses=source.get('generated_lane_addresses'),source_pins_unchanged=True)
    except BaseException as e:
        result['error']=type(e).__name__+': '+str(e)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)  # Failure cleanup only, never a run deadline.
            except subprocess.TimeoutExpired:child.kill();child.wait()
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(returncode=child.returncode if child else None,wall_seconds=time.monotonic()-started,
            cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime))
        result['wall_minutes']=result['wall_seconds']/60;result['cpu_minutes']=result['cpu_seconds']/60
        (a.output/'finish.json').write_text(json.dumps(result,indent=2)+'\n')
        for sig,handler in handlers.items():signal.signal(sig,handler)
    print(json.dumps(result),flush=True)
    return 0 if result['status'].startswith('PASS_') else 1

if __name__=='__main__':raise SystemExit(main())
