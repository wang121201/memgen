#!/usr/bin/env python3
"""Consume expanded packed profiles in one original MemGen process.

Use the outer shared CPU lease controller. The frozen engine owns continuous
L2 state; this script never multiplies a source-layer traffic result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

F=Path('/home/xmu/nvidiagds/codex-runs/memgen-paper-ada-v1-20260916-01a08d87-r1')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expanded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=21600)
    p.add_argument('--allow-partial-diagnostic',action='store_true')
    a=p.parse_args();manifest=json.loads((a.expanded/'manifest.json').read_text())
    if not manifest['complete_full_model'] and not a.allow_partial_diagnostic:
        raise ValueError('Unsupported profiles remain; use explicit partial diagnostic or complete profile coverage')
    if not 1<=a.seconds<=86400:raise ValueError('Bounded runtime required')
    config=F/'config/RTX4000Ada.paper-v1.config';binary=F/'bin/hbserve'
    values={line.split()[0]:line.split()[1] for line in config.read_text().splitlines() if line.startswith('-')}
    l1=int(values['-gpgpu_l1d_cache_sets'])*int(values['-gpgpu_l1d_cache_associative'])*int(values['-gpgpu_l1d_cache_block_size'])
    slices=int(values['-gpgpu_num_memory_controllers'])*int(values['-gpgpu_num_sub_partition_per_memory_channel'])
    l2=slices*int(values['-gpgpu_l2d_cache_sets'])*int(values['-gpgpu_l2d_cache_associative'])*int(values['-gpgpu_l2d_cache_block_size'])
    if (l1,l2)!=(32768,41943040):raise ValueError('Expected common 32KiB L1 / 40MiB L2 geometry')
    a.output.mkdir(parents=True,exist_ok=False)
    command=[str(binary),'--mode','memgen','--profile-index',str(a.expanded/'profiles.index.jsonl'),
        '--app-config',str(a.expanded/'app.config'),'--issue-config',str(a.expanded/'issue.config'),
        '--hw-config',str(config),'--stats',str(a.output/'source-stats.json'),'--output-dir',str(a.output/'model'),
        '--include-local','false','--observe-cache','false']
    semantic=a.expanded/'semantic.ranges'
    if semantic.is_file() and semantic.stat().st_size:command+=['--semantic-file',str(semantic)]
    (a.output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('LD_PRELOAD',None)
    before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic();result=dict(status='FAIL')
    with (a.output/'stdout.log').open('xb') as out,(a.output/'stderr.log').open('xb') as err:
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
        try:
            child.wait(timeout=a.seconds)
            if child.returncode!=0:raise RuntimeError('MemGen returncode '+str(child.returncode))
            source=json.loads((a.output/'source-stats.json').read_text())
            if source['status']!='PASS' or source['materialized_raw_sass_bytes']!=0:raise RuntimeError('Original source closure failed')
            result.update(status='PASS_COMPLETE_SAMPLED_MODEL_CACHE' if manifest['complete_full_model'] else 'PASS_PARTIAL_MODEL_CACHE_DIAGNOSTIC',
                generated_memory_instructions=source['generated_memory_instructions'])
        except Exception as e:result['error']=type(e).__name__+': '+str(e)
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(returncode=child.returncode,wall_seconds=time.monotonic()-started,
        cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
        L1_bytes_per_SM=l1,L2_bytes=l2,dirty_writeback_bytes=32,
        binary_sha256=sha(binary),config_sha256=sha(config),expanded_manifest_sha256=sha(a.expanded/'manifest.json'),
        complete_declared_profile_stream=manifest['complete_declared_profile_stream'],
        full_native_address_coverage=False,allow_full_NCU_accuracy_comparison=False,
        complete_full_model=manifest['complete_full_model'],unsupported_launches=manifest['unsupported_launches'],
        unknown_private_allocations=manifest['unknown_private_allocations'],unknown_private_bytes=manifest['unknown_private_bytes'],
        hardware_accuracy_accepted=False,
        preserve_l1=False,preserve_l2=True,final_dirty_drain=False,
        cache_policy='original paper-v1 LRU control; L1 store bypass; line-miss-only write-sector policy; writeback; zero fill latency')
    result['cpu_minutes']=result['cpu_seconds']/60;result['wall_minutes']=result['wall_seconds']/60
    (a.output/'finish.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return 0 if result['status'].startswith('PASS_') else 1


if __name__=='__main__':raise SystemExit(main())
