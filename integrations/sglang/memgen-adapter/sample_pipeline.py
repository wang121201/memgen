#!/usr/bin/env python3
"""Run existing sparse sampler -> RAM pipe consumer -> packed profile fitter.

Invoke under the shared CPU/GPU lease controller. Children inherit its CPU
affinity and process group; all heavy numerical libraries are single-threaded.
Detailed sample/expanded records are never written to a regular file.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
# Single source of truth for the declared cases. Duplicating the choices here is
# what previously prevented a declared point from reaching host.py.
import matrix_workload as workload


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def birth(pid):return int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
def child_guard(parent):
    if os.getppid()!=parent or ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:os._exit(125)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--sampler-lib',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--python',default='/home/xmu/sgl/bin/python')
    workload.add_arguments(p)
    p.add_argument('--seconds',type=int,default=3600)
    a=p.parse_args()
    # Reject an undeclared (model, prefill, decode) triple here, in the parent.
    # Per-axis argparse choices alone cannot see a pair such as P32D128, and
    # without this the failure would only surface in the host child.
    workload.contract(a.model,a.prefill_length,a.decode_steps)
    if not 60<=a.seconds<=21600:raise ValueError('Explicit bounded runtime required')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('GPU must be assigned by outer lease controller')
    a.output.mkdir(parents=True,exist_ok=False)
    observer=a.output/'observer';observer.mkdir()
    plan=json.loads(a.plan.read_text())
    if not 0<sum(bool(r['fit_ctas']) for r in plan['launches'])<len(plan['launches']):
        raise ValueError('Requires sparse single-layer plan, never all-launch memory sampling')
    paths=list(HERE.glob('*.py'))+[HERE/'contract.json',a.sampler_lib,a.plan]
    paths+=list((a.upstream/'nvbit_sampler_r4').glob('*.py'))
    paths+=list((a.upstream/'template_adapter_r4').glob('*.py'))
    paths+=[a.upstream/'sglang_sample_to_packed.py',a.upstream/'full_source_postprocess.py']
    if (a.upstream/'profile_census.py').is_file():paths.append(a.upstream/'profile_census.py')
    before={str(x.resolve()):sha(x) for x in paths}
    # The projection policy decides whether a predicated global read is admitted or
    # refused. Refusing it is what turns a weight-streaming class into an unfitted
    # class, so which policy ran is part of the result and not an implementation
    # detail: it is validated against the adapter's own list here and passed as an
    # explicit argument, so a typo can never fall back to the default silently and
    # the run's own receipt records the value that was used.
    sys.path.insert(0,str(a.upstream/'template_adapter_r4'))
    from memory_projection import MODEL_POLICIES
    policy=os.environ.get('SG_TEMPLATE_MODEL_POLICY','strict')
    if policy not in MODEL_POLICIES:
        raise ValueError('unknown SG_TEMPLATE_MODEL_POLICY %r; choose one of %s'%(policy,MODEL_POLICIES))
    env=dict(os.environ)
    for k in list(env):
        if k=='LD_PRELOAD' or k.startswith(('HYFISS_','SG_NVBIT_','SG_SAMPLE_')):env.pop(k)
    env.pop('SG_TEMPLATE_MODEL_POLICY',None)
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MAX_JOBS='1')
    upstream_base=Path('/home/xmu/nvidiagds/codex-runs/hbserve-memgen-gtsim-alignment-20260914-01a09f50-r1/sglang-integration-r10/third_party')
    env.update(SG_HBSERVE_SOURCE_ROOT=str(upstream_base/'hbserve'),SG_MEMORYINST_CODEC=str(upstream_base/'hbserve_memory_template.py'))
    child_env=dict(env,CUDA_VISIBLE_DEVICES='')
    raw_read,raw_write=os.pipe();projection_read,projection_write=os.pipe()
    children=[];files=[];owned_births={};start=time.monotonic();result=dict(status='STARTED',full_raw_trace_bytes=0)
    old={s:signal.signal(s,lambda sig,frame:(_ for _ in ()).throw(InterruptedError(str(sig)))) for s in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT)}
    def spawn(name,argv,cenv,stdin=subprocess.DEVNULL,stdout=None,pass_fds=()):
        stderr=(a.output/(name+'.stderr')).open('xb');files.append(stderr)
        if stdout is None:stdout=(a.output/(name+'.stdout')).open('xb');files.append(stdout)
        parent=os.getpid()
        c=subprocess.Popen(argv,env=cenv,stdin=stdin,stdout=stdout,stderr=stderr,pass_fds=pass_fds,
                           preexec_fn=lambda:child_guard(parent))
        children.append(c);owned_births[c.pid]=birth(c.pid)
        save(a.output/(name+'-command.json'),dict(argv=argv,pid=c.pid,start_ticks=owned_births[c.pid]))
        return c
    try:
        post=spawn('profiles',[a.python,'-B',str(a.upstream/'full_source_postprocess.py'),'--output',str(a.output/'profiles'),
            '--transport-receipt',str(a.output/'consumer.json'),'--model-policy',policy],child_env,stdin=projection_read)
        os.close(projection_read);projection_read=None
        consumer=spawn('consumer',[a.python,'-B',str(a.upstream/'nvbit_sampler_r4/stream_consumer.py'),
            '--read-fd',str(raw_read),'--max-wire-bytes',str(plan['max_wire_bytes']),'--plan',str(a.plan),
            '--producer-exit',str(a.output/'producer-exit.json'),'--receipt',str(a.output/'consumer.json'),
            '--emit-projection-records'],child_env,stdout=projection_write,pass_fds=(raw_read,))
        os.close(raw_read);raw_read=None;os.close(projection_write);projection_write=None
        gen_env=dict(env,LD_PRELOAD=str(a.sampler_lib),SG_NVBIT_SCOPE_ABI='1',SG_NVBIT_OUTPUT_ROOT=str(observer),
            SG_NVBIT_MAX_BYTES=str(256<<20),ACK_CTX_INIT_LIMITATION='1',SG_SAMPLE_PIPE_FD=str(raw_write))
        gpu=spawn('host',[a.python,'-B',str(HERE/'host.py'),'--model',a.model,'--prefill-length',str(a.prefill_length),
            '--decode-steps',str(a.decode_steps),'--output',str(a.output/'host')],gen_env,pass_fds=(raw_write,))
        os.close(raw_write);raw_write=None
        marked=False
        while any(c.poll() is None for c in children):
            if time.monotonic()-start>a.seconds:raise TimeoutError('Sparse pipeline runtime budget')
            if gpu.poll() is not None and not marked:
                finishes=[]
                for path in observer.glob('process-*/finish.json'):
                    value=json.loads(path.read_text())
                    if value.get('pid')==gpu.pid and value.get('start_ticks')==owned_births[gpu.pid]:finishes.append(path)
                if len(finishes)!=1:raise RuntimeError('Same-process sampler closure missing')
                unchanged=before=={str(x.resolve()):sha(x) for x in paths}
                save(a.output/'source-identity.json',dict(before=before,unchanged=unchanged))
                save(a.output/'producer-exit.json',dict(schema='SG_SAMPLE_PRODUCER_EXIT_V1',returncode=gpu.returncode,
                    pid=gpu.pid,start_ticks=owned_births[gpu.pid],observer_finish=str(finishes[0]),source_pins_match=unchanged))
                marked=True
            for c in children:
                if c.poll() not in (None,0):raise RuntimeError('Owned child failed pid=%s rc=%s'%(c.pid,c.returncode))
            time.sleep(.2)
        if not marked:raise RuntimeError('Producer closure marker missing')
        transport=json.loads((a.output/'consumer.json').read_text())
        if transport['status']!='PASS_SAMPLED_TRANSPORT_ONLY':raise RuntimeError('Transport did not qualify')
        profiles=json.loads((a.output/'profiles/receipt.json').read_text())
        result.update(status='PASS_SINGLE_LAYER_SAMPLES_AND_PROFILE_FITTING',transport_status=transport['status'],
            selected_kernels=len(transport['kernels']),selected_records=transport['selected_records'],
            profiles_accepted=profiles['packed_models_accepted'],profiles_rejected=profiles['packed_models_missing'],
            full_layer_expansion_executed=False,cache_replay_executed=False,hardware_accuracy_accepted=False)
    except BaseException as e:
        result.update(status='FAIL',error=type(e).__name__+': '+str(e))
    finally:
        for fd in (raw_read,raw_write,projection_read,projection_write):
            if fd is not None:os.close(fd)
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=10)
            except subprocess.TimeoutExpired:c.kill();c.wait(timeout=10)
        for f in files:f.close()
        for sig,handler in old.items():signal.signal(sig,handler)
        result.update(elapsed_seconds=time.monotonic()-start,children=[dict(pid=c.pid,returncode=c.returncode) for c in children])
        save(a.output/'finish.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['status'].startswith('PASS_') else 1


if __name__=='__main__':raise SystemExit(main())
