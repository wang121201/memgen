"""Build private packet sampler from a frozen manifest plus actual sample plan.
Build only. No GPU invocation, deployment, shared source edit or implicit plan.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from compile_plan import compile_header
from packet_stream import need,parse_json

HERE=Path(__file__).resolve().parent

def pin(p):
    p=Path(p);need(p.is_file() and not p.is_symlink(),'regular build input '+str(p));b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nvbit',type=Path,default=Path('/home/xmu/nvidiagds/simulators/hyfiss/tracing-tool/nvbit'))
    p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda-12.8'))
    a=p.parse_args();out=a.output.resolve();manifest=parse_json((HERE/'manifest.json').read_bytes())
    inputs=[]
    for name,want in manifest['build_inputs'].items():
        got=pin(HERE/name);need(all(got[k]==want[k] for k in ('bytes','sha256')),'frozen source changed '+name);inputs.append(got)
    for name,want in manifest['nvbit_headers'].items():
        got=pin(a.nvbit/name);need(all(got[k]==want[k] for k in ('bytes','sha256')),'NVBit header differs '+name);inputs.append(got)
    inputs.extend([pin(a.cuda/'bin/nvcc'),pin(a.nvbit/'libnvbit.a'),pin(a.plan)])
    need(a.plan.stat().st_size<=256<<20,'plan size');header,plan_receipt=compile_header(parse_json(a.plan.read_bytes()))
    out.mkdir(mode=0o755,parents=False,exist_ok=False);(out/'tmp').mkdir(mode=0o700)
    (out/'sample_plan.h').write_text(header)
    nvcc=str(a.cuda/'bin/nvcc');flags=['-gencode','arch=compute_89,code=sm_89'];inc=['-I'+str(out),'-I'+str(HERE),'-I'+str(a.nvbit)]
    cmds=[('host',[nvcc,'-dc','-c','-std=c++11',*inc,'-Xptxas','-cloning=no','-Xcompiler','-Wall',*flags,'-O3','-Xcompiler','-fPIC',str(HERE/'sampler.cu'),'-o',str(out/'sampler.o')]),
          ('inject',[nvcc,*inc,'-maxrregcount=24','-Xptxas','-astoolspatch','--keep-device-functions',*flags,'-Xcompiler','-Wall','-Xcompiler','-fPIC','-c',str(HERE/'memory_inject_funcs.cu'),'-o',str(out/'memory_inject_funcs.o')]),
          ('link',[nvcc,*flags,'-O3',str(out/'sampler.o'),str(out/'memory_inject_funcs.o'),'-L'+str(a.nvbit),'-lnvbit','-L'+str(a.cuda/'lib64'),'-lcuda','-lcudart_static','-lcrypto','-lpthread','-ldl','-shared','-o',str(out/'sampler.so')]),
          ('symbols',['/usr/bin/nm','-D',str(out/'sampler.so')])]
    env=dict(PATH=str(a.cuda/'bin')+':/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',TMPDIR=str(out/'tmp'),LC_ALL='C.UTF-8',OMP_NUM_THREADS='1')
    result=dict(schema='SG_NATIVE_SAMPLER_BUILD_V1',status='FAIL_BUILD_ONLY',GPU_run=False,inputs=inputs,plan=plan_receipt,steps=[])
    def cancel(sig,frame):raise InterruptedError('build interrupted')
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,cancel)
    active=None
    try:
        for name,argv in cmds:
            start=time.monotonic()
            with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
                active=subprocess.Popen(argv,env=env,cwd=out,stdout=stdout,stderr=stderr,start_new_session=True)
                try:code=active.wait(timeout=180)
                finally:
                    if active.poll() is None:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
            active=None
            result['steps'].append(dict(name=name,argv=argv,returncode=code,seconds=time.monotonic()-start,
                stdout=pin(out/(name+'.stdout')),stderr=pin(out/(name+'.stderr'))))
            need(code==0,'build failed '+name)
        symbols=(out/'symbols.stdout').read_text()
        for name in ('set_scope','clear_scope','begin_epoch','end_epoch','get_status'):
            need(' T sg_nvbit_observer_'+name in symbols,'scope symbol missing '+name)
        after=[pin(x['path']) for x in inputs];need(after==inputs,'build inputs changed')
        result.update(status='PASS_BUILD_ONLY_NO_GPU',binary=pin(out/'sampler.so'),inputs_after=after)
    except BaseException as e:result['error']=type(e).__name__+': '+str(e)
    finally:
        if active is not None and active.poll() is None:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
        with (out/'build.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result.get(k) for k in ('status','binary','error')}));return 0 if result['status']=='PASS_BUILD_ONLY_NO_GPU' else 1
if __name__=='__main__':raise SystemExit(main())
