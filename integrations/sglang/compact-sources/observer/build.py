"""Build the metadata-only observer in a fresh directory. No GPU execution.

The caller supplies the CPU/resource lease. No original NVBit or SGLang file is
modified. Only this frozen source and real installed NVBit headers are compiled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE=Path(__file__).resolve().parent
SYMBOLS=('set_scope','clear_scope','begin_epoch','end_epoch','get_status')


def pin(path):
    p=Path(path)
    if not p.is_file() or p.is_symlink():raise ValueError('regular build input required: '+str(p))
    b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nvbit',type=Path,default=Path('/home/xmu/nvidiagds/simulators/hyfiss/tracing-tool/nvbit'))
    p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();source=HERE/'observer.cu'
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name in ('observer.cu','build.py'):
        actual=pin(HERE/name);want=manifest['build_inputs'][name]
        if any(actual[k]!=want[k] for k in ('bytes','sha256')):raise ValueError('frozen build source changed: '+name)
    inputs=[pin(source),pin(HERE/'build.py'),pin(a.cuda/'bin/nvcc'),pin(a.nvbit/'libnvbit.a')]
    for name,expected in manifest['nvbit_headers'].items():
        actual=pin(a.nvbit/name)
        if any(actual[k]!=expected[k] for k in ('bytes','sha256')):raise ValueError('NVBit header differs: '+name)
        inputs.append(actual)
    out.mkdir(mode=0o755,parents=False,exist_ok=False)
    (out/'tmp').mkdir(mode=0o700)
    flags=['-gencode','arch=compute_89,code=sm_89'];nvcc=str(a.cuda/'bin/nvcc')
    commands=[('host',[nvcc,'-dc','-c','-std=c++11','-I'+str(a.nvbit),'-Xptxas','-cloning=no',
                       '-Xcompiler','-Wall',*flags,'-O3','-Xcompiler','-fPIC',str(source),'-o',str(out/'observer.o')]),
              ('link',[nvcc,*flags,'-O3',str(out/'observer.o'),'-L'+str(a.nvbit),'-lnvbit',
                       '-L'+str(a.cuda/'lib64'),'-lcuda','-lcudart_static','-lcrypto','-lpthread','-ldl',
                       '-shared','-o',str(out/'observer.so')]),
              ('symbols',['/usr/bin/nm','-D',str(out/'observer.so')])]
    env=dict(PATH=str(a.cuda/'bin')+':/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',TMPDIR=str(out/'tmp'),
             LC_ALL='C.UTF-8',OMP_NUM_THREADS='1')
    result=dict(schema='SG_NVBIT_OBSERVER_BUILD_V1',status='FAIL_BUILD_ONLY',GPU_run=False,
                original_files_modified=False,inputs=inputs,steps=[])
    active=None
    def cancel(sig,frame):raise InterruptedError('build interrupted')
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,cancel)
    try:
        for name,argv in commands:
            start=time.monotonic()
            with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
                active=subprocess.Popen(argv,env=env,cwd=out,stdout=stdout,stderr=stderr,start_new_session=True)
                try:code=active.wait(timeout=180)
                finally:
                    if active.poll() is None:
                        os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
            active=None
            result['steps'].append(dict(name=name,argv=argv,returncode=code,seconds=time.monotonic()-start,
                stdout=pin(out/(name+'.stdout')),stderr=pin(out/(name+'.stderr'))))
            if code:raise RuntimeError('build failed: '+name)
        symbols=(out/'symbols.stdout').read_text()
        for name in SYMBOLS:
            if ' T sg_nvbit_observer_'+name not in symbols:raise RuntimeError('missing scope ABI: '+name)
        after=[pin(x['path']) for x in inputs]
        if after!=inputs:raise RuntimeError('build input identity changed')
        result.update(status='PASS_BUILD_ONLY_NO_GPU',binary=pin(out/'observer.so'),inputs_after=after)
    except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
        with (out/'build.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result.get(k) for k in ('status','binary','error')}))
    return 0 if result['status']=='PASS_BUILD_ONLY_NO_GPU' else 1


if __name__=='__main__':raise SystemExit(main())
