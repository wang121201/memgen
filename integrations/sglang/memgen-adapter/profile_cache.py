#!/usr/bin/env python3
"""CPU-only profile expansion then original MemGen, under run_job.py."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from followthrough import guard
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--sample',type=Path,required=True);p.add_argument('--bindings',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=21600,
  help='accepted for compatibility; the cache replay has no wall-clock deadline')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 if os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('CPU-only cache stage requires no GPU')
 result=dict(status='RUNNING',stages=[],hardware_accuracy_accepted=False);start=time.monotonic()
 try:
  cmds=[('expand',[sys.executable,'-B',str(HERE/'expand_profiles.py'),'--sample-output',str(a.sample),
    '--layer-bindings',str(a.bindings),'--output',str(a.output/'expanded')],1800),
   ('cache',[sys.executable,'-B',str(HERE/'run_memgen.py'),'--expanded',str(a.output/'expanded'),
    '--output',str(a.output/'cache')],None)]
  for name,argv,timeout in cmds:
   if name=='cache':
    m=json.loads((a.output/'expanded/manifest.json').read_text())
    if not m['complete_declared_profile_stream']:
     result.update(status='STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC',unsupported_launches=m['unsupported_launches']);break
   parent=os.getpid();t=time.monotonic()
   with (a.output/(name+'.stdout')).open('xb') as out,(a.output/(name+'.stderr')).open('xb') as err:
    c=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,preexec_fn=lambda:guard(parent))
    try:c.wait(timeout=timeout)
    finally:
     if c.poll() is None:
      c.terminate()
      try:c.wait(timeout=10)
      except subprocess.TimeoutExpired:c.kill();c.wait()
   result['stages'].append(dict(stage=name,returncode=c.returncode,wall_minutes=(time.monotonic()-t)/60,argv=argv))
   if c.returncode:raise RuntimeError(name+' failed')
  if result['status']=='RUNNING':result['status']='PASS_DECLARED_PROFILE_CACHE_MODEL'
 except BaseException as e:result.update(status='FAIL_DEPENDENCY',error=type(e).__name__+': '+str(e))
 result['wall_minutes']=(time.monotonic()-start)/60
 (a.output/'finish.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 return 0 if result['status'].startswith('PASS_') else 2
if __name__=='__main__':raise SystemExit(main())
