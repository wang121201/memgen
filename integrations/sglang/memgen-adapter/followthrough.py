#!/usr/bin/env python3
"""Sequential dependencies under the existing outer lease controller.

This is not a scheduler: it acquires no leases, launches no detached jobs and
never broadens CPU affinity. Use one fresh output directory per attempt.
"""
import argparse, ctypes, hashlib, json, os, resource, signal, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent

def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def guard(parent):
 if os.getppid()!=parent or ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:os._exit(125)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--journal',type=Path,required=True);p.add_argument('--host-finish',type=Path,required=True)
 p.add_argument('--sources',type=Path,required=True,help='compact-sources-r1 directory')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--stop-after',choices=('plan','build','sample','expand','memgen'),default='memgen')
 p.add_argument('--model-uncovered',choices=('refuse','modeled'),default='refuse',
  help='modeled lets a class with no admitted template complete the full model with an explicit numeric_modeled label')
 p.add_argument('--python',default='/home/xmu/sgl/bin/python')
 p.add_argument('--sample-seconds',type=int,default=7200);p.add_argument('--memgen-seconds',type=int,default=21600,
  help='accepted for compatibility; the replay has no wall-clock deadline')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 host=json.loads(a.host_finish.read_text());contract=host['input_contract'];stages=[];start=time.monotonic()
 result=dict(status='RUNNING',raw_trace_files=False,hardware_accuracy_accepted=False,stages=stages)
 def run(name,argv,timeout):
  usage=resource.getrusage(resource.RUSAGE_CHILDREN);t=time.monotonic();parent=os.getpid()
  with (a.output/(name+'.stdout')).open('xb') as out,(a.output/(name+'.stderr')).open('xb') as err:
   c=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,preexec_fn=lambda:guard(parent))
   try:c.wait(timeout=timeout)
   finally:
    if c.poll() is None:
     c.terminate()
     try:c.wait(timeout=10)
     except subprocess.TimeoutExpired:c.kill();c.wait()
  after=resource.getrusage(resource.RUSAGE_CHILDREN)
  stages.append(dict(stage=name,argv=argv,returncode=c.returncode,wall_minutes=(time.monotonic()-t)/60,
   cpu_minutes=(after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime)/60))
  save(a.output/'progress.json',result)
  if c.returncode:raise RuntimeError(name+' failed; inspect bounded stage logs')
  return a.stop_after==name
 try:
  done=run('plan',[a.python,'-B',str(HERE/'make_sample_plan.py'),'--journal',str(a.journal),
   '--host-finish',str(a.host_finish),'--output',str(a.output/'plan')],300)
  if not done:
   done=run('build',[a.python,'-B',str(a.sources/'upstream/nvbit_sampler_r4/build.py'),
    '--plan',str(a.output/'plan/sample-plan.json'),'--output',str(a.output/'sampler-build')],900)
  if not done:
   done=run('sample',[a.python,'-B',str(HERE/'sample_pipeline.py'),'--upstream',str(a.sources/'upstream'),
    '--sampler-lib',str(a.output/'sampler-build/sampler.so'),'--plan',str(a.output/'plan/sample-plan.json'),
    '--model',contract['model_key'],'--prefill-length',str(contract['prefill_length']),
    '--decode-steps',str(contract['decode_steps']),'--output',str(a.output/'sample'),
    '--seconds',str(a.sample_seconds),'--python',a.python],a.sample_seconds+120)
  if not done:
   done=run('expand',[a.python,'-B',str(HERE/'expand_profiles.py'),'--sample-output',str(a.output/'sample'),
    '--layer-bindings',str(a.output/'plan/layer-bindings.json'),'--output',str(a.output/'expanded'),
    '--model-uncovered',a.model_uncovered],1800)
  if not done:
   manifest=json.loads((a.output/'expanded/manifest.json').read_text())
   result.update(modeled_completion=manifest['modeled_completion'],exact_launches=manifest['exact_launches'],
    modeled_launches=manifest['modeled_launches'],modeled_fraction=manifest['modeled_fraction'],
    fully_exact=manifest['fully_exact'],modeled_by_cause=manifest['modeled_by_cause'])
   if not manifest['complete_full_model']:
    result.update(status='STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC',unsupported_launches=manifest['unsupported_launches'])
   else:
    run('memgen',[a.python,'-B',str(HERE/'run_memgen.py'),'--expanded',str(a.output/'expanded'),
     '--output',str(a.output/'cache')],None)
  if result['status']=='RUNNING':result['status']='PASS_THROUGH_'+stages[-1]['stage'].upper()
 except BaseException as e:result.update(status='FAIL_DEPENDENCY',error=type(e).__name__+': '+str(e))
 result.update(wall_minutes=(time.monotonic()-start)/60,input_contract=contract)
 save(a.output/'finish.json',result);print(json.dumps(result))
 return 0 if result['status'].startswith('PASS_') else 2
if __name__=='__main__':raise SystemExit(main())
