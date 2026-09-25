#!/usr/bin/env python3
"""One-shot census dependency waiter; all resource/cleanup control is run_job.py.

This script starts only the one explicitly specified case. It cannot select
another GPU/CPU, restart failures, run a partial model, or expand the matrix.
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
GPU3='GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13'
FROZEN=Path('/home/xmu/nvidiagds/codex-runs/memgen-paper-ada-v1-20260916-01a08d87-r1')

def pin(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def closed(root,status):
 paths=list(root.glob('process-*/finish.json'))
 if len(paths)!=1:raise ValueError('Require one process finish under '+str(root))
 value=json.loads(paths[0].read_text())
 if value['status']!=status:raise ValueError('Dependency did not close: '+str(paths[0]))
 return paths[0],value

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--census-job-finish',type=Path,required=True)
 p.add_argument('--observer-root',type=Path,required=True);p.add_argument('--host-root',type=Path,required=True)
 p.add_argument('--sources',type=Path,required=True);p.add_argument('--observer-binary',type=Path,required=True)
 p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--cache-directory',type=Path,required=True)
 p.add_argument('--python',default='/home/xmu/sgl/bin/python')
 p.add_argument('--engine',type=Path,
  help='frozen engine the cache stage replays through; forwarded to profile_cache.py, which '
       'builds it from release/source/tools/ when that stage runs. Without it the cache stage '
       'is refused instead of quietly using a binary this repository did not build.')
 p.add_argument('--wait-seconds',type=int,default=2400)
 p.add_argument('--sample-seconds',type=int,default=7200);p.add_argument('--memgen-seconds',type=int,default=21600)
 a=p.parse_args()
 if not 1<=a.wait_seconds<=86400:raise ValueError('Bounded dependency wait required')
 a.output.mkdir(parents=True,exist_ok=False);result=dict(status='WAITING_CENSUS',CPU_pool=list(range(16)),CPU=8,GPU=GPU3,stages=[])
 # Verify the reviewable adapter manifest before waiting, not after code can drift.
 deployment=json.loads((HERE/'deployment-files.json').read_text())
 for row in deployment['files']:
  got=pin(HERE/row['name'])
  if any(got[k]!=row[k] for k in ('bytes','sha256')):raise ValueError('Adapter deployment changed: '+row['name'])
 sources=[HERE/row['name'] for row in deployment['files']]+[HERE/'deployment-files.json',a.controller,
  a.controller.parent/'vendor/parent_controller.py',a.observer_binary]
 sources+=[x for x in a.sources.rglob('*') if x.is_file() and x.suffix in ('.py','.json','.h','.cu','.inc')]
 sources += [FROZEN/'bin/hbserve',FROZEN/'config/RTX4000Ada.paper-v1.config',FROZEN/'release-manifest.json']
 pins=[pin(x) for x in sorted(set(sources))];save(a.output/'source-pins.json',pins)
 start=time.monotonic();save(a.output/'progress.json',result)
 try:
  while not a.census_job_finish.exists():
   if time.monotonic()-start>a.wait_seconds:raise TimeoutError('Specified census did not finish within wait budget')
   time.sleep(10)
  job=json.loads(a.census_job_finish.read_text())
  if job['status']!='PASS_PROCESS_ONLY':raise ValueError('Specified census job failed; no successor launched')
  journal,observer=closed(a.observer_root,'PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE')
  host,host_value=closed(a.host_root,'PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE')
  if observer['epoch_begin_count']!=observer['epoch_end_count'] or observer['active_epoch']!=0:raise ValueError('Census epochs incomplete')
  if host.parent.name!='process-'+str(observer['pid']):raise ValueError('Host and observer must be the same process')
  if job['cpu']!=8 or job['gpu']!=GPU3 or job['case_id']!=host_value['input_contract']['case_id']:raise ValueError('Specified census resource/case mismatch')
  if pins!=[pin(x['path']) for x in pins]:raise ValueError('Pinned source changed while waiting')
  case=host_value['input_contract']['case_id']
  inputs=[a.census_job_finish,journal,host,journal.parent/'launch-journal.jsonl']
  pins+=list(map(pin,inputs));save(a.output/'qualified-input-pins.json',pins)
  def job_run(name,argv,gpu,seconds,extra):
   if pins!=[pin(x['path']) for x in pins]:raise ValueError('Pinned source/input drift')
   spec=dict(case_id=case,tool='memgen',input_kind='single_layer_profile',cpu=8,gpu=gpu,
    seconds=seconds,rss_limit_bytes=64<<30,cache_directory=str(a.cache_directory),argv=argv,
    sources=pins+list(map(pin,extra)))
   specfile=a.output/(name+'-spec.json');save(specfile,spec)
   with (a.output/(name+'-controller.stdout')).open('xb') as out,(a.output/(name+'-controller.stderr')).open('xb') as err:
    code=subprocess.call([a.python,'-B',str(a.controller),'--spec',str(specfile),'--output',str(a.output/(name+'-job')),'--execute'],stdout=out,stderr=err)
   receipt=json.loads((a.output/(name+'-job/job-finish.json')).read_text())
   result['stages'].append(dict(stage=name,controller_returncode=code,job_status=receipt['status']))
   save(a.output/'progress.json',result)
   if code or receipt['status']!='PASS_PROCESS_ONLY':raise RuntimeError(name+' did not pass; no successor launched')
  result['status']='RUNNING_SPARSE_SAMPLE';save(a.output/'progress.json',result)
  flow=a.output/'sample-followthrough'
  job_run('sample',[a.python,'-B',str(HERE/'followthrough.py'),'--journal',str(journal.parent),
   '--host-finish',str(host),'--sources',str(a.sources),'--output',str(flow),'--stop-after','sample',
   '--sample-seconds',str(a.sample_seconds),'--python',a.python],GPU3,a.sample_seconds+1500,[])
  sample_finish=flow/'sample/finish.json';f=json.loads(sample_finish.read_text())
  if f['status']!='PASS_SINGLE_LAYER_SAMPLES_AND_PROFILE_FITTING':raise ValueError('Sample closure failed')
  result['status']='RUNNING_CPU_PROFILE_CACHE';save(a.output/'progress.json',result)
  extra=[flow/'plan/layer-bindings.json',sample_finish]+[x for x in (flow/'sample').rglob('*')
   if x.is_file() and x.name in ('profiles.pack','profiles.index.jsonl','module_calls.json','tensor_metadata.json','launch-journal.jsonl','finish.json')]
  job_run('cache',[a.python,'-B',str(HERE/'profile_cache.py'),'--sample',str(flow/'sample'),
   '--bindings',str(flow/'plan/layer-bindings.json'),'--output',str(a.output/'profile-cache'),
   '--seconds',str(a.memgen_seconds)]
   + (['--engine',str(a.engine)] if a.engine else []),
   None,a.memgen_seconds+2000,extra)
  result['status']='PASS_DECLARED_PROFILE_CACHE_MODEL_NOT_NATIVE_ACCURACY'
 except BaseException as e:result.update(status='STOP_DEPENDENCY_FAILED',error=type(e).__name__+': '+str(e))
 result['wall_minutes']=(time.monotonic()-start)/60;save(a.output/'finish.json',result);print(json.dumps(result))
 return 0 if result['status'].startswith('PASS_') else 2
if __name__=='__main__':raise SystemExit(main())
