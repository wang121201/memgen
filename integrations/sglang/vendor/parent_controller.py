#!/usr/bin/env python3
"""Pinned single-CPU/GPU discovery controller; NCU commands are plan-only."""
import argparse
import contextlib
import csv
import ctypes
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

LABEL = 'tilegen-target-ncu-controller-v0.1.0-dev.1'
HOST_LABEL = 'tilegen-target-ncu-host-v0.1.0-dev.1'
HERE = Path(__file__).resolve().parent
HOST = HERE.parent/'tilegen-target-ncu-host-dev1'
RUN_ROOT = Path('/home/xmu/nvidiagds/codex-runs/tilegen-8b-sweep-20260917-r1')
INSTALL_ROOT = RUN_ROOT/'target-ncu-p1024d256-dev1'
GPU_LOCK_ROOT = Path('/home/xmu/nvidiagds/codex-runs/llm-footprint-v1/.locks')
PYTHON = '/home/xmu/sgl/bin/python'
NCU = '/usr/local/cuda-12.8/bin/ncu'
SMI = '/usr/bin/nvidia-smi'
MODEL = '/home/xmu/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct'
GPU_DEFAULT = 'GPU-7a22d253-7921-4e49-992e-0199eebd6f86'
RANGES = {'prefill':'TG_TARGET_P1024D256_PREFILL',
          'decode':'TG_TARGET_P1024D256_DECODE_COMBINED'}
THREAD_KEYS = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS','MAX_JOBS')
STOP = None


def need(ok, message):
    if not ok:
        raise ValueError(message)


def stop_handler(signum, frame):
    global STOP
    if STOP is None:
        STOP = signum


def alive_request():
    need(STOP is None, 'controller cancelled: '+str(STOP))


def pin(path, binary=False):
    path = Path(path)
    requested = str(path.absolute())
    if binary:
        path = path.resolve(strict=True)
    need(path.is_file() and not path.is_symlink(), 'regular file required: '+str(path))
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20), b''):
            h.update(chunk)
    after = path.stat()
    fields = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    need(all(getattr(before,k)==getattr(after,k) for k in fields), 'input changed during hash')
    result = dict(path=str(path.resolve()), bytes=after.st_size, sha256=h.hexdigest())
    if binary:
        need(os.access(path, os.X_OK), 'binary not executable')
        result['requested_path'] = requested
    return result


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def close_receipt(path, record):
    # A cancellation that arrives during serialization must not leave a PASS.
    while True:
        latched = STOP
        if latched is not None:
            record.update(status='FAIL_CANCELLED', termination_signal=latched)
        data = json.dumps(record, indent=2, allow_nan=False)+'\n'
        if latched == STOP:
            break
    tmp = Path(str(path)+'.tmp')
    with tmp.open('x') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    if STOP != latched:
        tmp.unlink()
        return close_receipt(path, record)
    need(not path.exists(), 'receipt already exists')
    os.replace(tmp,path)


def verify_component(expected_sha, here=HERE):
    need(isinstance(expected_sha,str) and re.fullmatch('[0-9a-f]{64}',expected_sha),
         'explicit --manifest-sha required for execution')
    manifest_path = here/'COMPONENT.json'
    identity = pin(manifest_path)
    need(identity['sha256']==expected_sha, 'controller manifest SHA mismatch')
    manifest = json.loads(manifest_path.read_text())
    need(manifest.get('schema')=='TILEGEN_TARGET_NCU_CONTROLLER_COMPONENT_V1' and
         manifest.get('component_label')==LABEL and manifest.get('status')=='SEALED',
         'controller manifest is not this sealed component')
    need(manifest.get('host_component_label')==HOST_LABEL, 'host label mismatch')
    files = manifest.get('files',[])
    need(type(files) is list and len(files)>=10, 'complete component inventory required')
    seen = set()
    for entry in files:
        rel = Path(entry['path'])
        need(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in seen and
             rel.parts[0] in ('tilegen-target-ncu-host-dev1','tilegen-target-ncu-controller-dev1'),
             'unsafe/duplicate/out-of-component manifest path')
        seen.add(str(rel))
        path = here.parent/rel
        need(path.resolve().is_relative_to(here.parent.resolve()), 'source escapes component root')
        actual = pin(path)
        need(all(actual[k]==entry[k] for k in ('bytes','sha256')), 'component source differs: '+str(rel))
    required = {str(Path('tilegen-target-ncu-host-dev1')/x) for x in
                ('host.py','contract.py','execution.py','vendor/parent_p512_driver.py')}
    required.add('tilegen-target-ncu-controller-dev1/controller.py')
    need(required<=seen and pin(manifest_path)==identity, 'missing dependency or changed manifest')
    actual_files=set()
    for directory in (here,here.parent/'tilegen-target-ncu-host-dev1'):
        for path in directory.rglob('*'):
            need(not path.is_symlink(),'component contains symlink')
            if path.is_file() and path!=manifest_path:
                actual_files.add(str(path.relative_to(here.parent)))
    need(actual_files==seen,'unlisted or missing component file')
    return dict(manifest=identity, file_count=len(files), files=files)


def cpu_lock_path(cpu):
    need(type(cpu) is int and 0<=cpu<=15, 'single CPU must be in shared 16-core pool 0..15')
    return RUN_ROOT/('cpu'+str(cpu)+'.lock')


def gpu_lock_path(uuid):
    need(isinstance(uuid,str) and re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',uuid),
         'canonical complete GPU UUID required')
    return GPU_LOCK_ROOT/('gpu-'+uuid+'.lock')


class ResourceBusy(RuntimeError):
    pass


@contextlib.contextmanager
def leases(paths):
    held = []
    try:
        for path in paths:
            # Existing canonical inode only: never replace/unlink/truncate a lock.
            fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
            held.append((Path(path),fd))
            need(stat.S_ISREG(os.fstat(fd).st_mode), 'canonical lock is not regular')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise ResourceBusy('canonical lease occupied: '+str(path))
            need((os.stat(path).st_dev,os.stat(path).st_ino)==
                 (os.fstat(fd).st_dev,os.fstat(fd).st_ino), 'canonical lock inode changed')
        yield held
    finally:
        for _,fd in reversed(held):
            os.close(fd)


def parse_online(text):
    result=set()
    for part in text.strip().split(','):
        values=[int(x) for x in part.split('-')]
        need(1<=len(values)<=2 and 0<=values[0]<=values[-1]<=65535,'invalid online CPU range')
        result.update(range(values[0],values[-1]+1))
    return result


def cpu_admission(cpu):
    cpu_lock_path(cpu)
    need(sys.platform=='linux' and hasattr(os,'sched_setaffinity'), 'GPU execution requires Linux affinity')
    allowed=set(os.sched_getaffinity(0))
    need(cpu in allowed and cpu in parse_online(Path('/sys/devices/system/cpu/online').read_text()),
         'selected CPU offline or outside inherited affinity')
    return dict(cpu_id=cpu, inherited_affinity=sorted(allowed), pool=list(range(16)))


def memory_available():
    rows=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(rows['MemAvailable'].strip().split()[0])*1024


def query_smi(args):
    completed=subprocess.run([SMI,*args], capture_output=True, text=True, timeout=5,
                             env={'PATH':'/usr/bin:/bin','LC_ALL':'C'})
    need(completed.returncode==0 and len(completed.stdout)<65536 and len(completed.stderr)<65536,
         'nvidia-smi failed or output exceeded bound')
    return completed.stdout


def gpu_apps(uuid):
    raw=query_smi(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'])
    result=[]
    for row in csv.reader(io.StringIO(raw)):
        if not row:
            continue
        need(len(row)==3, 'unrecognized nvidia-smi application row')
        if row[0].strip()==uuid:
            result.append(dict(gpu_uuid=uuid,pid=int(row[1].strip()),process_name=row[2].strip()))
    return result


def gpu_admission(uuid):
    raw=query_smi(['--query-gpu=uuid,name,compute_cap,memory.total,memory.used,utilization.gpu',
                   '--format=csv,noheader,nounits','-i',uuid])
    rows=list(csv.reader(io.StringIO(raw)))
    need(len(rows)==1 and len(rows[0])==6, 'one exact GPU required')
    value=[v.strip() for v in rows[0]]
    need(value[0]==uuid and 'RTX 4000 Ada' in value[1] and value[2]=='8.9', 'same-condition sm89 RTX4000 Ada required')
    result=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),uuid=value[0],name=value[1],
                compute_capability=value[2],total_memory_MiB=float(value[3]),
                used_memory_MiB=float(value[4]),utilization_percent=float(value[5]),active_compute_apps=gpu_apps(uuid))
    if result['active_compute_apps'] or result['used_memory_MiB']>=512 or result['utilization_percent']!=0:
        raise ResourceBusy('selected GPU not idle at fresh locked admission: '+json.dumps(result))
    return result


def environment(out, uuid):
    env=dict(PATH='/home/xmu/sgl/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin',CUDA_HOME='/usr/local/cuda-12.8',
        CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LC_ALL='C.UTF-8',
        NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT='1')
    env.update({key:'1' for key in THREAD_KEYS})
    # Pass no inherited LD_PRELOAD/PYTHONPATH/SG_* or credentials. Per-run caches.
    for key,name in [('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),
                     ('XDG_CACHE_HOME','xdg'),('FLASHINFER_WORKSPACE_BASE','flashinfer'),('TMPDIR','tmp')]:
        env[key]=str(out/'cache'/name)
    return env


def command_plan(mode, out):
    need(mode in ('discovery','prefill','decode'),'unsupported mode')
    command=[PYTHON,'-B',str(HOST/'host.py'),'--run-gpu','--output',str(out/'host'),
             '--model-path',MODEL,'--mem-fraction-static','0.80']
    if mode=='discovery':
        command.append('--discovery')
    else:
        command=[NCU,'--config-file','off','--rename-kernels','off','--disable-extra-suffixes',
            '--target-processes','application-only','--replay-mode','app-range','--cache-control','none',
            '--clock-control','none','--nvtx','--nvtx-include',RANGES[mode]+'/',
            '--profile-from-start','on','--metrics','dram__bytes_read.sum,dram__bytes_write.sum',
            '--export',str(out/'capture'),*command]
    return dict(component_label=LABEL,host_component_label=HOST_LABEL,mode=mode,argv=command,
        execution_enabled=(mode=='discovery'),NCU_execution_supported=False,
        NCU_final_qualification='FUTURE_3_INDEPENDENT_APP_RANGE_RUNS_PER_STAGE_MEDIAN_ALL_SAMPLES_REPORTED',
        complete_kernel_census_accepted=False,hardware_counter_acceptance=False)


def process_table():
    rows={}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            raw=(directory/'stat').read_text(); parts=raw[raw.rfind(')')+2:].split()
            pid=int(directory.name)
            rows[pid]=dict(pid=pid,ppid=int(parts[1]),pgid=int(parts[2]),start_ticks=int(parts[19]),
                           state=parts[0],rss_bytes=max(0,int(parts[21]))*os.sysconf('SC_PAGE_SIZE'))
        except (FileNotFoundError,ProcessLookupError):
            pass
    return rows


def descendants(rows, root):
    selected={root}
    while True:
        extra={pid for pid,row in rows.items() if row['ppid'] in selected}-selected
        if not extra:
            return {pid:rows[pid] for pid in selected if pid!=root}
        selected.update(extra)


def enable_subreaper():
    need(sys.platform=='linux','actual discovery controller requires Linux subreaper')
    libc=ctypes.CDLL(None,use_errno=True)
    need(libc.prctl(36,1,0,0,0)==0, 'PR_SET_CHILD_SUBREAPER failed')


def signal_exact(row, sig):
    now=process_table().get(row['pid'])
    if now and now['start_ticks']==row['start_ticks']:
        try:os.kill(row['pid'],sig)
        except ProcessLookupError:pass


def reap():
    while True:
        try:
            pid,_=os.waitpid(-1,os.WNOHANG)
            if not pid:return
        except ChildProcessError:return


def cleanup(child, known, linux):
    if not linux:
        # Portable CPU fixture path only; actual GPU execution requires Linux.
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        return dict(owned_descendants_empty=None,direct_child_reaped=child is None or child.poll() is not None,
                    scope='POSIX_CPU_FIXTURE_PROCESS_GROUP_NOT_LINUX_DESCENDANT_QUALIFICATION')
    known.update(descendants(process_table(),os.getpid()))
    for sig,seconds in ((signal.SIGTERM,3),(signal.SIGKILL,5)):
        end=time.monotonic()+seconds
        while True:
            live=descendants(process_table(),os.getpid());known.update(live)
            for row in live.values():signal_exact(row,sig)
            if child is not None:child.poll()
            reap()
            if not descendants(process_table(),os.getpid()):
                return dict(owned_descendants_empty=True,observed_identities=list(known.values()),
                            scope='LINUX_SUBREAPER_PID_START_TICKS_ONLY_THIS_CONTROLLER_DESCENDANTS')
            if time.monotonic()>=end:break
            time.sleep(.05)
    raise RuntimeError('owned descendants remain after cleanup')


def run_process(argv,out,env,timeout,*,held_fds=(),linux=True,monitor=None,rss_limit=64<<30):
    """One child tree. CPU tests inject only argv/environment, never a GPU bypass."""
    start=time.monotonic();child=None;known={}
    record=dict(status='RUNNING',argv=argv,started_unix_ns=time.time_ns(),wall_limit_seconds=timeout,
                linux_owned_descendant_protocol=linux,peak_owned_RSS_bytes=0,AS_limit_applied=False)
    try:
        alive_request()
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            watched=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)
            mask=signal.pthread_sigmask(signal.SIG_BLOCK,watched)
            try:
                child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env,start_new_session=True,
                    pass_fds=held_fds,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,mask))
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK,mask)
            if linux:known.update(descendants(process_table(),os.getpid()))
            record.update(child_pid=child.pid,child_pgid=child.pid,
                          child_identity=known.get(child.pid),inherited_lease_fd_count=len(held_fds))
            save(out/'process-start.json',record)
            last_monitor=0.0
            while child.poll() is None:
                alive_request()
                need(time.monotonic()-start<timeout,'owned child exceeded wall timeout')
                if linux:
                    live=descendants(process_table(),os.getpid());known.update(live)
                    need(len(known)<=4096,'owned descendant count bound')
                    rss=sum(row['rss_bytes'] for row in live.values())
                    record['peak_owned_RSS_bytes']=max(record['peak_owned_RSS_bytes'],rss)
                    need(rss<=rss_limit,'owned RSS bound exceeded')
                need(sum((out/name).stat().st_size for name in ('stdout.log','stderr.log'))<=256<<20,
                     'stdout/stderr aggregate quota exceeded')
                if monitor and time.monotonic()-last_monitor>=5:
                    monitor(known);last_monitor=time.monotonic()
                time.sleep(.1)
            record['returncode']=child.returncode
            alive_request()
            need(child.returncode==0,'child exited nonzero')
            need(time.monotonic()-start<timeout,'child completed after timeout')
            record['status']='PASS_PROCESS_ONLY'
    except BaseException as error:
        record.update(status='FAIL_PROCESS',error=type(error).__name__+': '+str(error))
    finally:
        if child is not None:record['returncode']=child.poll()
        try:record['cleanup']=cleanup(child,known,linux)
        except BaseException as error:record.update(status='FAIL_CLEANUP',cleanup_error=str(error))
        if child is not None:record['returncode']=child.returncode
        record.update(elapsed_seconds=time.monotonic()-start,finished_unix_ns=time.time_ns(),termination_signal=STOP)
        close_receipt(out/'process-exit.json',record)
    return record


def validate_host(out):
    host=json.loads((out/'host'/'host-receipt.json').read_text())
    phases=json.loads((out/'host'/'phase-metadata.json').read_text())
    need(host['component_label']==HOST_LABEL and
         host['status']=='HOST_COMPLETED_NOT_NCU_OR_KERNEL_CENSUS_ACCEPTED', 'host did not close')
    need(host['raw_discovery_profile_collected'] is True and host['NCU_counter_acceptance'] is False and
         host['full_kernel_census_acceptance'] is False,'host discovery qualification mismatch')
    names=['Prefill']+['Decode'+str(i) for i in range(1,257)]
    need(phases['phase_count']==257 and phases['warmup_forwards']==257 and phases['measured_forwards']==257 and
         [row['phase'] for row in phases['phases']]==names and phases['both_primary_ranges_closed'] is True,
         'host did not complete all257 phases')
    expected=[dict(kind=kind,range=name,phase_index=index) for kind,name,index in
        [('outer_push',RANGES['prefill'],0),('outer_pop',RANGES['prefill'],0),
         ('outer_push',RANGES['decode'],1),('outer_pop',RANGES['decode'],256)]]
    need(phases['range_journal']==expected,'outer range history differs')
    need(host['source_before']==host['source_after'] and host['weights_before']==host['weights_after'],
         'native source or weights changed')
    files=['host-receipt.json','phase-metadata.json','module-calls.json','tensor-roots.json','discovery.chrome.json']
    return dict(status='PASS_HOST_CLOSURE_ONLY_NOT_INDEPENDENT_KERNEL_CENSUS_OR_NCU',
                phase_count=257,files=[pin(out/'host'/name) for name in files])


def execute(args, out):
    need(args.mode=='discovery','NCU execution is not implemented by this component; plan only')
    alive_request()
    before=verify_component(args.manifest_sha)
    cpu=cpu_admission(args.cpu_id)
    binaries={name:pin(path,binary=True) for name,path in [('python',PYTHON),('nvidia_smi',SMI)]}
    need(not out.exists() and not out.is_symlink(),'fresh run directory required')
    with leases([cpu_lock_path(args.cpu_id),gpu_lock_path(args.gpu_uuid)]) as held:
        alive_request()
        gpu=gpu_admission(args.gpu_uuid)
        available=memory_available()
        need(available>=32<<30,'at least32GiB MemAvailable required')
        alive_request()
        need(not out.exists() and not out.is_symlink(),'fresh run directory required')
        need(HERE.parent.resolve()==INSTALL_ROOT.resolve(),'execution requires reviewed installation root')
        need(os.statvfs(INSTALL_ROOT).f_bavail*os.statvfs(INSTALL_ROOT).f_frsize>=8<<30,
             'at least8GiB free output capacity required')
        enable_subreaper()
        os.sched_setaffinity(0,{args.cpu_id})
        need(set(os.sched_getaffinity(0))=={args.cpu_id},'singleCPU affinity failed')
        out.mkdir(parents=True,exist_ok=False)
        record=dict(schema='TILEGEN_TARGET_NCU_CONTROLLER_RUN_V1',component_label=LABEL,
            host_component_label=HOST_LABEL,status='RUNNING_DISCOVERY',started_unix_ns=time.time_ns(),
            mode=args.mode,run_id=args.run_id,manifest_before=before,binaries_before=binaries,
            cpu_admission=cpu,actual_affinity=sorted(os.sched_getaffinity(0)),gpu_admission=gpu,
            memory_available_at_admission=available,AS_limit_applied=False,
            leases=[dict(path=str(path),device=os.fstat(fd).st_dev,inode=os.fstat(fd).st_ino) for path,fd in held],
            NCU_executed=False,independent_kernel_census_accepted=False,hardware_counter_acceptance=False)
        save(out/'controller-start.json',record)
        try:
            env=environment(out,args.gpu_uuid)
            for key in ('TRITON_CACHE_DIR','CUDA_CACHE_PATH','XDG_CACHE_HOME','FLASHINFER_WORKSPACE_BASE','TMPDIR'):
                Path(env[key]).mkdir(parents=True,exist_ok=False)
            plan=command_plan(args.mode,out)
            save(out/'command.json',dict(plan=plan,environment=env))
            def monitor(known):
                apps=gpu_apps(args.gpu_uuid);rows=process_table()
                # A helper may spawn while nvidia-smi is running: resample the
                # owned tree before classifying its fresh compute PID as foreign.
                known.update(descendants(rows,os.getpid()))
                foreign=[row for row in apps if row['pid'] not in known or row['pid'] not in rows or
                         known[row['pid']]['start_ticks']!=rows[row['pid']]['start_ticks']]
                need(not foreign,'foreign GPU compute process appeared; only this controller subtree will stop')
            process=run_process(plan['argv'],out,env,args.timeout_seconds,
                                held_fds=tuple(fd for _,fd in held),monitor=monitor)
            record['process']=process
            need(process['status']=='PASS_PROCESS_ONLY' and process['cleanup']['owned_descendants_empty'] is True,
                 'owned host process did not close successfully')
            record['host_closure']=validate_host(out)
            record['status']='PASS_DISCOVERY_HOST_CLOSURE_NOT_KERNEL_CENSUS_OR_NCU_ACCEPTANCE'
        except BaseException as error:
            record.update(status='FAIL_DISCOVERY',error=type(error).__name__+': '+str(error))
        finally:
            try:
                record['manifest_after']=verify_component(args.manifest_sha)
                record['binaries_after']={name:pin(path,binary=True) for name,path in [('python',PYTHON),('nvidia_smi',SMI)]}
                need(record['manifest_after']==before and record['binaries_after']==binaries,'source/binary identity changed')
                record['source_and_binary_identity_unchanged']=True
                record['gpu_apps_after']=gpu_apps(args.gpu_uuid)
                need(not record['gpu_apps_after'],'GPU compute owner remains after cleanup; never kill foreign owner')
            except BaseException as error:
                record.update(status='FAIL_IDENTITY_OR_GPU_CLOSURE',closure_error=str(error))
            record.update(finished_unix_ns=time.time_ns(),termination_signal=STOP)
            close_receipt(out/'controller-exit.json',record)
        return 0 if record['status'].startswith('PASS_') and STOP is None else (128+STOP if STOP else 1)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('discovery','prefill','decode'),default='discovery')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--run-id',default='discovery-r1')
    parser.add_argument('--cpu-id',type=int,default=3)
    parser.add_argument('--gpu-uuid',default=GPU_DEFAULT)
    parser.add_argument('--manifest-sha')
    parser.add_argument('--timeout-seconds',type=int,default=900)
    args=parser.parse_args(argv)
    need(re.fullmatch('[a-z0-9][a-z0-9-]{0,79}',args.run_id),'safe unique run-id required')
    need(1<=args.timeout_seconds<=86400,'bounded walltimeout required')
    cpu_lock_path(args.cpu_id);gpu_lock_path(args.gpu_uuid)
    out=INSTALL_ROOT/'runs'/args.run_id
    if not args.execute:
        print(json.dumps(dict(**command_plan(args.mode,out),dry_run=True,resources_not_probed=True,
                             output=str(out),cpu_id=args.cpu_id,gpu_uuid=args.gpu_uuid),indent=2))
        return 0
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop_handler)
    try:return execute(args,out)
    except ResourceBusy as error:
        need(not out.exists(),'busy rejection unexpectedly has output')
        cancelled=STOP is not None
        print(json.dumps(dict(schema='CANCELLED_BEFORE_OUTPUT' if cancelled else 'RESOURCE_BUSY_NO_OUTPUT',
            component_label=LABEL,run_id=args.run_id,error=str(error),termination_signal=STOP,
            output_created=False,child_started=False)),file=sys.stderr)
        return 128+STOP if cancelled else 75
    except BaseException as error:
        print(json.dumps(dict(schema='DISCOVERY_ADMISSION_REJECTED',component_label=LABEL,
                             error=type(error).__name__+': '+str(error),termination_signal=STOP)),file=sys.stderr)
        return 128+STOP if STOP else 1


if __name__=='__main__':
    raise SystemExit(main())
