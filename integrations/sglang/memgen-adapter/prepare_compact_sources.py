#!/usr/bin/env python3
"""Create task-private source derivatives; never edit upstream or run a GPU.

The only tracer behavior change omits redundant static-instruction JSON rows.
Decoded SASS hashing, launch census, selected-CTA gates and packet ABI remain.
Explicit host metadata/plan capacities cover the requested D128 census. Dynamic
packet/record/RAM caps remain unchanged; no silent subsampling or truncation.
"""
import argparse, hashlib, json, shutil
from pathlib import Path

OBSERVER=Path('/home/xmu/nvidiagds/codex-runs/hbserve-memgen-gtsim-alignment-20260914-01a09f50-r1/sglang-integration-r6/nvbit_observer_r3')
UPSTREAM=Path('/home/xmu/nvidiagds/codex-runs/l2-sglang-full-sparse-20260919-01a08d87-r5')


def ident(p):
 b=Path(p).read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def replace_one(s,a,b):
 if s.count(a)!=1:raise ValueError('Expected exactly one patch anchor: '+a)
 return s.replace(a,b)
def compact(s):
 before=s
 rows=s.splitlines(keepends=True)
 hits=[i for i,l in enumerate(rows) if 's.emit(s.instruction_file,' in l]
 if len(hits)!=1:raise ValueError('Static row emission identity')
 rows[hits[0]]='    // Task-private compact journal: decoded-SASS digest above is retained; static rows omitted.\n'
 s=''.join(rows)
 s=replace_one(s,'const uint64_t DEFAULT_CAP = 256ull << 20;','const uint64_t DEFAULT_CAP = 1024ull << 20;')
 s=replace_one(s,'MAX_LAUNCHES = 100000, MAX_FUNCTIONS = 4096','MAX_LAUNCHES = 262144, MAX_FUNCTIONS = 16384')
 s=replace_one(s,'metadata quota outside 2..256 MiB envelope','metadata quota outside 2..1024 MiB task envelope')
 # Record omission in lifecycle AND final closure metadata without schema change.
 s=s.replace('\\"module_binary_hash_available\\":false,','\\"module_binary_hash_available\\":false,\\"static_instruction_rows_emitted\\":false,')
 # Exact canonical digest calculation must survive byte for byte.
 for l in before.splitlines():
  if any(x in l for x in ('EVP_DigestUpdate(h,canonical.data()', 'EVP_DigestFinal_ex(h,result', 'd.code_hash=hex.str()')):
   if l not in s.splitlines():raise AssertionError('Decoded SASS digest changed')
 return s

def clone_manifest(src,dst,patches):
 m=json.loads((src/'manifest.json').read_text());old=ident(src/'manifest.json')
 for name,want in m['build_inputs'].items():
  if ident(src/name)!=want:raise ValueError('Upstream source differs: '+str(src/name))
  shutil.copy2(src/name,dst/name)
 for name,fn in patches.items():
  p=dst/name;p.write_text(fn(p.read_text()))
 m['build_inputs']={n:ident(dst/n) for n in m['build_inputs']}
 m.update(task_derivative='SGLANG_SINGLE_LAYER_MATRIX_COMPACT_R1',source_manifest=dict(path=str(src/'manifest.json'),**old),
          static_instruction_rows_emitted=False,decoded_sass_hash_unchanged=True,selected_cta_instrumentation_unchanged=True,
          task_plan_max_bytes=256<<20,task_selected_kernel_capacity=32768,task_max_launches=262144,
          task_max_functions=16384,task_metadata_max_bytes=1<<30)
 (dst/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 return dict(source=str(src),destination=str(dst),source_manifest=old,manifest=ident(dst/'manifest.json'))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--observer-source',type=Path,default=OBSERVER);p.add_argument('--upstream',type=Path,default=UPSTREAM)
 a=p.parse_args();a.output.mkdir(parents=False,exist_ok=False)
 observer=a.output/'observer';observer.mkdir();up=a.output/'upstream';up.mkdir();sampler=up/'nvbit_sampler_r4';sampler.mkdir()
 # Source only, no captures, binaries or prior results copied.
 for n in ('full_source_postprocess.py','sglang_sample_to_packed.py'):shutil.copy2(a.upstream/n,up/n)
 if (a.upstream/'profile_census.py').is_file():shutil.copy2(a.upstream/'profile_census.py',up/'profile_census.py')
 adapter=up/'template_adapter_r4';adapter.mkdir()
 for f in (a.upstream/'template_adapter_r4').iterdir():
  if f.is_file() and f.suffix in ('.py','.json'):shutil.copy2(f,adapter/f.name)
 def plan_capacity(s):
  s=s.replace('<=8<<20','<=256<<20')
  s=s.replace("integer(plan['max_selected_kernels'],1,4096)","integer(plan['max_selected_kernels'],1,32768)")
  s=s.replace('0<len(rows)<=100_000','0<len(rows)<=262_144')
  return s
 o=clone_manifest(a.observer_source,observer,{'observer.cu':compact})
 s=clone_manifest(a.upstream/'nvbit_sampler_r4',sampler,{'sampler.cu':compact,
   'build.py':plan_capacity,'compile_plan.py':plan_capacity,'stream_consumer.py':plan_capacity,
   'packet_stream.py':lambda s:replace_one(s,'MAX_KERNELS = 4096','MAX_KERNELS = 32768')})
 result=dict(status='PASS_PRIVATE_SOURCE_PREPARED_NOT_BUILT',observer=o,sampler=s,
  original_sources_modified=False,GPU_executed=False,dynamic_wire_cap_unchanged=8<<30,
  postprocess_record_cap_unchanged=12000000,warning='Host metadata bounds are capacities only; runtime budgets remain explicit.')
 (a.output/'prepare.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
