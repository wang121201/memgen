#!/usr/bin/env python3
"""Compare fixed service-cost diagnostics for one already closed paired replay."""
import argparse,csv,hashlib,importlib.util,json,pathlib,subprocess,sys
P=pathlib.Path;ROOT=P(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=P,required=True);p.add_argument('--prior-tool',type=P,required=True);p.add_argument('--output',type=P,required=True);a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);replay=a.replay.resolve()
 finish=json.loads((replay/'finish.json').read_text());assert finish['status']=='PASS_FULL_SOURCE_CACHE_REPLAY_NOT_HARDWARE_ACCEPTANCE' and finish['warmup_replayed']
 population=json.loads((replay/'population.json').read_text());byid={str(r['kernel_id']):r for r in population};assert len(byid)==len(population)
 selected={k:r for k,r in byid.items() if r['role']=='measurement'};assert len(selected)==finish['measurement_launches']
 phase=out/'measurement-phases.csv'
 with phase.open('w') as f:w=csv.DictWriter(f,fieldnames=['kernel_id','phase']);w.writeheader();w.writerows(dict(kernel_id=k,phase=r['phase']) for k,r in selected.items())
 source=[];results={};counts={};pins={str(replay/'finish.json'):sha(replay/'finish.json'),str(replay/'population.json'):sha(replay/'population.json')}
 costs=ROOT/'latency/fixtures/synthetic_latencies.json';weights=json.loads(costs.read_text());pins[str(costs)]=sha(costs)
 for model in ['legacy','r4-small-shared']:
  original=replay/model/'model/kernel_summary.csv';rows=list(csv.DictReader(original.open()));assert {r['kernel_id'] for r in rows}==set(byid)
  rows=[r for r in rows if r['kernel_id'] in selected];assert len(rows)==len(selected)
  for row in rows:
   assert int(row['dram_load_bytes'])==int(row['dram_load_sectors'])*32 and int(row['dram_store_bytes'])==int(row['dram_store_sectors'])*32
   assert int(row['l1_pending_hits'])==int(row['l2_pending_hits'])==0
  counts[model]={r['kernel_id']:(int(r['mem_insts']),int(r['lane_accesses'])) for r in rows}
  summary=out/(model+'-measurement.csv')
  with summary.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  pins[str(original)]=sha(original);source.append(json.loads((replay/model/'source-stats.json').read_text()))
  for version,tool in [('prior',a.prior_tool),('merged',ROOT/'latency/simple_latency.py')]:
   dest=out/(model+'-'+version);dest.mkdir();cmd=[sys.executable,'-B',str(tool),'--kernel-summary',str(summary),'--latency-config',str(costs),'--phase-map',str(phase),'--output-json',str(dest/'result.json'),'--output-csv',str(dest/'phases.csv')]
   with (dest/'log.txt').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
   assert r.returncode==0,(model,version)
   results[(model,version)]=json.loads((dest/'result.json').read_text());pins[str(tool)]=sha(tool)
  for field in ['whole','by_phase','kernel_count','latencies_ns']:assert results[(model,'prior')][field]==results[(model,'merged')][field],field
 assert counts['legacy']==counts['r4-small-shared']
 for field in ['semantic_digest_a','semantic_digest_b','generated_memory_instructions','generated_lane_addresses']:
  assert source[0][field]==source[1][field],field
 baseline=results[('legacy','merged')];candidate=results[('r4-small-shared','merged')]
 rows=[]
 for scope in ['whole',*baseline['by_phase']]:
  old=baseline['whole'] if scope=='whole' else baseline['by_phase'][scope];new=candidate['whole'] if scope=='whole' else candidate['by_phase'][scope]
  delta={k:new['event_counts'][k]-old['event_counts'][k] for k in weights};weighted={k:delta[k]*weights[k] for k in weights}
  total=new['serial_memory_work_ns']-old['serial_memory_work_ns'];assert sum(weighted.values())==total
  rows.append(dict(scope=scope,legacy_event_counts=old['event_counts'],r4_event_counts=new['event_counts'],event_delta=delta,weighted_delta_ns=weighted,
    legacy_serial_work_ns=old['serial_memory_work_ns'],r4_serial_work_ns=new['serial_memory_work_ns'],relative_change_percent=100*total/old['serial_memory_work_ns']))
 result=dict(status='PASS_PAIRED_TRAFFIC_WEIGHTED_DIAGNOSTIC',scope='measurement kernels selected from the same continuous warmup replay',
  kernel_count=len(selected),warmup_kernels_excluded=len(population)-len(selected),weights_ns=weights,weights_are_synthetic=True,
  same_input_prior_and_merged_tool_equivalent=True,hardware_accuracy_accepted=False,gpu_speedup_claim=False,rows=rows,input_sha256=pins)
 (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}))
if __name__=='__main__':main()
