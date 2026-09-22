#!/usr/bin/env python3
"""Synthetic negative-input checks; no hardware timing claim."""
import argparse,csv,json,math,pathlib,subprocess,sys
P=pathlib.Path;ROOT=P(__file__).resolve().parents[1]
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=P,required=True);args=a.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 base=list(csv.DictReader((ROOT/'latency/fixtures/kernel_summary.csv').open()));fields=list(base[0])
 costs=json.loads((ROOT/'latency/fixtures/synthetic_latencies.json').read_text());passed=[]
 cases=[('nan',dict(costs,l1_hit=float('nan')),base,None),('inf',dict(costs,l2_hit=float('inf')),base,None),
  ('negative-inf',dict(costs,dram_read_sector=-float('inf')),base,None),
  ('negative',dict(costs,l1_hit=-1),base,None),('overflow',dict(costs,l1_hit=1e308),base,None),
  ('conservation',costs,[dict(base[0],l1_requests=str(int(base[0]['l1_requests'])+1)),base[1]],None),
  ('duplicate-kernel',costs,[base[0],base[0]],None),('missing-phase',costs,base,[dict(kernel_id=base[0]['kernel_id'],phase='Prefill')])]
 for name,cfg,rows,phases in cases:
  d=out/name;d.mkdir();(d/'latencies.json').write_text(json.dumps(cfg))
  with (d/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
  cmd=[sys.executable,'-B',str(ROOT/'latency/simple_latency.py'),'--kernel-summary',str(d/'summary.csv'),'--latency-config',str(d/'latencies.json'),'--output-json',str(d/'result.json'),'--output-csv',str(d/'result.csv')]
  if phases:
   with (d/'phase.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=['kernel_id','phase']);w.writeheader();w.writerows(phases)
   cmd+=['--phase-map',str(d/'phase.csv')]
  with (d/'log.txt').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode!=0 and not (d/'result.json').exists(),name;passed.append(name)
 result=dict(status='PASS_SIMPLE_LATENCY_INVALID_INPUT_REJECTION',cases=passed,hardware_accuracy_accepted=False)
 (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
