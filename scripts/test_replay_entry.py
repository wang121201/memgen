#!/usr/bin/env python3
"""Exercise the actual replay executable and wrapper; no mocked cache results."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('cache_fixture',ROOT/'scripts/test_cache_core.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixture=out/'fixture';context=module.make_fixture(fixture)
    script=ROOT/'integrations/sglang/memgen-adapter/run_memgen.py'
    config=ROOT/'release/config/RTX4000Ada.r4.config';legacy=ROOT/'release/config/RTX4000Ada.paper-v1.config'
    base=[sys.executable,'-B',str(script),'--binary',str(a.binary.resolve()),'--profile-index',str(fixture/'profiles.index.jsonl'),
          '--app-config',str(fixture/'app.config'),'--issue-config',str(fixture/'issue.config')]
    checks=[]
    def run(name,args,success):
        with (out/(name+'.log')).open('x') as f:r=subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT)
        assert (r.returncode==0)==success,(name,r.returncode)
        checks.append(name)
    run('native-r4',base+['--hw-config',config,'--r4-context',fixture/'context.json','--output',out/'native-r4','--seconds','1'],True)
    finish=json.loads((out/'native-r4/finish.json').read_text())
    assert finish['status']=='PASS_PROFILE_STREAM_CACHE_REPLAY'
    assert finish['generated_memory_instructions']==6 and finish['generated_lane_addresses']==136
    assert finish['source_pins_unchanged'] and not finish['wallclock_cutoff']
    assert not finish['allow_full_NCU_accuracy_comparison'] and not finish['hardware_accuracy_accepted']
    assert finish['r4_context_sha256']==module.sha(fixture/'context.json')
    run('missing-context',base+['--hw-config',config,'--output',out/'missing-context'],False)
    assert not (out/'missing-context').exists()
    context['model_id']='wrong-model';(fixture/'wrong.json').write_text(json.dumps(context))
    run('bad-context',base+['--hw-config',config,'--r4-context',fixture/'wrong.json','--output',out/'bad-context'],False)
    assert json.loads((out/'bad-context/finish.json').read_text())['status']=='FAIL'
    assert not (out/'bad-context/source-stats.json').exists()
    # Existing output must not be overwritten, even with otherwise valid input.
    before=(out/'native-r4/finish.json').read_bytes()
    run('existing-output',base+['--hw-config',config,'--r4-context',fixture/'context.json','--output',out/'native-r4'],False)
    assert (out/'native-r4/finish.json').read_bytes()==before
    manifest=dict(complete_full_model=True,complete_declared_profile_stream=True,unsupported_launches=[],
                  unknown_private_allocations=0,unknown_private_bytes=0)
    (fixture/'manifest.json').write_text(json.dumps(manifest))
    expanded=[sys.executable,'-B',script,'--binary',a.binary.resolve(),'--expanded',fixture]
    run('legacy-expanded',expanded+['--hw-config',legacy,'--output',out/'legacy-expanded'],True)
    finish=json.loads((out/'legacy-expanded/finish.json').read_text())
    assert finish['status']=='PASS_COMPLETE_SAMPLED_MODEL_CACHE' and finish['L1_bytes_per_SM']==32768
    assert finish['input_scope']=='legacy_cross_layer_expansion' and not finish['full_native_address_coverage']
    run('synthetic-r4-reject',expanded+['--hw-config',config,'--output',out/'synthetic-r4-reject'],False)
    assert not (out/'synthetic-r4-reject').exists()
    rebound=out/'rebound-fixture'
    module.variant_fixture(rebound,lambda profile:profile.update(model=dict(layer_rebinding='synthetic translation')))
    explicit=[sys.executable,'-B',script,'--binary',a.binary.resolve(),'--profile-index',rebound/'profiles.index.jsonl',
              '--app-config',rebound/'app.config','--issue-config',rebound/'issue.config','--hw-config',config,
              '--r4-context',rebound/'context.json','--output',out/'explicit-rebound']
    run('explicit-rebound',explicit,False)
    assert json.loads((out/'explicit-rebound/finish.json').read_text())['status']=='FAIL'
    assert 'modeled layer rebinding' in (out/'explicit-rebound/stderr.log').read_text()
    # A synthetic FIFO blocks the real binary before reading a profile frame.
    # Cancel only this owned test process; verify wrapper cleanup reaps the child.
    blocked=out/'blocked-fixture';ctx=module.make_fixture(blocked)
    fifo=blocked/'profile.fifo';os.mkfifo(fifo)
    rows=[json.loads(line) for line in (blocked/'profiles.index.jsonl').read_text().splitlines()]
    rows[0]['path']=str(fifo)
    (blocked/'profiles.index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    ctx['profile_index_sha256']=module.sha(blocked/'profiles.index.jsonl');(blocked/'context.json').write_text(json.dumps(ctx))
    cancel=[sys.executable,'-B',script,'--binary',a.binary.resolve(),'--profile-index',blocked/'profiles.index.jsonl',
            '--app-config',blocked/'app.config','--issue-config',blocked/'issue.config','--hw-config',config,
            '--r4-context',blocked/'context.json','--output',out/'cancel']
    with (out/'cancel.log').open('x') as log:
        controller=subprocess.Popen(list(map(str,cancel)),stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(500):
                if (out/'cancel/process.json').exists():break
                if controller.poll() is not None:raise AssertionError('controller exited before cancellation')
                time.sleep(.01)
            else:raise AssertionError('controller did not start owned child')
            pid=json.loads((out/'cancel/process.json').read_text())['replay_pid']
            controller.send_signal(signal.SIGTERM)
            assert controller.wait(timeout=15)!=0
            assert json.loads((out/'cancel/finish.json').read_text())['status']=='FAIL'
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise AssertionError('cancelled replay child survived')
            checks.append('SIGTERM-reaps-owned-replay')
        finally:
            if controller.poll() is None:controller.kill();controller.wait()
    result=dict(status='PASS_REAL_BINARY_REPLAY_ENTRY',checks=checks,hardware_accuracy_accepted=False)
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
