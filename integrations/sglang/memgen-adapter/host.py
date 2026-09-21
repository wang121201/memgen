#!/usr/bin/env python3
"""NCU-matched SGLang host for metadata census and selected CTA sampling.

The preloaded observer/sampler chooses collection. The inference program always
executes the original full model; this host never enables a full raw collector.
"""
import argparse
import ctypes
import dataclasses
import json
import os
from pathlib import Path
import signal
import time

import matrix_common as reference
import matrix_workload as workload
from scope_observer import Observer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    workload.add_arguments(p)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    frozen = workload.from_args(a)
    if os.environ.get('SG_NVBIT_SCOPE_ABI') != '1':
        raise RuntimeError('Explicit native launch observer or sparse sampler required')
    parent = os.getppid()
    if parent <= 1 or ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        raise RuntimeError('Owned parent-death guard unavailable')
    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output / ('process-%d' % os.getpid())
    out.mkdir()
    import torch
    import sglang
    import sglang.bench_one_batch as bo
    packages = workload.check_packages()
    native_root = Path(sglang.__file__).resolve().parent
    before = reference.native_source_inventory(native_root)
    workload.check_native_sources(before)
    if reference.sha256(Path(frozen['model'])/'config.json') != workload.spec()['models'][a.model]['config_sha256']:
        raise RuntimeError('Checkpoint configuration changed')
    server = bo.ServerArgs(model_path=frozen['model'], dtype='bfloat16', load_format='safetensors',
        device='cuda', tp_size=1, pp_size=1, attention_backend='flashinfer', disable_cuda_graph=True,
        cuda_graph_max_bs=1, enable_torch_compile=False, disable_overlap_schedule=True,
        disable_radix_cache=True, mem_fraction_static=.90, max_total_tokens=frozen['max_total_tokens'],
        max_running_requests=1, random_seed=0, cpu_offload_gb=0)
    bo._set_envs_and_config(server)
    runner, _ = bo.load_model(server, bo.PortArgs.init_new(server), 0)
    if runner.cuda_graph_runner is not None or type(runner.model).__name__ != frozen['model_class']:
        raise RuntimeError('Full native eager model required')
    if runner.model_config.hf_config.num_hidden_layers != frozen['layers'] or runner.model_config.hf_config.torch_dtype != torch.bfloat16:
        raise RuntimeError('Original BF16 layer contract changed')
    fixed = [torch.tensor([v], dtype=torch.int64, device=runner.device) for v in frozen['decode_input_ids']]
    observer = Observer(torch, runner)
    parameters = [observer.registry.describe(v, 'parameter.'+k) for k,v in runner.model.named_parameters(remove_duplicate=False)]
    kv = runner.token_to_kv_pool
    kv_buffers = [observer.registry.describe(v, 'kv.%s.%d'%(kind,i))
                  for kind in ('k_buffer','v_buffer') for i,v in enumerate(getattr(kv,kind))]
    page_table = observer.registry.describe(runner.req_to_token_pool.req_to_token, 'req_to_token')
    observer.install()
    controls = []
    phase_rows = []
    try:
        with torch.no_grad():
            for role_index,role in enumerate(('warmup','measurement')):
                observer.role = role
                runner.req_to_token_pool.clear()
                runner.token_to_kv_pool_allocator.clear()
                batch = None
                for i,phase in enumerate(frozen['phases']):
                    torch.cuda.synchronize()
                    epoch = 1 + role_index*len(frozen['phases']) + i
                    observer.begin_native_epoch(epoch)
                    observer.stage = phase
                    observer.update_native_scope()
                    started = time.monotonic()
                    try:
                        if i == 0:
                            predicted, logits, batch = bo.extend(reference.make_request(bo,frozen),runner)
                        else:
                            predicted, logits = bo.decode(fixed[i-1],batch,runner)
                        if observer.last_graph_used:
                            raise RuntimeError('Unexpected CUDA Graph')
                        fb = observer.last_forward_batch
                        if int(fb.seq_lens_sum) != frozen['prefill_length'] + i:
                            raise RuntimeError('Context length changed')
                        if role == 'measurement':
                            controls.append(dict(phase=phase,seq_lens_sum=int(fb.seq_lens_sum),
                                input_ids_ref=fb.input_ids,positions_ref=fb.positions,cache_loc_ref=fb.out_cache_loc))
                    finally:
                        torch.cuda.synchronize()
                        observer.stage = None
                        observer.update_native_scope()
                        observer.end_native_epoch()
                    row = dict(role=role,phase=phase,epoch_id=epoch,instrumented_seconds=time.monotonic()-started)
                    phase_rows.append(row)
                    with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    finally:
        observer.finish()
    values=[]
    for i,row in enumerate(controls):
        v=dict(phase=row['phase'],seq_lens_sum=row['seq_lens_sum'],input_ids=row['input_ids_ref'].cpu().tolist(),
               positions=row['positions_ref'].cpu().tolist(),out_cache_loc=row['cache_loc_ref'].cpu().tolist())
        workload.validate_controls(frozen,i,v)
        values.append(v)
    if len(values)!=len(frozen['phases']) or before!=reference.native_source_inventory(native_root):
        raise RuntimeError('Workflow/source closure failed')
    layer_calls={role:{phase:0 for phase in frozen['phases']} for role in ('warmup','measurement')}
    for e in observer.events:
        if e['module_class'].endswith(('.LlamaDecoderLayer','.Qwen2DecoderLayer')):
            layer_calls[e['role']][e['phase']]+=1
    if any(v!=frozen['layers'] for rows in layer_calls.values() for v in rows.values()):
        raise RuntimeError('Full original layer coverage changed')
    reference.write_json(out/'module_calls.json',observer.events)
    reference.write_json(out/'tensor_metadata.json',dict(storage_roots=observer.registry.roots,parameters=parameters,
        kv_buffers=kv_buffers,page_table=page_table,kv_page_size=kv.page_size,kv_capacity=kv.size))
    finish=dict(schema='SGLANG_MATRIX_SAMPLED_MEMGEN_HOST_V1',status='PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE',
        input_contract=frozen,server_args=dataclasses.asdict(server),packages=packages,source_identity=before,
        control_values=values,layer_calls=layer_calls,phase_rows=phase_rows,
        numerical_acceptance='NOT_ASSESSED',profiling_seconds_are_hardware_runtime=False,
        full_raw_collection_requested=False,sample_selection='provided by compiled native sampler plan',
        driver_sha256=reference.sha256(__file__),scope_sha256=reference.sha256(Path(__file__).with_name('scope_observer.py')))
    reference.write_json(out/'finish.json',finish)
    print(json.dumps({'status':finish['status'],'case_id':frozen['case_id']}),flush=True)


if __name__=='__main__':
    main()
