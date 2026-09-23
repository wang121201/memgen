#!/usr/bin/env python3
"""SGLang matrix real-view metadata for a declared tilegraph, no NVBit required."""
import argparse
import ctypes
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
import matrix_common as common
import matrix_workload as workload
from metadata_host import Observer as BaseObserver


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


class Observer(BaseObserver):
    """Keep tensor pointers/strides only; additional small controls are explicit."""
    def install(self):
        super().install()
        # Base hooks are registered first, so a pre hook can attach to the
        # current event. Rotary may be a shared module named layer0 for all
        # layers; event.layer inherited from the active decoder is authoritative.
        for name, module in self.runner.model.named_modules():
            if name.endswith('.rotary_emb'):
                def rotary_post(mod, args, kwargs, output):
                    if self.stage is None:
                        return
                    event = self.events[-1]
                    need(event['module'].endswith('.rotary_emb') and event['completed'], 'rotary leaf ordering changed')
                    event['rotary'] = dict(
                        cache=self.registry.describe(mod.cos_sin_cache, 'rotary.cos_sin_cache'),
                        head_size=int(mod.head_size), rotary_dim=int(mod.rotary_dim),
                        is_neox_style=bool(mod.is_neox_style))
                self.handles.append(module.register_forward_hook(rotary_post, with_kwargs=True))
        processor = self.runner.model.logits_processor
        original = processor._get_logits
        code = original.__func__.__code__
        self.head_had_instance = '_get_logits' in vars(processor)
        self.head_previous = vars(processor).get('_get_logits')

        def head(*args, **kwargs):
            if self.stage is None:
                return original(*args, **kwargs)
            need(self.active and self.active[-1][0]['module'] == 'logits_processor', 'head must be inside observed module')
            event = self.active[-1][0]
            original_mm = self.torch.matmul
            owner = threading.get_ident()
            records = []

            def matmul(*a, **kw):
                selected = threading.get_ident() == owner and sys._getframe(1).f_code is code
                output = original_mm(*a, **kw)
                if selected:
                    records.append(dict(inputs=self.registry.walk(a, 'head.mm.args') + self.registry.walk(kw, 'head.mm.kwargs'),
                                        outputs=self.registry.walk(output, 'head.mm.output')))
                return output

            self.torch.matmul = matmul
            try:
                result = original(*args, **kwargs)
                need(len(records) == 1, 'exactly one original head matmul expected')
                event['head_linear_observation'] = dict(
                    schema='SGLANG_MATRIX_HEAD_LINEAR_V1', matmul=records[0],
                    get_logits_return=self.registry.walk(result, 'head.get_logits.return'),
                    original_method_and_return_unchanged=True, tensor_values_copied=False)
                return result
            finally:
                self.torch.matmul = original_mm

        processor._get_logits = head
        base_forward = self.runner.forward

        def forward(fb, *args, **kwargs):
            result = base_forward(fb, *args, **kwargs)
            if self.stage is not None:
                # Values can mutate at the next decode; read immediately in
                # this metadata-only run, never claim the time is NCU time.
                values = {k: getattr(fb, k).detach().cpu().tolist() for k in
                          ('input_ids', 'positions', 'out_cache_loc', 'seq_lens', 'req_pool_indices')}
                req = values['req_pool_indices'][0]
                count = int(fb.seq_lens_sum)
                slots = self.runner.req_to_token_pool.req_to_token[req, :count].detach().cpu().tolist()
                record = self.controls[-1]
                record['actual_controls'] = dict(values, seq_lens_sum=count,
                    kv_read_slots_in_logical_order=slots, kv_write_slots=values['out_cache_loc'])
                # Base retains immutable small tensors. Drop references now;
                # all required values have been copied, avoiding lifetime drift.
                for k in ('input_ids_ref', 'positions_ref', 'cache_loc_ref'):
                    record.pop(k, None)
            return result
        self.runner.forward = forward

    def finish(self):
        processor = self.runner.model.logits_processor
        if self.head_had_instance:
            processor._get_logits = self.head_previous
        else:
            delattr(processor, '_get_logits')
        super().finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workload.add_arguments(parser)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    need(not os.environ.get('LD_PRELOAD') and not os.environ.get('SG_NVBIT_SCOPE_ABI'), 'metadata route must not preload a tracer')
    frozen = workload.from_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / ('process-%d' % os.getpid())
    out.mkdir(exist_ok=False)
    artifacts = out / 'artifacts'
    artifacts.mkdir()
    parent = os.getppid()
    need(parent > 1 and ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) == 0 and os.getppid() == parent,
         'owned parent-death guard unavailable')
    import torch
    import sglang
    import sglang.bench_one_batch as bo
    packages = workload.check_packages()
    source = Path(sglang.__file__).resolve().parent
    before = common.native_source_inventory(source)
    workload.check_native_sources(before)
    need(common.sha256(Path(frozen['model']) / 'config.json') == workload.spec()['models'][args.model]['config_sha256'],
         'checkpoint config changed')
    server_args = bo.ServerArgs(model_path=frozen['model'], dtype='bfloat16', load_format='safetensors', device='cuda',
        tp_size=1, pp_size=1, attention_backend='flashinfer', disable_cuda_graph=True, cuda_graph_max_bs=1,
        enable_torch_compile=False, disable_overlap_schedule=True, disable_radix_cache=True,
        mem_fraction_static=0.90, max_total_tokens=frozen['max_total_tokens'], max_running_requests=1,
        random_seed=0, cpu_offload_gb=0)
    bo._set_envs_and_config(server_args)
    runner, _ = bo.load_model(server_args, bo.PortArgs.init_new(server_args), 0)
    need(runner.cuda_graph_runner is None and type(runner.model).__name__ == frozen['model_class'], 'native eager identity changed')
    hf = runner.model_config.hf_config
    need(hf.num_hidden_layers == frozen['layers'] and hf.torch_dtype == torch.bfloat16, 'full BF16 checkpoint required')
    fixed = [torch.tensor([token], dtype=torch.int64, device=runner.device) for token in frozen['decode_input_ids']]
    labels = [stage + '/' + phase for stage in ('Warmup', 'Measured') for phase in frozen['phases']]
    observation_contract = dict(frozen, phases=labels)
    observer = Observer(torch, runner, observation_contract)
    observer.install()
    parameters = [observer.registry.describe(t, 'parameter.' + name) for name, t in runner.model.named_parameters(remove_duplicate=False)]
    kv = runner.token_to_kv_pool
    kv_buffers = [observer.registry.describe(t, 'kv.%s.%d' % (kind, layer))
                  for kind in ('k_buffer', 'v_buffer') for layer, t in enumerate(getattr(kv, kind))]
    need(len(kv_buffers) == 2 * frozen['layers'], 'complete per-layer KV pools required')
    table = observer.registry.describe(runner.req_to_token_pool.req_to_token, 'req_to_token')
    phases = []
    wall0, cpu0 = time.monotonic(), time.process_time()
    try:
        with torch.no_grad():
            for stage in ('Warmup', 'Measured'):
                runner.req_to_token_pool.clear()
                runner.token_to_kv_pool_allocator.clear()
                batch = None
                for index, phase in enumerate(frozen['phases']):
                    ordinal = len(phases)
                    torch.cuda.synchronize()
                    observer.begin(stage + '/' + phase, ordinal)
                    start = time.monotonic()
                    if index == 0:
                        _, logits, batch = bo.extend(common.make_request(bo, frozen), runner)
                    else:
                        _, logits = bo.decode(fixed[index - 1], batch, runner)
                    torch.cuda.synchronize()
                    observer.end()
                    control = observer.controls[-1]
                    values = control['actual_controls']
                    workload.validate_controls(frozen, index, values)
                    slots = values['kv_read_slots_in_logical_order']
                    need(len(slots) == frozen['prefill_length'] + index and len(set(slots)) == len(slots), 'KV slot history length/uniqueness')
                    need(values['kv_write_slots'] == slots[-len(values['input_ids']):], 'KV current-write suffix differs')
                    phases.append(dict(stage=stage, phase=phase, phase_ordinal=ordinal, forward_index=index,
                        actual_controls=values, forward_tensor_descriptors=control['forward_tensor_descriptors'],
                        logits=observer.registry.describe(logits, stage + '/' + phase + '.logits'),
                        instrumented_wall_seconds=time.monotonic() - start))
                    common.write_json(out / 'progress.json', dict(status='RUNNING_METADATA_ONLY', completed_phases=len(phases), total_phases=len(labels),
                        stage=stage, phase=phase, wall_seconds=time.monotonic()-wall0))
    finally:
        observer.finish()
    need(all(x['completed'] for x in observer.events), 'unclosed module calls')
    need(all(counts == [1] * frozen['layers'] for counts in observer.layer_counts.values()), 'complete real layer coverage required')
    need(before == common.native_source_inventory(source), 'SGLang native source changed')
    manifest = dict(schema='SGLANG_FULL_MATRIX_TILEGRAPH_METADATA_V2', status='PASS_REAL_VIEW_METADATA_NO_MEMORY_TRACE',
        input_contract=frozen, hf_config=hf.to_dict(), parameters=parameters,
        kv_pool=dict(page_size=kv.page_size, size=kv.size, buffers=kv_buffers, page_table=table),
        packages=packages, native_source_files=before, server_args=dataclasses.asdict(server_args),
        coverage=dict(full_real_checkpoint_layers=frozen['layers'], warmup_and_measured=True,
            metadata_real_views=True, head_internal_views=True, request_KV_slots_observed=True,
            native_instruction_trace=False, backend_private_workspaces=False),
        qualification='Declared operator tile model input. Metadata observation changes host overhead; no native instruction or hardware-accuracy claim.',
        excluded_from_declared_tilegraph=['planner/split-merge workspace','sampling and reset helpers','metadata D2H observation'],
        source_pins=[dict(path=str(p),sha256=common.sha256(p),bytes=p.stat().st_size) for p in
                     (Path(__file__), HERE/'vendor/metadata_host.py',HERE/'vendor/matrix_common.py',HERE/'vendor/matrix_workload.py',HERE/'vendor/contract.json')])
    for name, value in [('manifest.json',manifest),('phases.json',phases),('module_calls.json',observer.events),('tensor_roots.json',observer.registry.roots)]:
        common.write_json(artifacts / name, value)
    receipt = dict(status=manifest['status'], process=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])),
        input_contract=frozen, phases=len(phases), module_calls=len(observer.events),
        wall_seconds=time.monotonic()-wall0, cpu_seconds=time.process_time()-cpu0,
        files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=common.sha256(p)) for p in sorted(artifacts.iterdir())])
    common.write_json(out / 'finish.json', receipt)
    print(json.dumps(dict(status=receipt['status'],output=str(out),phases=len(phases))),flush=True)


if __name__ == '__main__':
    main()
