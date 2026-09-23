#!/usr/bin/env python3
"""Complete frozen native workflow: tensor descriptors, module calls and kernel census.

This CPU-prepared host is not a dynamic instruction/address sampler. Activation,
weight and KV payloads are not copied. The same small controls as native_host
are read after the complete measured workflow, outside the profiler context.
"""
import argparse
import bisect
import ctypes
import dataclasses
import heapq
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import time
import weakref

import matrix_common as reference
import matrix_workload as workload


class TensorRegistry:
    def __init__(self, torch):
        self.torch = torch
        self.roots, self.by_storage, self.refs = [], {}, {}
        self.clock = 0

    def describe(self, tensor, label):
        self.clock += 1
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage._cdata, storage.data_ptr(), storage.nbytes())
        root = self.by_storage.get(key)
        if root is not None and not any(ref() is not None for ref in self.refs[root['id']]):
            root = None
        if root is None:
            root = dict(id='storage_%06d' % len(self.roots), first_label=label,
                        device=str(tensor.device), base_address=storage.data_ptr(),
                        storage_nbytes=storage.nbytes(), storage_identity=storage._cdata,
                        first_observation=self.clock, last_observation=self.clock)
            self.roots.append(root)
            self.by_storage[key] = root
            self.refs[root['id']] = []
        root['last_observation'] = self.clock
        refs = self.refs[root['id']]
        refs[:] = [ref for ref in refs if ref() is not None]
        if not any(ref() is tensor for ref in refs):
            refs.append(weakref.ref(tensor))
        return dict(root=root['id'], label=label, data_address=tensor.data_ptr(),
                    storage_offset_elements=tensor.storage_offset(),
                    storage_offset_bytes=tensor.storage_offset() * tensor.element_size(),
                    shape=list(tensor.shape), stride_elements=list(tensor.stride()),
                    stride_bytes=[x * tensor.element_size() for x in tensor.stride()],
                    dtype=str(tensor.dtype), element_size=tensor.element_size(),
                    logical_nbytes=tensor.numel() * tensor.element_size(), device=str(tensor.device))

    def walk(self, value, label):
        if isinstance(value, self.torch.Tensor):
            return [self.describe(value, label)]
        if isinstance(value, dict):
            return [row for key, child in value.items() for row in self.walk(child, label + '.' + str(key))]
        if isinstance(value, (tuple, list)):
            return [row for i, child in enumerate(value) for row in self.walk(child, label + '[%d]' % i)]
        return []


class Observer:
    def __init__(self, torch, runner, frozen):
        self.torch, self.runner, self.frozen = torch, runner, frozen
        self.registry = TensorRegistry(torch)
        self.events, self.active, self.handles, self.controls = [], [], [], []
        self.stage = None
        layers = list(runner.model.model.layers)
        if len(layers) != frozen['layers'] or len({id(x) for x in layers}) != len(layers):
            raise RuntimeError('Complete distinct decoder-layer modules required')
        self.layer_by_module = {id(module): i for i, module in enumerate(layers)}
        self.layer_counts = {phase: [0] * len(layers) for phase in frozen['phases']}
        self.native = None
        self.epoch = None
        if os.environ.get('SG_NVBIT_SCOPE_ABI') == '1':
            self.native = ctypes.CDLL(None)
            self.native.sg_nvbit_observer_set_scope.argtypes = [ctypes.c_uint64, ctypes.c_int64, ctypes.c_int32,
                                                                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
            self.native.sg_nvbit_observer_set_scope.restype = ctypes.c_int
            self.native.sg_nvbit_observer_clear_scope.argtypes = []
            self.native.sg_nvbit_observer_clear_scope.restype = ctypes.c_int
            for name in ('sg_nvbit_observer_begin_epoch', 'sg_nvbit_observer_end_epoch'):
                getattr(self.native, name).argtypes = [ctypes.c_uint64]
                getattr(self.native, name).restype = ctypes.c_int

    def scope(self):
        if self.native is None:
            return
        if self.stage is None:
            rc = self.native.sg_nvbit_observer_clear_scope()
        else:
            forward = self.frozen['phases'].index(self.stage)
            layer = next((event['layer'] for event, _, _ in reversed(self.active) if event['layer'] >= 0), -1)
            event = self.active[-1][0] if self.active else dict(call_id=10000000 + forward, module='<phase-global>')
            rc = self.native.sg_nvbit_observer_set_scope(event['call_id'], forward, layer,
                    self.stage.encode(), event['module'].encode(), b'measurement')
        if rc != 1:
            raise RuntimeError('Native scope marker rejected')

    def begin(self, phase, index):
        if self.active or self.stage is not None:
            raise RuntimeError('Unclosed module or phase')
        if self.native is not None:
            if self.native.sg_nvbit_observer_begin_epoch(index + 1) != 1:
                raise RuntimeError('Native epoch begin rejected')
            self.epoch = index + 1
        self.stage = phase
        self.scope()

    def end(self):
        if self.active:
            raise RuntimeError('Unclosed module hook stack')
        self.stage = None
        self.scope()
        if self.native is not None and self.epoch is not None:
            if self.native.sg_nvbit_observer_end_epoch(self.epoch) != 1:
                raise RuntimeError('Native epoch end rejected')
            self.epoch = None

    def install(self):
        for name, module in self.runner.model.named_modules():
            def pre(mod, args, kwargs, name=name):
                if self.stage is None:
                    return
                own_layer = self.layer_by_module.get(id(mod), -1)
                inherited = next((e['layer'] for e, _, _ in reversed(self.active) if e['layer'] >= 0), -1)
                event = dict(call_id=len(self.events), phase=self.stage, module=name or '<model>',
                             module_class=type(mod).__module__ + '.' + type(mod).__name__,
                             parent_call_id=self.active[-1][0]['call_id'] if self.active else None,
                             layer=own_layer if own_layer >= 0 else inherited,
                             is_decoder_layer=own_layer >= 0,
                             inputs=self.registry.walk(args, 'args') + self.registry.walk(kwargs, 'kwargs'),
                             completed=False)
                if own_layer >= 0:
                    self.layer_counts[self.stage][own_layer] += 1
                self.events.append(event)
                marker = self.torch.profiler.record_function('tilegraph/%s/%d/%s' % (self.stage, event['call_id'], name or '<model>'))
                marker.__enter__()
                self.active.append((event, marker, id(mod)))
                self.scope()

            def post(mod, args, kwargs, output):
                if self.stage is None:
                    return
                if not self.active or self.active[-1][2] != id(mod):
                    raise RuntimeError('Module hook nesting changed')
                event, marker, _ = self.active.pop()
                event['outputs'] = self.registry.walk(output, 'output')
                event['completed'] = True
                marker.__exit__(None, None, None)
                self.scope()

            self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(module.register_forward_hook(post, with_kwargs=True, always_call=True))
        self.original_forward = self.runner.forward

        def forward(fb, *args, **kwargs):
            if self.stage is None:
                return self.original_forward(fb, *args, **kwargs)
            fields = ('input_ids', 'positions', 'seq_lens', 'req_pool_indices', 'out_cache_loc',
                      'extend_seq_lens', 'extend_prefix_lens', 'extend_start_loc')
            record = dict(phase=self.stage, seq_lens_sum=int(fb.seq_lens_sum),
                          input_ids_ref=fb.input_ids, positions_ref=fb.positions, cache_loc_ref=fb.out_cache_loc,
                          forward_tensor_descriptors=[row for key in fields for row in
                              self.registry.walk(getattr(fb, key, None), 'forward_batch.' + key)])
            result = self.original_forward(fb, *args, **kwargs)
            if bool(result[1]):
                raise RuntimeError('Unexpected CUDA Graph execution')
            self.controls.append(record)
            return result
        self.runner.forward = forward

    def finish(self):
        self.stage = None
        while self.active:
            _, marker, _ = self.active.pop()
            marker.__exit__(None, None, None)
        self.scope()
        if self.native is not None and self.epoch is not None:
            self.native.sg_nvbit_observer_end_epoch(self.epoch)
            self.epoch = None
        self.runner.forward = self.original_forward
        for handle in self.handles:
            handle.remove()


def kernel_census(events, phases):
    """Join GPU events to CPU launch correlations and enclosing phase markers.

    Preserve all raw metadata. This is a profiler association, not a native
    function/ABI/PC or dynamic-memory witness.
    """
    phase_events = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('phase/')]
    if sorted(e['name'][6:] for e in phase_events) != sorted(phases):
        raise RuntimeError('Exactly one profiler phase marker per contract phase required')
    phases_by_thread = {}
    for e in phase_events:
        phases_by_thread.setdefault((e['pid'], e['tid']), []).append(e)
    for rows in phases_by_thread.values():
        rows.sort(key=lambda x: x['ts'])
        if any(a['ts'] + a['dur'] > b['ts'] for a, b in zip(rows, rows[1:])):
            raise RuntimeError('Overlapping phase markers')
    modules = {}
    for e in events:
        if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('tilegraph/'):
            modules.setdefault((e['pid'], e['tid']), []).append(e)
    runtime = {}
    for e in events:
        if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in e.get('args', {}):
            runtime.setdefault(e['args']['correlation'], []).append(e)
    kernels = [e for e in events if e.get('cat') == 'kernel']
    if not kernels:
        raise RuntimeError('Profiler emitted no native kernels')
    requests = {}
    for i, kernel in enumerate(kernels):
        candidates = runtime.get(kernel.get('args', {}).get('correlation'), [])
        if len(candidates) != 1:
            raise RuntimeError('Missing or ambiguous kernel/runtime correlation')
        e = candidates[0]
        requests.setdefault((e['pid'], e['tid']), []).append((e['ts'], i, e))
    rows = [None] * len(kernels)
    counts = {phase: 0 for phase in phases}
    for thread, queries in requests.items():
        pp = phases_by_thread.get(thread, [])
        starts = [e['ts'] for e in pp]
        mm = sorted(modules.get(thread, []), key=lambda e: e['ts'])
        active, pos = [], 0
        for ts, index, launch in sorted(queries):
            k = bisect.bisect_right(starts, ts) - 1
            if k < 0 or ts >= pp[k]['ts'] + pp[k]['dur']:
                raise RuntimeError('Kernel launch lies outside all measured phase markers')
            phase = pp[k]['name'][6:]
            while pos < len(mm) and mm[pos]['ts'] <= ts:
                e = mm[pos]
                heapq.heappush(active, (-e['ts'], pos, e['ts'] + e['dur'], e['name']))
                pos += 1
            while active and active[0][2] <= ts:
                heapq.heappop(active)
            marker = active[0][3] if active else None
            if marker is not None and marker.split('/', 3)[1] != phase:
                raise RuntimeError('Profiler module/phase association differs')
            kernel = kernels[index]
            rows[index] = dict(kernel_event_index=index, phase=phase, name=kernel['name'],
                               runtime_correlation=kernel['args']['correlation'],
                               enclosing_module_marker=marker,
                               module_call_id=int(marker.split('/', 3)[2]) if marker else None,
                               grid=kernel.get('args', {}).get('grid'), block=kernel.get('args', {}).get('block'),
                               native_code_sha256=None, dynamic_memory_witness=False)
            counts[phase] += 1
    if any(counts[p] == 0 for p in phases):
        raise RuntimeError('A measured phase has no native kernel events')
    return dict(kernel_count=len(kernels), phase_kernel_counts=counts, kernels=rows,
                association='torch.profiler correlation + same CPU thread enclosing markers; not NVBit native launch identity')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--keep-chrome-profile', action='store_true')
    workload.add_arguments(p)
    a = p.parse_args()
    frozen = workload.from_args(a)
    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output / ('process-%d' % os.getpid())
    out.mkdir(exist_ok=False)
    artifacts = out / 'artifacts'
    artifacts.mkdir()
    parent = os.getppid()
    if parent <= 1 or ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        raise RuntimeError('Owned parent-death guard unavailable')
    import torch
    import sglang
    import sglang.bench_one_batch as bo
    packages = workload.check_packages()
    source_root = Path(sglang.__file__).resolve().parent
    source_before = reference.native_source_inventory(source_root)
    workload.check_native_sources(source_before)
    model_spec = workload.spec()['models'][frozen['model_key']]
    if reference.sha256(Path(frozen['model']) / 'config.json') != model_spec['config_sha256']:
        raise RuntimeError('Checkpoint config changed')
    args = bo.ServerArgs(model_path=frozen['model'], dtype='bfloat16', load_format='safetensors',
        device='cuda', tp_size=1, pp_size=1, attention_backend='flashinfer',
        disable_cuda_graph=True, cuda_graph_max_bs=1, enable_torch_compile=False,
        disable_overlap_schedule=True, disable_radix_cache=True, mem_fraction_static=0.90,
        max_total_tokens=frozen['max_total_tokens'], max_running_requests=1, random_seed=0, cpu_offload_gb=0)
    bo._set_envs_and_config(args)
    runner, _ = bo.load_model(args, bo.PortArgs.init_new(args), 0)
    if runner.cuda_graph_runner is not None or type(runner.model).__name__ != frozen['model_class']:
        raise RuntimeError('Native eager model identity changed')
    if runner.model_config.hf_config.num_hidden_layers != frozen['layers'] or runner.model_config.hf_config.torch_dtype != torch.bfloat16:
        raise RuntimeError('Complete original BF16 checkpoint layers required')
    fixed = [torch.tensor([token], dtype=torch.int64, device=runner.device) for token in frozen['decode_input_ids']]
    with torch.no_grad():
        runner.req_to_token_pool.clear()
        runner.token_to_kv_pool_allocator.clear()
        _, _, batch = bo.extend(reference.make_request(bo, frozen), runner)
        for token in fixed:
            bo.decode(token, batch, runner)
        torch.cuda.synchronize()
        runner.req_to_token_pool.clear()
        runner.token_to_kv_pool_allocator.clear()
    observer = Observer(torch, runner, frozen)
    observer.install()
    stages = []
    try:
        parameters = [observer.registry.describe(t, 'parameter.' + name) for name, t in runner.model.named_parameters(remove_duplicate=False)]
        buffers = [observer.registry.describe(t, 'buffer.' + name) for name, t in runner.model.named_buffers(remove_duplicate=False)]
        kv = runner.token_to_kv_pool
        kv_buffers = []
        for kind in ('k_buffer', 'v_buffer'):
            tensors = list(getattr(kv, kind))
            if len(tensors) != frozen['layers']:
                raise RuntimeError('Complete per-layer KV buffer inventory required')
            kv_buffers += [observer.registry.describe(t, 'kv.%s.%d' % (kind, layer)) for layer, t in enumerate(tensors)]
        page_table = observer.registry.describe(runner.req_to_token_pool.req_to_token, 'req_to_token')
        with torch.no_grad(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            batch = None
            for i, phase in enumerate(frozen['phases']):
                torch.cuda.synchronize()
                observer.begin(phase, i)
                start = time.monotonic()
                with torch.profiler.record_function('phase/' + phase):
                    torch.cuda.nvtx.range_push('phase/' + phase)
                    try:
                        if i == 0:
                            predicted, logits, batch = bo.extend(reference.make_request(bo, frozen), runner)
                        else:
                            predicted, logits = bo.decode(fixed[i - 1], batch, runner)
                    finally:
                        torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                observer.end()
                stages.append(dict(phase=phase, forward_index=i, instrumented_elapsed_seconds=time.monotonic() - start,
                                   logits=observer.registry.describe(logits, phase + '.logits'), predictions_fed_back=False))
        if len(observer.controls) != len(frozen['phases']) or any(
                counts != [1] * frozen['layers'] for counts in observer.layer_counts.values()):
            raise RuntimeError('Complete forward/layer coverage did not close')
        if any(not event['completed'] for event in observer.events):
            raise RuntimeError('Incomplete module callbacks')
        for i, control in enumerate(observer.controls):
            if control['phase'] != frozen['phases'][i]:
                raise RuntimeError('Forward order changed')
            values = dict(phase=control['phase'], seq_lens_sum=control['seq_lens_sum'],
                          input_ids=control['input_ids_ref'].cpu().tolist(),
                          positions=control['positions_ref'].cpu().tolist(),
                          out_cache_loc=control['cache_loc_ref'].cpu().tolist())
            workload.validate_controls(frozen, i, values)
            stages[i].update(actual_controls=values, forward_tensor_descriptors=control['forward_tensor_descriptors'])
        raw = artifacts / 'torch_launch_metadata.chrome.json'
        profiler.export_chrome_trace(str(raw))
        raw_events = json.loads(raw.read_text()).get('traceEvents', [])
        events = [e for e in raw_events if e.get('cat') in (
            'kernel', 'gpu_memcpy', 'gpu_memset', 'cuda_runtime', 'cuda_driver', 'user_annotation') or e.get('ph') in ('s', 'f')]
        census = kernel_census(events, frozen['phases'])
        for row in census['kernels']:
            cid = row['module_call_id']
            if cid is not None and (not 0 <= cid < len(observer.events) or observer.events[cid]['phase'] != row['phase']):
                raise RuntimeError('Kernel/module ledger differs')
        reference.write_json(artifacts / 'kernel_launches.json', dict(
            source='torch.profiler CPU/CUDA metadata', profile_memory=False,
            memory_instruction_trace=False, events=events, **census))
        if not a.keep_chrome_profile:
            raw.unlink()
    finally:
        observer.finish()
    if source_before != reference.native_source_inventory(source_root):
        raise RuntimeError('Native source changed')
    process = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]))
    for name, value in [('tensor_roots.json', observer.registry.roots), ('module_calls.json', observer.events),
                        ('phases.json', stages)]:
        reference.write_json(artifacts / name, value)
    coverage = dict(full_checkpoint_layers=frozen['layers'], phase_layer_calls=observer.layer_counts,
        parameter_and_module_boundary_metadata=True, named_parameter_aliases_preserved=True,
        KV_buffers_complete_for_exposed_k_v_lists=True, backend_private_tensors_complete=False,
        native_scope_abi_enabled=observer.native is not None, kernel_launch_metadata=True,
        instruction_memory_trace_collected=False, tilegraph_complete=False, full_native_trace_admitted=False,
        activation_weight_KV_payloads_copied=False, small_control_values_read_after_measured_workflow=True,
        lifetime_tracking='Python tensor weakrefs and observation generations, not complete allocator/free or backend-private lifetime proof',
        address_meaning='Observed process virtual tensor pointers, not physical addresses or lane memory accesses',
        timing='Instrumented diagnostic only: profiler/module hooks alter overhead; not natural CUDA event or NCU duration')
    manifest = dict(schema='SGLANG_FULL_MATRIX_METADATA_V1', status='COMPLETE_METADATA_ONLY',
        process=process, guarded_parent_pid=parent, parent_death_signal='SIGKILL', input_contract=frozen,
        contract_sha256=reference.sha256(Path(workload.__file__).parent / 'contract.json'),
        driver_sha256=reference.sha256(__file__), workload_sha256=reference.sha256(workload.__file__),
        common_sha256=reference.sha256(reference.__file__), native_source_files=source_before,
        native_source_unchanged=True, packages=packages, platform=platform.platform(),
        server_args=dataclasses.asdict(args), hf_config=runner.model_config.hf_config.to_dict(),
        model_class=type(runner.model).__module__ + '.' + type(runner.model).__name__,
        attention_backend_class=type(runner.attn_backend).__module__ + '.' + type(runner.attn_backend).__name__,
        gpu=dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability())),
        checkpoint_identity_expected=model_spec, weight_content_hashes_complete=False,
        weight_hash_note='This host verifies config SHA; complete checkpoint content pins are supplied by frozen contract and must be independently verified by controller.',
        parameters=parameters, module_buffers=buffers, coverage=coverage,
        kv_pool=dict(class_name=type(kv).__module__ + '.' + type(kv).__name__, page_size=kv.page_size,
                     size=kv.size, dtype=str(kv.dtype), buffers=kv_buffers, page_table=page_table),
        kernel_count=census['kernel_count'], phase_kernel_counts=census['phase_kernel_counts'],
        warmup_runs=1, predictions_fed_back=False, numerical_acceptance='NOT_ASSESSED', native_cuda_kernels_modified=False)
    reference.write_json(artifacts / 'manifest.json', manifest)
    files = [dict(path=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=reference.sha256(path))
             for path in sorted(artifacts.iterdir()) if path.is_file()]
    reference.write_json(out / 'finish.json', dict(schema='SGLANG_FULL_MATRIX_METADATA_FINISH_V1',
        status='PASS_NATIVE_METADATA_ONLY', process=process, input_contract=frozen,
        input_contract_sha256=frozen['sha256'], kernel_count=census['kernel_count'],
        phases=frozen['phases'], phase_kernel_counts=census['phase_kernel_counts'],
        module_calls=len(observer.events), tensor_roots=len(observer.registry.roots),
        artifacts=files, coverage=coverage, driver_sha256=reference.sha256(__file__)))
    print(json.dumps(dict(status='PASS_NATIVE_METADATA_ONLY', output=str(out), kernel_count=census['kernel_count'])), flush=True)


if __name__ == '__main__':
    main()
