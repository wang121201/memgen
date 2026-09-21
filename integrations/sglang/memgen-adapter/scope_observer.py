"""Unchanged tensor/module scope classes from existing SGLang sparse driver."""
import ctypes, os, weakref

class TensorRegistry:
    def __init__(self, torch):
        self.torch = torch
        self.roots = []
        self.by_storage = {}
        self.tensor_refs = {}
        self.clock = 0

    def describe(self, tensor, label):
        self.clock += 1
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage._cdata, storage.data_ptr(), storage.nbytes())
        root = self.by_storage.get(key)
        # Do not retain tensors: doing so would change native allocator reuse.
        # A generation is a best-effort Python-visible lifetime, not a claim
        # that backend-private allocations have been completely enumerated.
        if root is not None and not any(ref() is not None for ref in self.tensor_refs[root["id"]]):
            root = None
        if root is None:
            root = {
                "id": "storage_%06d" % len(self.roots),
                "first_label": label,
                "device": str(tensor.device),
                "base_address": storage.data_ptr(),
                "storage_nbytes": storage.nbytes(),
                "storage_identity": storage._cdata,
                "first_observation": self.clock,
                "last_observation": self.clock,
            }
            self.roots.append(root)
            self.by_storage[key] = root
            self.tensor_refs[root["id"]] = []
        root["last_observation"] = self.clock
        refs = self.tensor_refs[root["id"]]
        if not any(ref() is tensor for ref in refs):
            refs[:] = [ref for ref in refs if ref() is not None]
            refs.append(weakref.ref(tensor))
        return {
            "root": root["id"], "label": label,
            "data_address": tensor.data_ptr(),
            "storage_offset_elements": tensor.storage_offset(),
            "storage_offset_bytes": tensor.storage_offset() * tensor.element_size(),
            "shape": list(tensor.shape), "stride_elements": list(tensor.stride()),
            "stride_bytes": [n * tensor.element_size() for n in tensor.stride()],
            "dtype": str(tensor.dtype), "element_size": tensor.element_size(),
            "logical_nbytes": tensor.numel() * tensor.element_size(),
            "device": str(tensor.device),
        }

    def walk(self, value, label):
        if isinstance(value, self.torch.Tensor):
            return [self.describe(value, label)]
        if isinstance(value, dict):
            return [r for k, v in value.items() for r in self.walk(v, label + "." + str(k))]
        if isinstance(value, (tuple, list)):
            return [r for i, v in enumerate(value) for r in self.walk(v, label + "[%d]" % i)]
        return []


class Observer:
    def __init__(self, torch, runner):
        self.torch, self.runner = torch, runner
        self.registry = TensorRegistry(torch)
        self.stage = None
        self.role = "measurement"
        self.events = []
        self.active = []
        self.handles = []
        self.last_forward_batch = None
        self.last_graph_used = None
        self.forward_descriptors = []
        self.native = None
        self.native_epoch = None
        if os.environ.get('SG_NVBIT_SCOPE_ABI') == '1':
            self.native = ctypes.CDLL(None)
            self.native.sg_nvbit_observer_set_scope.argtypes = [ctypes.c_uint64, ctypes.c_int64, ctypes.c_int32,
                                                                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
            self.native.sg_nvbit_observer_set_scope.restype = ctypes.c_int
            self.native.sg_nvbit_observer_clear_scope.argtypes = []
            self.native.sg_nvbit_observer_clear_scope.restype = ctypes.c_int
            for name in ('sg_nvbit_observer_begin_epoch', 'sg_nvbit_observer_end_epoch'):
                func = getattr(self.native, name)
                func.argtypes = [ctypes.c_uint64]
                func.restype = ctypes.c_int

    def begin_native_epoch(self, number):
        if self.native is not None:
            if self.native.sg_nvbit_observer_begin_epoch(number) != 1:
                raise RuntimeError('Native observer epoch begin failed')
            self.native_epoch = number

    def end_native_epoch(self):
        if self.native is not None and self.native_epoch is not None:
            if self.native.sg_nvbit_observer_end_epoch(self.native_epoch) != 1:
                raise RuntimeError('Native observer epoch end failed')
            self.native_epoch = None

    def update_native_scope(self):
        if self.native is None:
            return
        if self.stage is None:
            if self.native.sg_nvbit_observer_clear_scope() != 1:
                raise RuntimeError('Native observer scope clear failed')
            return
        forward_id = 0 if self.stage == 'Prefill' else int(self.stage.removeprefix('Decode'))
        layer = -1
        for event, _ in reversed(self.active):
            if event['module_class'].endswith(('.LlamaDecoderLayer', '.Qwen2DecoderLayer')):
                layer = int(event['module'].split('.layers.')[1].split('.')[0])
                break
        if self.active:
            event = self.active[-1][0]
            call_id, scope = event['call_id'], event['module']
        else:
            call_id, scope = 10000000 + forward_id, '<phase-global>'
        # A shared module (notably RoPE) keeps its canonical name, while
        # layer ownership comes from the active decoder invocation above.
        rc = self.native.sg_nvbit_observer_set_scope(call_id, forward_id, layer,
                self.stage.encode(), scope.encode(), self.role.encode())
        if rc != 1:
            raise RuntimeError('Native launch observer rejected scope marker')

    def install(self):
        for name, module in self.runner.model.named_modules():
            def pre(mod, args, kwargs, name=name):
                if self.stage is None:
                    return
                event = {
                    "call_id": len(self.events), "phase": self.stage, "role": self.role,
                    "module": name or "<model>",
                    "module_class": type(mod).__module__ + "." + type(mod).__name__,
                    "parent_call_id": self.active[-1][0]["call_id"] if self.active else None,
                    "inputs": self.registry.walk(args, "args") + self.registry.walk(kwargs, "kwargs"),
                }
                self.events.append(event)
                marker = self.torch.profiler.record_function(
                    "tilegraph/%s/%d/%s" % (self.stage, event["call_id"], name or "<model>")
                )
                marker.__enter__()
                self.active.append((event, marker))
                self.update_native_scope()

            def post(mod, args, kwargs, output):
                if self.stage is None:
                    return
                event, marker = self.active.pop()
                event["outputs"] = self.registry.walk(output, "output")
                marker.__exit__(None, None, None)
                self.update_native_scope()

            self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(module.register_forward_hook(post, with_kwargs=True, always_call=True))
        original = self.runner.forward

        def forward(forward_batch, *args, **kwargs):
            if self.stage is not None:
                self.last_forward_batch = forward_batch
                self.forward_descriptors = self.describe_control(forward_batch)
            output = original(forward_batch, *args, **kwargs)
            if self.stage is not None:
                self.last_graph_used = bool(output[1])
                if self.last_graph_used:
                    raise RuntimeError("Eager execution contract violated: CUDA Graph was used")
            return output

        self.runner.forward = forward
        self.original_forward = original

    def describe_control(self, batch):
        fields = ("input_ids", "positions", "seq_lens", "req_pool_indices", "out_cache_loc",
                  "extend_seq_lens", "extend_prefix_lens", "extend_start_loc")
        return [r for key in fields for r in self.registry.walk(getattr(batch, key, None), "forward_batch." + key)]

    def finish(self):
        self.stage = None
        self.update_native_scope()
        self.end_native_epoch()
        self.runner.forward = self.original_forward
        for handle in self.handles:
            handle.remove()

