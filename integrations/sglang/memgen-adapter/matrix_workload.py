"""24 explicit real-checkpoint cases. No extrapolation or GPU imports."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def spec():
    value = json.loads((HERE / 'contract.json').read_text())
    if value['source_status'] != 'PINNED_REAL_CHECKPOINTS_AND_NATIVE_SOURCES':
        raise RuntimeError('Unsealed model/source identities')
    return value


def add_arguments(parser):
    parser.add_argument('--model', choices=('qwen25_1p5b', 'llama3_8b'), required=True)
    parser.add_argument('--prefill-length', type=int, choices=(128, 256, 512, 1024), required=True)
    parser.add_argument('--decode-steps', type=int, choices=(32, 64, 128), required=True)


def contract(model, prefill_length, decode_steps):
    s = spec()
    if model not in s['models'] or prefill_length not in s['prefills'] or decode_steps not in s['decodes']:
        raise ValueError('Case outside the requested 24-case matrix')
    m = s['models'][model]
    if prefill_length + decode_steps + 1 > s['max_total_tokens']:
        raise ValueError('Insufficient declared KV capacity')
    value = dict(schema='SGLANG_FIXED_PD_INPUT_V2', case_id=f'{model}-p{prefill_length}-d{decode_steps}',
                 batch_size=1, prefill_length=prefill_length, decode_steps=decode_steps,
                 prompt_ids=list(range(1000, 1000 + prefill_length)),
                 decode_input_ids=[(944, 291)[i % 2] for i in range(decode_steps)],
                 phases=['Prefill'] + [f'Decode{i}' for i in range(1, decode_steps + 1)],
                 max_total_tokens=s['max_total_tokens'], output_feedback=False,
                 input_source='Frozen arithmetic prompt IDs; alternating 944/291 decode IDs; no tokenizer/chat template',
                 model=m['path'], model_key=model, model_class=m['class'], layers=m['layers'],
                 dtype='bfloat16', tp_size=1, pp_size=1, attention_backend='flashinfer',
                 native_eager=True, warmup_runs=1, mem_fraction_static=0.90, cuda_graph=False,
                 torch_compile=False, disable_overlap_schedule=True, disable_radix_cache=True,
                 sampling_retained=True, numerical_acceptance='NOT_ASSESSED',
                 weights='complete original checkpoint, no dummy layers/offload/quantization')
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return value


def from_args(args):
    return contract(args.model, args.prefill_length, args.decode_steps)


def validate_controls(c, index, values):
    if not 0 <= index <= c['decode_steps']:
        raise ValueError('Phase outside contract')
    p = c['prefill_length']
    expected = dict(input_ids=c['prompt_ids'] if index == 0 else [c['decode_input_ids'][index - 1]],
                    positions=list(range(p)) if index == 0 else [p + index - 1],
                    seq_lens_sum=p + index,
                    out_cache_loc=list(range(1, p + 1)) if index == 0 else [p + index])
    for name, value in expected.items():
        if values.get(name) != value:
            raise RuntimeError(f'Actual {name} differs at {c["phases"][index]}')


def check_packages():
    import importlib.metadata
    s = spec()
    actual = {name: importlib.metadata.version(name) for name in s['packages']}
    if actual != s['packages']:
        raise RuntimeError('Installed package identity changed: ' + repr(actual))
    return actual


def check_native_sources(rows):
    actual = {r['relative']: r['sha256'] for r in rows}
    expected = {r['relative']: r['sha256'] for r in spec()['native_sources']}
    if len(actual) != len(rows) or actual != expected:
        raise RuntimeError('Native source inventory changed')
