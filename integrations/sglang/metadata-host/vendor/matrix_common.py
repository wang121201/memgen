"""Small CPU helpers for the frozen native full-inference host."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def native_source_inventory(root):
    import matrix_workload
    return [dict(relative=row['relative'], path=str(root / row['relative']),
                 sha256=sha256(root / row['relative']))
            for row in matrix_workload.spec()['native_sources']]


def make_request(bo, frozen):
    ids = list(frozen['prompt_ids'])
    req = bo.Req(rid=0, origin_input_text='', origin_input_ids=ids,
                 sampling_params=bo.SamplingParams(temperature=0,
                                                   max_new_tokens=frozen['decode_steps'] + 1))
    req.prefix_indices = []
    req.fill_ids = req.origin_input_ids
    req.extend_input_len = len(ids)
    req.logprob_start_len = len(ids) - 1
    return [req]
