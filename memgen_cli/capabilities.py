"""Machine-readable capability boundary for memgen.

`memgen capabilities` answers, without running anything: which workload points
this snapshot declares, what each stage needs, what the pipeline can and cannot
be asked to produce, and what none of it may claim. The declared sets are read
from `contract.json`, so this file cannot drift from the enforcement.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import __version__
from ._repo import ADAPTER, ROOT, SGLANG

CAPABILITY_SCHEMA = {'name': 'memgen.capabilities', 'version': 1}


def _contract() -> dict:
    return json.loads((ADAPTER / 'contract.json').read_text())


def _gpu_pool() -> list[str]:
    found = set(re.findall(r'GPU-[0-9a-f-]{36}', (SGLANG / 'run_job.py').read_text()))
    return sorted(found)


def current_capabilities() -> dict[str, Any]:
    contract = _contract()
    admission = contract.get('basic_admission') or {}
    declared = {
        'scale_series': {
            'prefills': contract['prefills'],
            'decodes': contract['decodes'],
            'models': sorted(contract['models']),
            'cases': len(contract['prefills']) * len(contract['decodes']) * len(contract['models']),
        },
        'basic_admission': {
            'prefills': admission.get('prefills', []),
            'decodes': admission.get('decodes', []),
            'models': admission.get('models', []),
            'cases': (len(admission.get('prefills', [])) * len(admission.get('decodes', []))
                      * len(admission.get('models', []))),
        },
    }
    return {
        'schema': CAPABILITY_SCHEMA,
        'version': __version__,
        'hardware_accuracy_accepted': False,
        'admission_authority': 'validation/p32d2_branch_status.csv',
        'purpose': 'Sampled GPU memory-SASS to full-inference address stream to a functional '
                   'L1/L2 cache filter to aggregate DRAM counters, for comparison against NCU.',
        'declared': declared,
        'models': {key: {'display': row.get('display'), 'layers': row.get('layers'),
                         'dtype': contract['dtype']}
                   for key, row in contract['models'].items()},
        'stages': [
            {'name': 'census', 'device': 'gpu',
             'command': 'host.py under LD_PRELOAD=observer.so',
             'budget_seconds': 1800, 'budget': 'hard limit',
             'produces': 'ordered launch journal, measured phases, tensor metadata'},
            {'name': 'plan', 'device': 'cpu', 'command': 'make_sample_plan.py',
             'budget_seconds': 300, 'budget': 'hard limit',
             'produces': 'sample-plan.json, layer-bindings.json, census.json'},
            {'name': 'build', 'device': 'cpu', 'command': 'nvbit_sampler_r4/build.py',
             'budget_seconds': 900, 'budget': 'hard limit', 'produces': 'sampler.so'},
            {'name': 'sample', 'device': 'gpu', 'command': 'sample_pipeline.py',
             'budget_seconds': 7200, 'budget': 'hard limit, and 60..21600 inside the tool',
             'produces': 'packed profile stream'},
            {'name': 'expand', 'device': 'cpu', 'command': 'expand_profiles.py',
             'budget_seconds': None, 'budget': 'no limit',
             'produces': 'full-inference profile stream and its manifest'},
            {'name': 'replay', 'device': 'cpu', 'command': 'run_memgen.py',
             'budget_seconds': None, 'budget': 'no wall-clock deadline at any layer',
             'produces': 'kernel_summary.csv and cache counters'},
        ],
        'reachable_artifacts': [
            'launch journal', 'census and host receipts', 'sample plan', 'packed profile',
            'expansion manifest', 'kernel_summary.csv', 'cache_observation.json',
            'collect-receipt.json',
        ],
        'not_reachable': {
            'per_address_trace': 'hbserve_profile_stream_cache_semantic_r17.cpp sets '
                                 'output_format = "summary" unconditionally, so the address '
                                 'stream is consumed in memory and never written',
            'raw_sass': 'materialized_raw_sass_bytes stays 0',
            'ncu_reference_for_sglang_bf16': 'not present in this archive; the in-repo NCU tool '
                                             'belongs to the historical llama.cpp/Q8 stack',
            'decode_steps_4_8_16': 'not declared, so not selectable',
        },
        'not_modeled': [
            'DRAM bank timing', 'MSHR queueing, finite concurrency and merge',
            'NVIDIA hardware write-back and dirty-release behaviour',
            'GPU arrival concurrency and completion timing',
        ],
        'unreconciled': {
            'dram_write_gate': 'docs/BRANCH_AND_ACCEPTANCE_CONTRACT.md section 6 states at most '
                               '20%; evidence/sglang/L2_CACHE_STRATEGY_ACCURACY_REPORT.md states '
                               'strictly below 10%. Both are still in the archive and the '
                               'difference decides rows, so quote a number with its gate.',
        },
        'resources': {
            'cpu_ids': '0..15 shared pool, one per job',
            'gpu_pool': _gpu_pool(),
            'gpu_selection': 'by index into the admitted pool, as `memgen gpus` prints',
            'guarded_rss_bytes': 64 << 30,
        },
        'commands': {
            'check': 'host dependencies and archive pins',
            'cases': 'declared workload points',
            'gpus': 'admitted GPUs by index',
            'plan': 'write the job specs for a case and print the plan',
            'collect': 'run census, sample, expand and replay for a case',
            'replay': 'run an admitted profile stream through the cache model',
            'smoke': 'replay the frozen engine fixture on the CPU',
            'test': 'portable regression tests',
        },
        'entry_points': {
            'engine_source': 'release/source/tools/hbserve_profile_stream_cache_semantic_r17.cpp',
            'observer_source': 'integrations/sglang/compact-sources/observer/observer.so',
            'sampler_source': 'integrations/sglang/compact-sources/upstream/nvbit_sampler_r4/',
            'no_network': 'nothing here downloads anything',
        },
    }


def main(argv: list[str] | None = None) -> int:
    import sys
    if argv:
        print('memgen capabilities takes no options', file=sys.stderr)
        return 2
    print(json.dumps(current_capabilities(), indent=2, sort_keys=True))
    return 0
