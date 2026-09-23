#!/usr/bin/env python3
"""Tests for the single entry point: dispatch, capabilities and refusals.

These are portable. They need no GPU, no model and no NVBit; the GPU-dependent
verbs are only checked up to their argument validation.
"""
import json
import subprocess
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
MEMGEN = ROOT / 'memgen'
ADAPTER = ROOT / 'integrations/sglang/memgen-adapter'


def run(*args, **kwargs):
    return subprocess.run(['bash', str(MEMGEN), *args], capture_output=True, text=True, **kwargs)


class Dispatch(unittest.TestCase):
    def test_no_arguments_prints_usage(self):
        result = run()
        self.assertEqual(result.returncode, 0)
        for verb in ('check', 'cases', 'gpus', 'plan', 'collect', 'replay', 'smoke', 'test',
                     'capabilities'):
            self.assertIn(verb, result.stdout)

    def test_version(self):
        result = run('--version')
        self.assertEqual(result.returncode, 0)
        self.assertRegex(result.stdout.strip(), r'^\d+\.\d+\.\d+$')

    def test_unknown_command_is_refused(self):
        result = run('frobnicate')
        self.assertEqual(result.returncode, 2)
        self.assertIn('unknown command', result.stderr)

    def test_collect_refuses_dry_run(self):
        result = run('collect', '--model', 'qwen25_1p5b', '--prefill-length', '32',
                     '--decode-steps', '2', '--dry-run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('memgen plan', result.stderr)


class Capabilities(unittest.TestCase):
    def setUp(self):
        result = run('capabilities')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.payload = json.loads(result.stdout)

    def test_schema_and_boundary(self):
        self.assertEqual(self.payload['schema']['name'], 'memgen.capabilities')
        self.assertIs(self.payload['hardware_accuracy_accepted'], False)
        self.assertEqual(self.payload['admission_authority'],
                         'validation/p32d2_branch_status.csv')

    def test_declared_sets_cannot_drift_from_the_contract(self):
        contract = json.loads((ADAPTER / 'contract.json').read_text())
        declared = self.payload['declared']
        self.assertEqual(declared['scale_series']['cases'],
                         len(contract['prefills']) * len(contract['decodes'])
                         * len(contract['models']))
        self.assertEqual(declared['basic_admission']['cases'], 2)
        self.assertEqual(sorted(declared['basic_admission']['prefills']), [32])
        self.assertEqual(sorted(declared['basic_admission']['decodes']), [2])

    def test_unreachable_and_unreconciled_are_declared(self):
        self.assertIn('per_address_trace', self.payload['not_reachable'])
        self.assertIn('ncu_reference_for_sglang_bf16', self.payload['not_reachable'])
        self.assertIn('dram_write_gate', self.payload['unreconciled'])

    def test_every_stage_states_a_budget_policy(self):
        names = [stage['name'] for stage in self.payload['stages']]
        self.assertEqual(names, ['census', 'plan', 'build', 'sample', 'expand', 'replay'])
        by_name = {stage['name']: stage for stage in self.payload['stages']}
        for name in ('census', 'plan', 'build', 'sample'):
            self.assertIn('hard limit', by_name[name]['budget'])
        self.assertIn('no limit', by_name['expand']['budget'])
        self.assertIn('no wall-clock deadline', by_name['replay']['budget'])


class ReadOnlyVerbs(unittest.TestCase):
    def test_cases_lists_every_declared_point(self):
        result = run('cases')
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        start = next(index for index, line in enumerate(lines) if line.startswith('case id'))
        table = lines[start + 1:start + 1 + 26]
        self.assertTrue(all('Qwen2.5-1.5B' in line or 'Meta-Llama-3-8B' in line for line in table))
        note = [line for line in lines[start + 27:] if line]
        self.assertTrue(note and note[0].startswith('The model key'))

    def test_gpus_numbering_is_contiguous(self):
        result = run('gpus')
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split() for line in result.stdout.strip().splitlines()[1:]
                if line and not line.startswith('Pass')]
        self.assertEqual([row[0] for row in rows], ['0', '1', '2'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
