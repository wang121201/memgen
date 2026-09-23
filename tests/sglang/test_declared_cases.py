#!/usr/bin/env python3
"""Declared-case contract tests: what the adapter may build, and what it must refuse.

These tests are portable and need no GPU, no model and no NVBit. They pin the
shape of the declared matrices, the rejection of undeclared pairs, the
mirror-copy invariant and the deployment pin structure. They are not an
accuracy test and they do not admit any workload.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / 'integrations/sglang/memgen-adapter'
SGLANG = ROOT / 'integrations/sglang'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ADAPTER))
import matrix_workload as workload


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DeclaredCases(unittest.TestCase):
    def setUp(self):
        self.spec = workload.spec()
        self.declared = workload.declared_cases(self.spec)

    def test_declares_the_expected_matrices(self):
        self.assertEqual(self.spec['schema'], 'SGLANG_FULL_INFERENCE_V2')
        self.assertEqual(sorted(self.spec['prefills']), [128, 256, 512, 1024])
        self.assertEqual(sorted(self.spec['decodes']), [32, 64, 128])
        self.assertEqual(len(self.declared['scale_series']), self.spec['scale_series_cases'])
        self.assertEqual(len(self.declared['basic_admission']),
                         self.spec['basic_admission_cases'])
        self.assertEqual(self.declared['basic_admission'],
                         {('qwen25_1p5b', 32, 2), ('llama3_8b', 32, 2)})

    def test_matrices_do_not_overlap(self):
        self.assertEqual(self.declared['scale_series'] & self.declared['basic_admission'], set())

    def test_admission_point_is_accepted_and_named(self):
        for model in ('qwen25_1p5b', 'llama3_8b'):
            case = workload.contract(model, 32, 2)
            self.assertEqual(case['declared_matrix'], 'basic_admission')
            self.assertEqual(case['case_id'], f'{model}-p32-d2')
            self.assertEqual(len(case['prompt_ids']), 32)
            self.assertEqual(case['decode_input_ids'], [944, 291])
            self.assertEqual(case['phases'], ['Prefill', 'Decode1', 'Decode2'])

    def test_scale_point_keeps_its_matrix_name(self):
        case = workload.contract('qwen25_1p5b', 128, 32)
        self.assertEqual(case['declared_matrix'], 'scale_series')

    def test_undeclared_pairs_are_refused(self):
        for case in [('qwen25_1p5b', 32, 128), ('qwen25_1p5b', 32, 64),
                     ('qwen25_1p5b', 64, 2), ('qwen25_1p5b', 128, 2),
                     ('qwen25_1p5b', 256, 2), ('llama3_8b', 32, 16),
                     ('qwen25_1p5b', 1024, 128 + 1)]:
            with self.assertRaises(ValueError, msg=repr(case)):
                workload.contract(*case)

    def test_unknown_model_is_refused(self):
        with self.assertRaises(ValueError):
            workload.contract('qwen235B', 32, 2)

    def test_kv_capacity_is_enforced(self):
        largest = self.spec['max_total_tokens'] - 1
        self.assertLessEqual(max(self.spec['prefills']) + max(self.spec['decodes']) + 1, largest + 1)

    def test_case_is_deterministic(self):
        self.assertEqual(workload.contract('qwen25_1p5b', 32, 2),
                         workload.contract('qwen25_1p5b', 32, 2))

    def test_mirrored_copies_are_identical(self):
        for name in ('contract.json', 'matrix_workload.py'):
            self.assertEqual(sha256(ADAPTER / name), sha256(SGLANG / 'metadata-host/vendor' / name),
                             name)


class DeploymentPins(unittest.TestCase):
    def test_manifest_and_revisions_agree_with_the_files(self):
        deployment = json.loads((ADAPTER / 'deployment-files.json').read_text())['files']
        self.assertTrue(deployment)
        for row in deployment:
            path = ADAPTER / row['name']
            self.assertTrue(path.is_file(), row['name'])
            self.assertEqual(path.stat().st_size, row['bytes'], row['name'])
            self.assertEqual(sha256(path), row['sha256'], row['name'])

        revisions = json.loads((SGLANG / 'revisions.json').read_text())['revisions']
        self.assertTrue(revisions)
        for entry in revisions:
            self.assertTrue(entry['paths'] and entry['reason'] and entry['record'])
            for relative in entry['paths']:
                path = SGLANG / relative
                self.assertTrue(path.is_file(), relative)
                self.assertEqual(sha256(path), entry['current_sha256'], relative)
                self.assertNotEqual(entry['historical_sha256'], entry['current_sha256'], relative)

    def test_revision_paths_are_unique(self):
        revisions = json.loads((SGLANG / 'revisions.json').read_text())['revisions']
        paths = [relative for entry in revisions for relative in entry['paths']]
        self.assertEqual(len(paths), len(set(paths)))

    def test_pinned_verifier_agrees(self):
        result = subprocess.run([sys.executable, '-B', str(SGLANG / 'bootstrap_vendor.py'), '--check'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS_VENDOR_PINS_SATISFIED', result.stdout)

    def test_frozen_runtime_manifest_check_passes(self):
        # Exact replica of the loop at the top of wait_then_sample.py.
        deployment = json.loads((ADAPTER / 'deployment-files.json').read_text())
        for row in deployment['files']:
            path = ADAPTER / row['name']
            got = dict(bytes=path.stat().st_size, sha256=sha256(path))
            self.assertEqual((got['bytes'], got['sha256']), (row['bytes'], row['sha256']),
                             row['name'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
