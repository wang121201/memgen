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
import shutil
import subprocess
import sys
import tempfile
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

    def test_declared_models_carry_a_display_name(self):
        for key, row in self.spec['models'].items():
            self.assertTrue(row.get('display'), key)
            self.assertIn('BF16', row['display'])

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


class CollectionDriver(unittest.TestCase):
    """The driver must agree with the contract and must not bound the replay."""

    def setUp(self):
        sys.path.insert(0, str(SGLANG))
        import collect_case
        self.driver = collect_case
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_list_cases_prints_every_declared_case(self):
        result = subprocess.run([sys.executable, '-B', str(SGLANG / 'collect_case.py'),
                                 '--list-cases'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        table = result.stdout.split('\n\n')[0].strip().splitlines()[1:]
        self.assertEqual(len(table), 26)
        cases = [line.split()[0] for line in table]
        self.assertEqual(cases[0], 'llama3_8b-p32-d2')
        self.assertEqual(cases[-1], 'qwen25_1p5b-p1024-d128')
        matrices = {token for line in table for token in line.split()
                    if token in ('scale_series', 'basic_admission')}
        self.assertEqual(matrices, {'scale_series', 'basic_admission'})
        # The model key is terse, so the readable name must be visible too.
        self.assertIn('Qwen2.5-1.5B-Instruct', result.stdout)
        self.assertIn('Meta-Llama-3-8B-Instruct', result.stdout)

    def test_sample_budget_cannot_be_asked_to_run_unbounded(self):
        result = subprocess.run([sys.executable, '-B', str(SGLANG / 'collect_case.py'),
                                 '--model', 'qwen25_1p5b', '--prefill-length', '32',
                                 '--decode-steps', '2', '--sample-seconds', '0',
                                 '--work', '/tmp/collect-case-test-never-created',
                                 '--dry-run'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('60..21600', result.stdout + result.stderr)
        self.assertFalse(Path('/tmp/collect-case-test-never-created').exists())

    def test_driver_declares_the_same_cases_as_the_contract(self):
        expected = sorted(set().union(*workload.declared_cases(workload.spec()).values()))
        self.assertEqual(self.driver.declared_cases(), expected)

    def test_gpu_index_numbering_is_stable_and_covers_the_pool(self):
        table = self.driver.gpu_table()
        self.assertEqual([index for index, _, _ in table], list(range(len(table))))
        self.assertEqual({uuid for _, uuid, _ in table}, self.driver.gpu_pool())

    def test_collect_job_has_no_wall_clock_ceiling(self):
        import argparse
        args = argparse.Namespace(cpu=8, gpu='GPU-admitted-placeholder', job_seconds=0,
                                  python=sys.executable, sample_seconds=7200,
                                  journal_process='process-0000')
        spec = self.driver.collect_spec('qwen25_1p5b-p32-d2', self.tmp / 'work', args)
        self.assertEqual(spec['seconds'], 0)
        self.assertNotIn('--seconds', spec['argv'])
        self.assertNotIn('--memgen-seconds', spec['argv'])

    def test_census_job_stays_bounded(self):
        import argparse
        args = argparse.Namespace(cpu=8, gpu='GPU-admitted-placeholder', census_seconds=1800,
                                  python=sys.executable, observer='/nonexistent/observer.so')
        contract = workload.contract('qwen25_1p5b', 32, 2)
        spec = self.driver.census_spec('qwen25_1p5b-p32-d2', self.tmp / 'work',
                                       contract, args)
        self.assertEqual(spec['seconds'], 1800)
        self.assertEqual(spec['environment']['SG_NVBIT_SCOPE_ABI'], '1')


class ObserverOutputRoot(unittest.TestCase):
    """observer.cu needs its root to exist and to equal its own realpath.

    It initializes before the child interpreter runs, so `host.py` cannot create
    the directory and the driver has to. A symlinked `--work` must not leak into
    the variable either, because the observer compares the two as strings.
    """

    def setUp(self):
        sys.path.insert(0, str(SGLANG))
        import collect_case
        self.driver = collect_case
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.work = self.tmp / 'work'
        self.work.mkdir()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def spec(self, work):
        import argparse
        args = argparse.Namespace(cpu=8, gpu='GPU-admitted-placeholder', census_seconds=1800,
                                  python=sys.executable, observer='/nonexistent/observer.so')
        return self.driver.census_spec('qwen25_1p5b-p32-d2', work,
                                      workload.contract('qwen25_1p5b', 32, 2), args)

    def test_census_spec_names_an_existing_canonical_root(self):
        root = Path(self.spec(self.work)['environment']['SG_NVBIT_OUTPUT_ROOT'])
        self.assertTrue(root.is_dir())
        self.assertEqual(str(root), str(root.resolve()))
        self.assertEqual(root, self.work / 'observers' / 'qwen25_1p5b-p32-d2-census')

    def test_symlinked_work_yields_the_real_path(self):
        link = self.tmp / 'link'
        link.symlink_to(self.work)
        root = Path(self.spec(link)['environment']['SG_NVBIT_OUTPUT_ROOT'])
        self.assertEqual(str(root), str(root.resolve()))
        self.assertNotIn('link', str(root))

    def test_collect_spec_reads_the_same_root(self):
        import argparse
        args = argparse.Namespace(cpu=8, gpu='GPU-admitted-placeholder', job_seconds=0,
                                  python=sys.executable, sample_seconds=7200,
                                  journal_process='process-123')
        spec = self.driver.collect_spec('qwen25_1p5b-p32-d2', self.work, args)
        journal = Path(spec['argv'][spec['argv'].index('--journal') + 1])
        self.assertEqual(journal.parent,
                         self.work / 'observers' / 'qwen25_1p5b-p32-d2-census')
        self.assertTrue(journal.parent.is_dir())


class ReplayWrappersHaveNoDeadline(unittest.TestCase):
    """The wrappers must not kill a replay that run_memgen.py says is unbounded."""

    def test_followthrough_runs_the_replay_without_a_timeout(self):
        text = (ADAPTER / 'followthrough.py').read_text()
        self.assertIn("'--output',str(a.output/'cache')],None)", text)
        self.assertNotIn('a.memgen_seconds+60', text)

    def test_profile_cache_runs_the_replay_without_a_timeout(self):
        text = (ADAPTER / 'profile_cache.py').read_text()
        self.assertIn("'--output',str(a.output/'cache')],None)]", text)
        self.assertNotIn('a.seconds+60', text)

    def test_controller_can_express_an_unbounded_job(self):
        text = (SGLANG / 'run_job.py').read_text()
        self.assertIn('seconds == 0', text)
        self.assertIn('no wall-clock deadline', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
