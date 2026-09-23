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


def fake_census(work, case, gpu, pid=938490, ticks=918072691):
    """Write the census receipts a real run leaves, with its own values."""
    observer = work / 'observers' / f'{case}-census' / f'process-{pid}-{ticks}'
    host = work / 'runs' / f'{case}-census' / 'host' / f'process-{pid}'
    observer.mkdir(parents=True)
    host.mkdir(parents=True)
    (observer / 'finish.json').write_text(json.dumps(
        dict(status='PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE', pid=pid, start_ticks=ticks,
             epoch_begin_count=6, epoch_end_count=6, active_epoch=0,
             metadata_bytes_before_finish=10347345, max_total_bytes=268435456,
             launch_before_count=2194)))
    (host / 'finish.json').write_text(json.dumps(
        dict(status='PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE',
             input_contract=dict(case_id=case))))
    (work / 'runs' / f'{case}-census' / 'job-finish.json').write_text(json.dumps(
        dict(status='PASS_PROCESS_ONLY', cpu=8, gpu=gpu, case_id=case)))
    return observer, host


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
                                  python=sys.executable, sample_seconds=7200)
        spec = self.driver.collect_spec('qwen25_1p5b-p32-d2', self.tmp / 'work', args,
                                       'process-0001-000000002', 'process-0001')
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
                                  python=sys.executable, sample_seconds=7200)
        spec = self.driver.collect_spec('qwen25_1p5b-p32-d2', self.work, args,
                                       'process-938490-918072691', 'process-938490')
        journal = Path(spec['argv'][spec['argv'].index('--journal') + 1])
        self.assertEqual(journal.parent,
                         self.work / 'observers' / 'qwen25_1p5b-p32-d2-census')
        self.assertTrue(journal.parent.is_dir())


class CensusClosure(unittest.TestCase):
    """The gate that decides whether a census may be continued from.

    Field values and both directory names come from a real run: the observer
    writes `process-<pid>-<ticks>` because it does not know the controller's
    name, and the controller writes `process-<pid>`, so comparing the two names
    can never pass and comparing the pids always must.
    """

    PID = 938490
    TICKS = 918072691
    GPU = 'GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13'
    CASE = 'qwen25_1p5b-p32-d2'

    def setUp(self):
        sys.path.insert(0, str(SGLANG))
        import collect_case
        self.driver = collect_case
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.observer = dict(status='PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE', pid=self.PID,
                             start_ticks=self.TICKS, epoch_begin_count=6, epoch_end_count=6,
                             active_epoch=0, metadata_bytes_before_finish=10347345,
                             max_total_bytes=268435456, launch_before_count=2194)
        self.host = dict(status='PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE',
                         input_contract=dict(case_id=self.CASE))
        self.job = dict(status='PASS_PROCESS_ONLY', cpu=8, gpu=self.GPU, case_id=self.CASE)
        self.observer_path = self.write(
            f'observers/{self.CASE}-census/process-{self.PID}-{self.TICKS}/finish.json', self.observer)
        self.host_path = self.write(f'runs/{self.CASE}-census/host/process-{self.PID}/finish.json',
                                    self.host)
        self.job_path = self.write(f'runs/{self.CASE}-census/job-finish.json', self.job)

    def test_the_two_process_directories_are_named_differently(self):
        """Job 2 reads <journal> and <host>/process-<pid>/finish.json, nothing else."""
        self.assertEqual(self.observer_path.parent.name, f'process-{self.PID}-{self.TICKS}')
        self.assertEqual(self.host_path.parent.name, f'process-{self.PID}')
        self.assertNotEqual(self.observer_path.parent.name, self.host_path.parent.name)

    def write(self, relative, payload):
        path = self.tmp / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    def verify(self, observer=None, host=None, job=None, case=None, cpu=8, gpu=None):
        for path, payload in ((self.observer_path, observer), (self.host_path, host),
                              (self.job_path, job)):
            if payload is not None:
                path.write_text(json.dumps(payload))
        return self.driver.verify_census(self.observer_path, self.host_path, self.job_path,
                                        case or self.CASE, cpu, gpu or self.GPU)

    def test_one_process_with_two_directory_names_is_accepted(self):
        returned = self.verify()
        self.assertEqual(returned['pid'], self.PID)
        self.assertEqual(self.host_path.parent.name, f'process-{self.PID}')
        self.assertNotEqual(self.observer_path.parent.name, self.host_path.parent.name)

    def test_another_pid_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.verify(observer=dict(self.observer, pid=self.PID + 1))
        self.assertIn('not one process', str(caught.exception))

    def test_open_epochs_are_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.verify(observer=dict(self.observer, epoch_end_count=5, active_epoch=1))
        self.assertIn('epochs did not close', str(caught.exception))

    def test_exhausted_metadata_quota_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.verify(observer=dict(self.observer, metadata_bytes_before_finish=268435456))
        self.assertIn('metadata quota', str(caught.exception))

    def test_receipt_for_another_resource_is_refused(self):
        with self.assertRaises(SystemExit):
            self.verify(job=dict(self.job, cpu=9))
        with self.assertRaises(SystemExit):
            self.verify(job=dict(self.job, case_id='llama3_8b-p32-d2'))
        with self.assertRaises(SystemExit):
            self.verify(host=dict(self.host, input_contract=dict(case_id='llama3_8b-p32-d2')))


class GpuAdmissionWait(unittest.TestCase):
    """A device that was just released still reads as busy for a moment."""

    BUSY = ('parent_controller.ResourceBusy: selected GPU not idle at fresh locked '
            'admission: {"utc": "2026-09-23T16:58:55Z", "uuid": '
            '"GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13", '
            '"used_memory_MiB": 15.0, "utilization_percent": 38.0, '
            '"active_compute_apps": []}')

    def setUp(self):
        sys.path.insert(0, str(SGLANG))
        import collect_case
        self.driver = collect_case

    def test_the_real_refusal_is_recognised(self):
        self.assertTrue(self.driver.gpu_admission_is_busy(self.BUSY))
        self.assertTrue(self.driver.gpu_admission_is_busy('... ResourceBusy: selected GPU not idle '
                                                          'at fresh locked admission: {}'))

    def test_other_failures_are_not_retried(self):
        for text in ('', 'ValueError: child exited nonzero',
                     'FileNotFoundError: [Errno 2] no such file',
                     'the device is busy'):
            self.assertFalse(self.driver.gpu_admission_is_busy(text))

    def test_the_wait_budget_is_bounded_and_can_be_turned_off(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu-wait-seconds', type=int,
                            default=self.driver.GPU_ADMISSION_WAIT_SECONDS)
        self.assertEqual(parser.parse_args([]).gpu_wait_seconds, 600)
        self.assertEqual(parser.parse_args(['--gpu-wait-seconds', '0']).gpu_wait_seconds, 0)
        self.assertLessEqual(self.driver.GPU_ADMISSION_RETRY_SECONDS, 60)


class Resume(unittest.TestCase):
    """--resume continues a run whose census closed, on the census's own receipts."""

    CASE = 'qwen25_1p5b-p32-d2'
    GPU = 'GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13'

    def setUp(self):
        sys.path.insert(0, str(SGLANG))
        import collect_case
        self.driver = collect_case
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.work = self.tmp / 'work'
        observer_root = self.work / 'observers' / f'{self.CASE}-census' / 'process-938490-918072691'
        observer_root.mkdir(parents=True)
        host_root = self.work / 'runs' / f'{self.CASE}-census' / 'host' / 'process-938490'
        host_root.mkdir(parents=True)
        (observer_root / 'finish.json').write_text(json.dumps(
            dict(status='PASS_METADATA_OBSERVER_CLOSED_NOT_TRACE', pid=938490,
                 start_ticks=918072691, epoch_begin_count=6, epoch_end_count=6,
                 active_epoch=0, metadata_bytes_before_finish=10347345,
                 max_total_bytes=268435456, launch_before_count=2194)))
        (host_root / 'finish.json').write_text(json.dumps(
            dict(status='PASS_NATIVE_HOST_PENDING_OBSERVER_OR_SAMPLER_CLOSURE',
                 input_contract=dict(case_id=self.CASE))))
        (self.work / 'runs' / f'{self.CASE}-census' / 'job-finish.json').write_text(json.dumps(
            dict(status='PASS_PROCESS_ONLY', cpu=8, gpu=self.GPU, case_id=self.CASE)))

    def driver_run(self, *args, work=None):
        return subprocess.run([sys.executable, '-B', str(SGLANG / 'collect_case.py'),
                               '--model', 'qwen25_1p5b', '--prefill-length', '32',
                               '--decode-steps', '2', '--gpu', self.GPU,
                               '--work', str(work or self.work), *args],
                              capture_output=True, text=True)

    def test_reused_census_reads_the_receipts(self):
        finish, observer = self.driver.reused_census(self.work, self.CASE, 8, self.GPU)
        self.assertEqual(finish.parent.name, 'process-938490-918072691')
        self.assertEqual(observer['launch_before_count'], 2194)

    def test_the_two_process_names_are_derived_not_guessed(self):
        finish, observer = self.driver.reused_census(self.work, self.CASE, 8, self.GPU)
        journal, host = self.driver.census_process_names(observer, finish)
        self.assertEqual(journal, 'process-938490-918072691')
        self.assertEqual(host, 'process-938490')
        self.assertNotEqual(journal, host)

    def test_a_receipt_that_disagrees_with_its_directory_is_refused(self):
        finish, observer = self.driver.reused_census(self.work, self.CASE, 8, self.GPU)
        with self.assertRaises(SystemExit) as caught:
            self.driver.census_process_names(dict(observer, start_ticks=1), finish)
        self.assertIn('disagree', str(caught.exception))

    def test_collect_spec_points_at_both_real_directories(self):
        import argparse
        finish, observer = self.driver.reused_census(self.work, self.CASE, 8, self.GPU)
        journal, host = self.driver.census_process_names(observer, finish)
        args = argparse.Namespace(cpu=8, gpu=self.GPU, job_seconds=0, python=sys.executable,
                                  sample_seconds=7200)
        spec = self.driver.collect_spec(self.CASE, self.work, args, journal, host)
        given = dict(zip(spec['argv'], spec['argv'][1:]))
        self.assertTrue(Path(given['--journal']).is_dir())
        self.assertTrue(Path(given['--host-finish']).is_file())

    def test_reused_census_still_applies_the_gate(self):
        (self.work / 'runs' / f'{self.CASE}-census' / 'job-finish.json').write_text(json.dumps(
            dict(status='PASS_PROCESS_ONLY', cpu=9, gpu=self.GPU, case_id=self.CASE)))
        with self.assertRaises(SystemExit):
            self.driver.reused_census(self.work, self.CASE, 8, self.GPU)

    def test_previous_job_output_is_kept_not_deleted(self):
        stale = self.work / 'runs' / f'{self.CASE}-collect'
        (stale / 'followthrough').mkdir(parents=True)
        (stale / 'followthrough' / 'evidence.json').write_text('{}')
        moved = self.driver.stash_job_output(self.work, self.CASE)
        self.assertFalse(stale.exists())
        self.assertTrue((moved / 'followthrough' / 'evidence.json').is_file())
        self.assertIn('.attempt-', moved.name)
        self.assertIsNone(self.driver.stash_job_output(self.work, self.CASE))

    def test_resume_needs_an_existing_work(self):
        result = self.driver_run('--resume', work=self.tmp / 'absent')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('needs an existing --work', result.stdout + result.stderr)

    def test_resume_and_dry_run_do_not_combine(self):
        result = self.driver_run('--resume', '--dry-run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('do not combine', result.stdout + result.stderr)

    def test_an_existing_work_is_refused_without_resume(self):
        result = self.driver_run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refusing existing work directory', result.stdout + result.stderr)
        self.assertIn('--resume', result.stdout + result.stderr)


class PartialDiagnostic(unittest.TestCase):
    """A partial expansion must name its coverage, never pass as the case traffic."""

    CASE = 'qwen25_1p5b-p32-d2'
    GPU = 'GPU-69cebdc2-40c1-603a-aa3d-991cd3fbac13'

    def setUp(self):
        sys.path.insert(0, str(SGLANG))
        import collect_case
        self.driver = collect_case
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.work = self.tmp / 'work'
        fake_census(self.work, self.CASE, self.GPU)
        follow = self.work / 'runs' / f'{self.CASE}-collect' / 'followthrough'
        (follow / 'expanded').mkdir(parents=True)
        (follow / 'finish.json').write_text(json.dumps(
            dict(status='STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC', stages=[],
                 unsupported_launches=700)))
        (follow / 'expanded' / 'manifest.json').write_text(json.dumps(
            dict(complete_full_model=False, target_launches=2060, packed_launches=1360,
                 unsupported_launches=700)))
        self.follow = follow

    def resume(self, *extra):
        return subprocess.run([sys.executable, '-B', str(SGLANG / 'collect_case.py'),
                               '--model', 'qwen25_1p5b', '--prefill-length', '32',
                               '--decode-steps', '2', '--gpu', self.GPU,
                               '--work', str(self.work), '--resume', *extra],
                              capture_output=True, text=True)

    def test_a_closed_job_two_is_reused_instead_of_rerun(self):
        result = self.resume()
        self.assertIn('reused from --work (--resume)', result.stdout)
        self.assertIn('STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC', result.stdout)

    def test_the_stop_names_its_coverage_and_points_at_partial(self):
        result = self.resume()
        self.assertEqual(result.returncode, 2)
        self.assertIn('1360 of 2060 launches', result.stdout)
        self.assertIn('--partial', result.stdout)
        self.assertIn('no counters were produced', result.stdout)

    def test_the_receipt_records_the_coverage_it_came_from(self):
        self.resume()
        receipt = json.loads((self.work / 'collect-receipt.json').read_text())
        self.assertEqual(receipt['status'], 'STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC')
        self.assertIsNone(receipt['partial_diagnostic'])
        self.assertIs(receipt['hardware_accuracy_accepted'], False)
        self.assertIsNone(receipt['artifacts']['kernel_summary'])

    def test_the_declared_matrix_point_has_no_replay_to_reuse(self):
        self.assertFalse((self.follow / 'cache' / 'model' / 'kernel_summary.csv').exists())

    def test_counters_reader_sums_kernels_and_ratios_sums(self):
        summary = self.tmp / 'kernel_summary.csv'
        summary.write_text(
            'kernel_id,l1_requests,l1_hits,l2_requests,l2_hits,dram_load_bytes,'
            'dram_store_bytes,l2_writeback_dirty_sectors\n'
            '1,2,1,4,1,64,0,2\n'
            '2,6,3,4,3,192,128,0\n')
        counters = self.driver.kernel_counters(summary)
        self.assertEqual(counters['kernels'], 2)
        self.assertEqual(counters['l1_requests'], 8)
        self.assertEqual(counters['l1_hits'], 4)
        self.assertEqual(counters['l1_hit_rate'], '0.500000')
        self.assertEqual(counters['l2_hits'], 4)
        self.assertEqual(counters['l2_hit_rate'], '0.500000')
        self.assertEqual(counters['dram_load_bytes'], 256)
        self.assertEqual(counters['dram_store_bytes'], 128)
        self.assertEqual(counters['l2_writeback_dirty_sectors'], 2)

    def test_the_engine_is_built_from_the_archived_source(self):
        self.assertTrue(self.driver.ENGINE_SOURCE.is_file())
        self.assertIn('release/source/tools/', str(self.driver.ENGINE_SOURCE))


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
