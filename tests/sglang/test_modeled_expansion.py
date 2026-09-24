#!/usr/bin/env python3
"""Modeled completion for unpacked classes: geometry, evidence, and labels.

These are the rules that decide what a modeled profile may claim. They run
without a GPU, without the frozen sampler, and without any run directory.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / 'integrations/sglang/memgen-adapter'
sys.path.insert(0, str(ADAPTER))

import model_uncovered  # noqa: E402


def view(address, elements, element_size=2, shape=None, stride=None):
    shape = shape or [elements]
    return dict(data_address=address, shape=shape, element_size=element_size,
                stride_bytes=stride or [element_size], dtype='bf16')


def kernel(grid, name='modeled_kernel', phase='Prefill'):
    size = grid[0] * grid[1] * grid[2]
    return dict(name=name, grid_dims=list(grid), grid_size=size, block_size=128, phase=phase, id=0)


def launch(grid, records=64, call_id=7, code='c0ffee'):
    return dict(function_name='modeled_kernel', grid=list(grid), block=[128, 1, 1],
                phase='Prefill', call_id=call_id, code_sha256=code,
                selected_records=records)


def evidence(records=64, classified=None, key='epoch-1-launch-3', ctas=1, grid=1,
             phase='Prefill', census=None):
    """Per-class evidence as class_evidence() builds it from consumer.json."""
    return dict(records=classified if classified is not None else records, ctas=ctas,
                grid=grid, phase=phase, census=census, selected_records=records,
                grid_whole=ctas == grid, source_launch_key=key)


CALIB = dict(sampled_ctas=1.0, bytes_per_record=128.0,
             bytes_per_record_by_phase={'Prefill': 128.0, 'Decode1': 128.0},
             bytes_per_record_range=[4.0, 512.0], read_share=1.0,
             classes_calibrated=150, basis='run_calibrated_records_per_cta')


class AccessGeometry(unittest.TestCase):
    def test_a_wide_object_uses_full_warp_16_byte_lanes(self):
        self.assertEqual(model_uncovered.access_geometry(64 << 20), (16, 32))

    def test_a_narrow_object_narrows_the_access_before_the_warp(self):
        self.assertEqual(model_uncovered.access_geometry(300), (8, 32))
        self.assertEqual(model_uncovered.access_geometry(64), (2, 32))

    def test_an_object_smaller_than_a_warp_uses_fewer_lanes(self):
        width, lanes = model_uncovered.access_geometry(20)
        self.assertEqual((width, lanes), (16, 1))

    def test_no_geometry_ever_reads_past_its_object(self):
        for span in (1, 7, 16, 20, 63, 64, 255, 256, 511, 512, 513, 4096):
            width, lanes = model_uncovered.access_geometry(span)
            self.assertLessEqual(width * lanes, max(span, 1), span)

    def test_opcode_tokens_match_the_engine_width_table(self):
        wanted = {16: 'LDG.E.128', 8: 'LDG.E.64', 4: 'LDG.E.32', 2: 'LDG.E.16', 1: 'LDG.E.8'}
        for width, opcode in wanted.items():
            self.assertEqual(model_uncovered.opcode_for(width, 'read'), opcode)
        self.assertEqual(model_uncovered.opcode_for(16, 'write'), 'STG.E.128')


class Evidence(unittest.TestCase):
    def test_classified_records_prefer_the_projection_census(self):
        row = dict(selected_records=100, projection_classification=dict(classified_records=40))
        self.assertEqual(model_uncovered.classified_records(row), 40)

    def test_classified_records_never_exceed_the_selected_sample(self):
        row = dict(selected_records=10, projection_classification=dict(classified_records=40))
        self.assertEqual(model_uncovered.classified_records(row), 10)

    def test_record_count_without_a_classification_is_the_sample(self):
        self.assertEqual(model_uncovered.classified_records(dict(selected_records=11)), 11)

    def test_a_template_census_is_a_whole_grid_total_and_is_divided_by_its_grid(self):
        census = dict(native_read_lane_bytes=4096, native_write_lane_bytes=1024)
        volume = model_uncovered.per_cta_volume(CALIB, evidence(grid=8, ctas=8),
                                                phase='Prefill', census=census)
        self.assertEqual((volume['read'], volume['write']), (512, 128))
        self.assertEqual(volume['basis'], 'template_census_whole_grid_divided_by_grid')

    def test_a_record_count_is_divided_by_the_ctas_it_was_observed_on(self):
        volume = model_uncovered.per_cta_volume(CALIB, evidence(records=64, ctas=8, grid=8),
                                                phase='Prefill')
        self.assertEqual(volume['per_cta_records'], 8)
        self.assertEqual(volume['read'], 1024)

    def test_the_phase_median_is_used_when_the_run_has_one(self):
        calib = dict(CALIB, bytes_per_record_by_phase={'Decode1': 8.0}, bytes_per_record=128.0)
        volume = model_uncovered.per_cta_volume(calib, evidence(records=4, ctas=1), phase='Decode1')
        self.assertEqual(volume['bytes_per_record'], 8.0)

    def test_calibration_needs_at_least_one_fitted_class(self):
        with self.assertRaises(ValueError):
            model_uncovered.calibrate([], [])

    def test_calibration_reports_bytes_per_instruction_by_phase_and_its_range(self):
        kernels = [dict(source_launch_key='k', phase='Prefill', selected_records=8,
                        projection_classification=dict(classified_records=8)),
                   dict(source_launch_key='j', phase='Decode1', selected_records=4,
                        projection_classification=dict(classified_records=4))]
        packed = [dict(source_launch_key='k', status='PASS_PROFILE_EXACT_SAMPLES',
                       census=dict(mem_insts=4, native_read_lane_bytes=512,
                                   native_write_lane_bytes=0)),
                  dict(source_launch_key='j', status='PASS_PROFILE_EXACT_SAMPLES',
                       census=dict(mem_insts=2, native_read_lane_bytes=192,
                                   native_write_lane_bytes=64))]
        calib = model_uncovered.calibrate(kernels, packed)
        self.assertEqual(calib['bytes_per_record_by_phase'], {'Decode1': 128.0, 'Prefill': 128.0})
        self.assertEqual(calib['bytes_per_record_range'], [128.0, 128.0])
        self.assertAlmostEqual(calib['read_share'], 704 / 768)
        self.assertNotIn('sampled_ctas', calib)

    def test_class_evidence_uses_the_cta_count_the_sampler_really_selected(self):
        consumer = dict(kernels=[
            dict(source_launch_key='a', selected_records=240, selected_all_grid_ctas=False,
                 fit_ctas=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], holdout_ctas=[]),
            dict(source_launch_key='b', selected_records=768, selected_all_grid_ctas=True,
                 fit_ctas=[0], holdout_ctas=[])])
        kernels = [dict(source_launch_key='a', code_sha256='x', grid=[32, 1, 1], block=[128, 1, 1],
                        phase='Prefill', selected_records=240),
                   dict(source_launch_key='b', code_sha256='y', grid=[32, 1, 1], block=[128, 1, 1],
                        phase='Prefill', selected_records=768)]
        index = model_uncovered.class_evidence(consumer, kernels, [])
        subset = index[model_uncovered.class_shape(kernels[0])]
        whole = index[model_uncovered.class_shape(kernels[1])]
        self.assertEqual((subset['records'], subset['ctas'], subset['grid']), (240, 10, 32))
        self.assertTrue(whole['grid_whole'])
        self.assertEqual((whole['records'], whole['ctas'], whole['grid']), (768, 32, 32))

    def test_class_evidence_refuses_a_class_the_fitting_receipt_does_not_know(self):
        consumer = dict(kernels=[dict(source_launch_key='missing', selected_records=8,
                                      selected_all_grid_ctas=True, fit_ctas=[], holdout_ctas=[])])
        with self.assertRaises(ValueError):
            model_uncovered.class_evidence(consumer, [], [])


class ObjectSpans(unittest.TestCase):
    def test_inputs_and_outputs_are_separated(self):
        views = [(('m', 'inputs', 'w'), view(0x1000, 64)),
                 (('m', 'outputs', 'o'), view(0x2000, 32))]
        self.assertEqual(model_uncovered.object_spans(views, 'inputs'), [(0x1000, 0x1000 + 128)])
        self.assertEqual(model_uncovered.object_spans(views, 'outputs'), [(0x2000, 0x2000 + 64)])

    def test_overlapping_spans_merge(self):
        views = [(('m', 'inputs', 'a'), view(0x1000, 64)),
                 (('m', 'inputs', 'b'), view(0x1010, 64))]
        self.assertEqual(model_uncovered.object_spans(views, 'inputs'), [(0x1000, 0x1000 + 0x90)])

    def test_strided_views_use_their_real_extent(self):
        views = [(('m', 'inputs', 's'), view(0x1000, 4, shape=[4, 4], stride=[8, 2]))]
        (lo, hi), = model_uncovered.object_spans(views, 'inputs')
        self.assertEqual((lo, hi), (0x1000, 0x1000 + 3 * 8 + 3 * 2 + 2))


class BuildProfile(unittest.TestCase):
    def build(self, grid, views, records=64, census=None, calib=None, arena=(), ctas=1):
        return model_uncovered.build_profile(kernel(grid), launch(grid, records=records), views,
                                            calib or CALIB, 'missing_template_profile',
                                            evidence(records=records, ctas=ctas, grid=ctas),
                                            census=census, arena=arena)

    def test_a_grid_that_fits_tiles_the_object_privately(self):
        views = [(('m', 'inputs', 'w'), view(0x100000, 65536))]
        profile, summary = self.build([4, 1, 1], views)
        self.assertIn('numeric_modeled_private_tile', summary['policies'])
        profile = model_uncovered.validate(profile)
        # Four CTAs, each with its own disjoint tile of the object.
        used = [entry['address_rules'][0]['cta_x_stride'] for entry in profile['template']]
        self.assertEqual(len(set(used)), 1)
        self.assertEqual(used[0], summary['modeled_read_bytes_per_cta'])

    def test_a_grid_too_large_for_the_object_shares_one_arena(self):
        views = [(('m', 'inputs', 'w'), view(0x100000, 2048))]
        profile, summary = self.build([64, 1, 1], views, records=4096)
        self.assertIn('numeric_modeled_shared_arena', summary['policies'])
        strides = {entry['address_rules'][0]['cta_x_stride'] for entry in profile['template']}
        self.assertEqual(strides, {0})

    def test_every_modeled_issue_stays_inside_its_own_object(self):
        views = [(('m', 'inputs', 'w'), view(0x100000, 65536)),
                 (('m', 'outputs', 'o'), view(0x500000, 300))]
        profile, _ = self.build([8, 1, 1], views, calib=dict(CALIB, read_share=0.5))
        for entry in profile['template']:
            low, high = model_uncovered_width_bounds(entry, entry['address_rules'][0], [8, 1, 1])
            within_read = 0x100000 <= low and high <= 0x100000 + 131072
            within_write = 0x500000 <= low and high <= 0x500000 + 600
            self.assertTrue(within_read or within_write, (hex(low), hex(high)))

    def test_writes_are_modeled_only_from_output_objects(self):
        views = [(('m', 'inputs', 'w'), view(0x100000, 65536)),
                 (('m', 'outputs', 'o'), view(0x500000, 65536))]
        profile, _ = self.build([2, 1, 1], views, calib=dict(CALIB, read_share=0.5))
        directions = {entry['opcode'][:3] for entry in profile['template']}
        self.assertEqual(directions, {'LDG', 'STG'})

    def test_the_budget_caps_an_observed_record_flood(self):
        views = [(('m', 'inputs', 'w'), view(0x100000, 1 << 26))]
        profile, summary = self.build([1, 1, 1], views, records=10 ** 7)
        self.assertLessEqual(summary['issues_per_cta'], model_uncovered.MAX_ISSUES_PER_CTA)
        self.assertTrue(summary['capped_by_budget'])

    def test_a_launch_without_bound_objects_cannot_be_modelled_from_nothing(self):
        with self.assertRaises(ValueError):
            self.build([2, 1, 1], [], records=8)

    def test_an_arena_fallback_is_labelled_as_a_representative_arena(self):
        profile, summary = model_uncovered.build_profile(
            kernel([2, 1, 1]), launch([2, 1, 1]), [], CALIB, 'missing_template_profile',
            evidence(records=8), arena=[(0x800000, 0x800000 + 4096)])
        self.assertEqual(summary['address_basis'], 'representative_persistent_arena')
        self.assertEqual(profile['modeling']['address_basis'], 'representative_persistent_arena')

    def test_a_bound_object_is_modelled_from_the_launch_own_context(self):
        profile, summary = self.build([2, 1, 1], [(('m', 'inputs', 'w'), view(0x100000, 1024))])
        self.assertEqual(summary['address_basis'], 'target launch allocation context')

    def test_every_profile_is_labelled_modeled_and_disclaims_cross_layer_identity(self):
        profile, _ = self.build([2, 1, 1], [(('m', 'inputs', 'w'), view(0x100000, 1024))])
        self.assertEqual(profile['status'], model_uncovered.MODELED_STATUS)
        self.assertEqual(profile['schema'], model_uncovered.SCHEMA)
        self.assertFalse(profile['modeling']['exact_cross_layer_identity_claimed'])
        self.assertEqual(profile['modeling']['mode'], 'numeric_modeled')
        self.assertEqual(profile['modeling']['not_claimed'], model_uncovered.NOT_CLAIMED)
        self.assertNotIn('structural_classes', profile)
        self.assertEqual(profile['modeling']['evidence']['sampled_records'], 64)

    def test_validation_refuses_an_incomplete_lane_sequence(self):
        profile, _ = self.build([2, 1, 1], [(('m', 'inputs', 'w'), view(0x100000, 1024))])
        profile['template'][0]['groups'][0]['pairs'] = ['16:8']
        with self.assertRaises(ValueError):
            model_uncovered.validate(profile)

    def test_validation_refuses_a_rule_without_explicit_xyz_strides(self):
        profile, _ = self.build([2, 1, 1], [(('m', 'inputs', 'w'), view(0x100000, 1024))])
        del profile['template'][0]['address_rules'][0]['cta_z_stride']
        with self.assertRaises(ValueError):
            model_uncovered.validate(profile)


def model_uncovered_width_bounds(entry, rule, grid_dims):
    """Lane bounds of one modeled entry, in the engine's own arithmetic."""
    width = next(bits // 8 for bits in (128, 64, 32, 16, 8) if '.%d' % bits in entry['opcode'])
    offsets = [0]
    for token in entry['groups'][0]['pairs']:
        delta, count = map(int, token.split(':'))
        offsets.extend(offsets[-1] + delta for _ in range(count))
    active = [value for lane, value in enumerate(offsets) if int(entry['mask'], 0) >> lane & 1]
    base = rule['intercept']
    return base + min(active), base + max(active) + width


class RefusalIsStillTheDefault(unittest.TestCase):
    def test_expansion_help_states_both_modes_and_the_default(self):
        import subprocess
        result = subprocess.run([sys.executable, '-B', str(ADAPTER / 'expand_profiles.py'),
                                 '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--model-uncovered', result.stdout)
        self.assertIn('{refuse,modeled}', result.stdout)
        self.assertIn('refuse keeps a class with no admitted template', result.stdout)

    def test_an_unknown_mode_is_refused(self):
        import subprocess
        result = subprocess.run([sys.executable, '-B', str(ADAPTER / 'expand_profiles.py'),
                                 '--sample-output', '/nonexistent', '--layer-bindings', '/nonexistent',
                                 '--output', '/nonexistent', '--model-uncovered', 'guess'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=1)
