"""Native SGLang samples -> existing HBServe PreparedPhaseGenerator.

Thin format/admission adapter, not a new template engine. Rules stay pending
until every declared independent holdout warp matches exact memory semantics.
"""
import collections
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

from memory_projection import project_record, key, need, MODEL_POLICIES, LDG_PREDICATE_ESTIMATE_LABEL
from diagnostics import DiagnosticError, mismatch, training_failure
from storage_windows import StorageWindows

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[1] / 'hbserve-stream'
HBSERVE = Path(os.environ.get('SG_HBSERVE_SOURCE_ROOT', FROZEN / 'sources/hbserve')).resolve()
CODEC = Path(os.environ.get('SG_MEMORYINST_CODEC', FROZEN / 'sources/memgen/hbserve-q8-template-r3/hbserve_memory_template.py')).resolve()
CODEC_SHA = 'c4b90ea4f5739452db5e24222bbc2a6d549f6f8b2e19de59f902500b0d9184be'
GENERATOR = HBSERVE / 'hbserve/traces/_reference/phase_aware_cta_generator.py'
GENERATOR_SHA = '74bc9dbc352dfa41c4c673d37335275b052e3a8a83c5c61fbc61dd2c1fed7570'


def existing_core():
    need(hashlib.sha256(CODEC.read_bytes()).hexdigest() == CODEC_SHA, 'frozen MemoryInst adapter SHA mismatch')
    need(hashlib.sha256(GENERATOR.read_bytes()).hexdigest() == GENERATOR_SHA, 'frozen HBServe generator SHA mismatch')
    if str(HBSERVE) not in sys.path:
        sys.path.insert(0, str(HBSERVE))
    from hbserve.traces._reference.phase_aware_cta_generator import generated_bundle_records
    spec = importlib.util.spec_from_file_location('sglang_reused_memory_template_codec', CODEC)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    return codec, generated_bundle_records


class SampleCollector:
    """Retain only bounded normalized MemoryInst extras, using actual masks/refs."""
    def __init__(self, begin, max_records=100000, model_policy='strict'):
        need(model_policy in MODEL_POLICIES, 'unknown explicit memory model policy')
        need(begin['schema'] == 'SG_KERNEL_SAMPLE_BEGIN_V1', 'sample begin schema')
        need(type(max_records) is int and 0 < max_records <= 1000000, 'sample record budget')
        need(len(begin['grid']) == len(begin['block']) == 3 and
             all(type(v) is int and v > 0 for v in begin['grid'] + begin['block']), 'launch geometry')
        self.warps = (begin['block'][0] * begin['block'][1] * begin['block'][2] + 31) // 32
        need(self.warps <= 32, 'source CTA exceeds 1024 threads')
        grid_size = begin['grid'][0] * begin['grid'][1] * begin['grid'][2]
        fit, hold = begin['fit_ctas'], begin['holdout_ctas']
        need(fit and len(fit) == len(set(fit)) and len(hold) == len(set(hold)) and
             not set(fit) & set(hold) and all(type(c) is int and 0 <= c < grid_size for c in fit + hold),
             'invalid/duplicate/overlapping/out-of-grid sample selection')
        self.begin, self.max_records = begin, max_records
        self.samples = collections.defaultdict(list)
        self.last_ordinal = -1
        self.selected = set(begin['fit_ctas']) | set(begin['holdout_ctas'])
        self.count = 0
        self.async_records = 0
        self.model_policy = model_policy
        self.estimated_sites = set()
        self.sample_candidate_bytes = collections.Counter()
        self.qualified_projection = None
        if model_policy == "validated_ldg_source_predicate":
            from qualified_source_projection import GuardSourceLedger
            from qualified_source_projection import qualification
            if begin['code_sha256']==qualification().scope['code_sha256']:
                self.qualified_projection = GuardSourceLedger(begin)
        self.shared_only_census = collections.Counter()

    def record(self, record):
        need(key(record) == key(self.begin) and record['code_sha256'] == self.begin['code_sha256'], 'sample launch/code mismatch')
        direction, width, mask, ranges, async_read = project_record(record, self.model_policy)
        if len(record['refs']) == 1 and record['refs'][0]['shared_mask']:
            assert mask == 0 and not ranges and not async_read
            self.shared_only_census['records'] += 1
            self.shared_only_census['active_lanes'] += record['effective_mask'].bit_count()
            self.shared_only_census['lane_bytes'] += record['effective_mask'].bit_count() * width
        ordinal = record['original_received_ordinal']
        need(type(ordinal) is int and ordinal > self.last_ordinal,
             'sample delivery ordinal not strictly increasing')
        need(type(record['cta_warp_id']) is int and 0 <= record['cta_warp_id'] < self.warps,
             'sample warp outside actual block')
        need(self.count < self.max_records, 'HBServe sampled extras RAM record bound')
        self.last_ordinal = ordinal
        self.count += 1
        self.async_records += int(async_read)
        if record.get('projection_kind') == 'predicated_global_read_candidate':
            if self.model_policy == 'estimate_ldg_source_predicate':
                self.estimated_sites.add((record['function_id'], record['pc'], record['opcode']))
            self.sample_candidate_bytes['records'] += 1
            self.sample_candidate_bytes['source_mask_gated_read_bytes'] += bin(mask).count('1') * width
            self.sample_candidate_bytes['effective_mask_upper_bound_read_bytes'] += bin(record['effective_mask']).count('1') * width
        if self.qualified_projection is not None:
            self.qualified_projection.record(record, direction, width, mask, ranges)
        gx, gy, gz = self.begin['grid']
        x, y, z = record['cta']
        need(0 <= x < gx and 0 <= y < gy and 0 <= z < gz, 'source CTA outside grid')
        cta = x + gx * (y + gy * z)
        need(cta in self.selected, 'unexpected sampled CTA')
        lanes = []
        for lane in range(32):
            if mask >> lane & 1:
                addr = ranges[len(lanes)]['offset_bytes']
                lanes.append({'lane': lane, 'ref_id': 1 if async_read else 0, 'addr': addr,
                              'is_local': 0, 'local_offset': 0})
        m = {'kernel_id': 0, 'block_id': cta, 'sm_id': record['actual_sm'], 'seq': ordinal,
             'pc': record['pc'], 'opcode': record['opcode'], 'mask': mask if self.qualified_projection is not None else record['effective_mask'],
             'timestamp': ordinal, 'mem_width': width, 'op': ord('R' if direction == 'load' else 'W'),
             'has_space_metadata': 1, 'capture_seq': 0, 'full_clock': record['clock64'],
             'local_warp_owner': 0, 'cta_warp': record['cta_warp_id'], 'function_id': record['function_id'],
             'lanes': lanes}
        self.samples[cta, record['cta_warp_id']].append(m)

    def compile(self, end, qualified_transport, period=1):
        need(qualified_transport['status'] == 'PASS_SAMPLED_TRANSPORT_ONLY', 'producer/source transport gate required')
        row = next((r for r in qualified_transport['kernels'] if r['source_launch_key'] == self.begin['source_launch_key']), None)
        need(row is not None and row['selected_records'] == self.count, 'transport sample count mismatch')
        need(key(end) == key(self.begin) and end['source_closed'] and not end['overflow'] and
             end['selected_records'] == self.count and end['unknown_space_lane_references'] == 0, 'source sample closure')
        chosen = set(self.begin['fit_ctas']) | set(self.begin['holdout_ctas'])
        from cta_entry_contract import validate_entry_proof
        proof=validate_entry_proof(self.begin,end,sorted(set(c for c,_ in self.samples)))
        need(row['entry_proof']==end['entry_proof'],'transport entry proof mismatch')
        self.packetless_ctas=set(proof['packetless_ctas'])
        need(not self.packetless_ctas or all(k in ('CONSTANT','SHARED') or not v for k,v in end['omitted_static_memory_classes'].items()),'uncovered potential L2 client prevents packetless admission')
        need(set(c for c,_ in self.samples)|self.packetless_ctas==chosen,'selected CTA closure')
        need(not self.packetless_ctas or chosen==set(range(self.begin['grid'][0]*self.begin['grid'][1]*self.begin['grid'][2])),'packetless structural class requires complete grid')
        compiled = CompiledNativeTemplate(self, period, end)
        if self.estimated_sites:
            compiled.receipt['memory_model_label'] = LDG_PREDICATE_ESTIMATE_LABEL
            compiled.receipt['status'] = ('PASS_EXACT_CAPTURED_CTA_ANCHORS_FOR_MEMORY_MODEL_ESTIMATE' if compiled.exact else
                                          'PASS_HELDOUT_CTA_MEMORY_MODEL_ESTIMATE')
        return compiled


class CompiledNativeTemplate:
    def __init__(self, collector, period, end):
        self.begin, self.samples = collector.begin, collector.samples
        self.packetless_ctas=collector.packetless_ctas
        self.estimated_sites = collector.estimated_sites
        self.codec, self.generator = existing_core()
        self.grid_size = self.begin['grid'][0] * self.begin['grid'][1] * self.begin['grid'][2]
        fit, hold = set(self.begin['fit_ctas']), set(self.begin['holdout_ctas'])
        need(not fit & hold, 'fit and holdout overlap')
        self.exact = fit | hold == set(range(self.grid_size))
        self.receipt = {'schema': 'SGLANG_HBSERVE_NATIVE_TEMPLATE_ADMISSION_V1',
            'source_launch_key': self.begin['source_launch_key'], 'code_sha256': self.begin['code_sha256'],
            'source_grid': self.begin['grid'], 'source_block': self.begin['block'],
            'fit_ctas': sorted(fit), 'holdout_ctas': sorted(hold), 'sampled_records': collector.count,
            'async_global_source_records': collector.async_records, 'period_candidate': period,
            'engine': 'existing_HBServe_PreparedPhaseGenerator/generated_bundle_records',
            'MemoryInst_codec_sha256': CODEC_SHA, 'exact_anchor': self.exact,
            'HBServe_generator_sha256': GENERATOR_SHA,
            'semantics_compared': ['PC', 'opcode', 'effective_guard_mask', 'width', 'read_or_write',
                'global_projection_reference_and_lane', 'virtual_address', 'function', 'CTA_warp', 'per_warp_order'],
            'whole_hardware_instruction_trace': False, 'physical_SM_or_timing_reproduced': False,
            'omitted_static_memory_classes': end['omitted_static_memory_classes'],
            'address_object_domain': ('observed_storage_spans_with_explicit_opaque_fallback' if getattr(collector, 'storage_windows', None) is not None else 'unbound_process_VA_transport_fallback'),
            'compact_window_table_values': 'absolute_base_addresses',
            'cross_layer_expansion_qualified': False, 'ready_for_modeled_CTA_expansion': False}
        self.receipt.update(model_policy=collector.model_policy,
            strict_projection_compatible=not bool(self.estimated_sites),
            source_predicate_hardware_semantics_validated=False,
            sampled_candidate_traffic=dict(collector.sample_candidate_bytes),
            estimated_static_memory_sites=len(self.estimated_sites))
        if self.exact:
            self.receipt['address_object_domain'] = 'exact_original_sample_VAs_no_compact_mapping'
            self.receipt.update(status='PASS_EXACT_CAPTURED_CTA_ANCHORS', independent_holdout_prediction=False,
                                ready_for_modeled_CTA_expansion=True)
            return
        core_fit = {0, 1} if period == 1 else {0, 1, 4} if period == 4 else None
        need(core_fit is not None and core_fit <= fit, 'fit plan lacks required core coordinates for candidate period')
        need(hold, 'independent heldout CTA required')
        if period == 4:
            need({2, 3} <= {c % 4 for c in hold}, 'period4 untrained phases2/3 need independent holdouts')
        # Superset fit plans may include shape boundaries and interior probes.
        # The existing candidate engine needs only its core coordinates. Extra
        # declared fit samples validate that candidate without being relabeled
        # as independent holdouts.
        training = {k: v for k, v in self.samples.items() if k[0] in core_fit}
        try:
            storage_windows = getattr(collector, 'storage_windows', None)
            if storage_windows is None:
                self.prepared, self.source, self.bundles, self.extras, windows, self.rules = self.codec.prepare(
                    training, 0, (self.grid_size, 1, 1), period)
                self.windows = {index: window << 32 for index, window in windows.items()}
            else:
                need(type(storage_windows) is StorageWindows, 'unsupported compact storage mapper implementation')
                need(storage_windows.begin == self.begin, 'compact storage mapping belongs to another source begin')
                self.prepared, self.source, self.bundles, self.extras, self.windows, self.rules = storage_windows.prepare(
                    self.codec, training, 0, (self.grid_size, 1, 1), period)
                self.receipt['storage_window_mapping'] = storage_windows.receipt()
        except (KeyError, TypeError, ValueError) as error:
            detail = (training_failure(training, core_fit, str(error), self.codec) if storage_windows is None else
                      dict(stage='STORAGE_COMPACT_CORE_TEMPLATE_BUILD', reason=str(error), localized=False,
                           source_coordinates='real_VA_preserved_outside_temporary_compact_encoding',
                           full_dynamic_record_saved=False))
            raise DiagnosticError(str(error), detail) from error
        tested_warps = 0
        extra_fit_warps = 0
        for cta in sorted((fit - core_fit) | hold):
            sample_role = 'heldout' if cta in hold else 'additional fit'
            expected = {w: frames for (c, w), frames in self.samples.items() if c == cta}
            actual = collections.defaultdict(list)
            for m in self._cta(cta):
                actual[m['cta_warp']].append(m)
            if set(expected) != set(actual):
                raise DiagnosticError(sample_role + ' warp-program coverage differs',
                    dict(stage='CTA_VALIDATION', sample_role=sample_role, target_cta_linear=cta,
                         expected_warps=sorted(expected), generated_warps=sorted(actual)))
            for warp in expected:
                want, got = expected[warp], actual[warp]
                if len(want) != len(got):
                    raise DiagnosticError(sample_role + ' instruction counts differ',
                        dict(stage='CTA_VALIDATION', sample_role=sample_role, target_cta_linear=cta, cta_warp=warp,
                             expected_instruction_count=len(want), generated_instruction_count=len(got)))
                for ordinal, (a, b) in enumerate(zip(want, got)):
                    if self.codec.semantic_bytes(a) != self.codec.semantic_bytes(b):
                        raise DiagnosticError(sample_role + ' PC/opcode/mask/width/ref/lane/address/per-warp order mismatch',
                            dict(mismatch(a, b, stage='CTA_VALIDATION', cta=cta, warp=warp, ordinal=ordinal),
                                 sample_role=sample_role))
                tested_warps += int(cta in hold)
                extra_fit_warps += int(cta not in hold)
        self.receipt.update(status='PASS_HELDOUT_CTA_MEMORY_SEMANTICS_MODEL_ESTIMATE',
                            independent_holdout_prediction=True, heldout_warp_programs=tested_warps,
                            core_fit_ctas=sorted(core_fit), additional_fit_validation_ctas=sorted(fit - core_fit),
                            additional_fit_warp_programs=extra_fit_warps,
                            ready_for_modeled_CTA_expansion=True, translation_rules=len(self.rules),
                            compact_template_bytes=len(self.source.getbuffer()))

    def _cta(self, cta):
        need(0 <= cta < self.grid_size, 'target CTA outside admitted source grid')
        if self.exact:
            for (source_cta, warp), frames in sorted(self.samples.items()):
                if source_cta == cta:
                    yield from (copy.deepcopy(m) for m in frames)
            return
        translated = {}
        for bundle, records in self.generator(prepared=self.prepared, source=self.source,
                                              kernel_ordinal=0, target_cta_x=cta):
            index = bundle['extra_index']
            m = copy.deepcopy(self.extras[index])
            need(len(records) == len(m['lanes']), 'HBServe lane cardinality drift')
            for lane, record in zip(m['lanes'], records):
                obj, offset, kid, width, op, flags = record
                need(kid == 0 and width == m['mem_width'] and op == int(m['op'] == ord('W')), 'HBServe request metadata drift')
                lane['addr'] = self.windows[obj] + offset
            translated[index] = m
        for i, original in enumerate(self.extras):
            if i in translated:
                m = translated[i]
            else:
                need(not original['lanes'], 'missing nonempty HBServe bundle')
                m = copy.deepcopy(original)
            m['block_id'] = cta
            yield m

    def iter_memory_instructions(self, *, kernel_id, sm_for_cta, target_binding=None):
        """Stream same-grid estimates. Other-layer addresses require a verified mapper.

        target_binding is an external, evidence-qualified object binding, not a
        module association: {status,source_code_sha256,target_code_sha256,
        phase,grid,block,evidence_refs,map_address}. Candidate mappings reject.
        """
        need(self.receipt['ready_for_modeled_CTA_expansion'], 'template not admitted')
        if target_binding is not None:
            need(target_binding['status'] == 'PASS_SAMPLE_BOUND_ROOT_RELOCATION', 'cross-layer binding is still a candidate')
            need(target_binding['source_code_sha256'] == target_binding['target_code_sha256'] == self.begin['code_sha256'],
                 'cross-layer compiled code regime differs')
            need(target_binding['phase'] == self.begin['phase'] and target_binding['grid'] == self.begin['grid'] and
                 target_binding['block'] == self.begin['block'], 'cross-phase/shape/grid template transfer not admitted')
            need(target_binding['evidence_refs'] and callable(target_binding['map_address']), 'verified address mapping evidence required')
        seq = 0
        for cta in range(self.grid_size):
            sm = sm_for_cta(cta)
            need(type(sm) is int and 0 <= sm < 48, 'explicit modeled SM outside profile')
            for m in self._cta(cta):
                if target_binding is not None:
                    for lane in m['lanes']:
                        address, proof = target_binding['map_address'](lane['addr'], m['mem_width'])
                        need(proof and type(address) is int and 0 < address < address + m['mem_width'] < 1 << 64,
                             'unmapped/ambiguous/out-of-range cross-layer address')
                        lane['addr'] = address
                m.update(kernel_id=kernel_id, sm_id=sm, seq=seq, timestamp=seq,
                         capture_seq=0, full_clock=0, local_warp_owner=0)
                seq += 1
                yield m
