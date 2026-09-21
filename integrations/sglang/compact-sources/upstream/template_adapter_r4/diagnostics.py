"""Small bounded failure witnesses. Never serialize per-lane address arrays."""
import collections
import hashlib
import json


class DiagnosticError(ValueError):
    def __init__(self, message, detail):
        super().__init__(message)
        self.diagnostic = detail


def record_witness(record):
    fields = ['source_launch_key', 'function_id', 'pc', 'opcode', 'opcode_id', 'width',
              'projection_kind', 'is_load', 'is_store', 'ref_count', 'cta_warp_id',
              'original_received_ordinal', 'selected_ordinal', 'active_mask', 'predicate_mask',
              'effective_mask', 'transfer_width', 'transfer_policy', 'source_control_kind', 'source_read_mask']
    out = {k: record.get(k) for k in fields}
    out['cta'] = record.get('cta')
    out['reference_space_masks'] = [{k: ref.get(k) for k in ['global_mask', 'local_mask', 'shared_mask']}
                                    for ref in record.get('refs', [])[:2]]
    out['dynamic_addresses_in_witness'] = False
    return out


def memory_witness(m):
    fields = ['pc', 'opcode', 'mask', 'mem_width', 'op', 'has_space_metadata', 'cta_warp', 'function_id']
    refs, h = collections.Counter(), hashlib.sha256()
    for lane in m['lanes']:
        refs[lane['ref_id']] |= 1 << lane['lane']
        h.update(json.dumps([lane['lane'], lane['ref_id'], lane['addr'], lane['is_local'],
                             lane['local_offset']], separators=(',', ':')).encode())
    return dict({k: m[k] for k in fields}, lane_references=len(m['lanes']),
                reference_lane_masks=[dict(ref_id=k, mask=v) for k, v in sorted(refs.items())],
                lane_address_semantics_sha256=h.hexdigest(), dynamic_addresses_in_witness=False)


def mismatch(expected, generated, *, stage, cta, warp, ordinal):
    fields = ['pc', 'opcode', 'mask', 'mem_width', 'op', 'has_space_metadata', 'cta_warp', 'function_id']
    changed = [k for k in fields if expected[k] != generated[k]]
    lane_diffs = []
    if len(expected['lanes']) != len(generated['lanes']):
        changed.append('lane_reference_count')
    for i, (left, right) in enumerate(zip(expected['lanes'], generated['lanes'])):
        diff = [k for k in ['lane', 'ref_id', 'addr', 'is_local', 'local_offset'] if left[k] != right[k]]
        if diff:
            changed.extend('lane.' + k for k in diff)
            if len(lane_diffs) < 4:
                row = dict(lane_reference_index=i, expected_lane=left['lane'], generated_lane=right['lane'],
                           expected_ref=left['ref_id'], generated_ref=right['ref_id'], changed_fields=diff)
                if 'addr' in diff:
                    row['generated_minus_expected_address_bytes'] = right['addr'] - left['addr']
                lane_diffs.append(row)
    return dict(stage=stage, target_cta_linear=cta, cta_warp=warp, warp_program_ordinal=ordinal,
                changed_fields=sorted(set(changed)), expected=memory_witness(expected),
                generated=memory_witness(generated), first_lane_differences=lane_diffs,
                maximum_reported_lane_differences=4, full_dynamic_record_saved=False)


def training_failure(samples, core, reason, codec):
    """Localize existing-codec preconditions without replacing its fit logic."""
    detail = dict(stage='CORE_TEMPLATE_BUILD', core_fit_ctas=sorted(core), reason=reason,
                  localized=False, full_dynamic_record_saved=False)
    base_keys = sorted((c, w) for c, w in samples if c == 0)
    for target in sorted(core - {0}):
        expected_warps = {w for c, w in samples if c == 0}
        actual_warps = {w for c, w in samples if c == target}
        if expected_warps != actual_warps:
            return dict(detail, localized=True, target_cta_linear=target,
                        expected_warps=sorted(expected_warps), actual_warps=sorted(actual_warps))
        for _, warp in base_keys:
            left, right = samples[0, warp], samples[target, warp]
            if len(left) != len(right):
                return dict(detail, localized=True, target_cta_linear=target, cta_warp=warp,
                            expected_instruction_count=len(left), actual_instruction_count=len(right))
            for ordinal, (a, b) in enumerate(zip(left, right)):
                if codec.semantic_bytes(a, False) != codec.semantic_bytes(b, False):
                    return dict(detail, localized=True, first_difference=mismatch(a, b, stage='CORE_FIT_SEMANTICS',
                                cta=target, warp=warp, ordinal=ordinal))
                deltas = {}
                for lane_id, (x, y) in enumerate(zip(a['lanes'], b['lanes'])):
                    if x['addr'] >> 32 != y['addr'] >> 32:
                        return dict(detail, localized=True, target_cta_linear=target, cta_warp=warp,
                            warp_program_ordinal=ordinal, lane_reference_index=lane_id, address_window_changed=True,
                            source=memory_witness(a), target=memory_witness(b))
                    window, delta = x['addr'] >> 32, y['addr'] - x['addr']
                    if window in deltas and deltas[window] != delta:
                        return dict(detail, localized=True, target_cta_linear=target, cta_warp=warp,
                            warp_program_ordinal=ordinal, lane_reference_index=lane_id,
                            lane_translation_uniform=False, first_lane_delta_bytes=deltas[window],
                            differing_lane_delta_bytes=delta, source=memory_witness(a), target=memory_witness(b))
                    deltas[window] = delta
    return detail


class RejectionCensus:
    """Bounded rejected instruction classes and first scalar witnesses only."""
    def __init__(self, max_classes=32, max_witnesses=4):
        self.max_classes, self.max_witnesses = max_classes, max_witnesses
        self.classes, self.witnesses = {}, []
        self.classified_records, self.rejected_records, self.unbucketed_rejections = 0, 0, 0

    def observe(self, record, error=None):
        self.classified_records += 1
        if error is None:
            return
        self.rejected_records += 1
        fields = ['opcode', 'width', 'ref_count', 'is_load', 'is_store', 'projection_kind',
                  'transfer_width', 'transfer_policy', 'source_control_kind']
        signature = {k: record.get(k) for k in fields}
        signature['reason'] = str(error)
        key = json.dumps(signature, sort_keys=True)
        new_class = key not in self.classes
        if key not in self.classes and len(self.classes) < self.max_classes:
            self.classes[key] = dict(signature, records=0, active_lanes=0, effective_guard_lanes=0,
                source_mask_records=0, source_read_lanes=0, source_predicate_off_under_guard_lanes=0,
                source_zero_read_records=0, source_read_mask_outside_guard_records=0)
        if key in self.classes:
            bucket = self.classes[key]
            bucket['records'] += 1
            for field, count_field in [('active_mask', 'active_lanes'), ('effective_mask', 'effective_guard_lanes')]:
                mask = record.get(field)
                if type(mask) is int and 0 <= mask < 1 << 32:
                    bucket[count_field] += bin(mask).count('1')
            if record.get('projection_kind') in ['async_global_read', 'predicated_global_read_candidate']:
                source, guard = record.get('source_read_mask'), record.get('effective_mask')
                if type(source) is int and type(guard) is int and 0 <= source < 1 << 32 and 0 <= guard < 1 << 32:
                    bucket['source_mask_records'] += 1
                    bucket['source_read_lanes'] += bin(source).count('1')
                    bucket['source_predicate_off_under_guard_lanes'] += bin(guard & ~source).count('1')
                    bucket['source_zero_read_records'] += int(source == 0)
                    bucket['source_read_mask_outside_guard_records'] += int(bool(source & ~guard))
        else:
            self.unbucketed_rejections += 1
        if new_class and len(self.witnesses) < self.max_witnesses:
            self.witnesses.append(dict(reason=str(error), record=record_witness(record)))

    def receipt(self):
        return dict(classified_records=self.classified_records, rejected_records=self.rejected_records,
                    rejection_classes=list(self.classes.values()), unbucketed_rejections=self.unbucketed_rejections,
                    witnesses=self.witnesses, limits=dict(classes=self.max_classes, witnesses=self.max_witnesses),
                    full_dynamic_records_saved=False)
