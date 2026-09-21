#!/usr/bin/env python3
"""Stream sampled native memory records into a bounded GTSim global projection.

Consumes the sampler's normalized begin/record/end JSONL. Raw records and the
generated DAG remain in RAM. Only aggregate results are emitted/saved.
"""
import argparse
import bisect
import collections
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

MAX_U64 = (1 << 64) - 1
MODEL_POLICIES = ('strict', 'estimate_ldg_source_predicate', 'validated_ldg_source_predicate')
LDG_PREDICATE_ESTIMATE_FORMS = {
    'LDG.E.LTC128B.64.STRONG.GPU': 8,
    'LDG.E.LTC128B.128.STRONG.GPU': 16,
    'LDG.E.LTC128B.128.CONSTANT': 16,
}
LDG_PREDICATE_ESTIMATE_LABEL = 'ESTIMATED_SOURCE_PREDICATE_GATING_NOT_HARDWARE_VALIDATED'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def uint(value, bits=64):
    need(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 1 << bits,
         'invalid unsigned integer')
    return value


def key(message):
    return json.dumps(message['source_launch_key'], sort_keys=True)


class Attribution:
    """Optional SAME-PROCESS interval matches; overlapping lifetimes stay unknown."""
    def __init__(self, binding=None):
        self.binding = binding
        self.roots = sorted((binding or {}).get('roots', []), key=lambda r: r['base_address'])
        self.bases = [r['base_address'] for r in self.roots]
        self.ends, end = [], 0
        for r in self.roots:
            end = max(end, r['base_address'] + r['storage_nbytes'])
            self.ends.append(end)

    def match(self, begin, address, width):
        if (not self.binding or not isinstance(begin.get('process_id'), int) or
                begin['process_id'] <= 0 or begin['process_id'] != self.binding.get('source_process_id')):
            return 'opaque', []
        i, candidates = bisect.bisect_right(self.bases, address) - 1, []
        while i >= 0 and self.ends[i] > address:
            r = self.roots[i]
            if r['base_address'] <= address and address + width <= r['base_address'] + r['storage_nbytes']:
                candidates.append(r['id'])
            i -= 1
        return ('unique_recorded_interval' if len(candidates) == 1 else
                'ambiguous_recorded_intervals' if candidates else 'opaque'), candidates


def project_record(record, model_policy='strict'):
    need(model_policy in MODEL_POLICIES, 'unknown explicit memory model policy')
    need(record['schema'] == 'SG_MEMORY_PROJECTION_RECORD_V1', 'record schema')
    refs = record['refs']
    need(record['ref_count'] == len(refs), 'reference count mismatch')
    active = uint(record['active_mask'], 32)
    predicate = uint(record['predicate_mask'], 32)
    effective = uint(record['effective_mask'], 32)
    need(effective & ~active == 0, 'effective mask outside active lanes')
    kind = record.get('projection_kind', 'ordinary')
    if kind == 'predicated_global_read_candidate':
        if model_policy == 'validated_ldg_source_predicate':
            from qualified_source_projection import qualification
            qualification().record(record)
        need(model_policy in ('estimate_ldg_source_predicate', 'validated_ldg_source_predicate'),
             'unsupported/unqualified projection kind (including predicated read candidate)')
        need(record.get('source_control_kind') == 'ldg_predicate_candidate' and len(refs) == 1,
             'LDG predicate estimate requires exact one-ref static source-control candidate')
        width = LDG_PREDICATE_ESTIMATE_FORMS.get(record['opcode'])
        need(width is not None and record['width'] == width and record.get('transfer_width') == width and
             record.get('transfer_policy') == 2, 'LDG predicate estimate opcode/width/policy outside explicit whitelist')
        need(record['is_load'] is True and record['is_store'] is False, 'LDG predicate estimate requires load-only static direction')
        ref = refs[0]
        source_mask = uint(record['source_read_mask'], 32)
        need(effective == active & predicate and source_mask & ~effective == 0 and
             ref['global_mask'] == effective and ref['local_mask'] == ref['shared_mask'] == 0,
             'LDG predicate estimate source/guard/global-reference mask contract')
        mask, direction = source_mask, 'load'
    elif kind == 'async_global_read':
        need(len(refs) == 2, 'validated LDGSTS requires shared-destination/global-source refs')
        need(record['transfer_width'] in (4, 8, 16) and record['transfer_policy'] in (0, 1),
             'unsupported LDGSTS transfer contract')
        need(record.get('source_control_kind') == 'validated_sm89_ldgsts' and
             (record['transfer_policy'] == 0 or record['transfer_width'] == 16),
             'LDGSTS static source-control witness missing/unsupported')
        need(record['opcode'].split('.')[0] == 'LDGSTS', 'async projection outside validated LDGSTS opcode')
        need(refs[0]['global_mask'] == 0 and refs[0]['local_mask'] == 0 and
             refs[1]['local_mask'] == 0 and refs[1]['shared_mask'] == 0, 'mixed LDGSTS memory reference spaces')
        source_mask = uint(record['source_read_mask'], 32)
        need(source_mask & ~effective == 0 and refs[0]['shared_mask'] == effective and
             refs[1]['global_mask'] == effective, 'LDGSTS effective source/destination mask contract')
        ref, width, direction = refs[1], record['transfer_width'], 'load'
        mask = uint(ref['global_mask'], 32) & source_mask
    else:
        need(kind in ('ordinary', 'global_memory', 'ordinary_memory_operands'),
             'unsupported/unqualified projection kind (including predicated read candidate)')
        need(len(refs) == 1, 'ordinary multi-reference instruction not admitted')
        need(isinstance(record['is_load'], bool) and isinstance(record['is_store'], bool) and
             record['is_load'] != record['is_store'], 'RMW/atomic or directionless memory op not admitted')
        need(record['opcode'].split('.')[0] in ('LDG', 'STG', 'LD', 'ST'),
             'ordinary opcode not admitted as simple load/store')
        ref, width = refs[0], record['width']
        need(width in (1, 2, 4, 8, 16), 'unsupported memory width')
        if ref['shared_mask']:
            shared = uint(ref['shared_mask'], 32)
            need(record['opcode'].split('.')[0] in ('LD', 'ST') and
                 ref['global_mask'] == ref['local_mask'] == 0 and
                 effective == active & predicate and shared == effective,
                 'shared-only generic instruction requires all effective lanes proven shared')
            need(len(ref['addresses']) == 32, 'exactly32 shared lane addresses required')
            for lane in range(32):
                if shared >> lane & 1: uint(ref['addresses'][lane])
            return ('load' if record['is_load'] else 'store'), width, 0, [], False
        need(ref['local_mask'] == 0, 'local ordinary reference remains outside global projection')
        direction = 'load' if record['is_load'] else 'store'
        mask = uint(ref['global_mask'], 32)
        need(mask & ~(effective & predicate) == 0, 'global mask outside effective predicate')
    need(len(ref['addresses']) == 32, 'exactly32 lane addresses required')
    ranges = []
    for lane in range(32):
        if mask & (1 << lane):
            address = uint(ref['addresses'][lane])
            need(address > 0 and address <= MAX_U64 - width, 'null/overflowing active memory address')
            ranges.append({'offset_bytes': address, 'byte_count': width})
    return direction, width, mask, ranges, kind == 'async_global_read'


class StreamLowerer:
    def __init__(self, *, bridge=None, max_records=50000, timeout_seconds=120, attribution=None,
                 max_deferred_bytes=64 << 20):
        self.bridge, self.max_records, self.timeout_seconds = bridge, max_records, timeout_seconds
        self.attribution = attribution or Attribution()
        self.pending = {}
        self.jobs, self.results = [], []
        self.deferred_bytes, self.max_deferred_bytes = 0, max_deferred_bytes

    def accept(self, message):
        schema = message.get('schema')
        if schema == 'SG_KERNEL_SAMPLE_BEGIN_V1':
            k = key(message)
            need(k not in self.pending, 'duplicate open kernel sample')
            need(not self.pending and len(self.results) < 512, 'nested/excessive sample kernels')
            for dims in (message['grid'], message['block']):
                need(len(dims) == 3 and all(isinstance(n, int) and n > 0 for n in dims), 'invalid observed launch geometry')
            need(math.prod(message['block']) <= 1024, 'CUDA block exceeds1024 threads')
            need(len(message['code_sha256']) == 64, 'compiled code identity required')
            selected = message['fit_ctas'] + message['holdout_ctas']
            need(message['fit_ctas'] and len(selected) <= 4096 and len(set(selected)) == len(selected),
                 'nonempty disjoint bounded fit/holdout CTA plan required')
            chosen = set(message['fit_ctas']) | set(message['holdout_ctas'])
            need(all(isinstance(c, int) and 0 <= c < math.prod(message['grid']) for c in chosen), 'sample CTA outside launch')
            self.pending[k] = {'begin': message, 'records': [], 'received_selected_records': 0,
                               'errors': [], 'chosen': chosen, 'seen_ordinals': set()}
            return None
        if schema == 'SG_MEMORY_PROJECTION_RECORD_V1':
            k = key(message)
            need(k in self.pending, 'record without kernel begin')
            state = self.pending[k]
            state['received_selected_records'] += 1
            if state['received_selected_records'] > self.max_records:
                if not state['errors']:
                    state['errors'].append('bounded projection record budget exceeded')
                return None
            try:
                ordinal = uint(message['original_received_ordinal'])
                need(ordinal not in state['seen_ordinals'], 'duplicate original received ordinal')
                state['seen_ordinals'].add(ordinal)
                need(message['code_sha256'] == state['begin']['code_sha256'], 'record/launch compiled code identity mismatch')
                cta = message['cta']
                grid = state['begin']['grid']
                need(len(cta) == 3 and all(isinstance(c, int) and 0 <= c < n for c, n in zip(cta, grid)), 'CTA coordinates outside launch')
                flat = cta[0] + grid[0] * (cta[1] + grid[1] * cta[2])
                need(flat in state['chosen'], 'record CTA not selected by begin')
                need(0 <= uint(message['cta_warp_id'], 32) < math.ceil(math.prod(state['begin']['block']) / 32), 'warp outside CTA')
                projection = project_record(message)
                state['records'].append((flat, message, projection))
            except (ValueError, KeyError, TypeError) as error:
                state['errors'].append(str(error))
            return None
        if schema == 'SG_KERNEL_SAMPLE_END_V1':
            k = key(message)
            need(k in self.pending, 'end without kernel begin')
            result = self.finish(self.pending.pop(k), message)
            self.results.append(result)
            return result
        # Other sampler metadata remains owned by its producer. Explicitly
        # reject misspelled data/control records rather than dropping them.
        need(schema not in (None, '') and not schema.startswith(('SG_MEMORY_PROJECTION_', 'SG_KERNEL_SAMPLE_')),
             'unknown projection protocol message')
        return None

    def finish(self, state, end):
        begin, errors = state['begin'], state['errors']
        def check(ok, message):
            if not ok:
                errors.append(message)
        check(end['source_closed'] is True and end['overflow'] is False, 'sample source did not close without overflow')
        check(end['pushed_records'] == end['received_records'], 'sampler push/receive mismatch')
        check(end['selected_records'] == state['received_selected_records'], 'selected record protocol count mismatch')
        if end['omitted_records'] is None:
            check(end.get('whole_kernel_dynamic_census') is False and end['all_memory_active_ctas_seen_count'] is None,
                  'device CTA filter must declare unavailable full dynamic census')
            check(end['received_records'] == end['selected_records'], 'device-selected receive ledger mismatch')
        else:
            check(end['received_records'] == end['selected_records'] + end['omitted_records'], 'selected/omitted receive ledger mismatch')
        check(end['unknown_space_lane_references'] == 0, 'unresolved memory-space lanes')
        check(math.prod(begin['block']) % 32 == 0, 'current native bridge does not admit partial-warp blocks')
        if errors:
            return {'schema': 'SGLANG_NATIVE_MEMORY_PROJECTION_RESULT_V1', 'status': 'UNLOWERED',
                    'source_launch_key': begin['source_launch_key'], 'reasons': sorted(set(errors)),
                    'selected_record_count': state['received_selected_records'], 'GTSim_executed': False}
        groups, counters = collections.defaultdict(list), collections.Counter()
        ownership = collections.Counter()
        for flat, record, projection in sorted(state['records'], key=lambda item: item[1]['original_received_ordinal']):
            direction, width, mask, ranges, is_async = projection
            counters['accepted_records'] += 1
            counters['async_global_read_records'] += int(is_async)
            if not ranges:
                counters['zero_global_source_records'] += 1
                continue
            counters['global_' + ('read' if direction == 'load' else 'write') + '_requested_bytes'] += len(ranges) * width
            for region in ranges:
                match, _ = self.attribution.match(begin, region['offset_bytes'], width)
                ownership[match + '_lane_references'] += 1
                ownership[match + '_requested_bytes'] += width
            groups[flat].append((record, projection))
        source_ctas = sorted(groups)
        selected_seen = set(end['selected_ctas_seen'])
        check(selected_seen == state['chosen'], 'planned selected CTA coverage incomplete')
        check(set(flat for flat, _, _ in state['records']) == selected_seen, 'selected CTA seen ledger mismatch')
        if errors:
            return {'schema': 'SGLANG_NATIVE_MEMORY_PROJECTION_RESULT_V1', 'status': 'UNLOWERED',
                    'source_launch_key': begin['source_launch_key'], 'reasons': sorted(set(errors)), 'GTSim_executed': False}
        ctalist = []
        for compact, original in enumerate(source_ctas):
            nodes, previous = [], {}
            for record, projection in groups[original]:
                direction, width, mask, ranges, is_async = projection
                warp = record['cta_warp_id']
                node_id = len(nodes)
                nodes.append({'id': node_id, 'warp': warp, 'op': direction, 'dtype': 'INT8',
                    'tile': {'dims': [1, len(ranges) * width], 'offs': [0, 0]},
                    'depends_on': [], 'issue_depends_on': [previous[warp]] if warp in previous else [],
                    'root': 1, 'subops': [{'ranges': ranges, 'requested_bytes': len(ranges) * width}],
                    'metadata': {'source_CTA': original, 'function_id': record['function_id'], 'PC': record['pc'],
                                 'opcode': record['opcode'], 'memory_width_bytes': width, 'global_source_mask': mask,
                                 'original_received_ordinal': record['original_received_ordinal'],
                                 'actual_SM': record['actual_sm'], 'projection_kind': record.get('projection_kind', 'ordinary'),
                                 'async_shared_destination_unmodeled': is_async}})
                previous[warp] = node_id
            ctalist.append({'id': compact, 'source_original_CTA': original, 'nodes': nodes})
        full_grid_selected = state['chosen'] == set(range(math.prod(begin['grid'])))
        omission_classes = end['omitted_static_memory_classes']
        qualification = {'observed_global_addresses_and_lane_multiplicity_preserved': True,
                         'source_grid': begin['grid'], 'source_block': begin['block'],
                         'compact_fragment_to_original_CTA': source_ctas, 'full_grid_planned_selected': full_grid_selected,
                         'global_active_CTA_count_reported': end['all_memory_active_ctas_seen_count'],
                         'selected_CTA_source_closed': True, 'omitted_static_memory_classes': omission_classes,
                         'sample_transport_qualification': 'PENDING_PRODUCER_EXIT_AND_SOURCE_IDENTITY',
                         'complete_all_memory_instruction_domain': False,
                         'sampled_fragment_not_full_model_or_layer': True,
                         'cross_layer_expansion_performed': False, 'same_warp_observed_issue_order_preserved': True,
                         'true_register_and_cross_warp_dependencies_known': False,
                         'compute_shared_async_destination_and_timing_modeled': False,
                         'CTA_SM_mapping': 'compact_CTA_mod_48_model_not_observed_SM',
                         'address_mapping': 'process_virtual_address_domain_carrier_not_allocation_or_physical_address',
                         'cache_initial_state': 'independent_fragment_model_cold',
                         'kernel_operands_inferred_from_module_association': False}
        result = {'schema': 'SGLANG_NATIVE_MEMORY_PROJECTION_RESULT_V1',
                  'status': 'GLOBAL_MEMORY_FRAGMENT_LOWERED_NOT_EXECUTED',
                  'source_launch_key': begin['source_launch_key'], 'source_kernel_name': begin['kernel_name'],
                  'code_sha256': begin['code_sha256'], 'selected_record_count': state['received_selected_records'],
                  'counts': dict(counters), 'object_attribution': dict(ownership), 'qualification': qualification,
                  'GTSim_executed': False}
        if not ctalist:
            result['status'] = 'NO_ACTIVE_GLOBAL_MEMORY_IN_SELECTED_RECORDS'
            return result
        payload = {'schema': 'SGLANG_GTSIM_KERNEL_DAG_V1',
                   'provenance': {'producer': 'native_sample_global_memory_projection',
                                  'source_launch_key': begin['source_launch_key'], 'qualification': qualification},
                   'execution_contract': {'kernel_order': 'explicit_sequential', 'cross_stream_overlap': False},
                   'roots': [{'id': 1, 'name': 'opaque_process_virtual_address_domain', 'base': 0, 'bytes': MAX_U64}],
                   'kernels': [{'id': 'fragment', 'symbol': begin['kernel_name'], 'variant_id': begin['code_sha256'],
                                'source_identity': {'sample_begin': begin, 'sample_end': end},
                                'correlation': {'source_launch_key': begin['source_launch_key']},
                                'phase': begin['phase'], 'layer': begin['layer_id'], 'stream': begin['stream_u64'],
                                'mapping': 'modeled_cta_mod_sm', 'grid': [len(ctalist), 1, 1],
                                'block': begin['block'], 'ctas': ctalist}]}
        raw = json.dumps(payload, separators=(',', ':')).encode()
        result['in_memory_DAG_bytes'] = len(raw)
        result['in_memory_DAG_sha256'] = hashlib.sha256(raw).hexdigest()
        result['in_memory_DAG_node_count'] = sum(len(c['nodes']) for c in ctalist)
        if self.bridge and self.deferred_bytes + len(raw) <= self.max_deferred_bytes:
            self.jobs.append((result, raw))
            self.deferred_bytes += len(raw)
            result['status'] = 'GLOBAL_FRAGMENT_AWAITING_TRANSPORT_QUALIFICATION'
        elif self.bridge:
            result['status'] = 'GLOBAL_FRAGMENT_NATIVE_QUEUE_BUDGET_EXCEEDED'
        return result

    def finalize_transport(self, receipt):
        """Call only after read_stream, successful producer exit and identity gates.

        This is intentionally outside inline sample callbacks: native execution
        must not block the capture pipe while the GPU producer holds its lease.
        """
        need(not self.pending, 'cannot finalize with open sample kernel')
        need(receipt['status'] == 'PASS_SAMPLED_TRANSPORT_ONLY', 'producer/observer/source identity gate not passed')
        admitted = {json.dumps(k['source_launch_key'], sort_keys=True): k for k in receipt['kernels']}
        for result in self.results:
            launch = json.dumps(result['source_launch_key'], sort_keys=True)
            need(launch in admitted, 'lowered source launch absent in transport receipt')
            if 'selected_record_count' in result:
                need(admitted[launch]['selected_records'] == result['selected_record_count'], 'transport/projected selected count mismatch')
            if 'qualification' in result:
                result['qualification']['sample_transport_qualification'] = receipt['status']
        for result, raw in self.jobs:
            # All addresses and per-node material remain only in this bounded
            # RAM buffer and the pipe connected to the native executable.
            payload = json.loads(raw)
            payload['provenance']['qualification']['sample_transport_qualification'] = receipt['status']
            raw = json.dumps(payload, separators=(',', ':')).encode()
            result['in_memory_DAG_bytes'] = len(raw)
            result['in_memory_DAG_sha256'] = hashlib.sha256(raw).hexdigest()
            try:
                completed = subprocess.run([str(self.bridge), '/dev/stdin'], input=raw,
                                           capture_output=True, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                result['status'] = 'LOWERED_NATIVE_EXECUTION_TIMED_OUT'
                continue
            result['native_returncode'] = completed.returncode
            if completed.returncode:
                result['status'] = 'LOWERED_NATIVE_EXECUTION_FAILED'
                result['native_error'] = completed.stderr.decode(errors='replace')[:4000]
            else:
                native = json.loads(completed.stdout)
                result['status'] = 'NATIVE_GLOBAL_MEMORY_FRAGMENT_EXECUTED'
                result['GTSim_executed'] = True
                result['native_summary'] = native
        self.jobs.clear()
        self.deferred_bytes = 0
        return self.results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge', type=Path)
    parser.add_argument('--max-records-per-kernel', type=int, default=50000)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--same-process-root-binding', type=Path)
    parser.add_argument('--qualified-transport-receipt', type=Path)
    parser.add_argument('--max-deferred-native-bytes', type=int, default=64 << 20)
    args = parser.parse_args()
    need(args.max_records_per_kernel > 0 and args.timeout_seconds > 0, 'positive limits required')
    binding = json.loads(args.same_process_root_binding.read_text()) if args.same_process_root_binding else None
    lowerer = StreamLowerer(bridge=args.bridge, max_records=args.max_records_per_kernel,
                            timeout_seconds=args.timeout_seconds, attribution=Attribution(binding),
                            max_deferred_bytes=args.max_deferred_native_bytes)
    for line in sys.stdin:
        if line.strip():
            result = lowerer.accept(json.loads(line))
            if result is not None and not args.qualified_transport_receipt:
                print(json.dumps(result, separators=(',', ':')), flush=True)
    need(not lowerer.pending, 'EOF with unclosed sample kernel; no completion claim')
    if args.qualified_transport_receipt:
        for result in lowerer.finalize_transport(json.loads(args.qualified_transport_receipt.read_text())):
            print(json.dumps(result, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
