"""Lossless compact address coordinates from same-process observed storage spans.

A storage span is an address carrier, not proof of allocator lifetime, tensor
operand ownership or cross-layer relocation. Alias candidate ids remain explicit;
ambiguous generations never select one root. No dynamic address list is retained.
"""
import bisect
import collections
import dataclasses

U32 = 1 << 32
MAX_REMAP_BYTES = 512 << 20
RECORD_SCRATCH_CHARGE = 1024
LANE_SCRATCH_CHARGE = 384


def need(ok, why):
    if not ok: raise ValueError(why)


class StorageWindows:
    def __init__(self, attribution, begin, *, unbound_policy='transport_fallback', max_remap_bytes=MAX_REMAP_BYTES):
        need(unbound_policy in ('transport_fallback', 'reject'), 'explicit unbound address policy required')
        need(type(max_remap_bytes) is int and 0 < max_remap_bytes <= MAX_REMAP_BYTES, 'compact remap scratch must stay within 512 MiB hard cap')
        self.attribution = attribution; self.begin = begin
        self.row = attribution.bind(begin)
        self.process = dict(attribution.process)
        self.unbound_policy = unbound_policy; self.max_remap_bytes = max_remap_bytes
        self.roots = {}; self.slots = []; self.slot_index = {}; self.compact_objects = []; self.counters = collections.Counter()
        objects = attribution.objects; ids = set(self.row['candidate_ids']) | set(attribution.document['persistent_candidate_ids'])
        for cid in sorted(ids):
            candidate = objects[cid]; root = candidate['root']
            if not root['device'].startswith('cuda'): continue
            rid = root['id']; base = root['base_address']; size = root['storage_nbytes']
            need(type(base) is int and type(size) is int and 0 < base < base + size < 1 << 64, 'invalid observed storage span')
            if rid in self.roots:
                need((base, size) == (self.roots[rid]['base_address'], self.roots[rid]['extent_bytes']), 'same root id has conflicting storage geometry')
            else:
                self.roots[rid] = dict(root_id=rid, base_address=base, extent_bytes=size, alias_candidate_ids=[])
            self.roots[rid]['alias_candidate_ids'].append(cid)
        self.intervals = sorted((r['base_address'], r['base_address'] + r['extent_bytes'], rid) for rid, r in self.roots.items())
        self.starts = [row[0] for row in self.intervals]; self.prefix_max = []; maximum = -1
        for start, end, _ in self.intervals:
            maximum = max(maximum, end); self.prefix_max.append(maximum)

    def observed_roots(self, address, width):
        i = bisect.bisect_right(self.starts, address) - 1; matched = []
        while i >= 0 and self.prefix_max[i] >= address + width:
            start, end, rid = self.intervals[i]
            if address + width <= end: matched.append(rid)
            i -= 1
        return sorted(matched)

    def encode_address(self, address, width):
        need(type(address) is int and type(width) is int and 0 < address < address + width < 1 << 64 and width > 0,
             'invalid source VA or width')
        matches = self.observed_roots(address, width)
        reason = None
        if len(matches) == 1:
            root = self.roots[matches[0]]
            if root['extent_bytes'] > U32: reason = 'OBSERVED_STORAGE_EXCEEDS_COMPACT_U32'
            else:
                identity = ('OBSERVED_STORAGE', self.process['pid'], self.process['start_ticks'], matches[0])
                carrier = dict(root, kind='observed_storage_span', source_name=matches[0],
                    stable_name='%s:%s:%s' % (self.process['pid'], self.process['start_ticks'], matches[0]))
        else:
            reason = 'NO_OBSERVED_STORAGE_SPAN' if not matches else 'AMBIGUOUS_OBSERVED_STORAGE_GENERATIONS'
        if reason:
            need(self.unbound_policy == 'transport_fallback', reason)
            base = address // U32 * U32
            need(address + width <= base + U32, 'unbound lane crosses opaque transport fallback window')
            identity = ('OPAQUE_TRANSPORT_FALLBACK', base)
            carrier = dict(kind='opaque_transport_fallback', root_id=None, base_address=base, extent_bytes=U32,
                alias_candidate_ids=[], source_name='opaque source VA transport fallback',
                stable_name='opaque_transport_fallback_%d' % (base // U32))
        if identity not in self.slot_index:
            need(len(self.slots) < 65536, 'compact object index exceeds u16')
            self.slot_index[identity] = len(self.slots); self.slots.append(carrier)
        slot = self.slot_index[identity]; offset = address - carrier['base_address']
        need(0 <= offset < U32 and offset + width <= carrier['extent_bytes'], 'source offset outside compact storage extent')
        self.counters[carrier['kind'] + '_lane_references'] += 1
        self.counters[carrier['kind'] + '_requested_bytes'] += width
        if reason: self.counters['fallback_reason_' + reason] += 1
        # Synthetic coordinate is used only as the unchanged codec's compact
        # object selector; no source VA is changed in retained original samples.
        return ((slot + 1) << 32) | offset

    def decode_address(self, symbolic, width):
        slot = (symbolic >> 32) - 1; offset = symbolic & (U32 - 1)
        need(0 <= slot < len(self.slots), 'unknown compact storage slot')
        root = self.slots[slot]
        need(offset + width <= root['extent_bytes'], 'decoded compact offset outside storage extent')
        return root['base_address'] + offset

    def prepare(self, codec, training, kernel, shape, period):
        self.attribution.bind(self.begin)  # Same key/begin mutation check at each fit attempt.
        transformed = {}; retained_bound = 0
        for key, frames in training.items():
            out = []; transformed[key] = out
            for original in frames:
                retained_bound += RECORD_SCRATCH_CHARGE + LANE_SCRATCH_CHARGE * len(original['lanes'])
                need(retained_bound <= self.max_remap_bytes, 'bounded compact-coordinate remap scratch exceeds limit')
                lanes = [dict(lane, addr=self.encode_address(lane['addr'], original['mem_width'])) for lane in original['lanes']]
                out.append(dict(original, lanes=lanes))
        prepared, source, bundles, extras, symbols, rules = codec.prepare(transformed, kernel, shape, period)
        # The frozen generator continues to enforce u32 and actual extent bounds.
        objects = [dict(self.slots[symbols[i] - 1]) for i in range(len(symbols))]
        self.compact_objects = [dict(obj, compact_object_index=i) for i, obj in enumerate(objects)]
        extents = [obj['extent_bytes'] for obj in objects]
        prepared = dataclasses.replace(prepared, source_extents=extents, target_extents=extents,
                                       object_metadata=objects)
        bases = {index: objects[index]['base_address'] for index in range(len(objects))}
        for frame in extras:
            for lane in frame['lanes']:
                lane['addr'] = self.decode_address(lane['addr'], frame['mem_width'])
        self.counters['peak_compact_remap_scratch_bound_bytes'] = max(self.counters['peak_compact_remap_scratch_bound_bytes'], retained_bound)
        return prepared, source, bundles, extras, bases, rules

    def receipt(self):
        return dict(schema='SGLANG_OBSERVED_STORAGE_COMPACT_WINDOWS_V1', source_process=self.process,
            source_launch_key=self.begin['source_launch_key'], native_launch_key=self.row['source_launch_key'],
            coordinate_domain='observed_storage_base_plus_u32_offset_with_separate_opaque_fallback',
            compact_absolute_VA_formula='base_address + compact_u32_offset', unbound_policy=self.unbound_policy,
            objects=self.compact_objects, registered_coordinate_slots=len(self.slots),
            fitting_query_counts=dict(self.counters), max_compact_remap_scratch_bytes=self.max_remap_bytes,
            scratch_charge_per_record_bytes=RECORD_SCRATCH_CHARGE, scratch_charge_per_lane_bytes=LANE_SCRATCH_CHARGE,
            count_scope='core_training_address_encoding_attempts_only_not_sample_or_full_grid_traffic',
            raw_dynamic_addresses_retained=False, allocator_lifetime_proven=False,
            tensor_operand_ownership_proven=False, cross_layer_relocation_proven=False,
            storage_span_not_exact_tensor_view_membership=True, same_VA_generations_not_merged=True)
