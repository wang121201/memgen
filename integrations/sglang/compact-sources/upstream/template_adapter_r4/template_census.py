"""Count an admitted native HBServe template using its existing RAM census.

No address stream is saved or expanded across the full grid. This is a count
of the admitted MODEL_ESTIMATE, never a measured full hardware memory trace.
"""
import collections
import contextlib
import hashlib
import io
from pathlib import Path
import sys

ADAPTER_SHA = '679e0c8fb1509729447f4f071cf4ac8fc416d8fbdded296b32f0ab9cd677394e'
STORAGE_WINDOWS_SHA = '613136fde2a945727b7daa332e984ae4b26ae39e3baa63bad41d799255ffede6'
CODEC_SHA = 'c4b90ea4f5739452db5e24222bbc2a6d549f6f8b2e19de59f902500b0d9184be'
GENERATOR_SHA = '74bc9dbc352dfa41c4c673d37335275b052e3a8a83c5c61fbc61dd2c1fed7570'
REFERENCE_INPUTS = {
    '../__init__.py': 'c8f7de7b3ce2bab4488b7c66591449bc97a2b62f17fe35d5a86c78e202c1183e',
    '__init__.py': 'ef050b1102aa35d003107f406008a219f47ebe662fd1e79038a21fa93ebb92c3',
    'compact_request_template.py': '46d952ab23e95a0d9962764130de743c174043bb39bcc32882b8fbc20490cc49',
    'shape_aware_cta_rules.py': '4dcd235e048d3c3f90291686d801158b439039112ba10c818538b4c0419071dd',
    'semantic_embedding_generator.py': 'bacf07c553f35f8bcb0d94f0936af74bab78d611441e2fbdc784c8e99a9ce1ba',
}
DOMAIN = 'requested_global_lane_bytes_not_cache_or_DRAM_transactions'
LABEL = 'ESTIMATED_SOURCE_PREDICATE_GATING_NOT_HARDWARE_VALIDATED'
FIELDS = ('records', 'lane_references', 'read_bytes', 'write_bytes',
          'read_bytes_source_mask_gated', 'read_bytes_effective_mask_upper_bound',
          'candidate_ldg_source_mask_gated_read_bytes',
          'candidate_ldg_effective_mask_upper_bound_read_bytes')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def integer(n, minimum=0):
    need(type(n) is int and n >= minimum, 'nonnegative exact integer required')
    return n


def source_pin(path, expected):
    path = Path(path).resolve()
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    need(got == expected, 'count proof source changed: ' + str(path))
    return dict(path=str(path), bytes=len(raw), sha256=got)


def checked_sources(compiled):
    need('_cta' not in vars(compiled) and 'iter_memory_instructions' not in vars(compiled), 'instance method override has no count proof')
    owner = type(compiled)._cta.__globals__
    need(type(compiled) is owner['CompiledNativeTemplate'], 'unsupported compiled template class')
    native = sys.modules[compiled.generator.__module__]
    need(compiled.generator is native.generated_bundle_records, 'generator override has no count proof')
    pins = [source_pin(owner['__file__'], ADAPTER_SHA),
            source_pin(Path(owner['__file__']).resolve().parent / 'storage_windows.py', STORAGE_WINDOWS_SHA),
            source_pin(compiled.codec.__file__, CODEC_SHA),
            source_pin(native.__file__, GENERATOR_SHA)]
    reference = Path(native.__file__).resolve().parent
    pins.extend(source_pin(reference / name, expected) for name, expected in REFERENCE_INPUTS.items())
    return native, pins


class CompactRAMSource:
    """The existing API needs only .open('rb'); keep the same BytesIO in RAM."""
    def __init__(self, source):
        need(type(source) is io.BytesIO, 'only existing compact BytesIO source is accepted')
        self.source = source

    @contextlib.contextmanager
    def open(self, mode):
        need(mode == 'rb', 'RAM facade is read-only')
        position = self.source.tell()
        try:
            self.source.seek(0)
            yield self.source
        finally:
            self.source.seek(position)


def count_record(m, estimated_sites, totals):
    width = integer(m['mem_width'], 1)
    need(width in (1, 2, 4, 8, 16) and m['op'] in (ord('R'), ord('W')), 'unadmitted MemoryInst width/operation')
    mask = integer(m['mask'])
    need(mask < 1 << 32, 'MemoryInst lane mask width')
    lanes = m['lanes']
    need(isinstance(lanes, list) and len(lanes) <= 32, 'bounded projected lanes')
    ids = [integer(l['lane']) for l in lanes]
    need(len(ids) == len(set(ids)) and all(i < 32 and mask & (1 << i) for i in ids), 'lane identity/mask mismatch')
    need(all(not l['is_local'] and not l['local_offset'] for l in lanes), 'local lane outside admitted global projection')
    totals['records'] += 1
    totals['lane_references'] += len(lanes)
    amount = len(lanes) * width
    totals['write_bytes' if m['op'] == ord('W') else 'read_bytes'] += amount
    site = (m['function_id'], m['pc'], m['opcode'])
    if site in estimated_sites:
        need(m['op'] == ord('R'), 'source-predicate candidate cannot be a write')
    if m['op'] == ord('R'):
        upper = amount
        if site in estimated_sites:
            upper = bin(mask).count('1') * width
            need(upper >= amount, 'source predicate gated count exceeds effective mask')
            totals['candidate_ldg_source_mask_gated_read_bytes'] += amount
            totals['candidate_ldg_effective_mask_upper_bound_read_bytes'] += upper
        totals['read_bytes_source_mask_gated'] += amount
        totals['read_bytes_effective_mask_upper_bound'] += upper


def zero_counts():
    return collections.Counter({name: 0 for name in FIELDS})


def check_admission(compiled):
    receipt = compiled.receipt
    need(receipt['schema'] == 'SGLANG_HBSERVE_NATIVE_TEMPLATE_ADMISSION_V1' and receipt['ready_for_modeled_CTA_expansion'] is True, 'template not admitted')
    dims = compiled.begin['grid']
    need(len(dims) == 3 and all(type(x) is int and x > 0 for x in dims), 'source grid dimensions')
    need(compiled.grid_size == dims[0] * dims[1] * dims[2] and 0 < compiled.grid_size <= 65536, 'source grid count cap')
    need(receipt['source_grid'] == dims and receipt['source_block'] == compiled.begin['block'], 'admission shape differs')
    need(receipt['source_launch_key'] == compiled.begin['source_launch_key'] and receipt['code_sha256'] == compiled.begin['code_sha256'], 'admission source differs')
    need(receipt['exact_anchor'] is compiled.exact, 'exact/nonexact admission mismatch')
    if compiled.estimated_sites:
        need(receipt['memory_model_label'] == LABEL and receipt['source_predicate_hardware_semantics_validated'] is False, 'source predicate estimate is not explicit')
    if not compiled.exact:
        need(receipt['independent_holdout_prediction'] is True, 'non-exact count requires independent heldout admission')


def exact_count(compiled, max_source_records):
    need({key[0] for key in compiled.samples}|compiled.packetless_ctas == set(range(compiled.grid_size)), 'exact sampled CTA coverage')
    totals = zero_counts()
    for (cta, warp), records in sorted(compiled.samples.items()):
        integer(cta); integer(warp)
        for m in records:
            need(totals['records'] < max_source_records, 'bounded exact sample record census')
            need(m['block_id'] == cta and m['cta_warp'] == warp, 'anchor CTA/warp mismatch')
            count_record(m, compiled.estimated_sites, totals)
    need(totals['records'] == compiled.receipt['sampled_records'], 'exact sample record count differs from admission')
    return totals, dict(method='EXACT_CAPTURED_ANCHORS_IN_RAM', source_records_scanned=totals['records'],
        full_grid_MemoryInst_materialization=False, count_multiplier=1)


def phase_count(compiled, native, max_source_records):
    """Narrow proof for the exact phase_rules object produced by pinned prepare.

    generated_bundle_records returns the same bundles and same lane tuples for
    every CTA; affine/period4 rules alter only offsets. _cta then emits every
    extras entry once, including zero-lane entries that lack compact bundles.
    The original segment census checks all rule-phase address extrema before
    multiplying its compact request counts. Its totals cross-check our extras.
    """
    prepared = compiled.prepared
    need(type(prepared) is native.PreparedPhaseGenerator, 'unsupported prepared generator')
    need(set(prepared.policies) == {0} and set(prepared.by_kernel_x) == {(0, 0)}, 'unsupported source kernel/CTA policy')
    policy = prepared.policies[0]
    need(policy['kind'] == 'phase_rules' and policy['source_cta_x'] == 0, 'unsupported variable-cardinality policy; no constant-CTA proof')
    need(prepared.shadow_runtime_binding is None, 'shadow generator needs another count proof')
    period = integer(compiled.receipt['period_candidate'], 1)
    native_period = native.policy_period(policy)
    need(period in (1, 4) and native_period in (1, period), 'unproved policy period')
    need(policy['rules'] == compiled.rules and compiled.rules, 'rule identity differs')
    for rule in compiled.rules:
        need(rule['kind'] == ('affine' if period == 1 else 'tiled_swizzle'), 'unproved address rule kind')
        if period == 4:
            need(rule['period'] == 4 and len(rule['phase_offsets_bytes']) == 4, 'unproved periodic rule')
            need(type(rule['quotient_stride_bytes']) is int and all(type(x) is int for x in rule['phase_offsets_bytes']), 'noninteger periodic address rule')
        else:
            need(type(rule['x_stride_bytes']) is int, 'noninteger affine address rule')
    need(prepared.by_kernel_x[(0, 0)] == compiled.bundles, 'bundle source identity differs')
    need(type(compiled.source) is io.BytesIO and len(compiled.source.getbuffer()) <= 256 << 20, 'bounded compact RAM template')
    extras = compiled.extras
    need(isinstance(extras, list) and 0 < len(extras) <= max_source_records, 'bounded source extras')
    expected = {i for i, m in enumerate(extras) if m['lanes']}
    indices = [b['extra_index'] for b in compiled.bundles]
    need(len(indices) == len(set(indices)) and set(indices) == expected, 'nonempty extras/bundle bijection')
    cursor = 0
    position = compiled.source.tell()
    try:
        for bundle in compiled.bundles:
            i = bundle['extra_index']; m = extras[i]
            need(bundle['kernel_ordinal'] == 0 and bundle['cta_x'] == bundle['cta_y'] == bundle['cta_z'] == 0, 'not a flattened single source CTA')
            need(bundle['warp_in_cta'] == m['cta_warp'], 'extra/bundle warp differs')
            need(bundle['request_ordinal_begin'] == cursor, 'compact ranges have gap/overlap')
            records = native.read_bundle_records(compiled.source, bundle)
            cursor = bundle['request_ordinal_end_exclusive']
            need(len(records) == len(m['lanes']), 'compact/extra lane cardinality differs')
            for lane, record in zip(m['lanes'], records):
                obj, offset, kernel, width, operation, flags = record
                need(kernel == 0 and width == m['mem_width'] and operation == int(m['op'] == ord('W')), 'compact/extra request metadata differs')
                need(obj in compiled.windows and lane['addr'] == compiled.windows[obj] + offset, 'compact/extra source address differs')
                need(native.known_request_flags(flags), 'compact flags unsupported')
        need(cursor * native.RECORD_BYTES == len(compiled.source.getbuffer()), 'compact template tail not covered')
    finally:
        compiled.source.seek(position)
    per_cta = zero_counts()
    for m in extras:
        need(m['block_id'] == 0, 'extras must come from flattened CTA0')
        count_record(m, compiled.estimated_sites, per_cta)
    # Use the frozen existing analytic implementation, including every finite
    # phase's first/last address bounds. No path or actual file is opened.
    extrema = {0, compiled.grid_size - 1}
    for phase in range(period):
        first = phase
        last = compiled.grid_size - 1 - ((compiled.grid_size - 1 - phase) % period)
        if first < compiled.grid_size:
            extrema.add(first)
        if last >= 0:
            extrema.add(last)
    # The frozen policy_period handles top-level tiled_swizzle but returns1
    # for this codec's phase_rules containing tiled_swizzle rules. Supplement
    # that case with the original generator at every finite residue extreme.
    if native_period != period:
        with CompactRAMSource(compiled.source).open('rb') as source:
            for cta in sorted(extrema):
                for _bundle, records in native.generated_bundle_records(
                        prepared=prepared, source=source, kernel_ordinal=0, target_cta_x=cta):
                    for _record in records:
                        pass
    descriptor = native.PreparedSegmentGenerator(generator=prepared,
        ranges=[(0, 0, compiled.grid_size)], provenance={'source': 'admitted_RAM_compact_template'})
    original = native.segment_generation_census(prepared=descriptor, source_path=CompactRAMSource(compiled.source))
    compact = original['totals']
    totals = collections.Counter({k: per_cta[k] * compiled.grid_size for k in FIELDS})
    need(compact.get('requests', 0) == totals['lane_references'] and compact.get('bytes', 0) == totals['read_bytes'] + totals['write_bytes'], 'existing HBServe compact lane/byte census mismatch')
    for compact_field, own in [('r_bytes', 'read_bytes'), ('w_bytes', 'write_bytes')]:
        need(sum(obj.get(compact_field, 0) for obj in original['by_object'].values()) == totals[own], 'existing HBServe direction census mismatch')
    need(compact['target_ctas'] == compiled.grid_size and compact['active_grid_ctas'] == compiled.grid_size, 'existing HBServe grid count mismatch')
    proof = dict(method='EXISTING_HBSERVE_SEGMENT_GENERATION_CENSUS_RAM', source_records_scanned=len(extras),
        source_lane_requests=per_cta['lane_references'], count_multiplier=compiled.grid_size,
        cardinality_classes=1, address_rule_period=period, address_extrema_ctas=sorted(extrema),
        original_policy_period=native_period, supplemented_nested_period_extrema=native_period != period,
        native_compact_census=original, zero_lane_MemoryInst_records_per_cta=sum(not m['lanes'] for m in extras),
        full_grid_MemoryInst_materialization=False, shape_proof='phase_rules has no CTA filtering; fixed bundle lane lists and extras order; only offsets vary')
    return totals, proof


def census(compiled, *, max_source_records=1000000):
    """Return the postprocess census shape; failure remains INCOMPLETE/unknown.

    Call only after ordinary template/holdout admission. Unknown policy or
    failed bounds are not converted to zero or silently enumerated. A caller
    may separately request the original bounded enumeration as a fallback.
    """
    try:
        need(type(max_source_records) is int and 0 < max_source_records <= 1000000, 'source scan cap')
        native, pins = checked_sources(compiled)
        check_admission(compiled)
        if compiled.exact:
            totals, proof = exact_count(compiled, max_source_records)
        else:
            totals, proof = phase_count(compiled, native, max_source_records)
        return dict(totals, status='COMPLETE_ADMITTED_GRID_MODEL', traffic_domain=DOMAIN,
            memory_model_label=LABEL if compiled.estimated_sites else None, hardware_source_predicate_validated=False,
            qualification='MODEL_ESTIMATE', analytic_census=proof, count_proof_source_pins=pins,
            full_hardware_trace=False, raw_or_expanded_trace_saved=False, cache_or_DRAM_traffic=False)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return dict(status='INCOMPLETE_MODEL_CENSUS', reason=type(error).__name__ + ': ' + str(error),
            traffic_domain=DOMAIN, qualification='UNKNOWN_REQUESTED_TRAFFIC', records=None, lane_references=None,
            read_bytes=None, write_bytes=None, read_bytes_source_mask_gated=None,
            read_bytes_effective_mask_upper_bound=None, raw_or_expanded_trace_saved=False,
            bounded_enumeration_fallback_eligible=str(error).startswith('unsupported variable-cardinality policy;'))
