"""Bounded, loss-intolerant reader for unchanged fast_mem_access_t packets.

This is an in-process streaming interface, not a trace file format to persist.
A successful parse remains pending the producer's successful exit/context close.
Callbacks may retain bounded address templates; this reader retains no packets.
"""
import hashlib
import json
import struct
from cta_entry_contract import validate_entry_proof

MAGIC = b'SGPKT001'
HEADER = struct.Struct('<II')
PACKET = struct.Struct('<Q11I9i64QQII')
assert PACKET.size == 616
ENVELOPE = struct.Struct('<QQ')
HELLO, STATIC, BEGIN, RECORD, END, CLOSED = range(1, 7)
MAX_FRAME = 1 << 20
MAX_WIRE = 8 << 30
MAX_STATIC = 100_000
MAX_KERNELS = 32768
MAX_CTAS = 65_536
MAX_SELECTED_CTAS = 8192


def need(ok, why):
    if not ok:
        raise ValueError(why)


def integer(x, low=0, high=(1 << 64)-1):
    need(type(x) is int and low <= x <= high, 'integer outside domain')
    return x


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def parse_json(b):
    def pairs(p):
        out = {}
        for k, v in p:
            need(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    return json.loads(b, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def exact(f, n):
    out = bytearray()
    while len(out) < n:
        part = f.read(n-len(out))
        need(bool(part), 'truncated sample pipe')
        out.extend(part)
    return bytes(out)


def _product(d, limit):
    need(type(d) is list and len(d) == 3, 'xyz geometry')
    p = 1
    for n in d:
        p *= integer(n, 1, limit)
    need(p <= limit, 'geometry product bound')
    return p


def _decode(payload, begin, statics):
    need(len(payload) == ENVELOPE.size + PACKET.size, 'stable packet envelope size')
    received, selected = ENVELOPE.unpack_from(payload)
    p = PACKET.unpack_from(payload, ENVELOPE.size)
    capture_seq = p[0]
    width, policy, source_mask, fid, cta_warp = p[1:6]
    gm, lm, sm = p[6:8], p[8:10], p[10:12]
    smid, x, y, z, hw_warp, gwarp, opcode_id, pc, nref = p[12:21]
    a1, a2 = p[21:53], p[53:85]
    clock, active, predicate = p[85:]
    mask = active & predicate
    need(capture_seq == 0, 'device capture_seq must remain conservation-only zero')
    need(mask and nref in (1, 2), 'empty mask or reference count')
    need(0 <= smid < begin['sm_count'], 'actual SM outside device')
    need(0 <= cta_warp < (begin['threads_per_cta']+31)//32, 'logical CTA warp out of block')
    grid = begin['grid']
    need(0 <= x < grid[0] and 0 <= y < grid[1] and 0 <= z < grid[2], 'CTA coordinates')
    cta = x + grid[0]*(y+grid[1]*z)
    need(cta in begin['_selected_ctas'], 'packet outside sealed CTA selection')
    need(fid in begin['related_function_ids'], 'packet function not in selected launch')
    s = statics.get((fid, pc))
    need(s is not None and s['opcode_id'] == opcode_id, 'packet lacks exact static instruction')
    need(s['ref_count'] == nref, 'static reference count differs')
    need((width, policy) == (s['transfer_width'], s['transfer_policy']), 'static transfer/source-control mismatch')
    refs = []
    for ref, addresses in enumerate((a1, a2)):
        g, l, sh = gm[ref], lm[ref], sm[ref]
        if ref >= nref:
            need(g == l == sh == 0 and not any(addresses), 'unused reference not zero')
            continue
        need((g | l | sh) == mask and not(g & l or g & sh or l & sh), 'unknown/overlapping dynamic address space')
        need(all(not(addr) for lane, addr in enumerate(addresses) if not(mask >> lane & 1)), 'inactive address nonzero')
        refs.append(dict(global_mask=g, local_mask=l, shared_mask=sh, addresses=list(addresses)))
    projection = 'ordinary_memory_operands'
    if policy == 2:
        need(nref == 1 and width in (8, 16) and width == s['width'] and gm[0] == mask and not(source_mask & ~mask), 'LDG source-control packet')
        need(s['source_control_kind'] == 'ldg_predicate_candidate', 'LDG control static witness')
        need(width != 8 or s['opcode'] == 'LDG.E.LTC128B.64.STRONG.GPU', 'unobserved 64-bit LDG candidate opcode')
        projection = 'predicated_global_read_candidate'
    elif width:
        need(nref == 2 and width in (4, 8, 16) and policy in (0, 1) and (not policy or width == 16), 'async shape')
        need(sm[0] == mask and gm[1] == mask and not(source_mask & ~mask), 'async reference roles/mask')
        need(s['source_control_kind'] == 'validated_sm89_ldgsts', 'async static form not admitted')
        projection = 'async_global_read'
    else:
        need(policy == source_mask == 0, 'unclassified source-control packet')
    return dict(schema='SG_MEMORY_PROJECTION_RECORD_V1',
        source_launch_key=begin['source_launch_key'], code_sha256=begin['code_sha256'],
        function_id=fid, pc=pc, opcode_id=opcode_id, opcode=s['opcode'],
        width=s['width'], is_load=s['is_load'], is_store=s['is_store'],
        original_received_ordinal=received, selected_ordinal=selected,
        cta=[x, y, z], cta_linear_id=cta, cta_warp_id=cta_warp,
        actual_sm=smid, hardware_warp_id=hw_warp, global_hardware_warp_id=gwarp,
        clock64=clock, active_mask=active, predicate_mask=predicate, effective_mask=mask,
        ref_count=nref, refs=refs, transfer_width=width, transfer_policy=policy,
        source_read_mask=source_mask, source_control_kind=s['source_control_kind'],
        projection_kind=projection, sequence_semantics='host_delivery_same_warp_order_not_cross_warp_dependency')


def read_stream(f, *, max_wire_bytes, on_begin=None, on_record=None, on_end=None):
    """Consume a finite pipe; no packet is written to any file or accumulated.

    on_begin(begin) / on_record(record) / on_end(end) callbacks execute inline.
    Callback errors abort parsing; the owner must stop its producer process group.
    A consumer that stores templates must separately enforce its retained RAM cap.
    """
    integer(max_wire_bytes, 1024, MAX_WIRE)
    digest = hashlib.sha256(); wire = 0
    def read(n):
        nonlocal wire
        need(wire+n <= max_wire_bytes, 'wire byte quota')
        b = exact(f, n); wire += n; digest.update(b); return b
    need(read(8) == MAGIC, 'sample pipe magic')
    static = {}; begin = None; hello = None; kernels = []; total = 0
    seen_keys = set(); finished = False; prior = None; clock_by_warp = {}; sms = {}; records = 0
    while not finished:
        before_hash = digest.hexdigest()
        kind, n = HEADER.unpack(read(HEADER.size))
        need(kind in range(HELLO, CLOSED+1) and 0 < n <= MAX_FRAME, 'frame type/length')
        payload = read(n)
        if kind == RECORD:
            need(begin is not None, 'packet outside kernel')
            r = _decode(payload, begin, static)
            need(r['selected_ordinal'] == records and (prior is None or r['original_received_ordinal'] > prior), 'selected/delivery sequence')
            prior = r['original_received_ordinal']; cta = r['cta_linear_id']
            key = (cta, r['cta_warp_id'])
            need(key not in clock_by_warp or r['clock64'] >= clock_by_warp[key], 'same warp clock regressed')
            clock_by_warp[key] = r['clock64']
            need(cta not in sms or sms[cta] == r['actual_sm'], 'CTA migrated between SMs')
            sms[cta] = r['actual_sm']
            if on_record: on_record(r)
            records += 1; total += 1
            continue
        obj = parse_json(payload)
        if kind == HELLO:
            need(hello is None and not static and begin is None and not kernels, 'duplicate/out of order hello')
            need(obj['schema'] == 'SG_PACKET_STREAM_HELLO_V1' and obj['packet_bytes'] == PACKET.size, 'hello ABI')
            need(obj['device_packet_abi_unchanged'] is True and obj['device_cta_filter'] is True, 'stable packet/CTA filter contract')
            need(obj['max_wire_bytes'] == max_wire_bytes, 'sealed wire budget differs')
            hello = obj
        elif kind == STATIC:
            need(hello is not None and begin is None, 'static instruction ordering')
            need(obj['schema'] == 'SG_SAMPLE_STATIC_MEMORY_V1', 'static schema')
            key = (integer(obj['function_id'], 1), integer(obj['pc']))
            need(key not in static and len(static) < MAX_STATIC, 'duplicate/static capacity')
            integer(obj['width'], 1, 128); integer(obj['ref_count'], 1, 2)
            need(type(obj['is_load']) is bool and type(obj['is_store']) is bool, 'NVBit load/store flags')
            need(obj['is_load'] or obj['is_store'] or obj['source_control_kind']=='validated_sm89_ldgsts', 'memory direction unavailable')
            static[key] = obj
        elif kind == BEGIN:
            need(hello is not None and begin is None and len(kernels) < MAX_KERNELS, 'kernel nesting/count')
            need(obj['schema'] == 'SG_KERNEL_SAMPLE_BEGIN_V1', 'begin schema')
            key = obj['source_launch_key']; need(key not in seen_keys, 'duplicate launch key'); seen_keys.add(key)
            integer(obj['sm_count'], 1, 1024); integer(obj['stream_u64']); integer(obj['function_id'], 1)
            grid_size = _product(obj['grid'], MAX_CTAS); threads = _product(obj['block'], 1024)
            fit, hold = obj['fit_ctas'], obj['holdout_ctas']
            need(fit and len(fit)+len(hold) <= MAX_SELECTED_CTAS, 'bounded nonempty CTA plan')
            selected = fit+hold
            need(len(set(selected)) == len(selected), 'fit/holdout duplicate or overlap')
            for c in selected: integer(c, 0, grid_size-1)
            for fid in obj['related_function_ids']: integer(fid, 1)
            need(obj['function_id'] in obj['related_function_ids'], 'entry function absent')
            begin = dict(obj, _selected_ctas=set(selected), threads_per_cta=threads)
            records = 0; prior = None; clock_by_warp = {}; sms = {}
            if on_begin: on_begin(obj)
        elif kind == END:
            need(begin is not None and obj['schema'] == 'SG_KERNEL_SAMPLE_END_V1', 'end state/schema')
            need(obj['source_launch_key'] == begin['source_launch_key'], 'end launch binding')
            pushed, received, saved = (integer(obj[k]) for k in ('pushed_records', 'received_records', 'selected_records'))
            need(pushed == received == saved == records, 'selected-CTA packet conservation')
            need(obj['omitted_records'] is None and obj['whole_kernel_dynamic_census'] is False, 'unobserved CTA traffic must not be invented')
            need(prior is None or prior < received, 'delivery ordinal beyond received census')
            entry=validate_entry_proof(begin,obj,sorted(sms))
            need(obj['unknown_space_lane_references'] == 0 and obj['overflow'] is False and obj['source_closed'] is True, 'source packet errors/closure')
            need(obj['all_memory_active_ctas_seen_count'] is None, 'whole-grid dynamic CTA census was not collected')
            if on_end: on_end(obj)
            kernels.append(dict(source_launch_key=begin['source_launch_key'], selected_records=records,
                fit_ctas=begin['fit_ctas'], holdout_ctas=begin['holdout_ctas'], actual_cta_sm=sms,
                selected_all_grid_ctas=len(entry['executed_ctas']) == _product(begin['grid'], MAX_CTAS),
                entry_proof=obj['entry_proof'],cta_entry=entry,
                omitted_static_memory_classes=obj['omitted_static_memory_classes'],
                memory_projection_admitted=False, template_or_gtsim_admitted=False))
            begin = None
        elif kind == CLOSED:
            need(hello is not None and begin is None and obj['schema'] == 'SG_PACKET_STREAM_CLOSED_V1', 'source close state')
            need(obj['wire_sha256_before_close'] == before_hash, 'whole stream SHA mismatch')
            need(obj['kernels'] == len(kernels) and obj['selected_records'] == total, 'whole stream count')
            need(obj['contexts_closed'] is True and obj['errors'] == [], 'source context/error closure')
            need(obj['selected_plan_complete'] is True, 'not all planned selected kernels observed')
            finished = True
    need(not f.read(1), 'trailing sample stream bytes')
    return dict(schema='SG_PACKET_CONSUMER_RECEIPT_V1',
        status='SOURCE_CLOSED_AWAITING_PRODUCER_EXIT_AND_IDENTITY',
        stream_sha256=digest.hexdigest(), wire_bytes=wire, selected_records=total,
        kernels=kernels, static_instructions=len(static), raw_packets_retained_by_reader=0,
        frame_scratch_limit_bytes=MAX_FRAME, max_wire_bytes=max_wire_bytes,
        full_hardware_trace=False, memory_projection_admitted=False, template_or_gtsim_admitted=False)


def qualify_transport(receipt, *, producer_exit_code, observer_closed, source_pins_match, consumer_callbacks_closed):
    """Separate conjunctive gate. This never grants template/cache qualification."""
    need(receipt['status'] == 'SOURCE_CLOSED_AWAITING_PRODUCER_EXIT_AND_IDENTITY', 'not a parsed stream')
    need(producer_exit_code == 0 and observer_closed and source_pins_match and consumer_callbacks_closed,
         'producer/observer/source identity/consumer not closed')
    return dict(receipt, status='PASS_SAMPLED_TRANSPORT_ONLY')
