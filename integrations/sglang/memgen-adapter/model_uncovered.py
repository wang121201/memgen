"""Modeled completion for sampled classes the exact fitter refuses.

The exact path (fit template -> rebind to the target launch) covers most classes,
but the structurally irregular ones (index_select, ragged attention, cutlass
tiling variants, argmax reducers) have no admitted template: the frozen fitter
rejects them and the expansion refuses the launch.  HBServe never has a numeric
template for those classes either - its full-model plan records them as
``class_only`` (symbolic, excluded from the numeric replay) or models their arena
with an explicit ``numeric_modeled`` label and a ``not_claimed`` list.

This module gives an unpacked class that same status, with the strongest evidence
the exact path cannot use:

* addresses come from the launch's *own* allocation context (the same
  ``module_calls``/``tensor_metadata`` ledger the exact rebind walks), never from
  an invented offset table;
* the per-CTA byte volume comes from the strongest measured source the run has for
  that class, in this order: the class's own whole-grid census when the fitter did
  admit a template elsewhere in the run
  (``template_census_whole_grid_divided_by_grid``); otherwise the lane counts and
  issue width of its *own* refused records in the projection receipt, carried to
  the class's classified records while they cover most of them
  (``observed_refusal_width_records_per_cta``, see :func:`observed_lane_census`);
  otherwise the run median of bytes per recorded instruction
  (``run_calibrated_records_per_cta``, the one estimate this module makes);
* the walk is affine over the object: a private per-CTA tile when the object can
  hold the whole grid's tiles, otherwise HBServe's ``shared_template_arena`` where
  every CTA re-reads the same representative lines.

Nothing here may be promoted to the exact statuses: ``modeling.mode`` is always
``numeric_modeled``, ``exact_cross_layer_identity_claimed`` stays false, and the
manifest keeps the ``NOT_CLAIMED`` list.
"""

import math

LANES = 32
WIDTHS = (16, 8, 4, 2, 1)
# Widest possible issue. Only the issue budget is expressed in these bytes; each
# span then sizes its own issue by access_geometry(), so a span narrower than
# this does not over-read.
ISSUE_BYTES_ENVELOPE = LANES * WIDTHS[0]
MODELED_STATUS = 'PASS_MODELED_UNCOVERED_CLASS'
# Format identity, not provenance: the engine validates schema.name/version and
# reads provenance from ``status`` (exact and layer-rebound rows share this too).
SCHEMA = dict(name='hbserve.hyfiss_sampled_sass_profile', version=5)
MAX_ISSUES_PER_CTA = 4096

# Volume bases, strongest first.  Every one of them but the last is a measurement
# of the class itself: the engine never has to be told which one was used, but the
# manifest does, because only the last one carries a run-wide assumption.
TEMPLATE_CENSUS_BASIS = 'template_census_whole_grid_divided_by_grid'
OBSERVED_CENSUS_BASIS = 'observed_refusal_width_records_per_cta'
CALIBRATED_BASIS = 'run_calibrated_records_per_cta'
MEASURED_BASES = (TEMPLATE_CENSUS_BASIS, OBSERVED_CENSUS_BASIS)
# A refusal census covers only the records the projection refused.  Its bytes per
# record describe the class only while it covers most of the class's classified
# records; below this share it describes a residue whose opcode mix differs from
# the bulk (a kernel refused for ten atomic lanes still has its loads), and
# extending it would replace one wrong width with another.
OBSERVED_REFUSAL_COVERAGE_MIN = 0.5

NOT_CLAIMED = [
    'instruction-level issue order inside a modeled class',
    'exact access width, lane mapping, or predicate mask of a modeled class',
    'addresses outside the launch allocation context',
    'complete model trace before non-block segments are added',
    'generality beyond the bound shape',
]


def access_geometry(span_bytes):
    """Widest access whose issue stays inside the object, else a partial warp.

    Widths follow the engine's opcode tokens (.128/.64/.32/.16/.8); a narrow
    object uses fewer active lanes rather than reading past its own allocation.
    """
    for width in WIDTHS:
        if span_bytes >= width * LANES:
            return width, LANES
    for width in WIDTHS:
        if span_bytes >= width:
            return width, max(1, min(LANES, int(span_bytes // width)))
    return 1, 1


def opcode_for(width, direction):
    """Opcode token the engine's infer_width() reads back as this lane width.

    The width suffix is mandatory: an unqualified ``LDG.E`` is inferred as four
    bytes per lane, which would silently quarter a 16-byte modeled issue.
    """
    return ('LDG' if direction == 'read' else 'STG') + '.E.%d' % (width * 8)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def classified_records(row):
    """Observed records the dynamic projection accepted as global memory traffic."""
    classified = (row.get('projection_classification') or {}).get('classified_records')
    if classified is None:
        return row.get('selected_records')
    return min(classified, row.get('selected_records', classified))


def observed_lane_census(row):
    """Measured lane bytes of a class's own refused records, if the run kept them.

    When the projection refuses a record it still buckets it, and the bucket keeps
    the issue width in bytes per lane plus the lane counts it saw on the sampled
    CTAs.  For a class the fitter never produced a template for, that is the only
    place its real access width survives - and it is a measurement, not a median
    over other classes, so it displaces the run-wide bytes-per-instruction figure
    exactly where that figure is wrong by the width ratio (a 16 B/lane class is a
    quarter of a run whose median instruction is 4 B/lane).

    The buckets cover only the records the projection *refused*, so the caller has
    to weigh them against the class's classified records: see
    ``OBSERVED_REFUSAL_COVERAGE_MIN``.

    Direction follows the opcode: a load-only class is sized by the lanes whose
    *source* mask read (the projection tracks that only for read candidates), a
    store-only class by its active lanes, and an opcode that is both - an atomic -
    is carried as writes with that attribution named, because it has no direction
    of its own to attribute.  Returns ``None`` when the run recorded no such
    bucket for this class.
    """
    read_bytes = 0
    write_bytes = 0
    records = 0
    widths = set()
    directionless = []
    for item in (row.get('projection_classification') or {}).get('rejection_classes') or []:
        width = int(item.get('width') or 0)
        active = int(item.get('active_lanes') or 0)
        seen = int(item.get('records') or 0)
        if width <= 0 or active <= 0 or seen <= 0:
            continue
        widths.add(width)
        records += seen
        if item.get('is_load') and not item.get('is_store'):
            # source_read_lanes is the guard-masked count and only exists for the
            # buckets the projection tracked a source mask on; without that mask
            # the active lanes are the best measured count, not zero.
            tracked = int(item.get('source_mask_records') or 0) > 0
            lanes = int(item.get('source_read_lanes') or 0) if tracked else active
            read_bytes += min(lanes, active) * width
        else:
            write_bytes += active * width
            if item.get('is_load'):
                directionless.append(str(item.get('opcode')))
    if not records or not widths:
        return None
    return dict(read=float(read_bytes), write=float(write_bytes), records=records,
                widths=sorted(widths), directionless_opcodes=sorted(set(directionless)))


def class_shape(launch):
    """Identity of a kernel class: binary, grid, and block. Phase-independent."""
    return (launch['code_sha256'], tuple(int(x) for x in launch['grid']),
            tuple(int(x) for x in launch['block']))


def grid_size_of(launch):
    size = 1
    for extent in launch['grid']:
        size *= int(extent)
    return size


def class_evidence(consumer, kernels, packed):
    """Per-class observed selection: records, the CTA count they cover, census.

    ``consumer.json`` records, for every sampled class, whether the whole grid or
    only a CTA subset produced its record count, so a per-CTA figure is measured
    rather than inferred from a cross-class median.
    """
    receipt = {row['source_launch_key']: row for row in kernels}
    fitted = {row['source_launch_key']: row for row in packed}
    index = {}
    for row in consumer['kernels']:
        stats = receipt.get(row.get('source_launch_key'))
        need(stats is not None, 'Consumer class %r is absent from the fitting receipt'
             % row.get('source_launch_key'))
        grid = grid_size_of(stats)
        subset = list(row.get('fit_ctas') or []) + list(row.get('holdout_ctas') or [])
        ctas = grid if row.get('selected_all_grid_ctas') else len(subset)
        need(ctas > 0, 'Class %r selected no CTA' % stats.get('source_launch_key'))
        need(ctas <= grid, 'Class %r selected more CTAs than its grid has'
             % stats.get('source_launch_key'))
        index[class_shape(stats)] = dict(
            records=classified_records(stats) or stats.get('selected_records'),
            ctas=ctas, grid=grid, phase=stats.get('phase'),
            selected_records=stats.get('selected_records'),
            grid_whole=bool(row.get('selected_all_grid_ctas')),
            source_launch_key=stats.get('source_launch_key'),
            observed=observed_lane_census(stats),
            census=(fitted.get(stats.get('source_launch_key')) or {}).get('census'))
    return index


def calibrate(kernels, packed):
    """Per-run constants used to size a class that has no fitted profile.

    Every figure is per *instruction* or a ratio, so it does not depend on how
    many CTAs the sampler selected: the census is a whole-grid total and
    ``mem_insts`` scales with it.
    """
    fitted = {row['source_launch_key']: row for row in packed
              if str(row.get('status', '')).startswith('PASS_PROFILE')}
    per_phase = {}
    every = []
    read_bytes = 0
    write_bytes = 0
    for row in kernels:
        census = (fitted.get(row.get('source_launch_key')) or {}).get('census')
        instructions = (census or {}).get('mem_insts') or 0
        if not census or instructions <= 0:
            continue
        read = census['native_read_lane_bytes']
        write = census['native_write_lane_bytes']
        if read + write <= 0:
            continue
        every.append((read + write) / instructions)
        per_phase.setdefault(row.get('phase'), []).append((read + write) / instructions)
        read_bytes += read
        write_bytes += write
    need(every, 'No fitted class to calibrate a modeled volume from')
    return dict(basis=CALIBRATED_BASIS,
                bytes_per_record=median(every),
                bytes_per_record_by_phase={phase: median(values)
                                           for phase, values in sorted(per_phase.items())},
                bytes_per_record_range=[min(every), max(every)],
                read_share=read_bytes / (read_bytes + write_bytes),
                classes_calibrated=len(every))


def object_spans(views, direction):
    """Merged byte spans of a launch's own input (read) or output (write) tensors."""
    spans = []
    for key, view in views:
        if key[1] != direction:
            continue
        element = view['element_size']
        stride = view['stride_bytes'] if 'stride_bytes' in view else [x * element for x in view['stride_elements']]
        need(all(x >= 0 for x in stride), 'Negative strides unsupported')
        size = sum((d - 1) * s for d, s in zip(view['shape'], stride)) + element if all(view['shape']) else 0
        if size > 0:
            spans.append((int(view['data_address']), int(view['data_address']) + size))
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def per_cta_volume(calib, evidence, phase=None, census=None):
    """Read/write bytes one CTA of this class is modeled to move.

    All three bases are per-CTA figures of the same class, strongest evidence
    first: the census is a whole-grid total that is divided by the grid it was
    measured on, the projection census is a lane-byte total that is divided by the
    CTAs it was observed on, and the sample record count is divided by the CTA
    count the sampler actually selected.
    """
    if census is not None or evidence.get('census'):
        census = census or evidence['census']
        need('native_read_lane_bytes' in census, 'Template census lacks lane bytes')
        grid = evidence['grid']
        need(grid >= 1, 'Census needs the grid it was measured on')
        return dict(read=census['native_read_lane_bytes'] / grid,
                    write=census['native_write_lane_bytes'] / grid,
                    basis=TEMPLATE_CENSUS_BASIS)
    observed = evidence.get('observed')
    if observed:
        ctas = evidence.get('ctas')
        classified = evidence.get('records') or 0
        need(ctas, 'Observed refusal census needs the CTA count it covered')
        need(classified, 'Observed refusal census needs the class record count')
        coverage = observed['records'] / classified
        total = observed['read'] + observed['write']
        if coverage >= OBSERVED_REFUSAL_COVERAGE_MIN and total > 0:
            # The class's own measured width, carried to the whole class because a
            # width belongs to the code, not to the records the projection refused.
            per_cta = classified / ctas * (total / observed['records'])
            share = observed['read'] / total
            return dict(read=per_cta * share, write=per_cta * (1 - share),
                        basis=OBSERVED_CENSUS_BASIS,
                        per_cta_records=classified / ctas,
                        bytes_per_record=total / observed['records'],
                        observed_widths=observed['widths'],
                        observed_coverage=coverage,
                        directionless_opcodes=observed['directionless_opcodes'])
    records = evidence.get('records')
    need(records, 'Modeled class needs its observed sampled record count')
    need(evidence.get('ctas'), 'Modeled class needs the CTA count it was observed on')
    per_cta_records = records / evidence['ctas']
    per_instruction = (calib.get('bytes_per_record_by_phase') or {}).get(phase)
    if per_instruction is None:
        per_instruction = calib['bytes_per_record']
    total = per_cta_records * per_instruction
    return dict(read=total * calib['read_share'],
                write=total * (1 - calib['read_share']),
                basis=calib['basis'], per_cta_records=per_cta_records,
                bytes_per_record=per_instruction)


def _issue_counts(wanted, spans):
    """Split a per-CTA issue budget over object spans, proportionally by bytes."""
    sizes = [hi - lo for lo, hi in spans]
    total = sum(sizes)
    counts = [int(wanted * size / total) for size in sizes]
    remainder = wanted - sum(counts)
    for index in sorted(range(len(spans)), key=lambda i: -sizes[i])[:remainder]:
        counts[index] += 1
    return counts


def _span_entries(lo, hi, direction, wanted, kernel, first_ordinal, policy_log):
    grid = list(kernel['grid_dims'])
    size = kernel['grid_size']
    span_bytes = hi - lo
    width, lanes = access_geometry(span_bytes)
    issue_bytes = width * lanes
    room = span_bytes // issue_bytes
    if wanted > room:
        policy_log.append(dict(policy='numeric_modeled_issue_budget_truncated', direction=direction,
                               requested=wanted, granted=room, span_bytes=span_bytes))
    wanted = max(1, min(wanted, room))
    entries = []
    if wanted * size * issue_bytes <= span_bytes:
        # Every CTA walks its own contiguous tile of the object.
        tile = wanted * issue_bytes
        x_stride = tile
        y_stride = x_stride * grid[0]
        z_stride = y_stride * grid[1]
        bases = [lo + index * issue_bytes for index in range(wanted)]
        policy = 'numeric_modeled_private_tile'
    else:
        # HBServe shared_template_arena: every CTA re-reads the same lines.
        x_stride = y_stride = z_stride = 0
        bases = [lo + index * issue_bytes for index in range(wanted)]
        policy = 'numeric_modeled_shared_arena'
    for base in bases:
        need(lo <= base and base + issue_bytes <= hi, 'Modeled issue outside its object')
    policy_log.append(dict(policy=policy, direction=direction, issues=len(bases),
                           access_width=width, active_lanes=lanes,
                           modeled_bytes=len(bases) * issue_bytes,
                           span_bytes=span_bytes, first_issue_ordinal=first_ordinal))
    for base in bases:
        ordinal = first_ordinal + len(entries)
        entries.append(dict(address_rules=[dict(intercept=base, cta_x_stride=x_stride,
                                                cta_y_stride=y_stride, cta_z_stride=z_stride)],
                            groups=[dict(pairs=['%d:%d' % (width, LANES - 1)])],
                            mask='0x%x' % ((1 << lanes) - 1), opcode=opcode_for(width, direction),
                            pc='0x0', ordinal=ordinal, signature_rank=ordinal,
                            sampled_timestamp_delta=0))
    return entries


def build_profile(kernel, launch, views, calib, cause, evidence, census=None, arena=()):
    """Build a modeled profile for one target launch.

    ``evidence`` is the sampled census of the target's own class: the same kernel
    binary, grid, and block, observed in this run.  ``arena`` is used when the
    launch has no bound tensor object at all (phase-global reductions).
    Returns ``(profile, summary)``; ``summary`` is the evidence block that the
    expansion copies into its manifest rows.
    """
    read_spans = object_spans(views, 'inputs')
    write_spans = object_spans(views, 'outputs')
    address_basis = 'target launch allocation context'
    if not read_spans and not write_spans:
        need(arena, 'Modeled class has no bound object to walk and no arena to fall back to')
        read_spans = list(arena)
        address_basis = 'representative_persistent_arena'
    volume = per_cta_volume(calib, evidence, phase=kernel.get('phase'), census=census)
    entries = []
    policies = []
    capped = False
    for direction, spans, wanted in (('read', read_spans, volume['read']),
                                     ('write', write_spans, volume['write'])):
        if not spans or wanted <= 0:
            continue
        issues = int(math.ceil(wanted / ISSUE_BYTES_ENVELOPE))
        if issues > MAX_ISSUES_PER_CTA:
            issues = MAX_ISSUES_PER_CTA
            capped = True
        for span, count in zip(spans, _issue_counts(issues, spans)):
            if count <= 0:
                continue
            entries.extend(_span_entries(span[0], span[1], direction, min(count, MAX_ISSUES_PER_CTA),
                                         kernel, len(entries), policies))
    need(entries, 'Modeled class produced no issue')
    profile = dict(schema=dict(SCHEMA),
                   status=MODELED_STATUS,
                   kernel=dict(kernel),
                   source=dict(launch=launch, modeled=True),
                   template=entries,
                   model=dict(cache_entry='ONE_CONTINUOUS_WARMUP_THEN_MEASUREMENT_STREAM',
                              complete_model=False,
                              ordering='modeled affine object walk; intra-class issue order not observed'),
                   modeling=dict(mode='numeric_modeled', cause=cause,
                                 exact_cross_layer_identity_claimed=False,
                                 not_claimed=list(NOT_CLAIMED),
                                 volume_basis=volume['basis'],
                                 policies=policies,
                                 issues_per_cta=len(entries),
                                 capped_by_budget=capped,
                                 address_basis=address_basis,
                                 evidence=dict(source_launch_key=evidence.get('source_launch_key'),
                                               selected_records=evidence.get('selected_records'),
                                               sampled_records=evidence.get('records'),
                                               observed_ctas=evidence.get('ctas'),
                                               observed_grid=evidence.get('grid'),
                                               whole_grid_selected=evidence.get('grid_whole'),
                                               observed_phase=evidence.get('phase'),
                                               observed_widths=volume.get('observed_widths'),
                                               observed_coverage=volume.get('observed_coverage'),
                                               directionless_opcodes=volume.get('directionless_opcodes')),
                                 calibration=dict(calib)))
    summary = dict(cause=cause, mode='numeric_modeled', basis=volume['basis'],
                   issues_per_cta=len(entries),
                   policies=sorted({row['policy'] for row in policies}),
                   address_basis=address_basis,
                   per_cta_records=volume.get('per_cta_records'),
                   bytes_per_record=volume.get('bytes_per_record'),
                   observed_widths=volume.get('observed_widths'),
                   observed_coverage=volume.get('observed_coverage'),
                   directionless_opcodes=volume.get('directionless_opcodes'),
                   read_spans=len(read_spans), write_spans=len(write_spans),
                   modeled_read_bytes_per_cta=sum(row.get('modeled_bytes', 0) for row in policies
                                                  if row['direction'] == 'read'),
                   modeled_write_bytes_per_cta=sum(row.get('modeled_bytes', 0) for row in policies
                                                   if row['direction'] == 'write'),
                   requested_read_bytes_per_cta=volume['read'],
                   requested_write_bytes_per_cta=volume['write'],
                   capped_by_budget=capped)
    return profile, summary


def validate(profile):
    """Self-check a modeled profile against the entry contract the engine parses."""
    need(profile['schema'] == SCHEMA, 'Modeled profile schema differs from the engine contract')
    need(str(profile['status']).startswith('PASS_'), 'Modeled profile status is not PASS')
    grid = profile['kernel']['grid_dims']
    need(sum(grid) >= 1 and all(x >= 1 for x in grid), 'Modeled grid is empty')
    for entry in profile['template']:
        need(len(entry['groups']) == len(entry['address_rules']), 'Group/rule count differs')
        need(int(entry['mask'], 0) != 0, 'Empty lane mask')
        for group in entry['groups']:
            lanes = 1
            for token in group['pairs']:
                delta, count = map(int, token.split(':'))
                need(delta != 0 and count >= 1, 'Empty lane delta token')
                lanes += count
            need(lanes == LANES, 'Incomplete lane delta sequence')
        for rule in entry['address_rules']:
            need(all(key in rule for key in ('intercept', 'cta_x_stride', 'cta_y_stride', 'cta_z_stride')),
                 'Modeled rule must carry explicit xyz CTA strides')
    return profile
