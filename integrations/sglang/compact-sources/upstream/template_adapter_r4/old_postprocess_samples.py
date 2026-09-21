#!/usr/bin/env python3
"""Drain a qualified sampler replay into bounded RAM, then fit native templates.

stdin is normalized JSONL from stream_consumer.py after producer exit. No raw
records, expanded MemoryInst or templates are written to disk. Only the final
small analysis receipt is persisted. Unsupported kernels remain UNLOWERED.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time


def need(ok, why):
    if not ok:
        raise ValueError(why)


def deep_size(value):
    """Conservative retained-object accounting; sharing is not discounted."""
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(k) + deep_size(v) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(map(deep_size, value))
    return size


def encoded_size(raw):
    return sys.getsizeof(raw) + 16  # Frame bytes plus amortized list slot/capacity.


def analyze(stream, transport_path, *, max_records=400000, max_encoded_bytes=256 << 20,
            max_kernel_records=100000, max_decoded_bytes=256 << 20, max_generated_records=2000000,
            model_policy='strict', on_decoded_capture=None, on_compiled_template=None):
    from hbserve_adapter import SampleCollector, existing_core, CODEC, GENERATOR, HBSERVE
    from diagnostics import RejectionCensus, record_witness
    from memory_projection import project_record, MODEL_POLICIES, LDG_PREDICATE_ESTIMATE_LABEL
    need(model_policy in MODEL_POLICIES, 'unknown explicit memory model policy')
    codec, _ = existing_core()
    started = time.monotonic()
    captures, current = [], None
    keys, all_records, retained_records, encoded_bytes = set(), 0, 0, 0
    peak_encoded, peak_decoded = 0, 0
    for line in iter(lambda: stream.readline((1 << 20) + 1), b''):
        need(len(line) <= 1 << 20 and line.endswith(b'\n'), 'normalized record line bound/termination')
        row = json.loads(line)
        schema = row.get('schema')
        if schema == 'SG_KERNEL_SAMPLE_BEGIN_V1':
            need(current is None and len(captures) < 512, 'nested kernel or kernel-count cap')
            launch = row['source_launch_key']
            need(isinstance(launch, str) and launch and launch not in keys, 'missing/duplicate launch key')
            keys.add(launch)
            current = dict(begin=row, frames=[], records=0, packet_ctas=set(), error=None, rejection_census=RejectionCensus(),
                           first_failure_detail=None, model_rejected_records=0)
            try:
                current['collector'] = SampleCollector(row, max_records=max_kernel_records, model_policy=model_policy)
            except (KeyError, TypeError, ValueError) as e:
                current['error'] = type(e).__name__ + ': ' + str(e)
                current['first_failure_detail'] = dict(stage='SAMPLE_BEGIN', reason=str(e))
        elif schema == 'SG_MEMORY_PROJECTION_RECORD_V1':
            need(current is not None and row['source_launch_key'] == current['begin']['source_launch_key'],
                 'record outside its launch')
            current['records'] += 1
            current['packet_ctas'].add(row['cta_linear_id'])
            all_records += 1
            # A prior unsupported instruction invalidates this kernel but must
            # not hide the remaining rejected instruction classes. Classify
            # every selected record; retain only bounded scalar diagnostics.
            projection_error = None
            try:
                project_record(row)
                current['rejection_census'].observe(row)
            except (KeyError, TypeError, ValueError) as e:
                current['rejection_census'].observe(row, e)
                projection_error = e
                if model_policy != 'strict':
                    try:
                        project_record(row, model_policy)
                        projection_error = None
                    except (KeyError, TypeError, ValueError) as model_error:
                        projection_error = model_error
            if projection_error is not None:
                e = projection_error
                current['model_rejected_records'] += 1
                if current['error'] is None:
                    current['error'] = type(e).__name__ + ': ' + str(e)
                    current['first_failure_detail'] = dict(stage='MEMORY_PROJECTION', reason=str(e),
                                                           record=record_witness(row))
                    encoded_bytes -= sum(map(encoded_size, current['frames']))
                    retained_records -= len(current['frames'])
                    current['frames'].clear()
                    current.pop('collector', None)
            if current['error'] is not None:
                continue
            try:
                need(retained_records < max_records, 'total retained record cap')
                collector = current['collector']
                collector.record(row)
                gx, gy, _ = collector.begin['grid']
                x, y, z = row['cta']
                key = (x + gx * (y + gy * z), row['cta_warp_id'])
                m = collector.samples[key].pop()
                del collector.samples[key]
                raw = codec.encode(m)
                need(encoded_bytes + encoded_size(raw) <= max_encoded_bytes, 'encoded selected RAM byte cap')
                current['frames'].append(raw)
                encoded_bytes += encoded_size(raw)
                retained_records += 1
                peak_encoded = max(peak_encoded, encoded_bytes)
            except (KeyError, TypeError, ValueError) as e:
                current['error'] = type(e).__name__ + ': ' + str(e)
                current['first_failure_detail'] = dict(stage='SAMPLE_RETENTION', reason=str(e),
                                                       record=record_witness(row))
                encoded_bytes -= sum(map(encoded_size, current['frames']))
                retained_records -= len(current['frames'])
                current['frames'].clear()
                current.pop('collector', None)
        elif schema == 'SG_KERNEL_SAMPLE_END_V1':
            need(current is not None and row['source_launch_key'] == current['begin']['source_launch_key'],
                 'end outside its launch')
            need(row['selected_records'] == current['records'], 'local replay record-count closure')
            from cta_entry_contract import validate_entry_proof
            current['cta_entry']=validate_entry_proof(current['begin'],row,sorted(current['packet_ctas']))
            current['end'] = row
            captures.append(current)
            current = None
        else:
            raise ValueError('unknown normalized replay schema')
    need(current is None and captures, 'incomplete/empty replay')
    drained_seconds = time.monotonic() - started

    # Read after EOF only: waiting for this file before draining would deadlock
    # the upstream writer. It is small metadata, never an address trace.
    transport_path = Path(transport_path)
    need(transport_path.is_file() and not transport_path.is_symlink() and transport_path.stat().st_size <= 4 << 20,
         'qualified transport receipt missing/unbounded')
    raw_receipt = transport_path.read_bytes()
    transport = json.loads(raw_receipt)
    need(transport['status'] == 'PASS_SAMPLED_TRANSPORT_ONLY', 'upstream transport is not qualified')
    source_counts = {r['source_launch_key']: r['selected_records'] for r in transport['kernels']}
    need(len(source_counts) == len(transport['kernels']) and set(source_counts) == keys,
         'transport/replay launch bijection')
    need(transport['selected_records'] == all_records and
         all(source_counts[c['begin']['source_launch_key']] == c['records'] for c in captures),
         'transport/replay record census')
    for c in captures:
        entry=next(r for r in transport['kernels'] if r['source_launch_key']==c['begin']['source_launch_key'])
        need(entry['entry_proof']==c['end']['entry_proof'] and entry['cta_entry']==c['cta_entry'],'transport/replay entry proof differs')

    results = []
    for capture in captures:
        begin = capture['begin']
        result = dict(source_launch_key=begin['source_launch_key'], phase=begin.get('phase'),
                      layer_id=begin.get('layer_id'), kernel_name=begin.get('kernel_name'),
                      module_scope=begin.get('module_scope'), call_id=begin.get('call_id'),
                      epoch_id=begin.get('epoch_id'), epoch_launch_ordinal=begin.get('epoch_launch_ordinal'),
                      stream_u64=begin.get('stream_u64'), launch_attributes=begin.get('launch_attributes'),
                      code_sha256=begin['code_sha256'], grid=begin['grid'], block=begin['block'],
                      selected_records=capture['records'], status='UNLOWERED',
                      projection_classification=capture['rejection_census'].receipt(),
                      projection_classification_policy='strict', model_policy=model_policy,
                      model_policy_rejected_records=capture['model_rejected_records'],
                      first_failure_detail=capture['first_failure_detail'])
        errors = []
        details = []
        compiled = None
        if capture['error']:
            errors.append(capture['error'])
        else:
            collector = capture['collector']
            decoded = 0
            try:
                # Only one kernel is decoded at a time. Prior and future
                # kernels remain in the existing compact MemoryInst codec.
                for raw in capture['frames']:
                    m = codec.decode(raw)
                    decoded += deep_size(m) + 16
                    peak_decoded = max(peak_decoded, decoded)
                    need(decoded <= max_decoded_bytes, 'one-kernel decoded RAM byte cap')
                    collector.samples[m['block_id'], m['cta_warp']].append(m)
                peak_decoded = max(peak_decoded, decoded)
                if on_decoded_capture is not None:
                    try:
                        on_decoded_capture(collector, begin, capture['end'], transport)
                    except Exception as error:
                        raise RuntimeError('on_decoded_capture callback failed: ' + type(error).__name__ + ': ' + str(error)) from error
                for period in [1, 4]:
                    try:
                        compiled = collector.compile(capture['end'], transport, period=period)
                        break
                    except (KeyError, TypeError, ValueError) as e:
                        errors.append('period ' + str(period) + ': ' + type(e).__name__ + ': ' + str(e))
                        details.append(dict(period_candidate=period, reason=str(e),
                            detail=getattr(e, 'diagnostic', dict(stage='TEMPLATE_ADMISSION_CONTRACT', localized=False))))
            except (KeyError, TypeError, ValueError) as e:
                errors.append(type(e).__name__ + ': ' + str(e))
                details.append(dict(reason=str(e), detail=dict(stage='MEMORYINST_DECODE_OR_RAM_BUDGET', localized=False)))
            if compiled is not None:
                result.update(status=('ESTIMATED_NATIVE_GLOBAL_MEMORY_TEMPLATE' if compiled.estimated_sites else
                                      'QUALIFIED_NATIVE_GLOBAL_MEMORY_TEMPLATE'), admission=compiled.receipt)
                from template_census import census as analytic_template_census
                analytic = analytic_template_census(compiled)
                if analytic['status'] == 'COMPLETE_ADMITTED_GRID_MODEL':
                    result['generated_memory_trace_census'] = analytic
                elif not analytic.get('bounded_enumeration_fallback_eligible', False):
                    result['status'] = 'UNLOWERED'
                    result['generated_memory_trace_census'] = analytic
                else:
                    totals = collections.Counter({key: 0 for key in ['records', 'lane_references', 'read_bytes', 'write_bytes',
                        'read_bytes_source_mask_gated', 'read_bytes_effective_mask_upper_bound',
                        'candidate_ldg_source_mask_gated_read_bytes', 'candidate_ldg_effective_mask_upper_bound_read_bytes']})
                    try:
                        for m in compiled.iter_memory_instructions(kernel_id=0, sm_for_cta=lambda c: c % 48):
                            need(totals['records'] < max_generated_records, 'generated record census cap')
                            totals['records'] += 1
                            totals['lane_references'] += len(m['lanes'])
                            totals['write_bytes' if m['op'] == ord('W') else 'read_bytes'] += len(m['lanes']) * m['mem_width']
                            if m['op'] == ord('R'):
                                source_bytes = len(m['lanes']) * m['mem_width']
                                upper_bytes = source_bytes
                                if (m['function_id'], m['pc'], m['opcode']) in compiled.estimated_sites:
                                    upper_bytes = bin(m['mask']).count('1') * m['mem_width']
                                    totals['candidate_ldg_source_mask_gated_read_bytes'] += source_bytes
                                    totals['candidate_ldg_effective_mask_upper_bound_read_bytes'] += upper_bytes
                                totals['read_bytes_source_mask_gated'] += source_bytes
                                totals['read_bytes_effective_mask_upper_bound'] += upper_bytes
                        result['generated_memory_trace_census'] = dict(totals, status='COMPLETE_ADMITTED_GRID_MODEL',
                            traffic_domain='requested_global_lane_bytes_not_cache_or_DRAM_transactions',
                            memory_model_label=LDG_PREDICATE_ESTIMATE_LABEL if compiled.estimated_sites else None,
                            hardware_source_predicate_validated=False)
                    except (KeyError, TypeError, ValueError) as e:
                        result['status'] = 'UNLOWERED'
                        result['generated_memory_trace_census'] = dict(totals, status='INCOMPLETE_MODEL_CENSUS',
                                                                      reason=type(e).__name__ + ': ' + str(e))
        result['candidate_rejections'] = errors
        result['candidate_diagnostics'] = details
        if compiled is not None and on_compiled_template is not None:
            try:
                on_compiled_template(compiled, result)
            except Exception as error:
                raise RuntimeError('on_compiled_template callback failed: ' + type(error).__name__ + ': ' + str(error)) from error
        results.append(result)
        encoded_bytes -= sum(map(encoded_size, capture['frames']))
        capture['frames'].clear()
        capture.pop('collector', None)
        compiled = None
        collector = None
    files = [Path(__file__).resolve(), Path(__file__).resolve().with_name('hbserve_adapter.py'),
             Path(__file__).resolve().with_name('memory_projection.py'), Path(__file__).resolve().with_name('diagnostics.py'),
             Path(__file__).resolve().with_name('template_census.py'), Path(__file__).resolve().with_name('storage_windows.py'), CODEC, GENERATOR]
    files += [HBSERVE / 'hbserve/traces/_reference' / name for name in
              ['compact_request_template.py', 'shape_aware_cta_rules.py', 'semantic_embedding_generator.py']]
    return dict(schema='SGLANG_NATIVE_TEMPLATE_POSTPROCESS_V1', diagnostic_revision=2, model_policy=model_policy,
        census_execution_policy='existing_HBServe_analytic_RAM_census_with_bounded_enumeration_fallback',
        max_generated_records_scope='enumeration_fallback_only_not_analytic_modeled_output_count',
        status='PASS_SAMPLED_TEMPLATE_ANALYSIS_WITH_EXPLICIT_UNLOWERED',
        transport_receipt_sha256=hashlib.sha256(raw_receipt).hexdigest(), transport_qualified=True,
        source_pins=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
        source_launches=len(results), selected_records=all_records,
        qualified_template_launches=sum(r['status'] == 'QUALIFIED_NATIVE_GLOBAL_MEMORY_TEMPLATE' for r in results),
        estimated_template_launches=sum(r['status'] == 'ESTIMATED_NATIVE_GLOBAL_MEMORY_TEMPLATE' for r in results),
        modeled_template_launches=sum(r['status'] in ['QUALIFIED_NATIVE_GLOBAL_MEMORY_TEMPLATE',
                                                      'ESTIMATED_NATIVE_GLOBAL_MEMORY_TEMPLATE'] for r in results),
        unlowered_launches=sum(r['status'] == 'UNLOWERED' for r in results),
        replay_drain_seconds=drained_seconds, total_seconds=time.monotonic() - started,
        peak_retained_encoded_record_object_bytes=peak_encoded, peak_one_kernel_decoded_estimated_bytes=peak_decoded,
        limits=dict(max_records=max_records, max_encoded_bytes=max_encoded_bytes, max_kernel_records=max_kernel_records,
                    max_decoded_bytes=max_decoded_bytes, max_generated_records=max_generated_records),
        kernels=results, actual_layers_expanded=0, cross_layer_expansion=False,
        gtsim_executed=False, hardware_timing_qualified=False, complete_model_trace=False,
        raw_or_expanded_trace_saved=False)


def main():
    from memory_projection import MODEL_POLICIES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--transport-receipt', required=True, type=Path)
    p.add_argument('--hbserve-source-root', type=Path)
    p.add_argument('--memoryinst-codec', type=Path)
    p.add_argument('--max-records', type=int, default=400000)
    p.add_argument('--max-encoded-bytes', type=int, default=256 << 20)
    p.add_argument('--max-kernel-records', type=int, default=100000)
    p.add_argument('--max-decoded-bytes', type=int, default=256 << 20)
    p.add_argument('--max-generated-records', type=int, default=2000000)
    p.add_argument('--model-policy', choices=MODEL_POLICIES,
                   default=os.environ.get('SG_LDG_SOURCE_PREDICATE_POLICY', os.environ.get('SG_TEMPLATE_MODEL_POLICY', 'strict')))
    a = p.parse_args()
    if a.hbserve_source_root:
        os.environ['SG_HBSERVE_SOURCE_ROOT'] = str(a.hbserve_source_root)
    if a.memoryinst_codec:
        os.environ['SG_MEMORYINST_CODEC'] = str(a.memoryinst_codec)
    try:
        need(stat.S_ISFIFO(os.fstat(0).st_mode), 'normalized samples must arrive over a pipe, not a trace file')
        limits = {k: getattr(a, k) for k in ['max_records', 'max_encoded_bytes', 'max_kernel_records',
                                           'max_decoded_bytes', 'max_generated_records']}
        need(all(v > 0 for v in limits.values()), 'non-positive RAM/census limit')
        result = analyze(sys.stdin.buffer, a.transport_receipt, model_policy=a.model_policy, **limits)
        code = 0
    except Exception as e:
        result = dict(schema='SGLANG_NATIVE_TEMPLATE_POSTPROCESS_V1', status='FAIL_TEMPLATE_POSTPROCESS',
                      error=type(e).__name__ + ': ' + str(e), raw_or_expanded_trace_saved=False,
                      gtsim_executed=False, complete_model_trace=False)
        code = 1
    data = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    need(len(data) <= 4 << 20, 'small aggregate receipt cap')
    a.output.mkdir(parents=True, exist_ok=True)
    fd = os.open(a.output / 'receipt.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
