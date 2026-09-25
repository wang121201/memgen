#!/usr/bin/env python3
"""Bind admitted sampled profiles across layers and prepare one MemGen stream.

Only compact address rules are translated. A missing profile stays unsupported,
never a zero-traffic kernel, unless --model-uncovered modeled is given: then the
refused class receives an explicit numeric_modeled profile built from the target
launch allocation context (integrations/sglang/memgen-adapter/model_uncovered.py),
its launches are counted separately, and the manifest states that the stream is
not fully exact. Private addresses without tensor metadata retain
their relative offsets in an explicit target-scoped synthetic allocation.
"""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import model_uncovered


def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def neutral(name):return re.sub(r'(?<=layers\.)\d+', '<L>',name)
def key(r):return (int(r['epoch_id']),int(r['epoch_launch_ordinal']))


def launch_rows(journal):
    counters=Counter();result={}
    for line in Path(journal).open():
        r=json.loads(line)
        if r.get('type')=='launch' and r.get('edge')=='before' and r.get('epoch_id'):
            k=(r['epoch_id'],counters[r['epoch_id']]);counters[r['epoch_id']]+=1
            result[k]=r
    return result


def entries_with_domains(profile):
    if 'structural_classes' not in profile:
        return [(r,None) for r in profile['template']]
    classes={c['class_id']:c for c in profile['structural_classes']}
    need(len(classes)==len(profile['structural_classes']),'Duplicate structural class')
    grid=profile['kernel']['grid_dims'];size=profile['kernel']['grid_size']
    selector=profile.get('structural_class_selector')
    if selector is not None:
        need(selector['kind']=='categorical_y','Unsupported structural class selector')
        by_y=selector['class_by_y'];need(len(by_y)==grid[1],'Structural y coverage')
        mapping=[by_y[(c//grid[0])%grid[1]] for c in range(size)]
    else:
        mapping=profile['cta_class_by_id'];need(len(mapping)==size,'Structural CTA coverage')
    need(set(mapping)==set(classes),'Structural class membership coverage')
    domains=defaultdict(list)
    for c,cid in enumerate(mapping):domains[cid].append(c)
    for cid,c in classes.items():
        need(all(0<=k<size and mapping[k]==cid for k in c['ctas']),'Observed CTA class mismatch')
    return [(r,domains[cid]) for cid,c in classes.items() for r in c['template']]


def rule_value(rule,grid,c):
    kind=rule.get('kind','coordinate_affine');x=c%grid[0];y=(c//grid[0])%grid[1];z=c//(grid[0]*grid[1])
    if kind=='exact_cta_base_table':return rule['bases_by_cta'][str(c)]
    if kind=='coordinate_x_axis_permutation':
        need(grid[1:]==[1,1],'Axis permutation requires one-dimensional grid')
        ext=rule['cta_x_input_extents'];order=rule['cta_x_output_axis_order']
        need(len(ext)==3 and math.prod(ext)==grid[0] and sorted(order)==[0,1,2],'Axis permutation domain')
        coords=[x//(ext[1]*ext[2]),(x//ext[2])%ext[1],x%ext[2]];mapped=0
        for axis in order:mapped=mapped*ext[axis]+coords[axis]
        return rule['intercept']+rule['element_stride']*mapped
    if kind=='coordinate_x_floor_quotient':return rule['intercept']+(x//rule['cta_x_divisor'])*rule['cta_x_quotient_stride']
    if kind=='coordinate_x_quotient_remainder_y_table_z_partition':
        value=rule['intercept']+(x//rule['cta_x_divisor'])*rule['cta_x_quotient_stride']+(x%rule['cta_x_divisor'])*rule['cta_x_remainder_stride']
    elif any(k in rule for k in ('cta_y_stride','cta_y_offsets','cta_z_stride')):
        value=rule['intercept']+x*rule['cta_x_stride']
    else:return rule['intercept']+c*rule['cta_x_stride']
    value+=rule['cta_y_offsets'][y] if 'cta_y_offsets' in rule else y*rule.get('cta_y_stride',0)
    value+=z*rule.get('cta_z_stride',0)
    if rule.get('cta_z_partition') is not None and z>=rule['cta_z_partition']:value+=rule['cta_z_partition_stride']
    return value


def rule_bounds(rule,grid,ctas=None):
    if ctas is not None:
        need(bool(ctas),'Empty structural class domain');values=[rule_value(rule,grid,c) for c in ctas]
        return min(values),max(values)
    kind=rule.get('kind','coordinate_affine')
    if kind=='exact_cta_base_table':
        values=list(rule['bases_by_cta'].values());return min(values),max(values)
    intercept=rule['intercept'];lo=hi=intercept
    def add(values):
        nonlocal lo,hi
        lo+=min(values);hi+=max(values)
    if kind=='coordinate_x_axis_permutation':
        add([0,(grid[0]-1)*rule['element_stride']]);return lo,hi
    if kind in ('coordinate_x_quotient_remainder_y_table_z_partition','coordinate_x_floor_quotient'):
        d=rule['cta_x_divisor'];q=rule['cta_x_quotient_stride'];r=rule.get('cta_x_remainder_stride',0)
        add([(x//d)*q+(x%d)*r for x in range(grid[0])])
    else:
        limit=grid[0] if any(k in rule for k in ('cta_y_stride','cta_y_offsets','cta_z_stride')) else math.prod(grid)
        add([0,(limit-1)*rule['cta_x_stride']])
    if 'cta_y_offsets' in rule:add(rule['cta_y_offsets'])
    else:add([0,(grid[1]-1)*rule.get('cta_y_stride',0)])
    zstride=rule.get('cta_z_stride',0)
    if 'cta_z_partition' in rule:
        d=rule['cta_z_partition'];stride=rule['cta_z_partition_stride']
        add([z*zstride+(stride if z>=d else 0) for z in range(grid[2])])
    else:add([0,(grid[2]-1)*zstride])
    return lo,hi


def normalize_backend_rules(profile):
    count=0;grid=profile['kernel']['grid_dims']
    for entry,_ in entries_with_domains(profile):
        for rule in entry['address_rules']:
            kind=rule.get('kind','coordinate_affine')
            if kind in ('coordinate_x_floor_quotient','coordinate_x_axis_permutation'):
                need(grid[1:]==[1,1],'Native backend special x rule requires one-dimensional grid')
            if kind=='coordinate_affine' and not any(k in rule for k in ('cta_y_stride','cta_y_offsets','cta_z_stride')) and grid[1:]!=[1,1]:
                # Frozen Python fitter uses flat CTA here. Lower exactly to
                # the frozen C++ generator's explicit xyz representation.
                rule['cta_y_stride']=rule['cta_x_stride']*grid[0]
                rule['cta_z_stride']=rule['cta_x_stride']*grid[0]*grid[1];count+=1
    return count


def lane_bounds(entry,group):
    offsets=[0]
    for value in group['pairs']:
        delta,count=map(int,value.split(':'))
        for _ in range(count):offsets.append(offsets[-1]+delta)
    need(len(offsets)==32,'Complete lane delta sequence required')
    mask=int(entry['mask'],0) if isinstance(entry['mask'],str) else entry['mask']
    active=[v for i,v in enumerate(offsets) if mask>>i&1]
    if not active:return 0,0
    bits=re.search(r'\.(?:U|S|B)?(128|64|32|16|8)(?:\.|$)',entry['opcode'])
    width=int(bits.group(1))//8 if bits else 4
    return min(active),max(active)+width


def translate(rule,delta):
    if rule.get('kind')=='exact_cta_base_table':
        rule['bases_by_cta']={k:v+delta for k,v in rule['bases_by_cta'].items()}
    else:rule['intercept']+=delta


def view_span(v):
    shape=v['shape']
    stride=v['stride_bytes'] if 'stride_bytes' in v else [x*v['element_size'] for x in v['stride_elements']]
    need(all(x>=0 for x in stride),'Negative strides unsupported')
    size=sum((d-1)*s for d,s in zip(shape,stride))+v['element_size'] if all(shape) else 0
    return int(v['data_address']),int(v['data_address'])+size


def layout(v):return (tuple(v['shape']),tuple(v.get('stride_bytes',v.get('stride_elements',[]))),v['dtype'],v['element_size'])


class Binder:
    def __init__(self,calls,metadata):
        self.calls={x['call_id']:x for x in calls};self.meta=metadata
        self.parameters={x['label']:x for x in metadata['parameters']}
        self.kv={x['label']:x for x in metadata['kv_buffers']}
        observed_ends=[r['base_address']+r['storage_nbytes'] for r in metadata.get('storage_roots',[])]
        observed_ends += [view_span(v)[1] for v in list(self.parameters.values())+list(self.kv.values())]
        self.unknown_cursor=max(0x600000000000,((max(observed_ends,default=0)+(1<<40)+4095)//4096)*4096)
        self.unknown_allocations=[]

    def contexts_with_hop(self,call_id):
        """Context views of a call, tagged with the call hop that reached them.

        Hop 0 is the launch's own call, which is where its operands live; higher
        hops are ancestor modules that inherited and reused the same buffers.
        """
        rows=[];call=self.calls.get(call_id);hop=0
        while call is not None:
            for field in ('inputs','outputs'):
                for v in call.get(field,[]):rows.append((hop,(neutral(call['module']),field,v['label']),v))
            call=self.calls.get(call.get('parent_call_id'));hop+=1
        return rows

    def contexts(self,call_id):
        return [(k,v) for _,k,v in self.contexts_with_hop(call_id)]

    def mapping(self,source,target):
        pairs=[]
        def add(s,t,kind,label,hop=0):
            if layout(s)==layout(t):
                lo,hi=view_span(s);tl,th=view_span(t)
                if hi>lo and th-tl==hi-lo:pairs.append(dict(begin=lo,end=hi,delta=tl-lo,kind=kind,label=label,hop=hop))
        layer=source['layer_id'];target_layer=target['layer_id']
        for name,s in self.parameters.items():
            match=re.search(r'\.layers\.(\d+)(?=\.|$)',name)
            if match and int(match.group(1))!=layer:continue
            target_name=re.sub(r'(?<=layers\.)\d+',str(target_layer),name) if match else name
            if target_name in self.parameters:add(s,self.parameters[target_name],'weights',target_name)
        for name,s in self.kv.items():
            if name.endswith('.'+str(layer)):
                target_name=name.rsplit('.',1)[0]+'.'+str(target_layer)
                if target_name in self.kv:add(s,self.kv[target_name],'KV',target_name)
        dst=defaultdict(list)
        for hop,k,v in self.contexts_with_hop(target['call_id']):dst[k].append((hop,v))
        for shop,k,s in self.contexts_with_hop(source['call_id']):
            for thop,t in dst[k]:add(s,t,'activation',repr(k),max(shop,thop))
        return pairs

    def rebind(self,profile,source,target):
        value=copy.deepcopy(profile);counts=Counter();details=[]
        normalized=normalize_backend_rules(value)
        if source['epoch_id']==target['epoch_id'] and source['epoch_launch_ordinal']==target['epoch_launch_ordinal']:
            return value,dict(status='SOURCE_OWN_PROFILE',rules={},normalized_legacy_flat_cta_rules=normalized)
        pairs=self.mapping(source,target);pending=[];resolutions=Counter()
        for ei,(entry,domain) in enumerate(entries_with_domains(value)):
            for gi,(rule,group) in enumerate(zip(entry['address_rules'],entry['groups'])):
                lo,hi=rule_bounds(rule,value['kernel']['grid_dims'],domain);ll,lh=lane_bounds(entry,group);lo+=ll;hi+=lh
                candidates=[m for m in pairs if m['begin']<=lo and hi<=m['end']]
                if candidates:
                    deltas={m['delta'] for m in candidates}
                    if len(deltas)>1:
                        # An ancestor module context inherits the buffers its children
                        # use, so one source address can appear both as an operand of
                        # the launch and as an inherited input of a parent module. The
                        # inherited view's counterpart in the target layer can sit
                        # elsewhere, which makes the union of containing views
                        # ambiguous even though the kernel's own operand is not.
                        # Attribute at the kernel: among the containing views reached
                        # in the fewest call hops the translation must be unique, and
                        # only then is it used. If it is not unique the rule is still
                        # refused, so nothing is guessed.
                        finest=min(m['hop'] for m in candidates)
                        innermost=[m for m in candidates if m['hop']==finest]
                        inner={m['delta'] for m in innermost}
                        if len(inner)>1:
                            raise ValueError('Ambiguous observed tensor binding: conflicting target deltas')
                        candidates=innermost;deltas=inner
                        resolutions['deepest_callsite_unique']+=1
                        resolutions['deepest_callsite_hop_%d'%finest]+=1
                    delta=deltas.pop();translate(rule,delta)
                    kind=next((m['kind'] for m in candidates if m['kind'] in ('weights','KV')), 'activation')
                    counts[kind]+=1
                else:pending.append((lo,hi,rule,ei,gi,'unobserved_private_object'))
        # Merge overlapping source address intervals before allocating. All
        # rules for an inferred private interval receive the same relocation.
        clusters=[]
        for item in sorted(pending,key=lambda x:x[0]):
            if clusters and item[0]<=clusters[-1]['end']:
                clusters[-1]['end']=max(clusters[-1]['end'],item[1]);clusters[-1]['items'].append(item)
            else:clusters.append(dict(begin=item[0],end=item[1],items=[item]))
        for c in clusters:
            source_base=c['begin']//4096*4096;size=((c['end']-source_base+4095)//4096)*4096
            base=self.unknown_cursor;self.unknown_cursor+=size+4096;delta=base-source_base
            need(self.unknown_cursor<(1<<63),'Modeled private address domain exhausted')
            for lo,hi,rule,ei,gi,why in c['items']:translate(rule,delta);counts['UNKNOWN_private_modeled']+=1
            row=dict(source_begin=c['begin'],source_end=c['end'],target_base=base,target_bytes=size,
                source_page_base=source_base,delta=delta,source_key=key(source),target_key=key(target),
                rules=len(c['items']),reason=sorted({x[-1] for x in c['items']}),
                assumption='target-kernel private allocation; preserves source offsets/alignment, no unobserved reuse asserted')
            details.append(row);self.unknown_allocations.append(row)
        # Observed counts remain evidence about the source template only.
        for field in ('native_reference_digest','independent_source_census'):
            if field in value:value['source_template_'+field]=value.pop(field)
        value['model'].update(layer_rebinding='same-process tensor view base translation',
            unknown_private_policy='disjoint target-kernel allocation preserving page offset',
            target_addresses_hardware_observed=False,hardware_accuracy_accepted=False)
        if resolutions:
            # Record which rule placed these addresses, so a reader can see that a
            # translated address came from the kernel's own operand view and not
            # from an inherited ancestor buffer.
            value['model']['ambiguous_binding_resolution']=dict(
                rule='innermost_callsite_view_must_be_unique',rules=dict(resolutions),
                scope='layer-to-layer rebinding only; translated addresses are not hardware-observed')
        return value,dict(status='PASS_MODELED_LAYER_PROFILE_BINDING',rules=dict(counts),private_allocations=details,
            ambiguous_binding_resolution=dict(resolutions),normalized_legacy_flat_cta_rules=normalized)


MODEL_CAUSE = (('Source profile', 'missing_template_profile'),
               ('Ambiguous observed tensor binding', 'ambiguous_address_binding'),
               ('Target kernel regime changed', 'template_regime_mismatch'))


def model_cause(message):
    for prefix, label in MODEL_CAUSE:
        if message.startswith(prefix):
            return label
    return 'other_exact_path_refusal'


def class_shape(launch):
    """Identity of a kernel class: binary, grid, and block. Phase-independent."""
    return model_uncovered.class_shape(launch)


def evidence_index(fitting, consumer):
    """Per-class evidence: the sampled record count with its CTA count, plus the
    fitted per-CTA census when the class was admitted."""
    return model_uncovered.class_evidence(consumer, fitting['kernels'], fitting['packed_profiles'])


def exact_binding(binder, launches, profiles, tk, sk):
    need(tk in launches and sk in profiles, 'Source profile missing/unsupported')
    source = profiles[sk]['source']['launch']
    target = dict(launches[tk], epoch_launch_ordinal=tk[1])
    for field in ('function_name', 'code_sha256', 'grid', 'block'):
        need(source[field] == target[field], 'Target kernel regime changed ' + field)
    value, receipt = binder.rebind(profiles[sk], source, target)
    return value, receipt, target


def storage_arena(binder):
    """Largest persistent allocation of the sampled host, used only as a labeled
    representative arena for a launch that binds no tensor object of its own."""
    roots=[(int(r['base_address']),int(r['base_address'])+int(r['storage_nbytes']))
           for r in binder.meta.get('storage_roots',[]) if int(r.get('storage_nbytes',0))>0]
    return [max(roots,key=lambda span:span[1]-span[0])] if roots else []


def modeled_binding(binder, launches, tk, sk, cause, calibration, evidence_by_shape):
    """No template is admitted for this class: model it from its own objects."""
    need(tk in launches, 'Modeled target absent from the launch census')
    target = dict(launches[tk], epoch_launch_ordinal=tk[1])
    shape = class_shape(target)
    evidence = evidence_by_shape.get(shape)
    need(evidence is not None, 'Modeled class %s absent from the fitting census' % (shape,))
    kernel = dict(name=target['function_name'], grid_dims=[int(x) for x in target['grid']],
                  grid_size=int(math.prod(target['grid'])), block_size=int(math.prod(target['block'])),
                  phase=target['phase'], id=0)
    profile, summary = model_uncovered.build_profile(
        kernel, target, binder.contexts(target['call_id']), calibration, cause,
        evidence, census=evidence.get('census'), arena=storage_arena(binder))
    return model_uncovered.validate(profile), summary, target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-output',type=Path,required=True)
    p.add_argument('--layer-bindings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-uncovered',choices=('refuse','modeled'),default='refuse',
        help='refuse keeps a class with no admitted template unpacked; modeled gives it '
             'a numeric_modeled object-volume profile from the target allocation context')
    a=p.parse_args();sample=a.sample_output
    finish=json.loads((sample/'finish.json').read_text());need(finish['status']=='PASS_SINGLE_LAYER_SAMPLES_AND_PROFILE_FITTING','Sample pipeline incomplete')
    modeling=a.model_uncovered=='modeled'
    calibration=None;evidence_by_shape={}
    if modeling:
        fitting=json.loads((sample/'profiles/receipt.json').read_text())
        need(fitting['status'].startswith('PASS_'),'Fitting receipt is not a passing run')
        consumer=json.loads((sample/'consumer.json').read_text())
        calibration=model_uncovered.calibrate(fitting['kernels'],fitting['packed_profiles'])
        evidence_by_shape=evidence_index(fitting,consumer)
    host=list((sample/'host').glob('process-*/finish.json'));need(len(host)==1,'Exactly one sampled host')
    hostdir=host[0].parent;h=json.loads(host[0].read_text())
    binding=json.loads(a.layer_bindings.read_text());need(h['input_contract']==binding['input_contract'],'Census/sample input contract differs')
    journal=list((sample/'observer').glob('process-*/launch-journal.jsonl'));need(len(journal)==1,'Same-process launch census required')
    launches=launch_rows(journal[0]);binder=Binder(json.loads((hostdir/'module_calls.json').read_text()),json.loads((hostdir/'tensor_metadata.json').read_text()))
    profiles={}
    for line in (sample/'profiles/profiles.index.jsonl').open():
        row=json.loads(line)
        with Path(row['path']).open('rb') as f:f.seek(row['offset']);raw=f.read(row['bytes'])
        need(hashlib.sha256(raw).hexdigest()==row['sha256'],'Sample profile content identity')
        profile=json.loads(raw);source=profile['source']['launch'];profiles[key(source)]=profile
    a.output.mkdir(parents=True,exist_ok=False);a.output=a.output.resolve()
    pack=a.output/'profiles.pack';index=a.output/'profiles.index.jsonl';app=a.output/'app.config';issue=a.output/'issue.config'
    accepted=[];unsupported=[];offset=0;app_rows=[];sm_rows=defaultdict(list)
    with pack.open('xb') as out,index.open('x') as ix:
        for row in binding['bindings']:
            tk,sk=tuple(row['target_key']),tuple(row['template_key']);mode='exact';summary=None
            try:
                v,receipt,target=exact_binding(binder,launches,profiles,tk,sk)
            except (ValueError,KeyError) as e:
                if not modeling:
                    unsupported.append(dict(target_key=tk,template_key=sk,phase=row['phase'],role=row['role'],reason=str(e)))
                    continue
                try:
                    v,summary,target=modeled_binding(binder,launches,tk,sk,model_cause(str(e)),calibration,evidence_by_shape)
                    mode='numeric_modeled'
                except (ValueError,KeyError) as inner:
                    unsupported.append(dict(target_key=tk,template_key=sk,phase=row['phase'],role=row['role'],
                        reason='%s | modeled completion refused: %s'%(e,inner)))
                    continue
            kid=len(accepted)+1;phase=('warmup/' if row['role']=='warmup' else '')+row['phase']
            v['kernel'].update(id=kid,phase=phase)
            v['status']='PASS_MODELED_LAYER_PROFILE_BINDING' if mode=='exact' else v['status']
            v['model'].update(cache_entry='ONE_CONTINUOUS_WARMUP_THEN_MEASUREMENT_STREAM',complete_model=False)
            v['model']['target_launch_key']='epoch-%d-launch-%d'%tk
            if mode!='exact':
                # Keep the r4 original-allocation guard able to refuse modeled rows
                # without a release re-pin: it already rejects any profile that
                # reports unobserved target addresses.
                v['model']['target_addresses_hardware_observed']=False
                v['model']['modeling_mode']='numeric_modeled'
            raw=canonical(v);out.write(raw)
            ix.write(json.dumps(dict(kernel_id=kid,path=str(pack),offset=offset,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),status=v['status']))+'\n');offset+=len(raw)
            k=v['kernel'];g=k['grid_dims']
            fields=dict(kernel_name=k['name'],llama_phase=phase,grid_dim_x=g[0],grid_dim_y=g[1],grid_dim_z=g[2],grid_size=k['grid_size'],block_size=k['block_size'])
            app_rows.extend('-kernel_%d_%s %s\n'%(kid,n,vv) for n,vv in fields.items())
            for cta in range(k['grid_size']):sm_rows[cta%48].append('(%d,%d,%x)'%(kid,cta,(cta//48)*1000000))
            entry=dict(kernel_id=kid,target_key=tk,template_key=sk,phase=phase,role=row['role'],binding=receipt,mode=mode)
            if summary is not None:entry['modeling']=summary
            accepted.append(entry)
    app.write_text(''.join(app_rows));issue.write_text(''.join('-trace_issued_sm_id_%d %s\n'%(sm,' '.join(values)) for sm,values in sorted(sm_rows.items())))
    # Only persistent classes have a sound global address-range interpretation.
    # Reused module I/O addresses are recorded in binding receipts, not promoted
    # to static whole-run activation ownership without a lifetime observer.
    ranges=[]
    for kind,vs in [('weights',binder.meta['parameters']),('KV',binder.meta['kv_buffers'])]:
        for view in vs:
            lo,hi=view_span(view)
            if lo<hi:ranges.append((lo,hi,kind))
    (a.output/'semantic.ranges').write_text(''.join('RANGE 0x%x 0x%x kind=%s\n'%r for r in sorted(set(ranges))))
    complete=not unsupported
    modeled_rows=[row for row in accepted if row['mode']!='exact']
    causes=Counter(row['modeling']['cause'] for row in modeled_rows)
    policies=Counter(policy for row in modeled_rows for policy in row['modeling']['policies'])
    # Which evidence sized the modeled traffic, and how much of it rests on the one
    # run-wide estimate. A reader must be able to see the measured share without
    # re-deriving it, and a class with a measured width must not look like a class
    # that only had the run median.
    basis_names=sorted({row['modeling']['basis'] for row in modeled_rows})
    volume_by_basis={b:dict(launches=sum(1 for row in modeled_rows if row['modeling']['basis']==b),
                            per_cta_bytes=int(sum((row['modeling'].get('requested_read_bytes_per_cta') or 0)
                                                  +(row['modeling'].get('requested_write_bytes_per_cta') or 0)
                                                  for row in modeled_rows if row['modeling']['basis']==b)))
                     for b in basis_names}
    measured_bytes=sum(v['per_cta_bytes'] for b,v in volume_by_basis.items()
                       if b in model_uncovered.MEASURED_BASES)
    modeled_bytes=sum(v['per_cta_bytes'] for v in volume_by_basis.values())
    modeled_ctas=sum(int(math.prod(launches[tuple(row['target_key'])]['grid'])) for row in modeled_rows)
    manifest=dict(schema='SGLANG_SAMPLED_LAYER_PACKED_EXPANSION_V1',status='PASS_COMPLETE_MODELED_PROFILE_EXPANSION' if complete else 'PARTIAL_PROFILE_EXPANSION_UNSUPPORTED_RETAINED',
        input_contract=h['input_contract'],sample_finish_sha256=sha(sample/'finish.json'),bindings_sha256=sha(a.layer_bindings),
        target_launches=len(binding['bindings']),packed_launches=len(accepted),unsupported_launches=len(unsupported),
        exact_launches=len(accepted)-len(modeled_rows),modeled_launches=len(modeled_rows),
        modeled_fraction=(len(modeled_rows)/len(accepted) if accepted else 0.0),modeled_ctas=modeled_ctas,
        modeled_by_cause=dict(causes),modeled_policies=dict(policies),
        modeled_volume_basis=volume_by_basis,
        modeled_measured_bytes_per_cta=measured_bytes,
        modeled_estimated_bytes_per_cta=modeled_bytes-measured_bytes,
        modeled_estimated_share=(0.0 if modeled_bytes==0 else (modeled_bytes-measured_bytes)/modeled_bytes),
        modeled_completion=a.model_uncovered,fully_exact=(complete and not modeled_rows),
        exact_cross_layer_identity_claimed=False,not_claimed=list(model_uncovered.NOT_CLAIMED),
        modeled_calibration=calibration,
        complete_declared_profile_stream=complete,complete_full_model=complete,full_native_address_coverage=False,
        hardware_accuracy_accepted=False,postcache_counts_multiplied=False,
        scheduling='CTA round robin across 48 SMs; fixed per-SM CTA time spacing, modeled not measured',
        cache_state='cold before warmup, L1 resets per kernel, L2 persists all layers/phases, no final dirty drain',
        semantic_scope='persistent weights/KV exact address membership; remaining static classification unknown; tensor/private binding ledger retained',
        unknown_private_allocations=len(binder.unknown_allocations),unknown_private_bytes=sum(x['target_bytes'] for x in binder.unknown_allocations),
        rejected_profiles_are_zero=False,allow_full_NCU_accuracy_comparison=False,
        allow_declared_estimate_NCU_comparison=complete,
        qualification='Profile-expanded estimate with modeled CTA placement and unknown private objects; not full native address accuracy',
        accepted=accepted,unsupported=unsupported)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('status','target_launches','packed_launches','unsupported_launches',
        'exact_launches','modeled_launches','modeled_fraction','modeled_by_cause','unknown_private_allocations')}))


if __name__=='__main__':main()
