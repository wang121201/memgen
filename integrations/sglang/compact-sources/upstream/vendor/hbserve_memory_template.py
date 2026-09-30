"""Train a current-capture CTA template in RAM and stream a held-out-checked kernel.

This first admission experiment handles affine translation along actual CTA-x,
with separate CTA-y/z strata. It does not assume BF16/Q8 compatibility, layer
repetition, or a new context's shape. Full source is read as an independent
holdout oracle; only x=0/1 columns are retained. No template or trace is saved.
"""
from pathlib import Path
import collections,copy,hashlib,io,json,struct,sys,zlib

def need(ok,why):
    if not ok:raise ValueError(why)

def exact(stream,n):
    b=bytearray()
    while len(b)<n:
        piece=stream.read(n-len(b));need(bool(piece),'truncated pipe');b.extend(piece)
    return bytes(b)

def uint(stream,n):return int.from_bytes(exact(stream,n),'little')

def decode(b):
    pos=0
    def get():
        nonlocal pos
        value=0
        for shift in range(0,64,7):
            need(pos<len(b),'truncated varint');c=b[pos];pos+=1
            need(shift!=63 or c<=1,'varint overflow');value|=(c&127)<<shift
            if c<128:
                need(shift==0 or c!=0,'noncanonical varint');return value
        raise ValueError('bad varint')
    m={}
    for key in ['kernel_id','block_id','sm_id','seq','pc']:m[key]=get()
    m['kernel_id']-=2147483648
    n=get();need(n<=4096 and pos+n<=len(b),'opcode bound');m['opcode']=b[pos:pos+n].decode();pos+=n
    for key in ['mask','timestamp','mem_width','op','has_space_metadata','capture_seq','full_clock','local_warp_owner','cta_warp','function_id']:m[key]=get()
    n=get();need(n<=8192,'lane bound');lanes=[];previous=0
    for _ in range(n):
        code=get();address=get()^previous;previous=address;flags=get();need(flags<=3,'lane flags')
        lanes.append(dict(lane=code&31,ref_id=code>>5,addr=address,is_local=flags&1,local_offset=get() if flags&2 else 0))
    need(pos==len(b),'trailing instruction bytes');m['lanes']=lanes;return m

def encode(m):
    b=bytearray()
    def put(n):
        need(0<=n<1<<64,'integer out of range')
        while n>=128:b.append((n&127)|128);n>>=7
        b.append(n)
    put(m['kernel_id']+2147483648)
    for key in ['block_id','sm_id','seq','pc']:put(m[key])
    opcode=m['opcode'].encode();put(len(opcode));b.extend(opcode)
    for key in ['mask','timestamp','mem_width','op','has_space_metadata','capture_seq','full_clock','local_warp_owner','cta_warp','function_id']:put(m[key])
    put(len(m['lanes']));previous=0
    for lane in m['lanes']:
        put((lane['ref_id']<<5)|lane['lane']);put(lane['addr']^previous);previous=lane['addr']
        put(int(bool(lane['is_local']))|(2 if lane['local_offset'] else 0))
        if lane['local_offset']:put(lane['local_offset'])
    need(len(b)<=1<<20,'frame bound');return bytes(b)

def semantic_bytes(m,addresses=True):
    # Per-warp program order remains in the digest update order. Deliberately
    # excludes capture delivery sequence, clocks, SM and target CTA identity.
    keys=['pc','opcode','mask','mem_width','op','has_space_metadata','cta_warp','function_id']
    value={key:m[key] for key in keys}
    value['lanes']=[{k:v for k,v in lane.items() if addresses or k!='addr'} for lane in m['lanes']]
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()+b'\n'

def read_kernel(stream,kernel,shape,period):
    gx,gy,gz=shape;grid=gx*gy*gz
    fitting=[0,1] if period==1 else [0,1,period]
    need(uint(stream,4)==kernel,'kernel order')
    samples=collections.defaultdict(list);digests={};counts=collections.Counter();sms={};records=0;sample_bytes=0
    while True:
        n=uint(stream,4)
        if not n:break
        need(n<=1<<20,'source frame bound');crc=uint(stream,4);raw=exact(stream,n)
        need(zlib.crc32(raw)==crc,'source CRC');m=decode(raw);need(encode(m)==raw,'lossless codec roundtrip')
        cta=m['block_id'];need(m['kernel_id']==kernel and 0<=cta<grid,'source target identity')
        need(cta not in sms or sms[cta]==m['sm_id'],'inconsistent CTA/SM');sms[cta]=m['sm_id']
        key=cta,m['cta_warp'];digests.setdefault(key,hashlib.sha256()).update(semantic_bytes(m));counts[key]+=1
        if cta%gx in fitting:
            sample_bytes+=len(raw);need(sample_bytes<=64<<20,'fitting samples exceed 64 MiB')
            samples[key].append(m)
        records+=1
    need(uint(stream,8)==records,'source footer')
    need(set(sms)==set(range(grid)),'source does not contain every target CTA')
    need(gx>max(fitting) and gx>len(fitting),'experiment requires fitting coordinates and a non-fitting CTA-x column')
    return samples,{k:v.hexdigest() for k,v in digests.items()},counts,sms,records,sample_bytes

def prepare(samples,kernel,shape,period):
    from hbserve.traces._reference.phase_aware_cta_generator import PreparedPhaseGenerator
    compact=struct.Struct('<HIHHBB');body=bytearray();bundles=[];extras=[];rules=[];windows={}
    def object_index(address):
        window=address>>32
        if window not in windows:windows[window]=len(windows)
        return windows[window]
    gx,gy,gz=shape
    need(period in [1,4],'only declared affine or source-informed period-4 candidates are implemented')
    fitting=[0,1] if period==1 else [0,1,period]
    source_keys=sorted((c,w) for c,w in samples if c%gx==0)
    for phase in fitting[1:]:
        need({(c+phase,w) for c,w in source_keys}=={(c,w) for c,w in samples if c%gx==phase},'training warp programs differ')
    for cta,warp in source_keys:
        cta_y=(cta//gx)%gy;cta_z=cta//(gx*gy)
        a=samples[cta,warp];training=[samples[cta+phase,warp] for phase in fitting]
        need(all(len(a)==len(b) for b in training),'training instruction counts differ')
        for ordinal,left in enumerate(a):
            others=[b[ordinal] for b in training]
            need(all(semantic_bytes(left,False)==semantic_bytes(right,False) for right in others),'training instruction/mask/width/role differs')
            need(left['op'] in [ord('R'),ord('W')],'unsupported source operation')
            begin=len(body)//12;strides={}
            for lane_index,x in enumerate(left['lanes']):
                need(not x['is_local'] and not x['local_offset'],'local source requires another template contract')
                addresses=[right['lanes'][lane_index]['addr'] for right in others]
                need(all(x['addr']>>32==v>>32 for v in addresses),'training address window changes')
                obj=object_index(x['addr']);deltas=tuple(v-x['addr'] for v in addresses)
                need(obj not in strides or strides[obj]==deltas,'nonuniform lane translation in bundle');strides[obj]=deltas
                offset=x['addr']&0xffffffff;need(offset+left['mem_width']<=1<<32,'lane crosses address window')
                body.extend(compact.pack(obj,offset,kernel,left['mem_width'],int(left['op']==ord('W')),int('.EF' in left['opcode'])))
            extra=copy.deepcopy(left);extras.append(extra)
            if not left['lanes']:continue
            row=dict(bundle_index=len(bundles),kernel_ordinal=kernel,cta_x=0,cta_y=cta_y,cta_z=cta_z,
                     warp_in_cta=warp,warp_program_ordinal=ordinal,request_ordinal_begin=begin,request_ordinal_end_exclusive=len(body)//12)
            row['extra_index']=len(extras)-1;bundles.append(row)
            for obj,deltas in strides.items():
                rule=dict(object_index=obj,cta_y=cta_y,cta_z=cta_z,warp_in_cta=warp,warp_program_ordinal=ordinal,
                          validation=dict(status='UNVALIDATED_CANDIDATE'))
                if period==1:rule.update(kind='affine',x_stride_bytes=deltas[1])
                else:rule.update(kind='tiled_swizzle',period=period,quotient_stride_bytes=deltas[2],phase_offsets_bytes=[i*deltas[1] for i in range(period)])
                rules.append(rule)
    need(body and bundles,'no generated memory requests')
    objects=[dict(source_name='address transport window',stable_name='address transport window',kind='transport_window') for _ in windows]
    policy=dict(kind='phase_rules',source_cta_x=0,rules=rules)
    prepared=PreparedPhaseGenerator([1<<32]*len(windows),[1<<32]*len(windows),objects,{kernel:policy},{(kernel,0):bundles})
    # Only the low-level in-memory generator is invoked here. Its file-based
    # prepare_generator validation is not claimed. Candidate rules must pass
    # the independent non-fitting CTA oracle below before output is emitted.
    return prepared,io.BytesIO(body),bundles,extras,{v:k for k,v in windows.items()},rules

def generated(prepared,source,bundles,extras,windows,kernel,shape,sms):
    from hbserve.traces._reference.phase_aware_cta_generator import generated_bundle_records
    seq=0
    gx,gy,gz=shape
    for target_x in range(gx):
        translated={}
        for bundle,records in generated_bundle_records(prepared=prepared,source=source,kernel_ordinal=kernel,target_cta_x=target_x):
            i=bundle['extra_index'];m=copy.deepcopy(extras[i]);need(len(records)==len(m['lanes']),'generated lane cardinality')
            for lane,record in zip(m['lanes'],records):
                obj,offset,kid,width,op,flags=record
                need(kid==kernel and width==m['mem_width'] and op==int(m['op']==ord('W')),'generated lane metadata drift')
                lane['addr']=(windows[obj]<<32)|offset
            translated[i]=m
        for i,original in enumerate(extras):
            m=translated.get(i)
            if m is None:need(not original['lanes'],'missing generated bundle');m=copy.deepcopy(original)
            cta=original['block_id']+target_x
            m.update(block_id=cta,sm_id=sms[cta],seq=seq,timestamp=seq,capture_seq=0,full_clock=0,local_warp_owner=0)
            seq+=1;yield m

def run(stream,target,kernels,metadata,receipt,period=1):
    fitting=[0,1] if period==1 else [0,1,period]
    need(exact(stream,8)==b'HBFONL01' and uint(stream,8)==len(kernels),'source stream census')
    target.write(b'HBFONL01'+struct.pack('<Q',len(kernels)));rows=[]
    for kernel in kernels:
        shape=tuple(metadata[kernel]['grid_dim_'+axis] for axis in 'xyz');gx,gy,gz=shape;grid=gx*gy*gz
        need(grid==metadata[kernel]['grid_size'],'actual grid shape mismatch')
        samples,expected,counts,sms,source_count,sample_bytes=read_kernel(stream,kernel,shape,period)
        prepared,source,bundles,extras,windows,rules=prepare(samples,kernel,shape,period)
        observed={};generated_counts=collections.Counter()
        for m in generated(prepared,source,bundles,extras,windows,kernel,shape,sms):
            key=m['block_id'],m['cta_warp'];observed.setdefault(key,hashlib.sha256()).update(semantic_bytes(m));generated_counts[key]+=1
        actual={k:v.hexdigest() for k,v in observed.items()}
        receipt['candidate_diagnostics']=dict(period=period,source_records=source_count,generated_records=sum(generated_counts.values()),
            count_mismatch_warps=sum(counts[k]!=generated_counts[k] for k in set(counts)|set(generated_counts)),
            semantic_mismatch_warps=sum(expected.get(k)!=actual.get(k) for k in set(expected)|set(actual)))
        need(counts==generated_counts and expected==actual,'independent held-out CTA source semantics mismatch')
        target.write(struct.pack('<I',kernel));count=0
        for m in generated(prepared,source,bundles,extras,windows,kernel,shape,sms):
            raw=encode(m);target.write(struct.pack('<II',len(raw),zlib.crc32(raw))+raw);count+=1
        target.write(struct.pack('<IQ',0,count));target.flush()
        rows.append(dict(kernel_id=kernel,status='PASS_CURRENT_KERNEL_ALL_NONFITTING_CTA_SEMANTICS',grid_ctas=grid,
                         fitting_ctas=len(fitting)*gy*gz,heldout_ctas=(gx-len(fitting))*gy*gz,source_records=source_count,generated_records=count,
                         sample_records=sum(map(len,samples.values())),sample_encoded_bytes=sample_bytes,
                         retained_template_bytes=len(source.getbuffer()),translation_rules=len(rules),period=period,
                         generation_order='CTA-x then source CTA-y/z and warp program order',
                         source_timestamps_reproduced=False,source_sm_placement_reused=True))
        print(json.dumps(dict(stage='current-q8-template',kernel=kernel,records=count,heldout_ctas=(gx-len(fitting))*gy*gz)),file=sys.stderr,flush=True)
    need(stream.read(1)==b'','trailing source bytes');target.flush()
    receipt.update(status='PASS_SELECTED_CURRENT_Q8_KERNEL_TEMPLATES_NOT_FULL_MODEL_OR_HARDWARE',kernels=rows)
