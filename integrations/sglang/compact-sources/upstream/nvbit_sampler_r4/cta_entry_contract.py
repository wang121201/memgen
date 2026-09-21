"""Independent selected CTA entry proof; packetless does not mean no L2 client."""
import math
SCHEMA='SG_CTA_WARP_ENTRY_V1'
SCOPE='selected_CTA_entry_and_instrumented_memory_packets_only'

def require(ok,message):
    if not ok:raise ValueError(message)

def validate_entry_proof(begin,end,packet_ctas):
    require(begin.get('entry_proof_schema')==SCHEMA,'entry protocol missing at begin')
    p=end.get('entry_proof');require(isinstance(p,dict) and p.get('schema')==SCHEMA,'entry proof absent')
    require(p.get('source_launch_key')==begin['source_launch_key']==end['source_launch_key'],'entry launch mismatch')
    require(p.get('entry_function_id')==begin['function_id'],'entry function mismatch')
    require(p.get('scope')==SCOPE and p.get('completion_sync_passed') is True,'entry proof scope/completion')
    selected=sorted(begin['fit_ctas']+begin['holdout_ctas'])
    require(len(selected)==len(set(selected)) and selected,'selected CTA duplicate/empty')
    threads=math.prod(begin['block']);require(0<threads<=1024,'entry block bound')
    warps=(threads+31)//32;mask=(1<<warps)-1
    require(p.get('warps_per_cta')==warps and p.get('columns')==['cta','calls','seen_warps','duplicate_warps','bad_lane_warps'],'entry proof geometry/columns')
    rows=p.get('rows');require(isinstance(rows,list) and len(rows)==len(selected),'entry row count')
    for c,row in zip(selected,rows):
        require(isinstance(row,list) and len(row)==5 and all(type(v) is int and 0<=v<(1<<64) for v in row),'entry row values')
        require(row==[c,warps,mask,0,0],'missing/duplicate warp entry or partial lane mask')
    packets=sorted(packet_ctas)
    require(packets==sorted(set(packets)) and set(packets)<=set(selected),'packet CTA outside executed set')
    require(end['selected_ctas_seen']==packets,'packet CTA census differs')
    empty=sorted(set(selected)-set(packets))
    require(p.get('packetless_ctas')==empty,'packetless partition mismatch')
    return dict(schema=SCHEMA,scope=SCOPE,executed_ctas=selected,packet_ctas=packets,packetless_ctas=empty,warp_entries=len(selected)*warps)
