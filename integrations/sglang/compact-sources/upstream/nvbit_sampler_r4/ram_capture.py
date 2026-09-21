"""Optional bounded selected-sample RAM handoff; no kernel simulation during GPU.
Owner calls capture() while its GPU child runs, joins producer/observer externally,
then invokes replay() only after qualify_transport. No file path or spill exists.
"""
import hashlib
from packet_stream import read_stream, qualify_transport, need, MAX_WIRE

CHUNK=1<<20

class SampleRAM:
    def __init__(self,max_wire_bytes):
        need(type(max_wire_bytes) is int and 1024<=max_wire_bytes<=MAX_WIRE,'sample RAM cap')
        self.limit=max_wire_bytes;self.chunks=[];self.size=0;self.receipt=None;self.qualified=None
    def append(self,b):
        need(self.size+len(b)<=self.limit,'selected-sample RAM quota')
        v=memoryview(b)
        while v:
            index,offset=divmod(self.size,CHUNK)
            if index==len(self.chunks):
                # Final chunk is exact remaining capacity; never round beyond cap.
                self.chunks.append(bytearray(min(CHUNK,self.limit-index*CHUNK)))
            n=min(len(v),len(self.chunks[index])-offset)
            self.chunks[index][offset:offset+n]=v[:n];self.size+=n;v=v[n:]
    def capture(self,pipe):
        need(self.receipt is None and self.size==0,'sample RAM reuse forbidden')
        owner=self
        class Tee:
            def read(self,n):
                b=pipe.read(n)
                if b:owner.append(b)
                return b
        try:
            self.receipt=read_stream(Tee(),max_wire_bytes=self.limit)
            return dict(self.receipt,retained_selected_wire_bytes=self.size,
                        allocated_sample_ram_bytes=sum(map(len,self.chunks)))
        except BaseException:
            self.clear();raise
    def qualify(self,**gates):
        self.qualified=qualify_transport(self.receipt,**gates)
        return self.qualified
    def replay(self,**callbacks):
        need(self.qualified is not None,'GPU producer must exit and transport must qualify before downstream replay')
        owner=self
        class Cursor:
            pos=0
            def read(self,n):
                out=bytearray();n=min(n,owner.size-self.pos)
                while n:
                    i,o=divmod(self.pos,CHUNK);take=min(n,len(owner.chunks[i])-o)
                    out.extend(memoryview(owner.chunks[i])[o:o+take]);self.pos+=take;n-=take
                return bytes(out)
        result=read_stream(Cursor(),max_wire_bytes=self.limit,**callbacks)
        need(result==self.receipt,'RAM stream differs from initial capture')
        return result
    def clear(self):
        self.chunks.clear();self.size=0;self.receipt=None;self.qualified=None
