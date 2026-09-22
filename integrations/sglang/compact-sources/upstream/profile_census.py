"""Exact sector census without enumerating invariant address residues.

This only aggregates already-fitted rules. It never relaxes native lane-address
validation or changes the address stream sent to the C++ cache backend.
"""
from functools import lru_cache

FIELDS=('mem_insts','lane_accesses','read_sector_requests','write_sector_requests',
        'atomic_sector_requests','native_read_lane_bytes','native_write_lane_bytes','native_atomic_lane_bytes')

def invariant_modulo(rule,grid,ctas):
 kind=rule.get('kind')
 if kind=='exact_cta_base_table':
  values=rule['bases_by_cta']
  assert all(str(c) in values for c in ctas),'exact rule omits a generated CTA'
  return len({int(values[str(c)])%32 for c in ctas})==1
 if kind=='coordinate_x_quotient_remainder_y_table_z_partition':
  assert rule['cta_x_divisor']>0
  strides=[rule['cta_x_quotient_stride'],rule['cta_x_remainder_stride'],rule['cta_z_stride']]
  offsets=rule['cta_y_offsets'];assert len(offsets)==grid[1]
 elif kind=='coordinate_x_axis_permutation':
  # Validity of the geometry is also checked by predict_bases below.
  strides=[rule['element_stride']];offsets=[0]
 elif kind=='coordinate_x_floor_quotient':
  assert rule['cta_x_divisor']>0
  strides=[rule['cta_x_quotient_stride']];offsets=[0]
 elif kind is None:
  strides=[rule['cta_x_stride']];offsets=[0]
  if any(k in rule for k in ['cta_y_stride','cta_y_offsets','cta_z_stride']):
   strides.append(rule['cta_z_stride'])
   if 'cta_y_offsets' in rule:offsets=rule['cta_y_offsets'];assert len(offsets)==grid[1]
   else:strides.append(rule['cta_y_stride'])
 else:return False
 if 'cta_z_partition' in rule and rule['cta_z_partition'] is not None:strides.append(rule['cta_z_partition_stride'])
 return all(value%32==0 for value in strides) and len({value%32 for value in offsets})==1

def count_profile(profile,adapter,*,fast=True):
 expected=dict.fromkeys(FIELDS,0);grid=tuple(profile['kernel']['grid_dims'])
 domains={}
 for c,template in enumerate(adapter.template_by_cta(profile)):
  key=id(template)
  if key not in domains:domains[key]=(template,[])
  domains[key][1].append(c)
 @lru_cache(maxsize=4096)
 def counts(mask,pairs,width):
  offsets=[off for lane,off in adapter.lane_offsets(mask,pairs)]
  return len(offsets),tuple(len({sector for off in offsets for sector in range((mod+off)//32,(mod+off+width-1)//32+1)}) for mod in range(32))
 for template,ctas in domains.values():
  n=len(ctas)
  for entry in template:
   assert len(entry['groups'])==len(entry['address_rules'])==1
   direction,width=adapter.direction_width(entry['opcode']);prefix={'R':'read','W':'write','A':'atomic'}[direction]
   lanes,sectors=counts(entry['mask'],tuple(entry['groups'][0]['pairs']),width)
   expected['mem_insts']+=n;expected['lane_accesses']+=n*lanes;expected['native_'+prefix+'_lane_bytes']+=n*lanes*width
   rule=entry['address_rules'][0]
   if fast and invariant_modulo(rule,grid,ctas):
    total=n*sectors[adapter.rules.predict_bases(entry,ctas[0],grid)[0]%32]
   else:total=sum(sectors[adapter.rules.predict_bases(entry,c,grid)[0]%32] for c in ctas)
   expected[prefix+'_sector_requests']+=total
 return expected
