#!/usr/bin/env python3
"""Differential census tests against independent per-lane byte enumeration."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys
import types
import unittest

ROOT=Path(__file__).resolve().parents[2]
UP=ROOT/'integrations/sglang/compact-sources/upstream'
sys.path.insert(0,str(UP))
sys.path.insert(0,str(UP/'template_adapter_r4'))
from profile_census import count_profile
from memory_projection import project_record
from hbserve_adapter import SampleCollector
import sglang_sample_to_packed as packed
spec=importlib.util.spec_from_file_location('rules',ROOT/'release/workflow/hyfiss_sampled_sass_trace_profile_rules_r15.py')
rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)

def templates(profile):
    if 'templates' in profile:return profile['templates']
    return [profile['template']]*profile['kernel']['grid_size']

def direction_width(opcode):
    direction,bits=opcode.split('.')
    return {'LDG':'R','STG':'W','ATOM':'A'}[direction],int(bits)//8

adapter=types.SimpleNamespace(template_by_cta=templates,direction_width=direction_width,
    lane_offsets=lambda mask,pairs:rules.active_lane_offsets(mask,list(pairs)),rules=rules)

def naive(profile):
    result=dict.fromkeys(('mem_insts','lane_accesses','read_sector_requests','write_sector_requests',
        'atomic_sector_requests','native_read_lane_bytes','native_write_lane_bytes','native_atomic_lane_bytes'),0)
    for c,template in enumerate(templates(profile)):
        for entry in template:
            direction,width=direction_width(entry['opcode']);name={'R':'read','W':'write','A':'atomic'}[direction]
            base=rules.predict_bases(entry,c,tuple(profile['kernel']['grid_dims']))[0]
            offsets=list(rules.active_lane_offsets(entry['mask'],entry['groups'][0]['pairs']))
            sectors={((base+off+b)//32) for _,off in offsets for b in range(width)}
            result['mem_insts']+=1;result['lane_accesses']+=len(offsets)
            result[name+'_sector_requests']+=len(sectors)
            result['native_'+name+'_lane_bytes']+=len(offsets)*width
    return result

def profile(rule,grid=(37,1,1),width=32,mask='ffffffff',stride=4):
    entry=dict(opcode='LDG.'+str(width),mask=mask,groups=[dict(pairs=[str(stride)+':31'])],address_rules=[rule])
    return dict(kernel=dict(grid_dims=list(grid),grid_size=grid[0]*grid[1]*grid[2]),template=[entry])

class Census(unittest.TestCase):
    def check_profile(self,p):
        want=naive(p)
        self.assertEqual(count_profile(p,adapter),want)
        self.assertEqual(count_profile(p,adapter,fast=False),want)

    def test_residues_masks_widths_and_direction(self):
        rng=random.Random(7)
        for i in range(180):
            p=profile(dict(intercept=(1<<32)+rng.randrange(32),cta_x_stride=rng.choice([-64,-4,0,1,2,4,16,31,32,64])),
                width=rng.choice([8,16,32,64,128]),mask=f'{rng.randrange(1,1<<32):08x}',stride=rng.choice([-8,0,1,4,8,33]))
            p['template'][0]['opcode']=rng.choice(['LDG','STG','ATOM'])+'.'+p['template'][0]['opcode'].split('.')[1]
            self.check_profile(p)

    def test_rule_families(self):
        grid=(12,3,2)
        for stride in [1,32,-64]:
            self.check_profile(profile(dict(intercept=65537,cta_x_stride=stride,cta_y_stride=64,cta_z_stride=128,cta_z_partition=1,cta_z_partition_stride=96),grid))
            self.check_profile(profile(dict(intercept=65537,cta_x_stride=stride,cta_y_offsets=[0,32,64],cta_z_stride=128),grid))
            self.check_profile(profile(dict(kind='coordinate_x_quotient_remainder_y_table_z_partition',intercept=65537,cta_x_divisor=4,
                cta_x_quotient_stride=stride,cta_x_remainder_stride=64,cta_y_offsets=[0,32,64],cta_z_stride=128,cta_z_partition=1,cta_z_partition_stride=96),grid))
            self.check_profile(profile(dict(kind='coordinate_x_floor_quotient',intercept=65537,cta_x_divisor=4,cta_x_quotient_stride=stride),grid))
            for order in itertools.permutations(range(3)):
                self.check_profile(profile(dict(kind='coordinate_x_axis_permutation',intercept=65537,element_stride=stride,
                    cta_x_input_extents=[3,4,2],cta_x_output_axis_order=list(order)),(24,1,1)))
            self.check_profile(profile(dict(kind='exact_cta_base_table',bases_by_cta={str(c):65537+c*stride for c in range(37)})))

    def test_structural_empty_and_shared_templates(self):
        p=profile(dict(intercept=65537,cta_x_stride=32))
        second=copy.deepcopy(p['template']);second[0]['opcode']='STG.128'
        p['templates']=[p['template'] if c%3==0 else second if c%3==1 else [] for c in range(37)]
        self.check_profile(p)

    def test_bad_exact_table_rejected(self):
        p=profile(dict(kind='exact_cta_base_table',bases_by_cta={'0':65536}))
        with self.assertRaises((AssertionError,ValueError)):count_profile(p,adapter)

    def test_does_not_modify_profile(self):
        p=profile(dict(intercept=65537,cta_x_stride=32));before=copy.deepcopy(p)
        self.check_profile(p);self.assertEqual(p,before)

    def test_validated_ldgsts_uses_observed_source_mask(self):
        begin=dict(schema='SG_KERNEL_SAMPLE_BEGIN_V1',grid=[1,1,1],block=[32,1,1],
            fit_ctas=[0],holdout_ctas=[],source_launch_key='launch-1',code_sha256='a'*64)
        record=dict(schema='SG_MEMORY_PROJECTION_RECORD_V1',source_launch_key='launch-1',
            code_sha256='a'*64,ref_count=2,active_mask=3,predicate_mask=3,effective_mask=3,
            projection_kind='async_global_read',source_control_kind='validated_sm89_ldgsts',
            transfer_width=16,transfer_policy=1,opcode='LDGSTS.E.BYPASS.128',width=16,
            is_load=False,is_store=False,source_read_mask=1,
            refs=[dict(global_mask=0,local_mask=0,shared_mask=3,addresses=[0]*32),
                  dict(global_mask=3,local_mask=0,shared_mask=0,addresses=[4096,8192]+[0]*30)],
            original_received_ordinal=1,cta_warp_id=0,actual_sm=0,clock64=7,cta=[0,0,0],
            function_id=1,pc=16)
        direction,width,mask,ranges,is_async=project_record(record)
        self.assertEqual((direction,width,mask,is_async),('load',16,1,True))
        collector=SampleCollector(begin)
        collector.record(record)
        frame=collector.samples[0,0][0]
        self.assertEqual(frame['effective_guard_mask'],3)
        self.assertEqual(frame['mask'],1)
        self.assertEqual(packed.normalize(collector.samples)[0][0]['mask'],'0x1')

    def test_shared_only_record_uses_effective_guard_for_no_global_census(self):
        begin=dict(schema='SG_KERNEL_SAMPLE_BEGIN_V1',grid=[1,1,1],block=[32,1,1],
            fit_ctas=[0],holdout_ctas=[],source_launch_key='launch-shared',code_sha256='b'*64)
        record=dict(schema='SG_MEMORY_PROJECTION_RECORD_V1',source_launch_key='launch-shared',
            code_sha256='b'*64,ref_count=1,active_mask=1,predicate_mask=1,effective_mask=1,
            projection_kind='ordinary_memory_operands',opcode='LD.E.32',width=4,is_load=True,is_store=False,
            refs=[dict(global_mask=0,local_mask=0,shared_mask=1,addresses=[4096]+[0]*31)],
            original_received_ordinal=1,cta_warp_id=0,actual_sm=0,clock64=7,cta=[0,0,0],
            function_id=1,pc=16)
        collector=SampleCollector(begin)
        collector.record(record)
        frame=collector.samples[0,0][0]
        self.assertEqual((frame['mask'],frame['effective_guard_mask']),(0,1))
        census=packed.projection_record_census(collector)
        self.assertEqual(census['observed_shared_only_records'],1)
        self.assertEqual(census['no_global_effective_nonzero_records'],1)
        self.assertEqual(census['no_global_effective_zero_records'],0)

    def test_atomic_is_one_native_rmw_record(self):
        record=dict(schema='SG_MEMORY_PROJECTION_RECORD_V1',ref_count=1,active_mask=1,
            predicate_mask=1,effective_mask=1,projection_kind='ordinary_memory_operands',
            opcode='ATOMG.E.ADD.STRONG.GPU',width=4,is_load=True,is_store=True,
            refs=[dict(global_mask=1,local_mask=0,shared_mask=0,addresses=[4096]+[0]*31)])
        direction,width,mask,ranges,is_async=project_record(record)
        self.assertEqual((direction,width,mask,is_async),('atomic',4,1,False))
        frames={(0,0):[dict(lanes=[dict(lane=0,addr=4096,is_local=0)],
            op=ord('A'),opcode=record['opcode'],mem_width=4,mask=1,pc=32)]}
        normalized=packed.normalize(frames)[0]
        self.assertEqual(len(normalized),1)
        self.assertEqual(packed.direction_width(normalized[0]['opcode']),('A',4))

if __name__=='__main__':unittest.main()
