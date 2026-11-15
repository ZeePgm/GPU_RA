"""Paired full-CUPTI timeline, same instrumentation for zero/inject."""
import csv,shutil
from common import read,write,rows,sha
from study import ROOT,invoke
p=ROOT/'output/gpu-a1/20261102-formal-001/multi-1000';out=ROOT/'output/gpu-a1/20261102-timeline-002';assert not out.exists();out.mkdir(parents=True)
binary=ROOT/'build-s3/bin/gpu_a1.exe';write(out/'protocol.json',{'role':'supplementary single-pair GPU timeline; no token-wall equivalence acceptance','order':['zero','inject'],'binary_sha256':sha(binary),'window':'CUPTI stamps bracket steady_clock delay; marker overhead included','units':'ns; CUPTI and steady_clock never subtracted'})
shutil.copyfile(binary,out/binary.name);shutil.copyfile(ROOT/'experiments/gpu_s/native_a1.cpp',out/'native_a1.cpp');shutil.copyfile(ROOT/'experiments/gpu_s/a1_segments.hpp',out/'a1_segments.hpp')
res={}
for mode in ['zero','inject']:
 c=read(p/'model.json');c['a1'].update(mode=mode,lean=False);write(out/(mode+'.json'),c)
 invoke([binary,c['model_path'],ROOT/'output/gpu-s/20261101-s8-002/G3/prompt.txt',out/mode,8,'timing',out/(mode+'.json')],out/'logs'/mode)
 assert (out/mode/'logits.f32').read_bytes()==(p/'model/logits.f32').read_bytes()
 d=out/mode;gates=read(d/'a1-clocks.json')['gates'];windows=rows(d/'segments.jsonl');apis=rows(d/'apis.jsonl');acts=rows(d/'kernels.jsonl')+rows(d/'transfers.jsonl');rs=[]
 for g in gates:
  lo,hi=g['cupti_begin_ns'],g['cupti_end_ns'];w=next(x for x in windows if x['kind']=='node' and x['node_id']==g['node_id'])
  members=[a for a in acts if w['first_activity_id']<=a['activity_id']<w['first_activity_id']+w['activity_count']]
  previous=[a for a in acts if a['activity_id']<w['first_activity_id']];prev=max(a['end_ns'] for a in previous)
  kernel=[a for a in members if 'name' in a];assert kernel
  launch=min(a['start_ns'] for a in kernel)
  overlaps=[a for a in acts if a['start_ns']<hi and a['end_ns']>lo]
  submits=[a for a in members for v in apis if v['correlation']==a['correlation'] and v['return_value']==0 and w['start_ns']<=v['start_ns']<=v['end_ns']<=w['end_ns']]
  assert len(submits)>=len(members)
  r={'node_id':g['node_id'],'step':g['step'],'pre_ns':lo-prev,'gate_ns':hi-lo,'post_ns':launch-hi,'gap_ns':launch-prev,'steady_wait_ns':g['end_ns']-g['start_ns'],'overlap_count':len(overlaps),'kernel_after_gate':launch>=hi,'enqueue_window_verified':True}
  assert r['gap_ns']==r['pre_ns']+r['gate_ns']+r['post_ns'];rs.append(r)
 res[mode]=rs;print(mode,'overlaps',[r['overlap_count'] for r in rs],flush=True)
paired=[]
for z,j in zip(res['zero'],res['inject']):
 assert z['node_id']==j['node_id']
 paired.append({'node_id':j['node_id'],'step':j['step'],'delta_pre_ns':j['pre_ns']-z['pre_ns'],'delta_gate_ns':j['gate_ns']-z['gate_ns'],'delta_post_ns':j['post_ns']-z['post_ns'],'delta_gap_ns':j['gap_ns']-z['gap_ns'],'target_ns':1000000})
write(out/'result.json',{'records':res,'paired':paired,'single_pair_only':True})
with (out/'paired.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
print('timeline finished')
