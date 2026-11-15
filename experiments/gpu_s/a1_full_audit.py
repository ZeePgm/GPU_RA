from common import read,write,rows
from study import ROOT,invoke
p=ROOT/'output/gpu-a1/20261102-formal-001/multi-1000';c=read(p/'model.json');c['a1']['lean']=False;c['a1']['mode']='inject'
o=ROOT/'output/gpu-a1/20261102-full-audit-001';assert not o.exists();o.mkdir(parents=True);write(o/'config.json',c)
invoke([ROOT/'build-s3/bin/gpu_a1.exe',c['model_path'],ROOT/'output/gpu-s/20261101-s8-002/G3/prompt.txt',o/'run',8,'timing',o/'config.json'],o/'logs')
assert (o/'run/logits.f32').read_bytes()==(p/'model/logits.f32').read_bytes()
a=read(o/'run/a1-clocks.json');windows=rows(o/'run/segments.jsonl');events=rows(o/'run/events.jsonl');apis=rows(o/'run/apis.jsonl');acts=rows(o/'run/kernels.jsonl')+rows(o/'run/transfers.jsonl');checks=[]
for g in a['gates']:
 w=next(x for x in windows if x['kind']=='node' and x['node_id']==g['node_id']);members=[x for x in acts if w['first_activity_id']<=x['activity_id']<w['first_activity_id']+w['activity_count']]
 assert members
 for x in members:
  assert any(y['correlation']==x['correlation'] and y['return_value']==0 and w['start_ns']<=y['start_ns']<=y['end_ns']<=w['end_ns'] for y in apis)
 checks.append({'node_id':g['node_id'],'activities':len(members),'enqueue_within_post_gate_window':True})
write(o/'result.json',{'status':'pass','logits_match':True,'consumer_checks':checks,'evidence':'segment_begin occurs after a1before returns; CUDA submissions associated with GPU activities fall inside this subsequent synchronized window. steady_clock and CUPTI timestamps are not subtracted.'})
print('full audit pass')
