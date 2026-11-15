"""A1 initial functional pilot only; no statistical acceptance."""
import copy
from pathlib import Path
from common import read,write,rows,sha
from study import ROOT,invoke
p=ROOT/'output/gpu-s/20261101-s8-002/G3'
out=ROOT/'output/gpu-a1/20261102-pilot-001'
assert not out.exists();out.mkdir(parents=True)
cfg=read(p/'configs/native-verify.json')
gates=[{'node_id':r['node_id'],'step':r['step'],'wait_ns':1000000} for r in rows(p/'G3-timing-1/gates.jsonl')]
assert len(gates)==16
write(out/'protocol.json',{'status':'functional_pilot_only','gates':gates,'wall_clock':'steady_clock ns','model_clock':'integer ps; no wall feedback','binary_sha256':sha(ROOT/'build-s3/bin/gpu_a1.exe'),'no_formal_equivalence_claim':True})
records={}
for mode in ['zero','model','inject']:
 c=copy.deepcopy(cfg);c['a1']={'mode':mode,'gates':gates};write(out/(mode+'.json'),c)
 invoke([ROOT/'build-s3/bin/gpu_a1.exe',c['model_path'],p/'prompt.txt',out/mode,8,'timing',out/(mode+'.json')],out/'logs'/mode)
 records[mode]=read(out/mode/'a1-clocks.json')
 assert (out/mode/'logits.f32').read_bytes()==(p/'G3-timing-1/logits.f32').read_bytes()
 print(mode,'finished',flush=True)
checks=[]
for i,(z,m,j) in enumerate(zip(records['zero']['tokens'],records['model']['tokens'],records['inject']['tokens'])):
 expected=sum(g['wait_ns'] for g in gates if g['step']<=i)*1000
 assert m['model_wait_ps']==expected and z['model_wait_ps']==j['model_wait_ps']==0
 assert z['model_compute_ps']==m['model_compute_ps']==j['model_compute_ps']
 actual=sum(g['end_ns']-g['start_ns'] for g in records['inject']['gates'] if g['step']<=i)
 checks.append({'step':i,'target_ps':expected,'actual_injected_ns':actual,'wall_difference_ns':j['wall_relative_ns']-z['wall_relative_ns']})
write(out/'pilot-result.json',{'functional_clock_and_logits_checks':'pass','statistical_acceptance':'not_started','paired_tokens':checks})
print(checks,flush=True)
