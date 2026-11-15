"""Independent A1 record analysis. No simulator counters generate expected sums."""
from pathlib import Path
import numpy as np
from scipy.stats import t
from common import read,write,rows,sha
from study import ROOT
base=ROOT/'output/gpu-s/20261101-s8-002/G3';out=ROOT/'output/gpu-a1/20261102-formal-001'
table=read(base/'compute-table.json')['segments'];gate_ids=[(r['node_id'],r['step']) for r in rows(base/'G3-timing-1/gates.jsonl')]
results=[]
for pattern in ['single','multi']:
 for us in [10,100,1000]:
  d=out/f'{pattern}-{us}';pairs=read(d/'pairs.json');assert len(pairs)==30
  expected=np.array([sum(us*1000 for k,g in enumerate(gate_ids) if g[1]<=i and (pattern=='multi' or k==0)) for i in range(8)])
  for f in d.glob('*/a1-clocks.json'):
   r=read(f);assert [(g['node_id'],g['step']) for g in r['gates']]==gate_ids
   for i,x in enumerate(r['tokens']):
    comp=sum(a['duration_ps'] for a in table if a['step']<=i)
    wait=int(expected[i])*1000 if r['mode']=='model' else 0
    assert x['session_tick_ps']==comp+wait and x['model_compute_ps']==comp and x['model_wait_ps']==wait
   for g in r['gates']:
    assert g['end_ns']>=g['start_ns']
    if r['mode']=='inject':assert g['end_ns']-g['start_ns']>=g['target_ns']
  e=np.array([[x['error_ns'] for x in p['tokens']] for p in pairs]);mean=e.mean(0);half=t.ppf(1-.05/(2*48),29)*e.std(0,ddof=1)/np.sqrt(30);tol=expected*.1
  bounds=np.stack([mean-half,mean+half],axis=1);equiv=np.all(np.abs(bounds)<=tol[:,None],axis=1)
  outside=(bounds[:,0]>tol)|(bounds[:,1]<-tol)
  results.append({'pattern':pattern,'us':us,'clock_and_injection_checks':'pass','mean_error_ns':mean.tolist(),'ci_ns':bounds.tolist(),'margin_ns':tol.tolist(),'per_token':['pass' if a else 'fail' if b else 'inconclusive' for a,b in zip(equiv,outside)],'conditional_inference':'Student-t approximation; process-pair dependence/drift not ruled out'})
write(out/'analysis.json',{'results':results,'full_A1_acceptance':False,'limitations':['Timing runs lack full CUPTI consumer submission audit','Injector resolution and dedicated negative controls not yet complete','Confidence intervals assume independent approximately well-behaved paired means; heavy tails/drift require caution']})
print([(r['pattern'],r['us'],r['per_token']) for r in results])
