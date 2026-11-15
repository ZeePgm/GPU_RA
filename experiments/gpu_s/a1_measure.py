"""A1 independent clocks: retained pilot and balanced process-paired sampling."""
import argparse,copy,random,time
import numpy as np
from common import read,write,rows,sha
from study import ROOT,invoke
p=ROOT/'output/gpu-s/20261101-s8-002/G3'
a=argparse.ArgumentParser();a.add_argument('--formal',action='store_true');a.add_argument('--study',required=True);a.add_argument('--pairs',type=int,default=6);a.add_argument('--us',type=int,default=1000);a.add_argument('--pattern',choices=['multi','single','late'],default='multi');args=a.parse_args()
if args.formal:
 raise SystemExit('P0-A: legacy formal sampling disabled; dose qualification, mode identity and protocol evidence are not yet wired into this runner. Use pilot only.')
out=ROOT/args.study;assert not out.exists();out.mkdir(parents=True)
cfg=read(p/'configs/native-verify.json');base=rows(p/'G3-timing-1/gates.jsonl');binary=ROOT/'build-s3/bin/gpu_a1.exe'
gates=[{'node_id':r['node_id'],'step':r['step'],'wait_ns':args.us*1000 if (args.pattern=='multi' or (args.pattern=='single' and k==0) or (args.pattern=='late' and k==2)) else 0} for k,r in enumerate(base)]
write(out/'protocol.json',{'phase':'formal' if args.formal else 'pilot','pairs':args.pairs,'us':args.us,'pattern':args.pattern,'warmup_runs':4,'sequence':'balanced shuffled pair order, seed 4102','clock':'steady_clock ns; Session integer ps','binary_sha256':sha(binary),'scope':'lean synchronized; new process and model/context per run; no CUPTI activity trace','acceptance':'Per token mean error simultaneous 95% Student-t CI, Bonferroni 48 endpoints; entirely within +/-10% cumulative target. 30 pairs fixed budget; no optional stopping. Target-zero endpoint not formally assessed.' if args.formal else 'No statistical acceptance in pilot'})
expected=(p/'G3-timing-1/logits.f32').read_bytes()
def run(name,mode):
 c=copy.deepcopy(cfg);c['a1']={'mode':mode,'gates':gates,'lean':True,'session_config':str(p/'effective-H0.ini')};write(out/(name+'.json'),c)
 invoke([binary,c['model_path'],p/'prompt.txt',out/name,8,'timing',out/(name+'.json')],out/'logs'/name)
 assert (out/name/'logits.f32').read_bytes()==expected
 r=read(out/name/'a1-clocks.json');assert len(r['tokens'])==8 and len(r['gates'])==16
 for i,t in enumerate(r['tokens']):
  w=sum(g['wait_ns']*1000 for g in gates if g['step']<=i) if mode=='model' else 0
  assert t['model_wait_ps']==w and t['session_tick_ps']==t['model_compute_ps']+w
 return r
for i in range(4):run('warmup-'+str(i),'zero')
run('model','model')
orders=[['zero','inject'] if i%2==0 else ['inject','zero'] for i in range(args.pairs)];random.Random(4102).shuffle(orders)
records=[]
for k,order in enumerate(orders):
 rs={mode:run(f'pair-{k}-{mode}',mode) for mode in order};z,j=rs['zero'],rs['inject'];check=[]
 for i,(zt,jt) in enumerate(zip(z['tokens'],j['tokens'])):
  actual=sum(g['end_ns']-g['start_ns'] for g in j['gates'] if g['step']<=i)
  target=sum(g['wait_ns'] for g in gates if g['step']<=i)
  diff=jt['wall_relative_ns']-zt['wall_relative_ns']
  check.append({'step':i,'actual_ns':actual,'target_ns':target,'difference_ns':diff,'error_ns':diff-actual})
 records.append({'pair':k,'order':order,'tokens':check});write(out/'pairs.json',records)
 print(k,'last error ms',check[-1]['error_ns']/1e6,flush=True)
err=np.array([[t['error_ns'] for t in r['tokens']] for r in records]);write(out/'result.json',{'status':'analyzed_separately' if args.formal else 'pilot_only','session_clock_and_logits':'pass','median_error_ns':np.median(err,axis=0).tolist(),'std_error_ns':np.std(err,axis=0,ddof=1).tolist(),'pairs':len(records)})
