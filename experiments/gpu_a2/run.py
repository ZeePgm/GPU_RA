"""Independent analytical oracle and reproducible A2 mechanism experiment."""
import argparse,copy,hashlib,itertools,json,random,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def reference(case):
 # Pure scalar recurrence, no queue, no simulator output or helper implementation.
 ordered=sorted(enumerate(case['requests']),key=lambda p:(p[1]['arrival_ps'],p[0]));end=0;result=[]
 for _,j in ordered:
  start=max(j['arrival_ps'],end);end=start+j['service_ps']
  assert end<2**64
  result.append(dict(j,start_ps=start,completion_ps=end,wait_ps=start-j['arrival_ps']))
 return result

def audit(case,actual):
 expected=reference(case)
 assert actual['name']==case['name'] and actual['completed']==expected
 assert actual['pending']==0 and actual['executed_events']==2*len(expected)
 assert actual['final_tick_ps']==(expected[-1]['completion_ps'] if expected else 0)
 events=actual['events'];assert len(events)==3*len(expected)
 byid={r['id']:r for r in expected};states={};order=[];last=0;busy=None
 for e in events:
  assert e['id'] in byid and e['tick_ps']>=last;last=e['tick_ps'];r=byid[e['id']];kind=e['kind'];old=states.get(e['id'])
  if kind=='arrival':assert old is None and e['tick_ps']==r['arrival_ps'];states[e['id']]='arrived'
  elif kind=='start':assert old=='arrived' and busy is None and e['tick_ps']==r['start_ps'];busy=e['id'];states[e['id']]='started'
  elif kind=='complete':assert old=='started' and busy==e['id'] and e['tick_ps']==r['completion_ps'];busy=None;states[e['id']]='completed';order.append(e['id'])
  else:raise AssertionError('unknown event')
 assert busy is None and order==[r['id'] for r in expected]
 return len(expected)

def run():
 ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);args=ap.parse_args();out=ROOT/args.study;assert not out.exists();out.mkdir(parents=True)
 exe=ROOT/'build-a2/a2_fifo.exe';assert exe.exists()
 cases=[]
 def add(name,arrivals,services):cases.append({'name':name,'requests':[{'id':i+1,'arrival_ps':a,'service_ps':s} for i,(a,s) in enumerate(zip(arrivals,services))]})
 add('empty',[],[]);add('one',[0],[7]);add('idle',[0,100,300],[10,20,30]);add('simultaneous',[4]*8,[0,1,2,0,4,3,0,7]);add('backlog',list(range(20)),[100]*20);add('exact_completion',[0,5,8],[5,3,7]);add('out_of_order',[9,0,3,0],[1,2,3,0]);add('zero_chain',[0]*30,[0]*30);add('uint64_boundary',[2**64-10,2**64-5],[5,4])
 for k,(arr,srv) in enumerate(itertools.product(itertools.product(range(3),repeat=3),itertools.product([0,1,3],repeat=3))):add(f'exhaustive-{k}',arr,srv)
 for seed in range(200):
  rng=random.Random(2026110300+seed);add(f'random-{seed}',[rng.randrange(300) for _ in range(64)],[rng.choice([0,1,2,10,100]) for _ in range(64)])
 inp={'cases':cases};write(out/'input.json',inp);write(out/'reference.json',{'cases':[{'name':c['name'],'completed':reference(c)} for c in cases]})
 sources=[Path(__file__),Path(__file__).with_name('fifo.cpp'),Path(__file__).with_name('CMakeLists.txt'),ROOT/'third_party/HBF-EventLab/src/infrastructure/event_queue.cpp',ROOT/'third_party/HBF-EventLab/include/hbf_eventlab/infrastructure/event_queue.hpp',ROOT/'third_party/HBF-EventLab/include/hbf_eventlab/common/types.hpp',exe]
 write(out/'protocol.json',{'status':'frozen_before_execution','scope':'experimental single-server FIFO adapter using actual production EventQueue; no Host/Device/H0 instantiated','oracle':'Python max recurrence; stable arrival then input-position tie break; integer ps; zero service valid','conditions':{'nonpreemptive':True,'retry':False,'parallel_servers':1,'same_tick':'production Completion/Arrival rounds; FIFO arrival ties preserve input order'},'cases':len(cases),'requests':sum(len(c['requests']) for c in cases),'random_seeds':'2026110300..2026110499','acceptance':'exact per-request and event audit, final drain, repeat byte-identical; any mismatch fails','MG1':'optional extension not executed; deterministic finite traces only','identity':{str(p.relative_to(ROOT)):sha(p) for p in sources}})
 for p in sources:
  dest=out/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 def execute(inputfile,resultfile,log):
  with log.open('w',encoding='utf-8') as f:return subprocess.run([str(exe),str(inputfile),str(resultfile)],stdout=f,stderr=f,timeout=120).returncode
 for k in range(2):assert execute(out/'input.json',out/f'actual-{k}.json',out/f'run-{k}.log')==0
 assert sha(out/'actual-0.json')==sha(out/'actual-1.json')
 actual=json.loads((out/'actual-0.json').read_text());assert len(actual['cases'])==len(cases)
 count=sum(audit(c,r) for c,r in zip(cases,actual['cases']))
 q=actual['queue_probes'];assert q=={'same_tick_phases':['completion','arrival','deadline'],'same_tick_dynamic_round':['arrival1','arrival2','dynamic_completion'],'dynamic_final_tick':0,'cancel':{'first':True,'second':False,'called':False,'executed':0,'pending':0},'idle_endpoint_unexecuted':True,'invalid_time_rejected':[True,True,True],'limit_before_resume':{'rejected':True,'pending':1,'executed':1,'tick':1},'limit_after_resume':{'pending':0,'executed':2,'tick':2}}
 negative=[]
 basecase=cases[4];baseres=actual['cases'][4]
 for kind in ['wrong_order','duplicate','missing_completion','wrong_time','wrong_wait','early_start','unknown_event']:
  r=copy.deepcopy(baseres)
  if kind=='wrong_order':r['completed'][0],r['completed'][1]=r['completed'][1],r['completed'][0]
  if kind=='duplicate':r['completed'].append(copy.deepcopy(r['completed'][0]))
  if kind=='missing_completion':r['events']=[e for e in r['events'] if not(e['kind']=='complete' and e['id']==1)]
  if kind=='wrong_time':r['completed'][0]['completion_ps']+=1
  if kind=='wrong_wait':r['completed'][1]['wait_ps']+=1
  if kind=='early_start':next(e for e in r['events'] if e['kind']=='start' and e['id']==2)['tick_ps']-=1
  if kind=='unknown_event':r['events'][0]['kind']='bogus'
  rejected=False
  try:audit(basecase,r)
  except AssertionError:rejected=True
  assert rejected;write(out/'negative-audit'/(kind+'.json'),r);negative.append({'case':kind,'rejected':True})
 illegal=[]
 for kind in ['negative_arrival','negative_service','fraction','duplicate_id','overflow','non_integer']:
  j=[{'id':1,'arrival_ps':0,'service_ps':1}]
  if kind=='negative_arrival':j[0]['arrival_ps']=-1
  if kind=='negative_service':j[0]['service_ps']=-1
  if kind=='fraction':j[0]['service_ps']=.5
  if kind=='duplicate_id':j*=2
  if kind=='overflow':j[0].update(arrival_ps=2**64-1,service_ps=1)
  if kind=='non_integer':j[0]['arrival_ps']='zero'
  f=out/'negative-input'/(kind+'.json');write(f,{'cases':[{'name':kind,'requests':j}]});dest=f.with_suffix('.result');rc=execute(f,dest,f.with_suffix('.log'));assert rc!=0 and not dest.exists();illegal.append({'case':kind,'exit_code':rc,'rejected':True})
 result={'status':'pass','scope':'production EventQueue + experimental FIFO mechanism, not H0 or Host/Device validation','cases':len(cases),'requests':count,'request_fields_exact':True,'event_records_checked':3*count,'max_abs_residual_ps':0,'all_cases_drained':True,'byte_identical_replay':True,'event_queue_probes':q,'audit_negatives':negative,'executor_negatives':illegal,'MG1_executed':False,'protocol_sha256':sha(out/'protocol.json'),'input_sha256':sha(out/'input.json'),'actual_sha256':sha(out/'actual-0.json')}
 write(out/'result.json',result);write(out/'artifact-index.json',{'files':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact-index.json']});print(json.dumps(result,indent=2))
if __name__=='__main__':run()
