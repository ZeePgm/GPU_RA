"""R3 qualification, real-tensor coverage, and predeclared paired performance."""
import argparse,hashlib,json,math,os,random,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from run_llama_integrated import environment

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['qualify','coverage','hybrid','pilot','performance'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--build',type=Path,default=ROOT/'build-r3-current-20261107');ap.add_argument('--pilot',type=Path);args=ap.parse_args()
 build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);env=environment(build);exe=build/'r3_run.exe'
 files=[p for base in ['src','include','experiments/readonly_r3','third_party/HBF-EventLab/src','third_party/HBF-EventLab/include'] for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ['.cpp','.hpp','.cu','.cuh','.py']]+[ROOT/'experiments/readonly_r3/CMakeLists.txt']
 inputs=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files]
 binaries=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [exe,build/'r3_runtime.lib',build/'eventlab/hbf_eventlab.lib',build/'CMakeCache.txt']]
 protocol={'stage':args.stage,'sources':inputs,'binaries':binaries,'rule':'exact contiguous active-lane addresses AND one registered interval -> merge; otherwise scalar; PTX warp only validated straight-line whole-warps; control-flow kernels -> PTX scalar or CUDA','unit':'independent process; one warmup then mean of three timed launches for performance; no GPU timer instrumentation in performance','primary':'within-route scalar-minus-merge wall_ns median of independent process-pair differences','seed':20261108,'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.current.sm','--format=csv'],text=True),'scope':'transport adapter prepares complete selected immutable tensors before unchanged native matrix/norm consumer; actual GGUF bytes; no claim of full model execution integration'}
 write(out/'protocol.json',protocol)
 def run(name,cfg,timeout=120):
  config=out/(name+'.config.json');result=out/(name+'.result.json');write(config,cfg)
  with (out/(name+'.log')).open('w',encoding='utf-8') as log:subprocess.run([str(exe),str(config),str(result)],env=env,stdout=log,stderr=log,check=True,timeout=timeout)
  return read(result)
 records=[]
 if args.stage=='qualify':
  cases=[('aligned',{}),('sector',{'shift':60}),('dlu',{'shift':4092}),('tail',{'mask':8191}),('sparse',{'mask':1431655765}),('empty',{'mask':0}),('single',{'mask':2147483648}),('duplicate',{'layout':'duplicate'}),('reverse',{'layout':'reverse'}),('hole',{'layout':'hole'}),('strided',{'stride':3}),('adjacent_ranges',{'split':64}),('two_warps',{'lanes':64}),('two_ctas',{'lanes':512})]
  for width in [4,16]:
   for name,values in cases:
    for route in ['ptx','cuda']:
     results=[]
     for merge in [False,True]:
      cfg=dict(strategy=route,merge=merge,width=width,lanes=32,repeats=1,timed=True,**{k:v for k,v in values.items() if k!='lanes'})
      cfg['lanes']=values.get('lanes',32)
      if width==16 and name in ['sector','dlu']:cfg['shift']=48 if name=='sector' else 4080
      r=run(f'{width}-{name}-{route}-{merge}',cfg);assert r['status']=='pass' and r['drained']
      # Independently expand actual audited spans into byte multiplicities.
      actual={}
      for parent in r.get('parents',[]):
       for b in range(parent['offset'],parent['offset']+parent['bytes']):actual[b]=actual.get(b,0)+1
      expected={};mask=cfg.get('mask',0xffffffff)
      for i in range(cfg['lanes']):
       lane=i%32
       if not (mask>>lane)&1:continue
       index=(i//32)*32+(mask&((1<<lane)-1)).bit_count()
       if cfg.get('layout')=='duplicate':index=(i//32)*32
       if cfg.get('layout')=='reverse':index=(i//32)*32+31-lane
       if cfg.get('layout')=='hole' and lane==5:index+=1
       offset=cfg.get('shift',0)+index*cfg.get('stride',1)*width
       for b in range(offset,offset+width):expected[b]=expected.get(b,0)+1
      assert actual==expected,'byte multiplicities changed'
      assert r['dependency_checks']==r['logical_accesses']
      results.append({'merge':merge,'requests':r['samples'][0]['requests'],'sectors':r['samples'][0]['sectors']})
     if name=='aligned':assert results[0]['requests']==32 and results[1]['requests']==1 and results[1]['sectors']==width//2
     records.append({'width':width,'case':name,'route':route,'status':'pass','results':results})
    print('qualification',width,name,'passed',flush=True)
  faults=[]
  for route in ['ptx','cuda']:
   for merge in [False,True]:
    for mode in ['fault','range']:
     r=run(f'failure-{route}-{merge}-{mode}',dict(strategy=route,merge=merge,mode=mode,lanes=256,repeats=1,width=4));assert r['status']=='expected_failure' and r['cuda_drained'];faults.append(r)
  write(out/'failures.json',faults)
 elif args.stage in ['coverage','hybrid']:
  sys.path.insert(0,str(ROOT/'third_party/llama.cpp-integrated/gguf-py'));from gguf import GGUFReader
  choices=[('qwen3-0.6b',[10],1,'development'),('smollm2-135m',[0,23],3,'holdout'),('qwen3-1.7b',[27],7,'holdout')]
  if args.stage=='hybrid':choices=[('smollm2-135m',[29],5,'fresh_policy_holdout')]
  manifest=[];(out/'tensors').mkdir()
  for profile_name,layers,tokens,role in choices:
   profile=read(ROOT/'configs/integrated'/f'{profile_name}.json');model=Path(profile['model_path']);assert sha(model)==profile['gguf_sha256'];reader=GGUFReader(str(model))
   for layer in layers:
    for kind in ['ffn_norm','ffn_down']:
     name=f'blk.{layer}.{kind}.weight';tensor=next(t for t in reader.tensors if t.name==name);expected=next(t for t in profile['tensors'] if t['name']==name)
     data=tensor.data.tobytes();shape=[int(v) for v in tensor.shape];assert shape==expected['shape'] and len(data)==expected['bytes']
     path=out/'tensors'/f'{profile_name}-{layer}-{kind}.bin';path.write_bytes(data)
     manifest.append({'model':profile_name,'model_sha256':profile['gguf_sha256'],'tensor':name,'role':role,'layer':layer,'tokens':tokens,'input_file':str(path),'bytes':len(data),'columns':shape[0],'rows':shape[1] if len(shape)>1 else 1,'dtype':expected['dtype'],'sha256':sha(path)})
  write(out/'coverage-manifest.json',manifest)
  for item in manifest:
   outputs=[]
   for route in (['hybrid'] if args.stage=='hybrid' else ['ptx','cuda']):
    for merge in ([True] if args.stage=='hybrid' else [False,True]):
     cfg={k:item[k] for k in ['input_file','bytes','columns','rows','dtype','tokens']};cfg.update(strategy=route,merge=merge,width=16,repeats=1,timed=True)
     name=f"{item['model']}-{item['layer']}-{item['dtype']}-{route}-{merge}";print('coverage starting',name,item['bytes'],flush=True)
     r=run(name,cfg,timeout=1800);assert r['status']=='pass' and r['covered_bytes']==item['bytes'] and r['input_sha256']==item['sha256']==r['mirror_sha256']
     assert r['dependency_checks']==item['bytes']//16
     outputs.append(r['consumer_output_sha256']);records.append({'model':item['model'],'tensor':item['tensor'],'role':item['role'],'route':route,'merge':merge,'bytes':r['covered_bytes'],'requests':r['requests'],'sectors':r['sectors'],'cpu_checks':r['cpu_consumer_checks'],'consumer_values':r['consumer_values'],'consumer_output_sha256':r['consumer_output_sha256'],'status':'pass'})
     write(out/'progress.json',records);print('coverage passed',name,flush=True)
     if args.stage=='hybrid':
      assert r['selected_strategy']=='ptx'
      alternate=dict(cfg,control_flow_fixture=True)
      fallback=run(name+'-fallback',alternate,timeout=1800)
      assert fallback['status']=='pass' and fallback['selected_strategy']=='cuda' and fallback['mirror_sha256']==item['sha256'] and fallback['consumer_output_sha256']==r['consumer_output_sha256']
      records.append({'model':item['model'],'tensor':item['tensor'],'role':item['role'],'route':'hybrid_to_cuda','bytes':fallback['covered_bytes'],'requests':fallback['requests'],'status':'pass'})
   assert len(set(outputs))==1
 else:
  rng=random.Random(20261108)
  if args.stage=='pilot':
   for route in ['ptx','cuda']:
    for pair in range(8):
     values=[]
     for j in range(2):
      r=run(f'aa-{route}-{pair}-{j}',dict(strategy=route,merge=False,width=4,lanes=256,repeats=3,timed=False));values.append(statistics.mean(s['wall_ns'] for s in r['samples']))
     records.append({'route':route,'pair':pair,'difference_ns':values[0]-values[1]})
   # Separate A/B development pilot estimates the scale needed for qualification.
   pilot_ab=[]
   for route in ['ptx','cuda']:
    for pair in range(4):
     order=[False,True];rng.shuffle(order);values={}
     for merge in order:
      r=run(f'ab-pilot-{route}-{pair}-{merge}',dict(strategy=route,merge=merge,width=4,lanes=256,repeats=3,timed=False));values[merge]=statistics.mean(s['wall_ns'] for s in r['samples'])
     pilot_ab.append({'route':route,'pair':pair,'difference_ns':values[False]-values[True]})
   write(out/'ab-pilot.json',pilot_ab)
  else:
   assert args.pilot,'pilot required';pilot=read(args.pilot/'result.json');ab=read(args.pilot/'ab-pilot.json')
   qual={}
   for route in ['ptx','cuda']:
    noise=max(abs(r['difference_ns']) for r in pilot['records'] if r['route']==route)
    gain=statistics.median(r['difference_ns'] for r in ab if r['route']==route)
    qual[route]={'aa_max_abs_ns':noise,'pilot_median_gain_ns':gain,'resolves_gain':gain>noise}
   protocol.update(pilot_qualification=qual,confirm_pairs_per_primary_route=32,secondary_pairs=12,primary_family=['ptx/aligned','cuda/aligned'],interval='distribution-free paired median interval from binomial sign order statistics; family alpha .05 Bonferroni over 2 primary comparisons; secondary intervals descriptive',delta_min='0 for claim of any positive median saving; no practical-effect claim or automatic application default switch',sample_basis='32 independent pairs; sign-test rejection and finite-sample median interval; qualification checks A/A spread against development A/B gain; fixed budget, no optional stopping')
   write(out/'protocol.json',protocol)
   conditions=[('aligned',dict(lanes=256),32),('strided',dict(lanes=256,stride=3),12),('single',dict(lanes=32,mask=1),12),('sparse',dict(lanes=256,mask=0x55555555),12),('dlu',dict(lanes=256,shift=4092),12)]
   schedule=[]
   for route in ['ptx','cuda']:
    for case,options,n in conditions:
     orders=[[False,True] for _ in range(n//2)]+[[True,False] for _ in range(n//2)];rng.shuffle(orders)
     for pair,order in enumerate(orders):schedule.append((route,case,options,pair,order))
   rng.shuffle(schedule);write(out/'schedule.json',schedule)
   for route,case,options,pair,order in schedule:
    values={}
    for merge in order:
     cfg=dict(strategy=route,merge=merge,width=4,repeats=3,timed=False,**options);r=run(f'{route}-{case}-{pair}-{merge}',cfg);values[str(merge)]=statistics.mean(s['wall_ns'] for s in r['samples'])
    records.append({'route':route,'case':case,'pair':pair,'order':order,'scalar_ns':values['False'],'merged_ns':values['True'],'difference_ns':values['False']-values['True']});write(out/'pairs.json',records);print('pair',route,case,pair,'passed',flush=True)
   summaries=[]
   for route in ['ptx','cuda']:
    for case,_,n in conditions:
     selected=[r for r in records if r['route']==route and r['case']==case];d=sorted(r['difference_ns'] for r in selected);alpha=.025 if case=='aligned' else .05
     k=max([k for k in range(1,n//2+1) if 2*sum(math.comb(n,j) for j in range(k))/2**n<=alpha] or [0]);ci=[d[k-1],d[n-k]] if k else [None,None]
     summaries.append({'route':route,'case':case,'pairs':n,'scalar_median_ns':statistics.median(r['scalar_ns'] for r in selected),'merged_median_ns':statistics.median(r['merged_ns'] for r in selected),'paired_median_difference_ns':statistics.median(d),'median_interval_ns':ci,'positive_pairs':sum(v>0 for v in d),'primary':case=='aligned','gain_established':case=='aligned' and qual[route]['resolves_gain'] and ci[0]>0})
   write(out/'analysis.json',summaries)
 for e in inputs+binaries:assert sha(ROOT/e['path'])==e['sha256'],f"input changed: {e['path']}"
 write(out/'result.json',{'status':'pass','stage':args.stage,'records':records})
 print(args.stage,'complete',flush=True)
if __name__=='__main__':main()
