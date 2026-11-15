import sys,json,subprocess,time,hashlib,os
from pathlib import Path
R=Path.cwd();O=R/'output/readonly-f-compare-a-20261108';B=R/'build-readonly-f-compare-20261108';sys.path.insert(0,str(R/'tools'));from platform_environment import runtime_environment
sys.path.insert(0,str(R/'experiments/gpu_ffn'));import run as f
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l]
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
# GPU runs serial, baseline process must finish before candidates start.
while not (O/'F2-old-cost.json').exists():time.sleep(10)
env=runtime_environment(B,R,'D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0');env['PATH']=str(R/'build-r2-final-20261107/bin')+os.pathsep+env['PATH'];env['GGML_CUDA_DISABLE_FUSION']='1';env['GGML_CUDA_CUBLAS_COMPUTE_TYPE']='f32'
ref=R/'output/gpu-ffn/20261105-new-backend-f1-f3/F1';profile=read(ref/'native-config.json');identity={str(p.relative_to(R)):sha(p) for base in [R/'experiments/readonly_f_compare',B] for p in base.rglob('*') if p.is_file() and p.suffix in ['.cpp','.cu','.hpp','.cuh','.py','.exe','.lib']};write(O/'candidate-identity.json',identity)
def invoke(exe,cfg,name,timeout=7200):
 target=O/name
 if (target.is_dir() and (target/'result.json').exists()) or (target.is_file() and read(target).get('status')=='pass'):
  assert read(O/(name+'.config.json'))==cfg,'resume config differs'
  print('VERIFIED EXISTING',name,flush=True);return
 path=O/(name+'.config.json');write(path,cfg);t=time.perf_counter();print('START',name,flush=True)
 with (O/(name+'.log')).open('w') as log:subprocess.run([str(exe),str(path),str(O/name)],env=env,stdout=log,stderr=log,check=True,timeout=timeout)
 write(O/(name+'-process.json'),{'wall_s':time.perf_counter()-t});print('DONE',name,flush=True)
for scale,layers in [('F1',[10]),('F2',[0,10,20,27])]:
 base=dict(model=profile['model_path'],prompt=str(ref/'prompt.txt'),layers=layers,outputs=8,route='cuda',merge=True)
 for label,bypass,verify in [('capture-perf',True,False),('live',False,True)]:
  name=f'candidate-{scale}-{label}';invoke(B/'f_candidate.exe',dict(base,bypass=bypass,verify=verify),name)
  assert sha(O/name/'logits.f32')==sha(O/'native-old/logits.f32'),'complete logits mismatch'
  result=read(O/name/'result.json');assert result['tokens']==read(O/'native-old/summary.json')['raw_tokens'] and len(result['step_ms'])==8
 # Build common virtual clock actions from candidate's own activity trace.
 cap=O/f'candidate-{scale}-capture-perf';gpu=rows(cap/'gpu.jsonl');marks=rows(cap/'batches.jsonl');actions=[]
 for step in range(8):
  events=sorted([v for v in gpu if v['step']==step],key=lambda v:v['start']);batch=[v for v in marks if v['step']==step];cursor=0;normal=[]
  def flush():
   if normal:actions.append({'kind':'compute','ps':1000*f.union_ns(normal)});normal.clear()
  previous_end=0
  for pos,event in enumerate(events):
   earlier_end=previous_end;previous_end=max(previous_end,event["end"])
   assert event['end']>event['start']
   if 'read_plan' in event.get('name',''):
    flush();m=batch[cursor];cursor+=1
    assert earlier_end<=event['start'] and (pos+1==len(events) or events[pos+1]['start']>=event['end']),'read batch overlaps GPU work'
    actions.append({'kind':'read','tensor':m['tensor'],'offset':m['offset'],'bytes':4096});actions.append({'kind':'compute','ps':1000*(event['end']-event['start'])})
   else:normal.append((event['start'],event['end']))
  flush();assert cursor==len(batch)==len(layers)*18*1024**2//4096
  actions.append({'kind':'token','step':step})
 regions=read(R/f'output/gpu-ffn/20261105-new-backend-f1-f3/{scale}/regions.json')
 # complete independent coverage per step/tensor
 for step in range(8):
  for region in regions:
   actual=[m['offset'] for m in marks if m['step']==step and m['tensor']==region['tensor']];assert actual==list(range(0,region['valid_bytes'],4096))
 cfg={'ini':str(ref/'H0-512MiB.ini'),'regions':regions,'chunk':512,'actions':actions}
 invoke(B/'f_clock.exe',cfg,f'candidate-{scale}-clock.json')
 r=read(O/f'candidate-{scale}-clock.json');assert r['bytes']==len(layers)*18*1024**2*8 and len(r['tokens'])==8
 # Validate old trace using the identical virtual clock implementation/backend.
 oldcfg=read(O/f'old-{scale}-config.json');gates=rows(O/f'old-{scale}/gates.jsonl');segments=rows(O/f'old-{scale}/segments.jsonl');actions=[]
 for step in range(8):
  for segment in [v for v in segments if v['step']==step]:
   if segment['kind']=='node':
    for g in gates:
     if g['step']==step and g['node_id']==segment['node_id']:actions.append({'kind':'read','tensor':g['tensor'],'offset':0,'bytes':g['bytes']})
   actions.append({'kind':'compute','ps':segment['duration_ps']})
  actions.append({'kind':'token','step':step})
 invoke(B/'f_clock.exe',dict(cfg,chunk=4096,actions=actions),f'old-{scale}-clock-check.json')
 r=read(O/f'old-{scale}-clock-check.json');oldtokens=rows(O/f'old-{scale}/tokens.jsonl');assert [v['ps'] for v in r['tokens']]==[v['relative_ps'] for v in oldtokens],'common clock baseline mismatch'
for path,h in identity.items():assert sha(R/path)==h,'source drift '+path
write(O/'completed.json',{'status':'pass','model_outputs_equal':True,'common_clock_baseline_exact':True,'clock_semantics':'GPU active durations with own schedule; concurrent cohort of eight 512B requests per 4KiB synchronization batch; finite read-kernel duration charged after batch service, conservative within-batch overlap assumption','sampling':'one paired run per scale, descriptive only'})
