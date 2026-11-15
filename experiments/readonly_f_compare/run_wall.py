"""Stable, serial live runs after all builds and offline clock replays finish."""
import json, os, sys, time, subprocess, hashlib, shutil, threading, atexit
from pathlib import Path
R=Path.cwd();O=R/'output/readonly-f-compare-a-20261108';B=R/'build-readonly-f-compare-20261108'
sys.path.insert(0,str(R/'experiments/gpu_ffn'));import run as f
sys.path.insert(0,str(R/'tools'));from platform_environment import runtime_environment
from process_probe import processes
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
assert read(O/'completed.json')['status']=='pass','finish qualification first'
sealed=read(O/'candidate-identity.json')
for p,h in sealed.items():assert sha(R/p)==h,'candidate source/binary drift '+p
old=R/'build-sf-f-entry-20261105';ref=R/'output/gpu-ffn/20261105-new-backend-f1-f3/F1'
identity=read(ref/'build-identity.json');assert sha(old/'bin/gpu_ffn.exe')==identity['binary_sha256']
for name,h in identity['runtime_dlls'].items():assert sha(old/'bin'/name)==h
env=runtime_environment(B,R,'D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0');env['PATH']=str(R/'build-r2-final-20261107/bin')+os.pathsep+env['PATH'];env['GGML_CUDA_DISABLE_FUSION']='1';env['GGML_CUDA_CUBLAS_COMPUTE_TYPE']='f32'
allowed={str(old/'bin/gpu_ffn.exe').replace('\\','/').lower(),str(B/'f_candidate.exe').replace('\\','/').lower()}
observations=[];conflicts=[];observing=True
def foreign_gpu():
 rows=processes();assert any(row['pid']==4 and row['name']=='System' for row in rows),'process probe is sandbox-limited; run this batch outside the process sandbox'
 found=[]
 for row in rows:
  path=row['path'].replace('\\','/').lower()
  if (path.startswith(str(R).replace('\\','/').lower()+'/') and path not in allowed) or row['name'].lower() in ['nvcc.exe','cl.exe','ninja.exe']:found.append(row)
 return found
while foreign_gpu():print('WAIT other workspace GPU experiment',flush=True);time.sleep(10)
def observe():
 while observing:
  try:
   foreign=foreign_gpu();entry={'time':time.time(),'foreign_workspace_processes':foreign};observations.append(entry)
   with (O/'wall3-process-observations.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
   if foreign:conflicts.append(entry)
  except Exception as e:conflicts.append({'time':time.time(),'probe_error':str(e)})
  time.sleep(5)
atexit.register(lambda:write(O/'wall3-process-observations.json',{'interval_s':5,'samples':observations,'conflicts':conflicts}))
watch=threading.Thread(target=observe,daemon=True);watch.start()
write(O/'wall3-preflight.json',{'time':time.time(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True),'order':['old-F1-wall3','candidate-F1-wall3','candidate-F2-wall3','old-F2-wall3'],'condition':'no concurrent build, offline simulation or other task GPU process','kv':'native F16, no adapter configured','candidate_verify':False,'candidate_audit':'parent coverage, completion, drain remain enabled; byte/dependency audit passed in separate live qualification'})
def audit_old(scale,name):
 a=O/'audit-wall3'/scale;a.mkdir(parents=True,exist_ok=True);refscale=ref.parent/scale
 write(a/'plan.json',{'stage':scale});write(a/'layout-all-ffn.json',read(refscale/'layout-all-ffn.json'))
 for src,dst in [(O/'native-old',a/'native'),(O/name,a/scale)]:
  dst.mkdir(exist_ok=True)
  for p in src.iterdir():
   if p.is_file():shutil.copy2(p,dst/p.name)
 result=f.audit(a,read(refscale/'regions.json'),read(O/'old-compute-table.json'));write(a/'acceptance.json',result)
def run_old(scale):
 name=f'old-{scale}-wall3';cfg=read(O/f'old-{scale}-config.json');print('START',name,flush=True)
 cost=f.execute(old/'bin/gpu_ffn.exe',Path(cfg['model_path']),ref/'prompt.txt',O/name,8,cfg,old,'D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0',7200)
 audit_old(scale,name);write(O/(name+'-process.json'),cost);print('DONE',name,flush=True)
def run_candidate(scale):
 name=f'candidate-{scale}-wall3';cfg=read(O/f'candidate-{scale}-live.config.json');cfg['verify']=False;p=O/(name+'.config.json');write(p,cfg);print('START',name,flush=True);t=time.perf_counter()
 with (O/(name+'.log')).open('w') as log:subprocess.run([str(B/'f_candidate.exe'),str(p),str(O/name)],env=env,stdout=log,stderr=log,check=True,timeout=7200)
 cost=time.perf_counter()-t;assert sha(O/name/'logits.f32')==sha(O/'native-old/logits.f32')
 x=read(O/name/'result.json');assert x['drained'] and x['tokens']==read(O/'native-old/summary.json')['raw_tokens'];assert read(O/name/'input.json')['tokens']==read(O/'native-old/summary.json')['prompt_ids']
 full=read(O/f'candidate-{scale}-live/result.json');assert x['requests']==full['requests'] and x['covered_bytes']==full['covered_bytes']
 write(O/(name+'-process.json'),{'wall_s':cost,'logits_bitwise_equal':True,'input_tokens_equal':True,'requests':x['requests'],'covered_bytes':x['covered_bytes']});print('DONE',name,flush=True)
run_old('F1');run_candidate('F1');run_candidate('F2');run_old('F2')
observing=False;watch.join(timeout=10);write(O/'wall3-process-observations.json',{'interval_s':5,'samples':observations,'conflicts':conflicts});assert not conflicts,'concurrent workspace GPU experiment; wall batch is not accepted'
for p,h in sealed.items():assert sha(R/p)==h,'candidate drift '+p
write(O/'wall3-completed.json',{'status':'pass','runs':4,'sampling':'one independent stable run per system per scale; descriptive, not statistical confirmation'})
