"""Frozen context holdout: native llama activations, complete-tensor R3 replay."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from run_llama_integrated import environment
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
out=ROOT/'output/r3-context-v2-20261108';out.mkdir(exist_ok=False)
build=ROOT/'build-r3-context-20261108';env=environment(build,cuda_root='D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0')
env['PATH']=str(ROOT/'build-r2-final-20261107/bin')+os.pathsep+env['PATH']
profile=read(ROOT/'configs/integrated/smollm2-135m.json');assert sha(profile['model_path'])==profile['gguf_sha256']
inputs=list((ROOT/'experiments/readonly_r3_model').glob('*.*'))+list(build.glob('*.exe'))+list((ROOT/'build-r2-final-20261107/bin').glob('*.dll'))+[ROOT/'build-r3-current-20261107/r3_runtime.lib',ROOT/'build-r3-current-20261107/eventlab/hbf_eventlab.lib']
hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
write(out/'protocol.json',{'contexts':[31,32,33],'reason':'one below / at / above fixed native ubatch=32; prefill and first decode','model':profile['model_path'],'model_sha256':profile['gguf_sha256'],'layer':29,'strategies':['ptx','cuda'],'merge':True,'rule':'unchanged from R3 final frozen policy','scope':'real contextual activation holdout; full tensor read then native arithmetic consumer replay, not end-to-end HBF inference','acceptance':'exact complete tensor bytes, complete GPU consumer output bytes versus original weights, CPU checks and all-lane dependency checks; cross-route identical activation and result hashes','sources_and_binaries':hashes})
weights=read(ROOT/'output/r3-20261107/hybrid-final/coverage-manifest.json')
results=[]
for context in [31,32,33]:
 cap=out/f'context-{context}'
 with (out/f'capture-{context}.log').open('w') as log:subprocess.run([str(build/'r3_capture.exe'),profile['model_path'],str(context),str(cap)],env=env,stdout=log,stderr=log,check=True,timeout=180)
 records=read(cap/'manifest.json')['records'];assert records
 # Last prefill microbatch includes the actual U+1 tail; decode sees all prior tokens.
 selected={}
 for item in records:selected[(item['phase'],item['tensor'])]=item
 assert len(selected)==4
 for (phase,tensor),activation in selected.items():
  item=next(w for w in weights if w['tensor']==tensor);assert activation['columns']==item['columns']
  expected=sha(activation['activation_file']);outputs=[]
  for route in ['ptx','cuda']:
   cfg={k:item[k] for k in ['input_file','bytes','columns','rows','dtype']}
   cfg.update(strategy=route,merge=True,width=16,timed=True,repeats=1,tokens=activation['tokens'],activation_file=activation['activation_file'])
   name=f'{context}-{phase}-{item["dtype"]}-{route}';config=out/f'{name}.config.json';result=out/f'{name}.result.json';write(config,cfg)
   print('starting',name,flush=True)
   with (out/f'{name}.log').open('w') as log:subprocess.run([str(build/'r3_context_replay.exe'),str(config),str(result)],env=env,stdout=log,stderr=log,check=True,timeout=300)
   r=read(result);assert r['status']=='pass' and r['drained'] and r['activation_sha256']==expected and r['mirror_sha256']==item['sha256'] and r['dependency_checks']==r['logical_accesses']
   outputs.append(r['consumer_output_sha256'])
   results.append({'context':context,'phase':phase,'tensor':tensor,'tokens':activation['tokens'],'route':route,'activation_sha256':expected,'output_sha256':outputs[-1],'result':str(result.relative_to(ROOT))})
  assert outputs[0]==outputs[1]
assert all(sha(ROOT/p)==h for p,h in hashes.items())
# Context-dependent activations must actually differ; equal batch sizes alone do not prove context holdout.
for tensor in {r['tensor'] for r in results}:
 assert len({r['activation_sha256'] for r in results if r['tensor']==tensor and r['phase']=='decode'})==3
write(out/'result.json',{'status':'pass','runs':len(results),'contexts':[31,32,33],'records':results,'source_hashes_unchanged':True})
print('context holdout passed',len(results),flush=True)

