"""Real executor rejection tests; output remains retained."""
import copy,subprocess
from common import read,write
from study import ROOT,environment
p=ROOT/'output/gpu-a1/20261102-formal-001/single-10';cfg=read(p/'model.json');out=ROOT/'output/gpu-a1/20261102-negative-001';assert not out.exists();out.mkdir(parents=True)
result=[]
for case in ['missing','reordered','wrong_step','bad_mode']:
 c=copy.deepcopy(cfg)
 if case=='missing':c['a1']['gates'].pop()
 if case=='reordered':c['a1']['gates'][0],c['a1']['gates'][1]=c['a1']['gates'][1],c['a1']['gates'][0]
 if case=='wrong_step':c['a1']['gates'][0]['step']=7
 if case=='bad_mode':c['a1']['mode']='invalid'
 f=out/(case+'.json');write(f,c)
 with (out/(case+'.log')).open('w') as log:
  r=subprocess.run([str(ROOT/'build-s3/bin/gpu_a1.exe'),c['model_path'],str(ROOT/'output/gpu-s/20261101-s8-002/G3/prompt.txt'),str(out/case),'8','timing',str(f)],env=environment(),stdout=log,stderr=log,timeout=90)
 assert r.returncode!=0 and not (out/case/'summary.json').exists()
 result.append({'case':case,'exitcode':r.returncode,'rejected':True})
write(out/'result.json',result);print(result)
