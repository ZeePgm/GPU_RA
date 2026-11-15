"""Fixed-budget formal A1 matrix. Launched after independent lean pilot."""
import subprocess,sys
from study import ROOT
from common import write
out=ROOT/'output/gpu-a1/20261102-formal-001';assert not out.exists();out.mkdir(parents=True)
write(out/'preregistration.json',{'conditions':[[p,u] for p in ['single','multi'] for u in [10,100,1000]],'pairs_each':30,'alpha_family':.05,'endpoints':48,'equivalence':'10% cumulative target ns','estimator':'paired mean residual using actual injection; t CI Bonferroni across 48 endpoints','budget':'fixed 30, no extension or sample exclusion','reason':'pilot suggests insufficient precision; limited-budget characterization, preserve inconclusive; no hardware accuracy claim'})
for pattern in ['single','multi']:
 for us in [10,100,1000]:
  subprocess.run([sys.executable,str(ROOT/'experiments/gpu_s/a1_measure.py'),'--formal','--study',str(out/f'{pattern}-{us}'),'--pairs','30','--us',str(us),'--pattern',pattern],check=True)
print('MATRIX FINISHED',flush=True)
