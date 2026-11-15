"""Fit only the preregistered training sizes, then evaluate sealed holdouts."""
import argparse,csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,read,write,sha
ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);args=ap.parse_args();out=ROOT/args.study
assert read(out/'counts-result.json')['status']=='pass'
with (out/'measurements.csv').open() as f:data=list(csv.DictReader(f))
points=[{**r,**{k:int(r[k]) for k in ['bytes','wait_ps','repeat','fragments','read_globals','capacity_rejections','read_bank_operations','read_sense_paths','read_cache_paths']}} for r in data if int(r['repeat'])==0]
knots=read(out/'protocol.json')['fit']['knots_bytes'];bounds=[0]+knots+[65536];models=[]
for lo,hi in zip(bounds,bounds[1:]):
 subset=[p for p in points if p['role']=='train' and lo<p['bytes']<=hi];assert len(subset)>=2
 x=np.array([p['bytes'] for p in subset]);y=np.array([p['wait_ps'] for p in subset]);m,b=np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0];err=y-(m*x+b)
 models.append({'lower_exclusive_B':lo,'upper_inclusive_B':hi,'slope_ps_per_byte':float(m),'intercept_ps':float(b),'training_points':len(x),'training_max_abs_residual_ps':float(np.max(np.abs(err)))})
fitrows=[]
for p in points:
 if p['role'] not in ['train','holdout']:continue
 m=next(m for m in models if m['lower_exclusive_B']<p['bytes']<=m['upper_inclusive_B']);prediction=m['slope_ps_per_byte']*p['bytes']+m['intercept_ps'];res=p['wait_ps']-prediction
 fitrows.append({'case':p['case'],'role':p['role'],'bytes':p['bytes'],'wait_ps':p['wait_ps'],'predicted_ps':prediction,'residual_ps':res,'relative_error':abs(res)/p['wait_ps'],'inside_training_residual_band':abs(res)<=m['training_max_abs_residual_ps']+1e-6})
hold=[p for p in fitrows if p['role']=='holdout'];passed=all(p['relative_error']<=.05 for p in hold)
result={'count_audit':read(out/'counts-result.json'),'response_model_status':'pass' if passed else 'fail','response_criterion':'all heldout relative errors <=5%, frozen before execution','models':models,'heldout':hold,'max_heldout_relative_error':max(p['relative_error'] for p in hold),'error_band':'deterministic training residual envelope, not confidence interval or hardware error bound','observed_min_wait_ps':min(p['wait_ps'] for p in points if p['role'] in ['train','holdout']),'observed_max_wait_ps':max(p['wait_ps'] for p in points if p['role'] in ['train','holdout'])}
write(out/'result.json',result)
with (out/'fit.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(fitrows[0]));w.writeheader();w.writerows(fitrows)
fig,axes=plt.subplots(3,1,figsize=(9,9),layout='constrained')
for role,marker in [('train','o'),('holdout','x')]:
 ps=sorted([p for p in points if p['role']==role],key=lambda x:x['bytes']);axes[0].scatter([p['bytes']/1024 for p in ps],[p['wait_ps']/1e6 for p in ps],marker=marker,label=role)
for m in models:
 x=np.linspace(max(64,m['lower_exclusive_B']),min(32832,m['upper_inclusive_B']),100);y=(m['slope_ps_per_byte']*x+m['intercept_ps'])/1e6;band=m['training_max_abs_residual_ps']/1e6;axes[0].plot(x/1024,y,color='gray',lw=1);axes[0].fill_between(x/1024,y-band,y+band,color='gray',alpha=.12)
axes[0].set(ylabel='Modeled read wait (µs)',title='Fixed H0; fresh Session; identical 64 KiB initialization');axes[0].legend()
ps=sorted([p for p in points if p['role'] in ['train','holdout']],key=lambda x:x['bytes']);grid=np.arange(64,32833,64);axes[1].step(grid/1024,np.ceil(grid/4096),where='post',label='Independent chunk rule');axes[1].scatter([p['bytes']/1024 for p in ps],[p['read_globals'] for p in ps],s=15);axes[1].set(ylabel='Read Globals (audited)')
for role,marker in [('train','o'),('holdout','x')]:
 rs=[p for p in fitrows if p['role']==role];axes[2].scatter([p['bytes']/1024 for p in rs],[p['residual_ps']/1e6 for p in rs],marker=marker,label=role)
axes[2].axhline(0,color='gray');axes[2].set(xlabel='Effective bytes (KiB)',ylabel='Observed − fitted wait (µs)');axes[2].legend();axes[2].set_ylim(-1e-6,1e-6);axes[2].set_title('Residual axis +/-1 ps; floating-point roundoff is below this scale')
fig.savefig(out/'response.png',dpi=160);fig.savefig(out/'response.svg')
print(json.dumps({'response_model_status':result['response_model_status'],'holdout':hold,'models':models},indent=2))
