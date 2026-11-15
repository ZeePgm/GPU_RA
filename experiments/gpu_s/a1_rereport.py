"""Re-report frozen A1 samples; no threshold changes, no exclusions."""
import csv
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read,write
from study import ROOT
out=ROOT/'output/gpu-a1/20261102-formal-001';dest=ROOT/'docs/validation/gpu-a1';dest.mkdir(exist_ok=True)
summary=[];distribution=[]
for pattern in ['single','multi']:
 for us in [10,100,1000]:
  pairs=read(out/f'{pattern}-{us}'/'pairs.json');n=len(pairs);assert n==30
  for i in range(8):
   v=np.array([p['tokens'][i]['difference_ns']/1e6 for p in pairs]);actual=np.array([p['tokens'][i]['actual_ns']/1e6 for p in pairs]);err=v-actual;half=float(t.ppf(1-.05/(2*48),29)*np.std(err,ddof=1)/np.sqrt(n));target=pairs[0]['tokens'][i]['target_ns']/1e6
   summary.append({'pattern':pattern,'gate_us':us,'token':i+1,'n':n,'target_ms':target,'mean_delta_ms':v.mean(),'median_delta_ms':np.median(v),'mean_actual_wait_ms':actual.mean(),'mean_residual_ms':err.mean(),'ci_residual_low_ms':err.mean()-half,'ci_residual_high_ms':err.mean()+half,'ci_halfwidth_ms':half,'frozen_equivalence':'inconclusive','reporting':'below_effective_wall_and_injection_resolution' if us in [10,100] else 'quantitative_equivalence_unestablished'})
  if pattern=='multi' and us==1000:
   for p in pairs:distribution.append({'pair':p['pair'],'order':'zero-first' if p['order'][0]=='zero' else 'inject-first','delta_ms':p['tokens'][-1]['difference_ns']/1e6,'actual_wait_ms':p['tokens'][-1]['actual_ns']/1e6})
for name,data in [('dose-summary.csv',summary),('multi-16ms-distribution.csv',distribution)]:
 with (dest/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
v=np.array([r['delta_ms'] for r in distribution]);groups=[]
for order in ['zero-first','inject-first']:
 a=np.array([r['delta_ms'] for r in distribution if r['order']==order]);groups.append({'order':order,'n':len(a),'mean_ms':a.mean(),'median_ms':np.median(a),'min_ms':a.min(),'max_ms':a.max()})
write(dest/'reporting-summary.json',{'overall':'mechanism_and_accounting_checks_pass; wall_quantitative_equivalence_unestablished_underpowered; original_inconclusive_retained','multi16':{'n':30,'target_ms':16,'mean_ms':v.mean(),'ratio':v.mean()/16,'median_ms':np.median(v),'quantiles':dict(zip(['min','p10','p25','p75','p90','max'],np.quantile(v,[0,.1,.25,.75,.9,1]).tolist())),'orders':groups},'detection_note':'CI halfwidth is an operational precision scale, not a one-sided upper bound on true delay; clocks timestamp granularity is not established injector accuracy.'})
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained');ax[0].hist(v,bins=10,color='#4477aa',edgecolor='white');ax[0].axvline(16,color='black',label='Target 16 ms');ax[0].axvline(v.mean(),color='#cc6677',label=f'Mean {v.mean():.2f}');ax[0].axvline(np.median(v),color='#228833',label=f'Median {np.median(v):.2f}');ax[0].set(xlabel='Paired last-token wall difference (ms)',ylabel='Pairs');ax[0].legend(fontsize=8)
for k,order in enumerate(['zero-first','inject-first']):
 a=[r['delta_ms'] for r in distribution if r['order']==order];ax[1].scatter(np.full(len(a),k)+np.linspace(-.12,.12,len(a)),a,alpha=.7)
ax[1].axhline(16,color='black');ax[1].set(xticks=[0,1],xticklabels=['Zero first','Inject first'],ylabel='Paired difference (ms)',title='All 30 pairs; no exclusions')
f=ROOT/'docs/figures/gpu-a1';f.mkdir(exist_ok=True);fig.savefig(f/'multi16-distribution.png',dpi=160);fig.savefig(f/'multi16-distribution.svg');print(groups)
