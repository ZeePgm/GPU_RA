from pathlib import Path
import shutil,csv
from common import read,write,sha
from study import ROOT
p=ROOT/'output/gpu-a1/20261102-timeline-002';r=read(p/'result.json')
for mode in ['zero','inject']:
 assert read(p/mode/'summary.json')['cupti_dropped']==0
 assert len(r['records'][mode])==16
 assert all(x['overlap_count']==0 and x['kernel_after_gate'] and x['enqueue_window_verified'] for x in r['records'][mode])
for x in r['paired']:
 assert x['delta_gap_ns']==x['delta_pre_ns']+x['delta_gate_ns']+x['delta_post_ns']
write(p/'audit.json',{'status':'pass','gates_each':16,'both_runs_zero_overlapping_activities':True,'post_gate_kernels_and_correlated_enqueue':True,'decomposition_integer_residual_ns':0,'cupti_dropped':0,'scope':'one zero/inject full-instrumentation pair, not 30-pair wall equivalence'})
shutil.copyfile(p/'result.json',ROOT/'docs/validation/gpu-a1/timeline.json');shutil.copyfile(p/'paired.csv',ROOT/'docs/validation/gpu-a1/timeline-paired.csv')
allrows=[]
for mode,records in r['records'].items():
 for x in records:allrows.append({'mode':mode,**x})
with (ROOT/'docs/validation/gpu-a1/timeline-windows.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
for name in ['a1_timeline.py','a1_rereport.py']:shutil.copyfile(ROOT/'experiments/gpu_s'/name,p/name)
write(p/'artifact-index.json',{'files':[{'path':str(f.relative_to(p)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in p.rglob('*') if f.is_file() and f.name!='artifact-index.json']})
a=read(ROOT/'docs/validation/gpu-a1.json');a['reporting_revision']='2026-11-02-r2';a['reporting_summary']=read(ROOT/'docs/validation/gpu-a1/reporting-summary.json');a['timeline_supplement']=read(p/'audit.json');a['overall_status']='mechanism/accounting checks pass; wall quantitative equivalence unestablished (underpowered), inconclusive retained';write(ROOT/'docs/validation/gpu-a1.json',a)
