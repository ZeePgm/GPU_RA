import json,hashlib,sys
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R/'experiments/gpu_ffn'));import run as f
O=R/'output/readonly-f-compare-a-20261108';O.mkdir(exist_ok=False)
B=R/'build-sf-f-entry-20261105';ref=R/'output/gpu-ffn/20261105-new-backend-f1-f3/F1';identity=f.read(ref/'build-identity.json');assert f.sha(B/'bin/gpu_ffn.exe')==identity['binary_sha256']
for name,h in identity['runtime_dlls'].items():assert f.sha(B/'bin'/name)==h
c=f.read(ref/'native-config.json');assert f.sha(Path(c['model_path']))==c['gguf_sha256']
f.write(O/'preflight.json',{'baseline':'second F 20261105','binary_sha256':identity['binary_sha256'],'dlls_verified':True,'model_verified':True,'kv':'native F16; HBFLAB_M4_PAGED=0 in baseline; no KV hooks configured in candidate','outputs':8,'clock':'A, per-system schedules; GPU active durations plus model service; excludes CPU simulation compute'})
cost=f.execute(B/'bin/gpu_ffn.exe',Path(c['model_path']),ref/'prompt.txt',O/'native-old',8,c,B,'D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0',7200)
header={'model_sha256':c['gguf_sha256'],'prompt_sha256':f.sha(ref/'prompt.txt'),'binary_sha256':f.sha(B/'bin/gpu_ffn.exe'),'native_config_sha256':f.sha(O/'native-old-config.json'),'native_events_sha256':f.sha(O/'native-old/events.jsonl')};table=f.capture_table(O/'native-old',header);f.write(O/'old-compute-table.json',table)
for scale in ['F1','F2']:
 cfg=f.read(R/f'output/gpu-ffn/20261105-new-backend-f1-f3/{scale}/{scale}-config.json');cfg.update(compute_table_path=str(O/'old-compute-table.json'),expected_compute_header=table['header']);print('START',scale,flush=True)
 cost=f.execute(B/'bin/gpu_ffn.exe',Path(c['model_path']),ref/'prompt.txt',O/('old-'+scale),8,cfg,B,'D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0',7200);f.write(O/(scale+'-old-cost.json'),cost);print('DONE',scale,flush=True)
