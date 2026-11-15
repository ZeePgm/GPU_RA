"""Independent acceptance and comparable metric extraction; no performance pass threshold."""
import json, sys, hashlib, collections
from pathlib import Path
R=Path.cwd();O=R/'output/readonly-f-compare-a-20261108'
sys.path.insert(0,str(R/'experiments/gpu_ffn'));import run as f
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def metrics(t):return {'ttft':t[0],'tpot':(t[-1]-t[0])/7,'token8':t[-1]}
assert read(O/'completed.json')['status']==read(O/'wall3-completed.json')['status']=='pass'
reference=read(O/'native-old/summary.json');logit_sha=sha(O/'native-old/logits.f32');results={}
for scale,mult in [('F1',1),('F2',4)]:
 regions=read(R/f'output/gpu-ffn/20261105-new-backend-f1-f3/{scale}/regions.json')
 expected=18*1024**2*mult*8
 for mode in ['qualification','wall3']:
  assert read(O/f'audit-{mode}/{scale}/acceptance.json')['status']=='pass'
 for name in [f'old-{scale}',f'old-{scale}-wall3',f'candidate-{scale}-capture-perf',f'candidate-{scale}-live',f'candidate-{scale}-wall3']:
  assert sha(O/name/'logits.f32')==logit_sha,name+' logits'
  if name.startswith('candidate'):
   runtime_log=(O/(name+'.log')).read_text(encoding='utf-8',errors='replace');assert 'K (f16):' in runtime_log and 'V (f16):' in runtime_log and 'CUDA0 KV buffer size =    56.00 MiB' in runtime_log,name+' native KV runtime log'
   assert read(O/name/'input.json')['tokens']==reference['prompt_ids'],name+' input'
   x=read(O/name/'result.json');assert x['tokens']==reference['raw_tokens'] and x['drained'] and x['covered_bytes']==expected
   counts=collections.Counter((c['step'],c['tensor']) for c in x['calls'])
   assert counts==collections.Counter({(step,r['tensor']):1 for step in range(8) for r in regions})
   assert all(c['bytes']==6291456 for c in x['calls'])
   if 'capture' not in name:assert x['requests']==expected//512
 full=read(O/f'candidate-{scale}-live/result.json');assert full['dependency_checks']==expected//16
 # Independent CUPTI-to-batch association, same timestamp domain, no hidden missing batch.
 cap=O/f'candidate-{scale}-capture-perf';events=f.rows(cap/'gpu.jsonl');marks=f.rows(cap/'batches.jsonl');categories=collections.defaultdict(list)
 for step in range(8):
  kernels=sorted([e for e in events if e['step']==step and 'read_plan' in e.get('name','')],key=lambda x:x['start']);batches=[m for m in marks if m['step']==step]
  assert len(kernels)==len(batches)==4608*mult
  for i,(k,m) in enumerate(zip(kernels,batches)):
   assert m['at']<=k['start']<k['end']
   if i+1<len(batches):assert k['end']<=batches[i+1]['at']
 for e in events:
  name=e.get('name','');kind='read_kernel' if 'read_plan' in name else 'scatter' if 'scatter' in name else 'model_other_kernel' if e['kind']=='kernel' else e['kind'];categories[kind].append((e['start'],e['end']))
 breakdown={k:{'count':len(v),'active_ms':f.union_ns(v)/1e6} for k,v in categories.items()}
 oldsim=read(O/f'old-{scale}-clock-check.json');newsim=read(O/f'candidate-{scale}-clock.json')
 assert [t['ps'] for t in oldsim['tokens']]==[t['relative_ps'] for t in f.rows(O/f'old-{scale}/tokens.jsonl')]
 for s in [oldsim,newsim]:
  assert s['status']=='pass' and s['bytes']==expected and s['read_fragments']==expected//64
  assert all(t['ps']==t['compute_ps']+t['wait_ps'] for t in s['tokens'])
 assert f.union_ns([(e['start'],e['end']) for e in events])*1000==newsim['compute_ps']
 oldevents=f.rows(O/f'old-{scale}-wall3/events.jsonl');start=next(e['timestamp_ns'] for e in oldevents if e['type']=='request_begin');ticks=[(e['timestamp_ns']-start)/1e9 for e in oldevents if e['type']=='token_emitted'];assert len(ticks)==8
 new=read(O/f'candidate-{scale}-wall3/result.json');oldsession=read(O/f'old-{scale}-wall3/session.json')
 entry={'selected_matrices':3*mult,'read_MiB':expected/1024**2,'old_matrix_parents':24*mult,'new_warp_parents':expected//512,'new_batch_count':expected//4096,'read_fragments_each':expected//64,'lane_dependency_checks':full['dependency_checks'],'old_sim_ms':metrics([t['ps']/1e9 for t in oldsim['tokens']]),'new_sim_ms':metrics([t['ps']/1e9 for t in newsim['tokens']]),'old_wall_s':metrics(ticks),'new_wall_s':metrics([t/1000 for t in new['token_wall_ms']]),'old_process_s':read(O/f'old-{scale}-wall3-process.json')['process_wall_seconds'],'new_process_s':read(O/f'candidate-{scale}-wall3-process.json')['wall_s'],'old_backend_init_s':oldsession['initialization_wall_ns']/1e9,'new_adapter_setup_s':new['setup_ms']/1000,'old_read_calls_s':sum(g['after_ns']-g['before_ns'] for g in oldsession['accesses'])/1e9,'new_gate_calls_s':sum(c['gate_ms'] for c in new['calls'])/1000,'old_compute_total_ms':oldsim['compute_ps']/1e9,'new_compute_total_ms':newsim['compute_ps']/1e9,'old_wait_total_ms':oldsim['wait_ps']/1e9,'new_wait_total_ms':newsim['wait_ps']/1e9,'gpu_activity_breakdown':breakdown,'common_core_old_replay_s':oldsim['run_wall_s'],'common_core_new_replay_s':newsim['run_wall_s']}
 entry['wall_speedup']={k:entry['old_wall_s'][k]/entry['new_wall_s'][k] for k in ['ttft','tpot']};entry['sim_slowdown']={k:entry['new_sim_ms'][k]/entry['old_sim_ms'][k] for k in ['ttft','tpot']};results[scale]=entry
for p,h in read(O/'candidate-identity.json').items():assert sha(R/p)==h,'source drift '+p
for p,h in read(O/'dependency-identity.json').items():assert sha(R/p)==h,'dependency drift '+p
for item in read(R/'output/readonly-f-compare-20261108/preflight-inputs.json')['inputs']:assert sha(R/item['path'])==item['sha256'],'prior input drift '+item['path']
profile=read(R/'output/gpu-ffn/20261105-new-backend-f1-f3/F1/native-config.json');assert sha(Path(profile['model_path']))==profile['gguf_sha256'],'model drift'
report={'status':'pass','baseline':'second F, 2026-11-05, sealed executable and DLLs','model':'Qwen3-0.6B F16','prompt_tokens':len(reference['prompt_ids']),'outputs':8,'logits_count':151936*8,'logits_sha256':logit_sha,'input_bytes':249,'kv':'original native F16','clock':'A: own GPU active work and own dependency policy; cohort approximation for 8 warp requests; no CPU solver elapsed time','sampling':'one stable wall run per system/scale and one GPU trace each; descriptive results, no confidence interval','cases':results,'acceptance':{'numerics':'bitwise exact logits and token IDs','inputs':'identical token IDs, CRLF bytes retained','coverage':'exact matrix, offset, byte and 64B fragment counts','dependencies':'old consumer API starts after gate completion; new lane time/data checks plus byte publication before graph submission','clock':'old token ps exactly reproduced on common backend; all token conservation residuals zero','kv':'native F16; no KV hooks','drain':'all sessions successful and drained'}}
probe=read(O/'wall3-process-observations.json');assert probe['samples'] and not probe['conflicts'],'resource contention in final batch'
failed=O/'candidate-F2-wall/logits.f32';prefix=failed.read_bytes();assert len(prefix)==5*151936*4 and prefix==(O/'native-old/logits.f32').read_bytes()[:len(prefix)]
report['status']='pass_retained_runs'
report['excluded_runs']=[{'batch':'wall','reason':'candidate F2 request timeout (status 4) after five exact outputs; K4 parallel build evidenced by object file timestamps','root_cause':'not uniquely localized; concurrent build and driver polling were removed before complete wall3 rerun','changed_gpu_binary':False,'changed_timeout_threshold':False,'failure_log':str(O/'candidate-F2-wall.log'),'partial_outputs':5}]
second=read(O/'wall2-interrupted.json');assert second['observed_conflicts']
report['excluded_runs'].append({'batch':'wall2','reason':second['reason'],'status':second['status'],'record':str(O/'wall2-interrupted.json')})
report['acceptance']['drain']='all retained completed sessions successful and drained; failed first wall batch explicitly excluded'
report['acceptance']['resource_window']={'native_process_samples':len(probe['samples']),'observed_foreign_workspace_or_build_processes':0,'interval_s':probe['interval_s']}
f.write(O/'comparison.json',report)
print(json.dumps(report,indent=2,ensure_ascii=False))
