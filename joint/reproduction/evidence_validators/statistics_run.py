"""Frozen 142-process paired timing confirmation; sealed model binaries unchanged."""
import argparse,hashlib,importlib.util,json,os,random,statistics,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
SEALED=ROOT/'integrated/20261112-dual-line-r1'
spec=importlib.util.spec_from_file_location('correctness',SEALED/'scripts/run_correctness.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
spec=importlib.util.spec_from_file_location('process_probe',ROOT/'experiments/readonly_f_compare/process_probe.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
MODES={
 'native':('native','native','native'),
 'readonly-bypass':('b2','native','bypass'),
 'readonly-modeled':('b2','native','modeled'),
 'kv-bypass':('native','paged','bypass'),
 'kv-modeled':('native','paged','modeled'),
 'joint-bypass':('b2','paged','bypass'),
 'joint-modeled':('b2','paged','modeled')}

def stamp():return datetime.now(timezone.utc).isoformat()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def event(kind,**kw):
 row={'utc':stamp(),'event':kind,**kw}
 with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps(row,ensure_ascii=False),flush=True)
def identities():
 paths=[*C.BIN.glob('*.exe'),*C.BIN.glob('*.dll'),SEALED/'scripts/run_correctness.py',ROOT/'experiments/readonly_f_compare/process_probe.py',SEALED/'configs/qwen3-0.6b.json',SEALED/'configs/prompt.txt']
 return [{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)]
def verify_identity(proto):
 assert sha(Path(__file__))==proto['runner_sha256'],'runner drift'
 assert identities()==proto['dependencies'],'dependency drift'
 assert sha(C.read(SEALED/'configs/qwen3-0.6b.json')['model_path'])==proto['model_sha256'],'model drift'
def prepare():
 assert not (OUT/'protocol.json').exists(),'protocol already frozen'
 for name in ['raw','reports','provenance']:(OUT/name).mkdir(exist_ok=True)
 rng=random.Random(131113);schedule=[];orders={}
 for scale in ['F1','F2']:
  warm=list(MODES);rng.shuffle(warm)
  aa=[False,True]*2;rng.shuffle(aa)
  blocks=[]
  for _ in range(4):
   order=list(MODES);rng.shuffle(order);blocks.extend([order,list(reversed(order))])
  rng.shuffle(blocks)
  orders[scale]={'warmup':warm,'AA_reverse':aa,'blocks':blocks}
  for mode in warm:schedule.append({'scale':scale,'stage':'warmup','block':0,'mode':mode,'replicate':'w'})
  for b,reverse in enumerate(aa):
   for rep in (['b','a'] if reverse else ['a','b']):schedule.append({'scale':scale,'stage':'AA','block':b,'mode':'joint-modeled','replicate':rep})
  for b,block in enumerate(blocks):
   for mode in block:schedule.append({'scale':scale,'stage':'confirm','block':b,'mode':mode,'replicate':'c'})
 assert len(schedule)==142
 for i,row in enumerate(schedule):row['index']=i+1;row['id']=f"{i+1:03d}-{row['scale']}-{row['stage']}-{row['block']:02d}-{row['replicate']}-{row['mode']}"
 proto={'version':1,'frozen_utc':stamp(),'task_start_utc':'2026-11-13T04:30:34.584553+00:00',
  'authorization':'User approved fixed 7-group, 8-block, 4-AA-pair-per-scale protocol on 2026-11-13',
  'scope':'Qwen3-0.6B F16, original 44-input/8-output prompt, single sequence, F1 then F2',
  'sealed_source':str(SEALED),'modes':MODES,'write_source':'device','audit':'Full existing byte/lane/KV audits and ordinary logs unchanged; CUPTI disabled',
  'primary_endpoint':'generation_wall_ms: 44-token prefill + 7 decode calls + final audit/drain; excludes model load/setup',
  'secondary_endpoints':['prefill_ms','decode_7_ms','decode_per_token_ms','full_process_wall_s'],
  'primary_contrasts':['readonly-modeled minus readonly-bypass','kv-modeled minus kv-bypass','joint-modeled minus joint-bypass'],
  'interval':'[min,max] of 8 paired block differences; distribution-free population median interval under independent blocks; coverage >=99.21875% per contrast, union bound >=95.3125% for 6 contrasts',
  'AA_noise':'maximum absolute difference in 4 joint-modeled A/A pairs per scale; common reference resolution, not individual-group variance estimate',
  'gate':'increase if interval lower > AA noise; decrease if upper < -AA noise; otherwise inconclusive at this resolution. Zero-exclusion also reported separately.',
  'interaction':'Descriptive only: joint modeled-minus-bypass minus sum of standalone modeled-minus-bypass; no extra confirmatory claim',
  'order':'4 random permutations plus their reverses, shuffled: each group pair AB/BA=4/4 within scale',
  'seed':131113,'processes':142,'warmups_per_group_per_scale':1,'blocks_per_scale':8,'AA_pairs_per_scale':4,
  'resource_policy':'Single model process; OS process snapshots each 5s; reject competing workspace experiment/compiler, preserve evidence and stop batch; no GPU driver queries during timed process',
  'stopping':'Fixed budget, no significance-based extension. Any numerical/timeout/resource conflict stops affected batch; no automatic sample replacement.',
  'timeout_s':3600,'runner_sha256':sha(__file__),'dependencies':identities(),
  'model_sha256':sha(C.read(SEALED/'configs/qwen3-0.6b.json')['model_path']),
  'orders':orders,'schedule':schedule}
 write(OUT/'protocol.json',proto)
 event('prepared',processes=len(schedule),protocol_sha256=sha(OUT/'protocol.json'))

def probe(allowed=None):
 rows=P.processes();assert any(p['pid']==4 and p['name']=='System' for p in rows),'host process visibility missing'
 allowed={os.getpid(),os.getppid(),allowed};root=str(ROOT).replace('\\','/').lower()+'/'
 conflicts=[p for p in rows if p['pid'] not in allowed and (p['name'].lower() in ('cl.exe','nvcc.exe','ninja.exe','realgpu_llama.exe') or (p['path'].replace('\\','/').lower().startswith(root) and p['name'].lower().endswith('.exe')))]
 return {'utc':stamp(),'visible_processes':len(rows),'conflicts':conflicts}
def gpu_snapshot(name):
 r=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv'],capture_output=True,text=True)
 (OUT/'provenance'/name).write_text(r.stdout+r.stderr,encoding='utf-8');assert r.returncode==0
def numerical(folder,scale,mode):
 gold=SEALED/'results/correctness-001'/f'{scale}-{mode}'
 s=read(folder/'summary.json');g=read(gold/'summary.json')
 assert s['prompt_tokens']==g['prompt_tokens'] and s['generated_tokens']==g['generated_tokens'],'tokens differ'
 assert not s['observed'] and s['sequence_count']==1 and not s['scenario']
 digest=sha(folder/'logits.f32');assert digest==sha(gold/'logits.f32'),'same-path logits drift'
 native=np.fromfile(SEALED/f'results/correctness-001/{scale}-native/logits.f32',dtype='<f4').astype('f8')
 values=np.fromfile(folder/'logits.f32',dtype='<f4').astype('f8');d=values-native
 max_abs=float(np.max(np.abs(d)));rms=float(np.sqrt(np.mean(d*d)))
 tol=read(SEALED/'configs/qwen3-0.6b.json')['numerical'];assert max_abs<=tol['max_abs'] and rms<=tol['rms']
 content=None
 if (folder/'kv-content.json').exists():
  content=read(folder/'kv-content.json')['sequences'];assert content==read(gold/'kv-content.json')['sequences'],'KV bytes drift'
 return {'status':'pass','tokens_equal':True,'logits_sha256':digest,'max_abs_vs_native':max_abs,'rms_vs_native':rms,'KV_content':content}
def run_one(row,proto,completed):
 folder=OUT/'raw'/row['id'];meta=OUT/'raw'/(row['id']+'.process.json')
 assert not folder.exists() and not meta.exists(),'refuse overwrite'
 before=probe();assert not before['conflicts'],before
 weights,kv,mode=MODES[row['mode']];env=C.environment()
 env.update(HBFLAB_JOINT_SCALE=row['scale'],HBFLAB_JOINT_WEIGHTS=weights,HBFLAB_JOINT_KV=kv,HBFLAB_KV_WRITE_MODE='device')
 cfg=read(SEALED/'configs/qwen3-0.6b.json')
 cmd=[str(C.BIN/'realgpu_llama.exe'),cfg['model_path'],str(SEALED/'configs/prompt.txt'),str(folder),'8','0',mode,str(SEALED/'configs/qwen3-0.6b.json')]
 started=stamp();t0=time.monotonic();observed=[];probes=[before];failure=None
 event('start',id=row['id'],completed=completed,total=142)
 next_probe=0;next_heartbeat=40
 with (OUT/'raw'/(row['id']+'.log')).open('w',encoding='utf-8') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
  try:
   while proc.poll() is None:
    elapsed=time.monotonic()-t0
    if elapsed>=next_probe:
     snap=probe(proc.pid);probes.append(snap);next_probe=elapsed+5
     assert not snap['conflicts'],f'resource conflict: {snap}'
    if not observed and elapsed>=.3:
     paths=C.modules(proc.pid)
     if any(Path(p).name.lower()=='ggml-cuda.dll' for p in paths):observed=paths
    if elapsed>=next_heartbeat:
     status={'state':'running','current':row,'process_elapsed_s':elapsed,'completed':completed,'total':142,'utc':stamp()}
     write(OUT/'status.json',status);event('heartbeat',id=row['id'],elapsed_s=round(elapsed,1),completed=completed);next_heartbeat=elapsed+40
    if elapsed>proto['timeout_s']:raise TimeoutError(row['id'])
    time.sleep(.1)
  except BaseException:
   failure=traceback.format_exc()
   if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
 elapsed=time.monotonic()-t0
 record={**row,'start_utc':started,'end_utc':stamp(),'wall_s':elapsed,'returncode':proc.returncode,'error':failure,'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('HBFLAB_','GGML_','CUDA_'))},'modules':observed,'resource_probes':probes}
 write(meta,record)
 assert failure is None and proc.returncode==0,(row['id'],failure,proc.returncode)
 dlls={Path(p).name.lower():Path(p) for p in observed};module_checks=[]
 for f in C.BIN.glob('*.dll'):
  if f.name.lower().startswith(('ggml','llama')):
   assert f.name.lower() in dlls and dlls[f.name.lower()].resolve()==f.resolve(),('wrong/missing DLL',f)
   module_checks.append({'path':str(f),'sha256':sha(f)})
 write(folder/'module-identity.json',module_checks)
 C.validate_case(folder,row['scale'],weights,kv,mode)
 validation=numerical(folder,row['scale'],row['mode']);write(folder/'validation.json',validation)
 s=read(folder/'summary.json');times=s['process_wall_ms'];assert len(times)==8 and all(v>0 for v in times)
 sample={**row,'generation_wall_ms':s['generation_wall_ms'],'prefill_ms':times[0],'decode_7_ms':sum(times[1:]),'decode_per_token_ms':sum(times[1:])/7,'wall_s':elapsed,'logits_sha256':validation['logits_sha256'],'status':'pass'}
 assert sample['generation_wall_ms']>=sum(times)
 event('pass',id=row['id'],wall_s=round(elapsed,3),generation_ms=round(sample['generation_wall_ms'],3))
 return sample

def distribution(values):
 return {'n':len(values),'median':statistics.median(values),'q1':float(np.quantile(values,.25)),'q3':float(np.quantile(values,.75)),'min':min(values),'max':max(values),'values':values}
def analyze(scale,samples):
 rows=[r for r in samples if r['scale']==scale];assert len(rows)==71 and all(r['status']=='pass' for r in rows)
 indexed={(r['stage'],r['block'],r['replicate'],r['mode']):r for r in rows}
 aa=[indexed['AA',b,'a','joint-modeled']['generation_wall_ms']-indexed['AA',b,'b','joint-modeled']['generation_wall_ms'] for b in range(4)]
 noise=max(map(abs,aa));contrasts={};deltas={}
 for line in ['readonly','kv','joint']:
  d=[indexed['confirm',b,'c',line+'-modeled']['generation_wall_ms']-indexed['confirm',b,'c',line+'-bypass']['generation_wall_ms'] for b in range(8)]
  deltas[line]=d
  contrasts[line]={'paired_delta_ms':distribution(d),'median_interval_ms':[min(d),max(d)],'zero_excluded':min(d)>0 or max(d)<0,
   'AA_reference_resolution_ms':noise,'decision':'increase' if min(d)>noise else 'decrease' if max(d)<-noise else 'inconclusive'}
 groups={}
 for mode in MODES:
  rr=[indexed['confirm',b,'c',mode] for b in range(8)]
  groups[mode]={key:distribution([r[key] for r in rr]) for key in ['generation_wall_ms','prefill_ms','decode_7_ms','decode_per_token_ms','wall_s']}
 interaction=[deltas['joint'][i]-deltas['readonly'][i]-deltas['kv'][i] for i in range(8)]
 result={'scale':scale,'status':'accepted','accepted_processes':71,'formal_samples':56,'warmups':7,'AA_processes':8,'AA_differences_ms':aa,'AA_resolution_ms':noise,'contrasts':contrasts,'groups':groups,'interaction_descriptive_ms':distribution(interaction)}
 write(OUT/f'reports/{scale}-statistics.json',result);event('scale-accepted',scale=scale,processes=71,decisions={k:v['decision'] for k,v in contrasts.items()})
 return result
def report(proto,samples,results):
 end=stamp();start=datetime.fromisoformat(proto['task_start_utc']);elapsed=(datetime.fromisoformat(end)-start).total_seconds()
 result={'status':'complete','start_utc':proto['task_start_utc'],'end_utc':end,'elapsed_s':elapsed,'accepted_model_processes':len(samples),'model_process_sum_s':sum(s['wall_s'] for s in samples),'protocol_sha256':sha(OUT/'protocol.json'),'scales':results}
 write(OUT/'reports/acceptance.json',result)
 lines=['# 双线联合统计性能确认','', 'Qwen3-0.6B F16，原始 44 输入 / 8 输出，单序列；F1 后 F2。封存二进制与完整审计配置保持一致。', '',f"完成 {len(samples)} 个模型进程；正确性、模块身份与资源排空全部通过。总耗时 {elapsed/60:.2f} 分钟，模型进程累计 {result['model_process_sum_s']/60:.2f} 分钟。",'', '主指标 generation_wall_ms 包括 prefill、7 次 decode 和最终审计/排空，排除模型加载与初始化。分阶段时间和进程耗时另见机器统计文件。', '', '| 规模 | 组别 | 生成时延中位数 / 秒 | Q1–Q3 / 秒 | 最小–最大 / 秒 |', '|---|---|---:|---:|---:|']
 for scale,r in results.items():
  for mode,g in r['groups'].items():
   d=g['generation_wall_ms'];lines.append(f"| {scale} | {mode} | {d['median']/1000:.3f} | {d['q1']/1000:.3f}–{d['q3']/1000:.3f} | {d['min']/1000:.3f}–{d['max']/1000:.3f} |")
 lines+=['','主比较均为同轮 modeled−bypass。区间为 8 个配对差值的最小值与最大值：在独立分块假设下，单项总体中位数覆盖率至少 99.21875%，6 项联合覆盖率下界 95.3125%。A/A 是联合组的共同分辨率参考，并不替代各组方差估计。','','| 规模 | 路径 | 差值中位数 / 秒 | 中位数区间 / 秒 | A/A 参考 / 秒 | 判定 |','|---|---|---:|---:|---:|---|']
 names={'increase':'时延增加，超过参考噪声','decrease':'时延减少，超过参考噪声','inconclusive':'当前分辨率下结论不明确'}
 for scale,r in results.items():
  for key,d in r['contrasts'].items():
   lo,hi=d['median_interval_ms'];lines.append(f"| {scale} | {key} | {d['paired_delta_ms']['median']/1000:.3f} | [{lo/1000:.3f}, {hi/1000:.3f}] | {d['AA_reference_resolution_ms']/1000:.3f} | {names[d['decision']]} |")
 lines+=['','叠加效应仅作描述性分析：联合增量−权重增量−KV 增量；未将其扩展为额外显著性结论。']
 for scale,r in results.items():lines.append(f"{scale} 叠加差值中位数为 {r['interaction_descriptive_ms']['median']/1000:.3f} 秒。")
 lines+=['','所有正式样本均保留，无按速度剔除或显著性追加。计时期间仅做 OS 进程监测，不轮询 GPU 驱动。完整审计、日志及短提示配置共同限定本次时延口径。','','记录：protocol.json 固化完整顺序与判定；samples.json 保留全部样本；raw/ 保留逐进程输出、日志、参数、模块身份、监测与正确性验收；provenance/ 保留封存身份与环境；reports/ 保留统计。', '',f"起止 UTC：{proto['task_start_utc']} 至 {end}。"]
 (OUT/'reports/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 write(OUT/'status.json',{'state':'complete','completed':len(samples),'total':142,'utc':end,'elapsed_s':elapsed})

def run():
 proto=read(OUT/'protocol.json');assert not (OUT/'samples.json').exists(),'refuse restart or overwrite'
 verify_identity(proto)
 verification=subprocess.run([sys.executable,str(SEALED/'scripts/verify.py')],capture_output=True,text=True)
 (OUT/'provenance/sealed-verification-before.log').write_text(verification.stdout+verification.stderr,encoding='utf-8');assert verification.returncode==0
 write(OUT/'provenance/host-processes-before.json',P.processes());gpu_snapshot('gpu-before-F1.csv')
 samples=[];results={};start=time.monotonic()
 for row in proto['schedule']:
  if row['scale']=='F2' and 'F1' not in results:
   results['F1']=analyze('F1',samples);verify_identity(proto);gpu_snapshot('gpu-before-F2.csv')
  sample=run_one(row,proto,len(samples));samples.append(sample);write(OUT/'samples.json',samples)
  write(OUT/'status.json',{'state':'running','last_pass':row['id'],'completed':len(samples),'total':142,'utc':stamp(),'run_elapsed_s':time.monotonic()-start})
 results['F2']=analyze('F2',samples)
 verify_identity(proto);gpu_snapshot('gpu-after.csv');write(OUT/'provenance/host-processes-after.json',P.processes())
 verification=subprocess.run([sys.executable,str(SEALED/'scripts/verify.py')],capture_output=True,text=True)
 (OUT/'provenance/sealed-verification-after.log').write_text(verification.stdout+verification.stderr,encoding='utf-8');assert verification.returncode==0
 report(proto,samples,results);event('complete',accepted=len(samples),run_elapsed_s=time.monotonic()-start)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);a=ap.parse_args()
 try:prepare() if a.action=='prepare' else run()
 except BaseException:
  failure=traceback.format_exc();write(OUT/'failure.json',{'utc':stamp(),'traceback':failure});event('failed',error=failure);raise
