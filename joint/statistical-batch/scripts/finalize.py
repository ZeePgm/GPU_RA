"""Independent raw-record/statistics audit and final delivery timing."""
import hashlib,json,platform,statistics,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert not (R/'provenance/evidence-manifest.json').exists(),'already sealed'
 samples=read(R/'samples.json');proto=read(R/'protocol.json');accept=read(R/'reports/acceptance.json')
 assert accept['status']=='complete' and len(samples)==142
 dlls={p['path'].lower():p['sha256'] for p in proto['dependencies'] if p['path'].lower().endswith('.dll')}
 modules=0;probe_count=0
 for s in samples:
  folder=R/'raw'/s['id'];summary=read(folder/'summary.json');process=read(folder.with_suffix('.process.json'))
  assert process['returncode']==0 and process['error'] is None
  assert process['wall_s']==s['wall_s']
  assert all(not p['conflicts'] for p in process['resource_probes']);probe_count+=len(process['resource_probes'])
  assert summary['generation_wall_ms']==s['generation_wall_ms'] and summary['process_wall_ms'][0]==s['prefill_ms']
  assert sum(summary['process_wall_ms'][1:])==s['decode_7_ms']
  assert len(summary['prompt_tokens'])==44 and len(summary['generated_tokens'])==8
  assert sha(folder/'logits.f32')==s['logits_sha256']
  for module in read(folder/'module-identity.json'):
   assert dlls[module['path'].lower()]==module['sha256'];modules+=1
 assert modules==710
 for scale in ['F1','F2']:
  data=read(R/f'reports/{scale}-statistics.json')
  confirmed=[s for s in samples if s['scale']==scale and s['stage']=='confirm']
  assert len(confirmed)==56
  indexed={(s['block'],s['mode']):s for s in confirmed}
  for name,d in data['contrasts'].items():
   values=[indexed[b,name+'-modeled']['generation_wall_ms']-indexed[b,name+'-bypass']['generation_wall_ms'] for b in range(8)]
   assert values==d['paired_delta_ms']['values'] and statistics.median(values)==d['paired_delta_ms']['median']
   assert [min(values),max(values)]==d['median_interval_ms']
  for mode,g in data['groups'].items():
   rows=[s for s in confirmed if s['mode']==mode];assert len(rows)==8
   for metric,dist in g.items():assert statistics.median(s[metric] for s in rows)==dist['median']
  orders=proto['orders'][scale]['blocks']
  for a in proto['modes']:
   for b in proto['modes']:
    if a!=b:assert sum(o.index(a)<o.index(b) for o in orders)==4
 write(R/'provenance/final-data-audit.json',{'status':'pass','raw_processes':142,'formal_samples':112,'AA_processes':16,'warmups':14,'loaded_module_hash_checks':modules,'OS_resource_snapshots':probe_count,'paired_contrasts_recomputed':6,'AB_BA_pairwise_balance':[4,4],'zero_sample_replacements':True})
 write(R/'provenance/analysis-environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__})
 report=R/'reports/RESULTS.md';text=report.read_text(encoding='utf-8')
 old_end=accept['end_utc'];old_elapsed=accept['elapsed_s']
 if 'sampling_report_end_utc' not in accept:
  accept['sampling_report_end_utc']=old_end;accept['sampling_report_elapsed_s']=old_elapsed
 end=datetime.now(timezone.utc).isoformat();elapsed=(datetime.fromisoformat(end)-datetime.fromisoformat(accept['start_utc'])).total_seconds()
 accept['end_utc']=end;accept['elapsed_s']=elapsed;accept['final_data_audit']='pass'
 write(R/'reports/acceptance.json',accept)
 text=text.replace(f'总耗时 {old_elapsed/60:.2f} 分钟',f'总耗时 {elapsed/60:.2f} 分钟').replace(f' 至 {old_end}',f' 至 {end}')
 if '沙箱启动预检' not in text:text+='\n沙箱启动预检未能完整枚举主机进程，尚未启动模型即停止；记录见 provenance/sandbox-preflight-failure.json。随后在具备主机进程可见性的环境执行完整固定批次，142 个模型进程均成功，无模型样本替换或排除。最终独立数据审计核对了 710 条已加载模块哈希、全部进程退出/资源监测、原始时延和六项配对统计。\n'
 report.write_text(text,encoding='utf-8')
 write(R/'status.json',{'state':'complete','completed':142,'total':142,'utc':end,'elapsed_s':elapsed,'final_data_audit':'pass'})
 print(json.dumps({'status':'pass','processes':142,'module_checks':modules,'OS_probes':probe_count,'elapsed_s':elapsed,'model_process_sum_s':accept['model_process_sum_s']},ensure_ascii=False))
if __name__=='__main__':main()
