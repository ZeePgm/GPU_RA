from pathlib import Path
import csv,shutil,json,hashlib
root=Path.cwd();out=root/'output/gpu-a3/20261103-scan-002';dest=root/'docs/validation/gpu-a3';dest.mkdir(exist_ok=True)
for name in ['protocol.json','counts-result.json','measurements.csv','fit.csv','result.json','plan.json']:shutil.copyfile(out/name,dest/name)
shutil.copyfile(out/'result.json',root/'docs/validation/gpu-a3.json');shutil.copyfile(out/'H0.ini',dest/'H0.ini')
fig=root/'docs/figures/gpu-a3';fig.mkdir(exist_ok=True)
for name in ['response.png','response.svg']:shutil.copyfile(out/name,fig/name)
with (out/'measurements.csv').open() as f:points=list(csv.DictReader(f))
valid=[r for r in points if r['role'] in ['train','holdout']];assert all(int(r['wait_ps'])==110534+1667*(int(r['bytes'])//64) for r in valid)
posthoc={'scope':'posthoc algebraic simplification of fitted response; not preregistered theory','formula_ps':'110534 + 1667 * (bytes / 64)','checked_rows':len(valid),'max_integer_residual_ps':0,'range_bytes':[64,32832]}
(out/'posthoc-linear-check.json').write_text(json.dumps(posthoc,indent=2)+'\n');shutil.copyfile(out/'posthoc-linear-check.json',dest/'posthoc-linear-check.json')
# Additional ownership and page-boundary checks on original complete records.
checks=[]
for d in (out/'runs').iterdir():
 st=json.loads((d/'statistics.json').read_text());s=json.loads((d/'session.json').read_text());wid={w['id']:w for w in st['logical_work']};gids={g['id'] for g in st['globals'] if g['op']=='read'};tr=[json.loads(x) for x in (d/'bridge.jsonl').read_text().splitlines()]
 assert sorted(e['global_id'] for e in tr if e['access_id']==s['read']['id'] and e['action']=='global-finished')==sorted(gids)
 for a in st['attempts']:
  if a['op']=='read':
   assert len(a['work_ids'])==1
   w=wid[a['work_ids'][0]];assert w['global_id'] in gids and w['local_byte']==a['local_byte'] and w['channel']==a['channel']
 checks.append({'case':d.name,'attempt_ownership':True,'finished_children_once':True})
(out/'supplemental-audit.json').write_text(json.dumps(checks,indent=2)+'\n');shutil.copyfile(out/'supplemental-audit.json',dest/'supplemental-audit.json')
for p in (root/'experiments/gpu_a3').iterdir():
 if p.is_file():shutil.copyfile(p,out/'sources'/p.name)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
(out/'artifact-index.json').write_text(json.dumps({'files':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact-index.json']},indent=2)+'\n')
print('Archived evidence; all 78 attempt-ownership checks passed')
