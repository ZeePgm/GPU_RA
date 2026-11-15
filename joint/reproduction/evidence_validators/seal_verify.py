"""Seal completed records or verify the immutable evidence manifest (no model runs)."""
import argparse,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['seal','verify']);a=ap.parse_args()
 target=R/'provenance/evidence-manifest.json'
 if a.action=='seal':
  assert not target.exists(),'refuse overwrite'
  status=read(R/'status.json');accept=read(R/'reports/acceptance.json');proto=read(R/'protocol.json');samples=read(R/'samples.json')
  assert status['state']=='complete' and accept['status']=='complete'
  assert status['completed']==accept['accepted_model_processes']==len(samples)==142
  assert [{k:r[k] for k in proto['schedule'][0]} for r in samples]==proto['schedule']
  assert sha(R/'scripts/run.py')==proto['runner_sha256']
  assert sha(R/'protocol.json')==accept['protocol_sha256']
  assert not (R/'failure.json').exists()
  assert all(s['status']=='pass' for s in samples)
  assert sum(s['stage']=='confirm' for s in samples)==112
  assert sum(s['stage']=='AA' for s in samples)==16
  assert sum(s['stage']=='warmup' for s in samples)==14
  assert all(s['scale']=='F1' for s in samples[:71]) and all(s['scale']=='F2' for s in samples[71:])
  f1last=read(R/'raw'/(samples[70]['id']+'.process.json'))['end_utc']
  f2first=read(R/'raw'/(samples[71]['id']+'.process.json'))['start_utc']
  assert f1last<f2first
  rows=[]
  for p in sorted(R.rglob('*')):
   if p.is_file() and p!=target and '__pycache__' not in p.parts:
    assert not p.name.endswith('.tmp'),p
    rows.append({'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
  target.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 else:
  rows=read(target)
  for row in rows:
   p=R/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
  actual={p.relative_to(R).as_posix() for p in R.rglob('*') if p.is_file() and p!=target and '__pycache__' not in p.parts}
  assert actual=={r['path'] for r in rows},'unmanifested or missing files'
 print('PASS',a.action,len(rows),'evidence files')
if __name__=='__main__':main()
