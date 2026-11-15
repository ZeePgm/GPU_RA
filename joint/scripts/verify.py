"""Read-only verification of the sealed source, build and evidence manifests."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
count=0
for name in ['final-source-build-manifest.json','evidence-manifest.json']:
    manifest=json.loads((R/'provenance'/name).read_text(encoding='utf-8'))
    for row in manifest:
        p=R/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p);count+=1
print('PASS',count,'source/build/evidence records; no files changed')
