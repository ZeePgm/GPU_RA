"""Copy verified experimental inputs into an independent source tree."""
from pathlib import Path
import hashlib, json, shutil, time
ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    source=DEST/'source'
    assert not source.exists()
    for name in ['configs','provenance','results','reports']: (DEST/name).mkdir(exist_ok=True)
    identity=json.loads((ROOT/'build-k5-r3-model/release-identity.json').read_text())
    records=[]
    for r in identity['sources']:
        p=Path(r['path']); assert sha(p)==r['sha256'],str(p)
    base=ROOT/'output/k5r3f'
    for p in base.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts: continue
        rel=p.relative_to(base)
        if rel.parts[0]=='configs': continue
        q=source/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        records.append({'path':str(rel),'source':str(p),'sha256':sha(q)})
    b=ROOT/'output/readonly-b2-20261110'
    ids=json.loads((b/'final-source-identity.json').read_text())
    for rel,h in ids.items(): assert sha(b/'source-final'/rel)==h,rel
    for name in ['runner.cpp','window.cu','fixture.cpp','PROTOCOL.md']:
        p=b/'source-final/experiments/readonly_b2'/name
        q=source/'experiments/readonly_b2'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        records.append({'path':str(q.relative_to(source)),'source':str(p),'sha256':sha(q)})
    # Preserve the original CRLF prompt exactly.
    manifest=json.loads((ROOT/'output/readonly-b-20261109/p-preflight-003/frozen-manifest.json').read_text())
    prompts=[x for x in manifest if x['original'].endswith('prompt.txt')]
    assert prompts
    p=ROOT/prompts[0]['original'];assert sha(p)==prompts[0]['sha256']
    shutil.copyfile(p,DEST/'configs/prompt.txt')
    assert (DEST/'configs/prompt.txt').stat().st_size==249
    profile=json.loads((base/'configs/integrated/qwen3-0.6b.json').read_text())
    model=Path(profile['model_path']);assert sha(model)==profile['gguf_sha256']
    profile['blocks_per_bank']=32768 # 1 GiB: complete 504 MiB FFN layout plus KV.
    write(DEST/'configs/qwen3-0.6b.json',profile)
    write(DEST/'provenance/source-origin.json',records)
    write(DEST/'provenance/task.json',{'start_utc':'2026-11-12T07:20:16Z','scope':'Qwen3-0.6B F16, 44 inputs, 8 outputs; F1 then F2; joint correctness only','k5_source_checks':len(identity['sources']),'b2_source_checks':len(ids),'model':str(model),'model_sha256':sha(model),'prompt_sha256':sha(DEST/'configs/prompt.txt')})
    print('COPIED',len(records),'files',flush=True)
if __name__=='__main__': main()
