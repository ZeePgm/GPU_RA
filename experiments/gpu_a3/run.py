"""A3 independent interval oracle, original H0 bridge execution, count-first audit."""
import argparse,copy,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from integrated_build_identity import inputs,verify_records

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def predict(region,request):
 # Independent scalar geometry; never imports bridge, splitter or backend statistics.
 lo=max(region['source_offset'],request['source_offset']);hi=min(region['source_offset']+region['valid_bytes'],request['source_offset']+request['source_bytes'])
 assert hi>lo and lo%64==hi%64==0
 addr=region['global_base']+lo-region['source_offset'];n=hi-lo
 return {'offset':lo-region['source_offset'],'bytes':n,'fragments':list(range(addr,addr+n,64)),
         'chunks':[{'address':a,'bytes':min(4096,addr+n-a),'fragments':list(range(a,min(a+4096,addr+n),64))} for a in range(addr,addr+n,4096)]}
def audit(region,expected,s,stats,trace):
 assert s['status']=='Completed' and all(s[k] for k in ['workload_success','input_closed','host_drained','device_drained'])
 assert s['mapped_offset']==expected['offset'] and s['mapped_bytes']==expected['bytes']
 parent=s['read'];assert parent['success'] and parent['complete'] and parent['unsubmitted_bytes']==0
 regs=[e['global_id'] for e in trace if e['action']=='global-registered' and e['access_id']==parent['id']]
 assert len(regs)==len(set(regs))==len(expected['chunks'])==parent['registered_globals']==parent['finished_globals']
 gs={g['id']:g for g in stats['globals']};readids={g['id'] for g in stats['globals'] if g['op']=='read'};assert set(regs)==readids
 assert all(g['result']=='success' and g['remaining']==0 and g['finished'] is not None for g in gs.values())
 assert parent['offered_ps']==s['request_t0_ps'] and parent['completed_ps']==max(gs[g]['finished'] for g in regs)
 work=stats['logical_work'];assert len({w['id'] for w in work})==len(work)
 for gid,e in zip(regs,expected['chunks']):
  g=gs[gid];assert g['bytes']==e['bytes']
  ws=[w for w in work if w['global_id']==gid]
  assert sorted(w['local_byte'] for w in ws)==e['fragments']
  assert all(w['channel']==0 and w['result']=='success' and w['content'] is None for w in ws)
  assert max(w['finished'] for w in ws)==g['finished']
 reads=[w for w in work if w['global_id'] in readids];assert sorted(w['local_byte'] for w in reads)==expected['fragments']
 writes=[w for w in work if gs[w['global_id']]['op']=='write'];assert sorted(w['local_byte'] for w in writes)==list(range(region['global_base'],region['global_base']+region['registered_bytes'],64))
 assert all(w['content']==1 and w['channel']==0 and w['result']=='success' and w['finished']<=s['request_t0_ps'] for w in writes)
 assert s['initialization']['success'] and s['initialization']['completed_ps']==s['request_t0_ps']
 ats=[a for a in stats['attempts'] if a['op']=='read'];assert len(ats)==len(expected['fragments'])
 assert sorted(a['local_byte'] for a in ats)==expected['fragments']
 assert all(a['bytes']==64 and a['channel']==0 and a['host_success'] and a['status']==0 and not a['replay'] for a in ats)
 assert sorted(k for a in ats for k in a['work_ids'])==sorted(w['id'] for w in reads)
 assert stats['summary']['logical_work']['read']['total']==len(expected['fragments']) and stats['summary']['globals']['read']['total']==len(expected['chunks'])
 assert stats['summary']['host_verification_failures']==0
 return {'bytes':expected['bytes'],'read_globals':len(expected['chunks']),'fragments':len(expected['fragments']),'wait_ps':parent['completed_ps']-parent['offered_ps'],'init_ps':s['request_t0_ps'],'capacity_rejections':parent['capacity_rejections'],'read_sense_paths':sum(r['service']=='sense' for r in stats['read_paths']),'read_cache_paths':sum(r['service']=='cache' for r in stats['read_paths']),'read_bank_operations':sum(b['issued_at']>=s['request_t0_ps'] for b in stats['bank_operations'])}
def run():
 ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);args=ap.parse_args();out=ROOT/args.study;assert not out.exists();out.mkdir(parents=True)
 identity=read(ROOT/'build-portability-model/release-identity.json');verify_records(identity['artifacts'])
 before={r['path']:r for r in identity['sources']};now={r['path']:r for r in inputs()}
 assert all(now.get(k)==v for k,v in before.items()),'sealed source changed or missing'
 added=set(now)-set(before);allowed={str(ROOT/'tools/restore_minexp_records.py'),str(ROOT/'tools/package_minexp_records.py')}
 assert added<=allowed,'unexpected new source'
 write(out/'dependency-build-identity.json',{'sealed_identity':identity,'inventory_exception':sorted(added),'rationale':'only unrelated archive tools added; every sealed source and artifact unchanged','linked_library_hashes':{n:sha(ROOT/'build-portability-model/eventlab'/n) for n in ['hbf_eventlab_tensor_bridge.lib','hbf_eventlab.lib']}})
 h0=ROOT/'output/gpu-s/20261101-s8-002/G3/effective-H0.ini';shutil.copyfile(h0,out/'H0.ini')
 train=[64,128,512,1024,1984,2048,2112,4032,4096,4160,8128,8192,8256,16320,16384,16448,32704,32768,32832];hold=[1536,3072,6144,12288,24576]
 region={'source_offset':128,'valid_bytes':65536,'registered_bytes':65536,'global_base':0,'content_id':1}
 cases=[{'name':f'n{n}','source_offset':128,'source_bytes':n,'role':'train' if n in train else 'holdout'} for n in train+hold]
 cases += [{'name':'offset64','source_offset':192,'source_bytes':8192,'role':'offset_only'},{'name':'left_clip','source_offset':64,'source_bytes':4160,'role':'intersection_only'}]
 plan={'region':region,'cases':cases};write(out/'plan.json',plan);expected={c['name']:predict(region,c) for c in cases};write(out/'expected.json',expected)
 write(out/'protocol.json',{'scope':'fixed uncalibrated S8 H0, real Bridge/Session/Host/Device; no GPU','h0_sha256':sha(h0),'repeats':3,'initial_state':'fresh Session; ordinary write initializes same 64KiB completely; no cache reset or warm/cold hardware claim','count_oracle':'raw source intersection, request-relative 4096B chunks, independent 64B addresses; single channel global_base=0','fit':{'knots_bytes':[2048,4096,8192,16384],'model':'independent affine least-squares in five size bins, discontinuities allowed; bins right-closed','train':train,'holdout':hold,'acceptance':'all five held-out relative errors <=5%; engineering predictive tolerance, not hardware accuracy','basis':'32-fragment threshold; 4096B chunks/pages, two/four chunk boundaries and 256-fragment residency; candidate, not guaranteed true breakpoints'},'invalid':'any independent count/address/content/drain failure excludes point from fitting and fails audit; no silent deletion','binary_sha256':sha(ROOT/'build-a3/a3_scan.exe')})
 snap=out/'sources';snap.mkdir()
 for p in [Path(__file__),Path(__file__).with_name('scan.cpp'),ROOT/'build-a3/a3_scan.exe',ROOT/'build-a3/build.log']:shutil.copyfile(p,snap/p.name)
 with (out/'run.log').open('w') as log:r=subprocess.run([str(ROOT/'build-a3/a3_scan.exe'),str(out/'H0.ini'),str(out/'plan.json'),str(out/'runs')],stdout=log,stderr=log,timeout=180)
 assert r.returncode==0,'execution failed, inspect run.log'
 results=[];t0=set()
 for c in cases:
  reference_hash=None
  for rep in range(3):
   d=out/'runs'/f'{c["name"]}-{rep}';s,st,tr=read(d/'session.json'),read(d/'statistics.json'),rows(d/'bridge.jsonl')
   result=audit(region,expected[c['name']],s,st,tr);result.update(case=c['name'],repeat=rep,role=c['role']);results.append(result);t0.add(result['init_ps'])
   hashes=[sha(d/f) for f in ['statistics.json','bridge.jsonl','effective.ini']]
   if reference_hash is None:reference_hash=hashes
   assert hashes==reference_hash,'non-deterministic repeat'
 assert len(t0)==1,'initialization changed with requested size'
 d=out/'runs/n4160-0';base=read(d/'statistics.json');session=read(d/'session.json');tr=rows(d/'bridge.jsonl');readids={g['id'] for g in base['globals'] if g['op']=='read'};neg=[]
 for kind in ['missing_fragment','duplicate_fragment','wrong_address','wrong_global_owner','wrong_global_size','bad_content_result']:
  st=copy.deepcopy(base);idx=next(i for i,w in enumerate(st['logical_work']) if w['global_id'] in readids)
  if kind=='missing_fragment':st['logical_work'].pop(idx)
  if kind=='duplicate_fragment':st['logical_work'].append(copy.deepcopy(st['logical_work'][idx]))
  if kind=='wrong_address':st['logical_work'][idx]['local_byte']+=64
  if kind=='wrong_global_owner':st['logical_work'][idx]['global_id']=max(readids)
  if kind=='wrong_global_size':next(g for g in st['globals'] if g['op']=='read')['bytes']+=64
  if kind=='bad_content_result':next(a for a in st['attempts'] if a['op']=='read')['host_success']=False
  rejected=False
  try:audit(region,expected['n4160'],session,st,tr)
  except AssertionError:rejected=True
  assert rejected;write(out/'negative'/f'{kind}.json',st);neg.append({'kind':kind,'rejected':True})
 write(out/'counts-result.json',{'status':'pass','points':len(cases),'runs':len(results),'independent_counts_addresses_content':True,'all_drained':True,'deterministic_repeats':True,'same_init_tick_ps':list(t0)[0],'negatives':neg})
 with (out/'measurements.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 print('Independent count audits PASS:',len(cases),'points',len(results),'sessions')
if __name__=='__main__':run()
