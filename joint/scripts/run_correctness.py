"""Serial correctness matrix; each scale must pass before the next starts."""
import argparse,ctypes,hashlib,json,os,subprocess,time
from ctypes import wintypes as W
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];OUT=R/'results';BIN=R/'build/model/bin'
CUDA=Path('D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def modules(pid):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    class M(ctypes.Structure):
        _fields_=[('dwSize',W.DWORD),('th32ModuleID',W.DWORD),('th32ProcessID',W.DWORD),('GlblcntUsage',W.DWORD),('ProccntUsage',W.DWORD),('modBaseAddr',ctypes.c_void_p),('modBaseSize',W.DWORD),('hModule',W.HMODULE),('szModule',W.WCHAR*256),('szExePath',W.WCHAR*260)]
    k.CreateToolhelp32Snapshot.argtypes=[W.DWORD,W.DWORD];k.CreateToolhelp32Snapshot.restype=W.HANDLE
    k.Module32FirstW.argtypes=[W.HANDLE,ctypes.POINTER(M)];k.Module32NextW.argtypes=[W.HANDLE,ctypes.POINTER(M)];k.CloseHandle.argtypes=[W.HANDLE]
    h=k.CreateToolhelp32Snapshot(0x18,pid)
    if h==W.HANDLE(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:
        m=M();m.dwSize=ctypes.sizeof(m);ok=k.Module32FirstW(h,ctypes.byref(m));result=[]
        while ok:result.append(m.szExePath);ok=k.Module32NextW(h,ctypes.byref(m))
        return result
    finally:k.CloseHandle(h)
def environment():
    env={k:v for k,v in os.environ.items() if not k.startswith(('HBFLAB_','GGML_'))}
    env['PATH']=os.pathsep.join(map(str,[BIN,CUDA/'bin',CUDA/'bin/x64',CUDA/'lib']))+os.pathsep+env.get('PATH','')
    env.update(CUDA_VISIBLE_DEVICES='0',GGML_CUDA_DISABLE_GRAPHS='1',GGML_CUDA_DISABLE_FUSION='1',GGML_CUDA_CUBLAS_COMPUTE_TYPE='f32',HBFLAB_KV_PAGING='async',HBFLAB_KV_STAGING_PAGES='4',HBFLAB_KV_CANCELLABLE='1',HBFLAB_KV_ASYNC_REPLICA='1',HBFLAB_KV_WRITE_MODE='device')
    return env
def run_case(scale,label,weights,kv,mode,write_source='device',batch='correctness-001'):
    base=OUT/batch;base.mkdir(exist_ok=True)
    name=f'{scale}-{label}';folder=base/name;record=base/(name+'.process.json')
    assert not folder.exists() and not record.exists(),name
    p=read(R/'configs/qwen3-0.6b.json');env=environment()
    env.update(HBFLAB_JOINT_SCALE=scale,HBFLAB_JOINT_WEIGHTS=weights,HBFLAB_JOINT_KV=kv,HBFLAB_KV_WRITE_MODE=write_source)
    command=[str(BIN/'realgpu_llama.exe'),p['model_path'],str(R/'configs/prompt.txt'),str(folder),'8','0',mode,str(R/'configs/qwen3-0.6b.json')]
    started=stamp();print('START',name,started,flush=True);start=time.monotonic();observed=[];failure=None
    with (base/(name+'.log')).open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=log)
        try:
            while proc.poll() is None:
                elapsed=time.monotonic()-start
                if not observed and elapsed>=.5:
                    paths=modules(proc.pid)
                    if any(Path(x).name.lower()=='ggml-cuda.dll' for x in paths):observed=paths
                if elapsed>3600:raise TimeoutError(name)
                time.sleep(.1)
        except Exception as e:
            failure=repr(e)
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
    data={'start_utc':started,'end_utc':stamp(),'wall_s':time.monotonic()-start,'returncode':proc.returncode,'error':failure,'command':command,'environment':{k:v for k,v in env.items() if k.startswith(('HBFLAB_','GGML_','CUDA_'))},'modules':observed}
    write(record,data)
    assert not failure and proc.returncode==0,(name,failure,proc.returncode)
    dlls={Path(x).name.lower():Path(x) for x in observed}
    for f in BIN.glob('*.dll'):
        if f.name.lower().startswith(('ggml','llama')):
            assert f.name.lower() in dlls,(name,'DLL missing',f.name)
            assert dlls[f.name.lower()].resolve()==f.resolve(),('wrong DLL',f)
    write(base/(name+'.modules.json'),[{'path':str(x),'sha256':sha(x)} for x in dlls.values() if x.name.lower().startswith(('ggml','llama'))])
    validate_case(folder,scale,weights,kv,mode)
    print('PASS',name,round(data['wall_s'],3),'seconds',flush=True)
def validate_case(folder,scale,weights,kv,mode):
    s=read(folder/'summary.json');j=read(folder/'joint-summary.json');p=read(R/'configs/qwen3-0.6b.json')
    assert s['status']=='success' and len(s['prompt_tokens'])==44 and len(s['generated_tokens'])==8
    assert s['logits_shape']==[8,151936] and (folder/'logits.f32').stat().st_size==4861952
    x=np.fromfile(folder/'logits.f32',dtype='<f4');assert np.isfinite(x).all()
    assert j['status']=='pass' and j['borrowed_unchanged'] and j['steps']==8
    mult=1 if scale=='F1' else 4
    expected_requests=294912*mult if weights=='b2' and mode=='modeled' else 0
    assert j['requests']==expected_requests and j['dependency_checks']==32*expected_requests
    assert j['covered_bytes']==(144*1024**2*mult if weights=='b2' else 0)
    assert j['publications']==(24*mult if weights=='b2' else 0)
    calls=[json.loads(line) for line in (folder/'weight-calls.jsonl').read_text().splitlines()]
    expected_layers=[10] if scale=='F1' else [0,10,20,27]
    from collections import Counter
    expected=Counter((step,f'blk.{l}.ffn_{role}.weight') for step in range(8) for l in expected_layers for role in ['gate','up','down']) if weights=='b2' else Counter()
    assert Counter((c['step'],c['tensor']) for c in calls)==expected
    if mode=='modeled':
        rt=j['runtime'];assert rt['drained'] and rt['submitted']==rt['completed']==expected_requests
        assert rt['registered_globals']==rt['completed_globals'] and rt['host_failed']==rt['audit_dropped']==0 and rt['peak_resident_fragments']<=256
        assert (rt['media_completed']>0)==(kv=='paged')
        init=read(folder/'initialization.json')
        assert init['programs']==(1536*3*mult if weights=='b2' else 0)+(p['logical_pages'] if kv=='paged' else 0),init
    if kv=='paged':
        k=read(folder/'kv-summary.json');release=read(folder/'kv-release.json');content=read(folder/'kv-content.json')
        assert k['steps']==8 and k['length']==51 and k['layers']==28 and k['transactions']==8
        assert k['appended_layer_tokens']==51*28 and k['frame_bytes']==16384
        assert k['cancellable_append'] and k['paging_mode']==1 and k['pinned_staging_bytes']==16384
        assert k['async_prepared']==k['async_retired'] and k['pins']==k['dirty_frames']==0
        assert all(v==0 for v in release.values()) and content['sequences']['0']['length']==51
        layout=read(folder/'address-layout.json');lo=layout['kv_base'];hi=lo+layout['kv_bytes']
        count=0
        with (folder/'kv-media.jsonl').open() as f:
            for line in f:
                a=json.loads(line);assert lo<=a['logical']<hi and a['end_ps']>=a['arrival_ps'];count+=1
        if mode=='modeled':assert count==j['runtime']['media_completed']
def compare_scale(scale,batch):
    base=OUT/batch;native=base/f'{scale}-native';n=read(native/'summary.json');ref=np.fromfile(native/'logits.f32',dtype='<f4')
    comparisons=[]
    for folder in sorted(base.glob(f'{scale}-*')):
        if not folder.is_dir():continue
        s=read(folder/'summary.json');j=read(folder/'joint-summary.json');x=np.fromfile(folder/'logits.f32',dtype='<f4');diff=x.astype('f8')-ref.astype('f8')
        assert s['prompt_tokens']==n['prompt_tokens'] and s['generated_tokens']==n['generated_tokens']
        max_abs=float(np.max(np.abs(diff)));rms=float(np.sqrt(np.mean(diff**2)))
        if j['kv_mode']=='native':assert sha(folder/'logits.f32')==sha(native/'logits.f32')
        else:
            tolerance=read(R/'configs/qwen3-0.6b.json')['numerical']
            assert max_abs<=tolerance['max_abs'] and rms<=tolerance['rms'],(folder,max_abs,rms)
        comparisons.append({'case':folder.name,'max_abs_vs_native':max_abs,'rms_vs_native':rms,'logits_sha256':sha(folder/'logits.f32'),'tokens_equal':True})
    for a,b in [('readonly-bypass','readonly-modeled'),('kv-bypass','kv-modeled'),('joint-bypass','joint-modeled'),('kv-modeled','joint-modeled'),('joint-modeled','joint-host')]:
        left=base/f'{scale}-{a}';right=base/f'{scale}-{b}'
        assert sha(left/'logits.f32')==sha(right/'logits.f32'),(a,b,'logits')
        if (left/'kv-content.json').exists():
            assert read(left/'kv-content.json')['sequences']==read(right/'kv-content.json')['sequences'],(a,b,'KV bytes')
    result={'status':'pass','scale':scale,'comparisons':comparisons,'same_arithmetic_logits_bitwise':True,'kv_content_sha256_equal':True,'statistical_performance_confirmation':False}
    write(base/f'{scale}-acceptance.json',result);print('ACCEPTED',scale,flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scale',choices=['F1','F2'],required=True);ap.add_argument('--batch',default='correctness-001');a=ap.parse_args()
    if a.scale=='F2':assert read(OUT/a.batch/'F1-acceptance.json')['status']=='pass'
    for args in [('native','native','native','native'),('readonly-bypass','b2','native','bypass'),('readonly-modeled','b2','native','modeled'),('kv-bypass','native','paged','bypass'),('kv-modeled','native','paged','modeled'),('joint-bypass','b2','paged','bypass'),('joint-modeled','b2','paged','modeled'),('joint-host','b2','paged','modeled')]:
        run_case(a.scale,*args,write_source='host' if args[0]=='joint-host' else 'device',batch=a.batch)
    compare_scale(a.scale,a.batch)
if __name__=='__main__':main()
