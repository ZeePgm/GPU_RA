"""Re-audit all V3 raw correctness/statistical records without CUDA or model execution."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import statistics
import types
import numpy as np

HERE=Path(__file__).resolve().parent
MODES={'native':('native','native','native'), 'readonly-bypass':('b2','native','bypass'),
       'readonly-modeled':('b2','native','modeled'), 'kv-bypass':('native','paged','bypass'),
       'kv-modeled':('native','paged','modeled'), 'joint-bypass':('b2','paged','bypass'),
       'joint-modeled':('b2','paged','modeled')}
MATRIX=6291456
MODEL_SHA='74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2'
PROMPT_SHA='ce4fa40becfa4a191064fbda15721985a953ae231557ea6f6f8f7b669e0d1a74'


def extended(p):
    p=Path(p).resolve()
    if os.name=='nt' and not str(p).startswith('\\\\?\\'):
        p=Path('\\\\?\\'+str(p))
    return p


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def historical(name, functions, namespace):
    path=HERE/'evidence_validators'/name
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in functions]
    assert {n.name for n in nodes}==set(functions)
    target=types.ModuleType(name.replace('.py',''))
    target.__dict__.update(namespace)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),target.__dict__)
    return target


def verify_manifest(folder):
    rows=read(folder/'provenance/evidence-manifest.json')
    for row in rows:
        p=folder/row['path'].replace('\\','/')
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
    if folder.name=='20261113-dual-line-stat-r1':
        actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and '__pycache__'not in p.parts and p!=folder/'provenance/evidence-manifest.json'}
        assert actual=={r['path'].replace('\\','/')for r in rows},'Unmanifested or missing statistical evidence'
    return len(rows)


def archived_path(root, original):
    path=PureWindowsPath(original)
    parts=path.parts
    # Protocol records intentionally retain their original workstation identity.
    # Resolve only the two explicitly archived subtrees, never arbitrary files.
    for prefix in ('integrated','experiments'):
        if prefix in parts:
            return root.joinpath(*parts[parts.index(prefix):])
    raise ValueError('Dependency is outside archived scope: '+original)


def environment(record,scale,weights,kv,write_source):
    env=record['environment']
    expected={'GGML_CUDA_DISABLE_GRAPHS':'1','GGML_CUDA_DISABLE_FUSION':'1',
              'GGML_CUDA_CUBLAS_COMPUTE_TYPE':'f32','HBFLAB_KV_PAGING':'async',
              'HBFLAB_KV_STAGING_PAGES':'4','HBFLAB_KV_CANCELLABLE':'1',
              'HBFLAB_KV_ASYNC_REPLICA':'1','HBFLAB_KV_WRITE_MODE':write_source,
              'HBFLAB_JOINT_SCALE':scale,'HBFLAB_JOINT_WEIGHTS':weights,'HBFLAB_JOINT_KV':kv}
    assert all(env.get(k)==v for k,v in expected.items()),record['command']
    assert record['returncode']==0 and record['error'] is None
    assert record['end_utc']>record['start_utc'] and record['wall_s']>0


def fixed_contract(first,second):
    cfg=read(first/'configs/qwen3-0.6b.json');task=read(first/'provenance/task.json');proto=read(second/'protocol.json')
    assert cfg['gguf_sha256']==task['model_sha256']==proto['model_sha256']==MODEL_SHA
    assert sha(first/'configs/prompt.txt')==task['prompt_sha256']==PROMPT_SHA
    assert cfg['framework_commit']=='b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9'
    assert cfg['numerical']['max_abs']==.2 and cfg['numerical']['rms']==.025 and cfg['numerical']['tokens_exact']
    assert {k:tuple(v)for k,v in proto['modes'].items()}==MODES
    assert proto['seed']==131113 and proto['processes']==142
    assert proto['warmups_per_group_per_scale']==1 and proto['blocks_per_scale']==8 and proto['AA_pairs_per_scale']==4
    assert proto['write_source']=='device' and proto['timeout_s']==3600
    for name in ['host-processes-before.json','host-processes-after.json']:
        rows=read(second/'provenance'/name)
        assert any(row['pid']==4 and row['name']=='System' for row in rows),name
    return {'model_sha256':MODEL_SHA,'prompt_sha256':PROMPT_SHA,'max_abs_tolerance':.2,'rms_tolerance':.025,
            'seed':131113,'processes':142,'per_scale_warmups_per_mode':1,'blocks_per_scale':8,'AA_pairs_per_scale':4,
            'host_System_PID4_visibility_before_after':True}


def raw_streams(folder,scale,weights,kv,mode):
    mult=1 if scale=='F1' else 4
    layout=read(folder/'address-layout.json')
    selected={r['tensor']:r for r in layout['selected']}
    assert layout['weight_layout_bytes']==28*3*MATRIX and layout['kv_base']==28*3*MATRIX
    assert layout['gap_initialization']=='none'
    layers=[10] if scale=='F1' else [0,10,20,27]
    for layer in layers:
        for role_id,role in enumerate(('gate','up','down')):
            row=selected[f'blk.{layer}.ffn_{role}.weight']
            assert row['base']==(layer*3+role_id)*MATRIX and row['bytes']==MATRIX
    assert len(selected)==3*mult
    calls=[json.loads(line)for line in (folder/'weight-calls.jsonl').read_text().splitlines()]
    for row in calls:
        assert row['logical_base']==selected[row['tensor']]['base']
        assert row['bytes']==row['publication_bytes']==MATRIX and row['mirror_verified']
        assert row['requests']==(MATRIX//512 if mode=='modeled' else 0) and row['kernels']==24
    bitmap={(step,t):bytearray(MATRIX//512)for step in range(8)for t in selected} if weights=='b2' and mode=='modeled' else {}
    previous_tickets=[0]*2048; launch_tickets={}; launches={}; current=None; count=0
    with (folder/'weight-requests.jsonl').open(encoding='utf-8') as f:
        for line in f:
            q=json.loads(line); key=(q['step'],q['tensor']); offset=q['offset']; producer=q['producer']
            assert key in bitmap and 0<=offset<MATRIX and offset%512==0
            index=offset//512
            assert bitmap[key][index]==0;bitmap[key][index]=1
            assert q['logical']==selected[q['tensor']]['base']+offset
            assert 0<=producer<2048 and producer%32==0 and offset%32768==producer*16
            launch=q['launch']; launch_key=(key,offset//262144)
            if current!=launch:
                for p,ticket in launch_tickets.items():previous_tickets[p]=ticket
                launch_tickets={};current=launch
                assert launch not in launches;launches[launch]=launch_key
            assert launches[launch]==launch_key
            batch=(offset%262144)//32768
            assert q['ticket']==previous_tickets[producer]+batch+1
            launch_tickets[producer]=max(launch_tickets.get(producer,0),q['ticket'])
            assert q['end_ps']>=q['arrival_ps']
            assert q['consumed_ns']>=q['arrival_ns']+(q['end_ps']-q['arrival_ps']+999)//1000
            count+=1
    assert count==(294912*mult if bitmap else 0)
    assert all(all(bits) for bits in bitmap.values())
    assert len(launches)==(576*mult if bitmap else 0)
    media=0;dispatch=set()
    if kv=='paged':
        lo,hi=layout['kv_base'],layout['kv_base']+layout['kv_bytes']
        with (folder/'kv-media.jsonl').open(encoding='utf-8')as f:
            for line in f:
                q=json.loads(line)
                assert lo<=q['logical'] and q['logical']+q['bytes']<=hi and q['logical']%4096==0
                assert q['dispatch'] not in dispatch;dispatch.add(q['dispatch'])
                # Step 8 is the final full-KV byte audit after the eight model calls.
                assert q['bytes']==4096 and 0<=q['step']<=8 and 0<=q['layer']<28
                if q['step']==8:assert not q['write']
                assert q['end_ps']>=q['arrival_ps'] and q['arrival_ns']>=q['floor_ns']
                assert q['delay_ns']==(q['end_ps']-q['arrival_ps']+999)//1000
                assert q['resumed_wall_ns']>=q['completion_wall_ns']>=q['ready_wall_ns']
                media+=1
        joint=read(folder/'joint-summary.json')
        assert media==(joint['runtime']['media_completed'] if mode=='modeled' else 0)
        k=read(folder/'kv-summary.json');content=read(folder/'kv-content.json')
        assert k['prefetch_issued']==k['prefetch_used']==k['prefetch_cancelled']==0
        assert content['pins']==content['active_leases']==0 and content['frame_bytes']==16384
    return {'weight_request_rows':count,'weight_launches':len(launches),'media_rows':media}


def correctness(first,C,dependency_hashes,out):
    count=0;modules=0;streams=Counter();processes=[]
    for scale in ('F1','F2'):
        for label,(weights,kv,mode)in {**MODES,'joint-host':MODES['joint-modeled']}.items():
            base=first/'results/correctness-001';folder=base/f'{scale}-{label}'
            record=read(base/f'{scale}-{label}.process.json')
            processes.append(record)
            environment(record,scale,weights,kv,'host'if label=='joint-host'else'device')
            C.validate_case(folder,scale,weights,kv,mode)
            for module in read(base/f'{scale}-{label}.modules.json'):
                assert dependency_hashes[module['path'].lower()]==module['sha256'];modules+=1
                assert module['path']in record['modules']
            streams.update(raw_streams(folder,scale,weights,kv,mode));count+=1
        C.compare_scale(scale,'correctness-001')
        recomputed=read(out/'correctness-recomputed'/f'{scale}-acceptance.json')
        assert recomputed==read(first/f'results/correctness-001/{scale}-acceptance.json')
    assert count==16 and modules==80
    processes.sort(key=lambda row:row['start_utc'])
    assert all(a['end_utc']<=b['start_utc']for a,b in zip(processes,processes[1:])), 'Overlapping correctness processes'
    original=read(first/'reports/acceptance.json')
    assert original['accepted_model_processes']==count and original['loaded_module_hash_checks']==modules
    assert original['model_process_sum_s']==sum(read(p)['wall_s']for p in (first/'results/correctness-001').glob('*.process.json'))
    fixture=read(first/'results/b2-fixture.json');assert len(fixture['cases'])==28
    assert 'terminal status 1'in(first/'results/b2-negative.log').read_text()
    assert '100% tests passed, 0 tests failed out of 608'in(first/'results/test-core.log').read_text()
    return {'status':'pass','model_processes':count,'loaded_module_hash_checks':modules,'raw_streams':dict(streams),'nonoverlapping_adjacent_processes':15,'GPU_fixture_cases_recorded':28,'historical_core_tests_recorded':608}


def statistical(root,first,second,C,out):
    proto=read(second/'protocol.json');samples=read(second/'samples.json');accept=read(second/'reports/acceptance.json')
    assert len(samples)==len(proto['schedule'])==accept['accepted_model_processes']==142
    assert sha(second/'protocol.json')==accept['protocol_sha256']
    assert sha(second/'scripts/run.py')==proto['runner_sha256']
    dependency_hashes={}
    for row in proto['dependencies']:
        p=archived_path(root,row['path'])
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
        dependency_hashes[row['path'].lower()]=row['sha256']
    original_write=write
    def redirected(path,value):original_write(out/'statistical-recomputed'/Path(path).name,value)
    S=historical('statistics_run.py',['distribution','analyze','numerical'],
                 {'np':np,'statistics':statistics,'MODES':MODES,'SEALED':first,'OUT':out/'statistical-recomputed','read':read,'write':redirected,'sha':sha,'event':lambda *a,**k:None})
    modules=0;probes=0;streams=Counter();recomputed=[];previous_process=None
    for row,scheduled in zip(samples,proto['schedule']):
        assert all(row[key]==value for key,value in scheduled.items()) and row['status']=='pass'
        folder=second/'raw'/row['id'];process=read(folder.with_suffix('.process.json'))
        weights,kv,mode=MODES[row['mode']]
        environment(process,row['scale'],weights,kv,'device')
        assert all(process[key]==value for key,value in scheduled.items())
        assert process['wall_s']==row['wall_s']
        if previous_process is not None:assert previous_process['end_utc']<=process['start_utc'],'Overlapping statistical processes'
        previous_process=process
        assert process['resource_probes'] and all(not p['conflicts'] and p['visible_processes']>4 for p in process['resource_probes'])
        probes+=len(process['resource_probes'])
        C.validate_case(folder,row['scale'],weights,kv,mode)
        numerical=S.numerical(folder,row['scale'],row['mode'])
        assert numerical==read(folder/'validation.json')
        for module in read(folder/'module-identity.json'):
            assert dependency_hashes[module['path'].lower()]==module['sha256'];modules+=1
            assert module['path']in process['modules']
        summary=read(folder/'summary.json');times=summary['process_wall_ms']
        assert len(times)==8 and all(v>0 for v in times)
        raw={**scheduled,'generation_wall_ms':summary['generation_wall_ms'],'prefill_ms':times[0],
             'decode_7_ms':sum(times[1:]),'decode_per_token_ms':sum(times[1:])/7,
             'wall_s':process['wall_s'],'logits_sha256':sha(folder/'logits.f32'),'status':'pass'}
        assert raw==row and raw['generation_wall_ms']>=sum(times)
        recomputed.append(raw);streams.update(raw_streams(folder,row['scale'],weights,kv,mode))
        if len(recomputed)%10==0:print('Statistical raw model processes rechecked',len(recomputed),'/ 142',flush=True)
    assert modules==710
    stages=Counter(s['stage']for s in samples);assert stages=={'warmup':14,'confirm':112,'AA':16}
    assert all(s['scale']=='F1'for s in samples[:71])and all(s['scale']=='F2'for s in samples[71:])
    assert read((second/'raw'/samples[70]['id']).with_suffix('.process.json'))['end_utc']<read((second/'raw'/samples[71]['id']).with_suffix('.process.json'))['start_utc']
    results={}
    for scale in ('F1','F2'):
        result=S.analyze(scale,recomputed)
        assert result==read(second/f'reports/{scale}-statistics.json')
        assert result==accept['scales'][scale]
        orders=proto['orders'][scale]['blocks']
        for a in MODES:
            for b in MODES:
                if a!=b:assert sum(order.index(a)<order.index(b)for order in orders)==4
        results[scale]=result
    # Min/max of n=8 paired observations covers the population median by at least
    # 1-2*(1/2)^8. Six contrasts use the same predeclared union-bound error budget.
    coverage=1-2/2**8;family_coverage=1-6*(1-coverage)
    assert coverage==.9921875 and family_coverage==.953125
    assert all(v['decision']=='increase'for result in results.values()for v in result['contrasts'].values())
    assert not(second/'failure.json').exists()
    assert 'host process visibility missing'in read(second/'provenance/sandbox-preflight-failure.json')['traceback']
    return {'status':'pass','model_processes':142,'frozen_dependency_hashes':len(dependency_hashes),
            'loaded_module_hash_checks':modules,'resource_probe_records':probes,'sample_stages':dict(stages),
            'nonoverlapping_adjacent_processes':141,
            'raw_streams':dict(streams),'paired_contrasts_recomputed':6,'all_group_distributions_recomputed':True,
            'AA_and_interaction_recomputed':True,'pairwise_AB_BA_balance':[4,4],
            'single_median_coverage_lower_bound':coverage,'family_coverage_lower_bound':family_coverage,
            'six_decisions':'increase','sandbox_preflight_failure_retained':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Restored root containing integrated/')
    p.add_argument('--report-dir',type=Path,required=True)
    args=p.parse_args();root=extended(args.root);out=extended(args.report_dir)
    out.mkdir(parents=True,exist_ok=False)
    first=root/'integrated/20261112-dual-line-r1';second=root/'integrated/20261113-dual-line-stat-r1'
    origin=read(HERE/'evidence_validators/origin.json')
    for row in origin['files']:assert sha(HERE/'evidence_validators'/row['path'])==row['sha256']
    verified={folder.name:verify_manifest(folder)for folder in (first,second)}
    print('Original sealed evidence manifests verified',verified,flush=True)
    contract=fixed_contract(first,second)
    dependency_hashes={r['path'].lower():r['sha256']for r in read(second/'protocol.json')['dependencies']}
    def redirected(path,value):write(out/'correctness-recomputed'/Path(path).name,value)
    # Original functions remain unchanged. Only their data roots and write destinations
    # are injected. No process launch, binary execution or GGUF acquisition is selected.
    C=historical('run_correctness.py',['validate_case','compare_scale'],
                 {'R':first,'OUT':first/'results','np':np,'json':json,'read':read,'sha':sha,'write':redirected})
    correctness_result=correctness(first,C,dependency_hashes,out)
    print('16 correctness processes rechecked',flush=True)
    stat=statistical(root,first,second,C,out)
    report={'status':'pass','GPU_executed':False,'original_evidence_manifest_records':verified,
            'fixed_contract':contract,'correctness':correctness_result,'statistical':stat,
            'scope':'All original correctness/statistical process output, exact logits/tokens/KV hashes, request address coverage/deadlines/tickets, media records, frozen binary identities, module/process/resource records, original statistical distributions/AA/paired confidence bounds/decisions rechecked.',
            'limitations':['Individual per-lane device timestamps and actual KV bytes are not exported by these historical programs. Raw logs retain the maximum consumer timestamp per warp plus in-process byte/32-lane audit counters and KV SHA-256 results; this offline check rederives everything present, not unrecorded individual lane samples or KV bytes.',
                           'Historical 608 CPU tests and 28 GPU fixture cases are evidence records checked here; GPU/CPU code is not re-executed by offline replay. Fresh source build and execution are separate migration validation.'],
            'historical_validator_adaptations':['Selected unchanged offline functions only; R/SEALED/OUT point to restored records, write/event functions emit only new replay reports.','Workstation model acquisition and OS process/module enumeration are not re-executed; their archived process/module/visibility records and actual frozen dependency binary hashes are rechecked.']}
    write(out/'acceptance.json',report)
    print('PASS full V3 offline audit:',out/'acceptance.json',flush=True)


if __name__=='__main__':main()
