"""Direct candidate scientific runner: one guarded C++ process per immutable RUN.

Uses the unchanged full-root resource guard and OS observer. No nested Python
scientific runner, background controller, or change to historical validators.
"""
import argparse
from collections import Counter
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import traceback
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/v3-three-routes-sampled-001'))
import stage1 as S
import sampled_run as X
from reader_validation import validate_reader
import numpy as np

read, write, need, sha = S.read, S.write, S.require, S.sha


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]


def receipt(build):
    write(build / 'candidate-receipt.json', {'source': S.source_identity(), 'build': S.audit_build(build),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'candidate_scope': 'five weights, four outputs; default historical 14/8 unchanged'})


def reader_summary(folder, route, expected_bytes, expected_requests):
    r = read(folder / 'summary.json')
    need(r['status'] == 'pass' and r['route'] == route and r['audit_mode'] == 'summary', 'Summary reader contract')
    need(r['requested_bytes'] == expected_bytes and r['requests'] == expected_requests, 'Independent byte/request extent')
    need(r['lane_checks'] * 16 == expected_bytes and r['all_runtime_lane_and_byte_checks'], 'Complete runtime lane checks')
    payload = expected_bytes if route in ('ptx', 'cuda') else 0
    need(r['payload_bytes'] == r['publication_bytes'] == payload and r['byte_checks'] == 2 * payload, 'Complete returned/published byte checks')
    need(r['request_log_bytes'] == r['lane_time_log_bytes'] == 0, 'Summary exported raw logs')
    need(not any((folder / n).exists() for n in ('requests.bin', 'lane-times.bin', 'chunks.jsonl')), 'Unexpected raw trace')
    return r


def validate_model(folder, route, reference):
    spec, inv = read(HERE / 'input-spec.json'), read(HERE / 'inventory.json')
    summary, adapter = read(folder / 'summary.json'), read(folder / 'sampled-summary.json')
    need(summary['status'] == 'pass' and summary['selected_tensors'] == 5 and summary['generated_steps'] == 4, 'Five/four model scope')
    need(summary['prompt_tokens'] == spec['prompt_tokens'] and summary['generated_tokens'] == spec['generated_tokens_reference'], 'Exact token contract')
    need(summary['arithmetic'] == 'F32 MUL_MAT' and summary['context'] == 512 and summary['batch'] == summary['ubatch'] == 256 and summary['threads'] == 8
         and not summary['graphs'] and not summary['fusion'] and not summary['flash'] and summary['embedding_buffer_override'] == 'CUDA0', 'Model configuration')
    need(adapter['status'] == 'pass' and adapter['steps'] == 4 and adapter['selected_weights'] == 5 and adapter['kv'] == 'native'
         and adapter['consumers'] == adapter['post_consumers'] == 20 and adapter['borrowed_unchanged'], 'Adapter scope and borrowed identity')
    wanted = {(step, w['name']) for step in range(4) for w in spec['weights']}
    gates, post = lines(folder / 'sampled-gates.jsonl'), lines(folder / 'sampled-consumers.jsonl')
    need(Counter((g['step'], g['tensor']) for g in gates) == Counter(wanted), 'Selected consumer coverage')
    need(Counter((g['step'], g['node']) for g in gates) == Counter((g['step'], g['node']) for g in post), 'Post-consumer coverage')
    observations = lines(folder / 'consumers.jsonl')
    need(Counter((g['step'], g['weight']) for g in observations) == Counter(wanted), 'Framework selected consumers')
    loaded = read(folder / 'loaded-tensors.json')
    need(len(loaded) == 311 and sum(t['selected'] for t in loaded) == 5, 'Five selected, 306 native')
    selected = [t for t in loaded if t['selected']]
    for index, a in enumerate(selected):
        need(a['cuda_memory_type'] == 2 and a['cuda_device'] == 0 and a['contiguous'], 'Native CUDA weight storage')
        for b in selected[index+1:]:
            need(a['address']+a['bytes'] <= b['address'] or b['address']+b['bytes'] <= a['address'], 'Selected storage aliases')
    independent = read(ROOT / 'experiments/v3-three-routes-sampled-001/demand-plan.json')['inputs']['original44']
    demands = {(d['step'], d['tensor']): d for d in independent['demands'] if (d['step'], d['tensor']) in wanted}
    posts = {(g['step'], g['node']): g for g in post}
    spans, total, requests = [], 0, 0
    for gate in gates:
        d = demands[gate['step'], gate['tensor']]
        need(gate['ranges'] == d['ranges'] and gate['op'] == d['op'] and gate['route'] == route, 'Frozen independent demand mismatch')
        after = posts[gate['step'], gate['node']]
        need(0 < gate['gate_release_ns'] == after['gate_release_ns'] <= after['consumer_finished_ns'], 'Publication/consumer ordering')
        size = sum(r['bytes'] for r in d['ranges'])
        need(gate['requested_bytes'] == (0 if route == 'native' else size), 'Requested extent')
        payload = size if route in ('ptx', 'cuda') else 0
        need(gate['payload_bytes'] == gate['publication_bytes'] == payload, 'Payload/publication extent')
        total += size
        for r in d['ranges']:
            spans.append({'tensor': gate['tensor'], 'offset': r['offset'], 'bytes': r['bytes']})
            requests += (r['bytes'] + 511)//512
    if route != 'native':
        reader_summary(folder / 'readonly', route, total, requests)
        need(adapter['requests'] == requests and adapter['requested_bytes'] == total, 'Adapter full accounting')
        r = adapter['runtime']
        need(r['drained'] and r['requests'] == requests and r['media_requests'] == 0 and r['peak_resident_fragments'] <= 256
             and r['peak_inflight'] <= 64 and r['peak_active_launches'] <= 1 and r['audit_limit'] == 513 and r['mailbox_slots'] == 2048
             and r['registered_globals'] == r['completed_globals'] and r['host_failed'] == r['audit_dropped'] == 0, 'Runtime resources/drain')
    else:
        need(adapter['requests'] == adapter['requested_bytes'] == 0 and adapter['runtime'] is None, 'Native reference issued requests')
    need(read(folder / 'native-kv-positions.json') == [{'step': s, 'min': 0, 'max': 43+s} for s in range(4)], 'Native KV positions')
    need(read(folder / 'framework-release.json') == {'status': 'pass', 'sequence': 0, 'empty': True}, 'Native KV release')
    logits = np.fromfile(folder / 'logits.f32', dtype='<f4')
    need(logits.size == 4*151936 and np.isfinite(logits).all(), 'Complete finite four-step logits')
    if reference:
        need(sha(folder / 'logits.f32') == sha(reference / 'logits.f32'), 'Full logits differ from native')
        need(sha(folder / 'native-kv-state.bin') == sha(reference / 'native-kv-state.bin'), 'Complete native KV bytes differ')
    return {'status': 'pass', 'consumers': 20, 'all_logits_and_native_kv_bitwise': bool(reference), 'native_reference': str(reference) if reference else None,
        'requested_bytes_expected': 0 if route == 'native' else total, 'requests_expected': 0 if route == 'native' else requests,
        'logits': S.identity(folder / 'logits.f32'), 'KV': S.identity(folder / 'native-kv-state.bin'),
        'ttft_ms': summary['ttft_wall_ms'], 'tpot_ms': summary['tpot_wall_ms'], 'initialization_ms': summary['initialization_wall_ms'],
        'generation_ms': summary['generation_until_last_token_wall_ms'], 'runtime_validation_in_timing': True,
        'independent_full_raw_offline_recheck': False}


def environment_probe():
    # Driver query is strictly outside the timed C++ process, as in the frozen policy.
    cmd=['nvidia-smi','--id=0','--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,temperature.gpu,clocks.sm,power.draw','--format=csv']
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
    need(p.returncode==0,'Pre/post GPU environment observation failed')
    return {'utc':datetime.now(timezone.utc).isoformat(),'command':cmd,'gpu':p.stdout.strip(),
        'platform':platform.platform(),'logical_cpu_count':os.cpu_count(),'python':sys.executable}


def formal_admission(args, identity):
    config=read(HERE/'config.json')['parameters']
    need(config['execution_authorized']['formal_matrix'] is True,'Formal execution not authorized')
    need(args.mode in ('weight','model') and args.audit=='summary','Formal mode/audit contract')
    matrix=read(HERE/'matrix.json')
    key='weight_process_schedule' if args.mode=='weight' else 'model_process_schedule'
    entries=[e for e in matrix[key] if e['sample_key']==args.sample_key]
    need(len(entries)==1,'Unique formal sample key required');entry=entries[0]
    need(entry['route']==args.route and args.output.parent.name==entry['future_run_id'],'Formal route/RUN binding')
    if args.mode=='weight':
        fixture=read(args.fixture);case=next(c for c in matrix['weight_cells'] if c['cell_id']==entry['cell_id'])
        need(fixture['scenario_id']==case['scenario_id'] and fixture['tensor']==case['tensor'],'Formal fixture binding')
        expected=next(f for f in read(HERE/'input-spec.json')['captured_native_fixtures'] if f['scenario_id']==fixture['scenario_id'])
        need(sha(args.fixture)==expected['descriptor_identity']['sha256'],'Formal native fixture identity')
        need(args.repeats==entry['repeated_calls'],'Formal repeated-call count')
    else:
        need(args.repeats==0 and entry['generated_tokens']==4 and entry['prefill_tokens']==44,'Formal model sequence scope')
        ref=read(HERE/'input-spec.json')['new_native_reference']
        need(sha(args.reference/'logits.f32')==ref['full_logits']['sha256'] and sha(args.reference/'native-kv-state.bin')==ref['kv_identity']['sha256'],'Formal native reference identity')
    return entry


def run(args):
    coordinator_start=time.process_time()
    formal_entry=None
    args.build_dir, args.output = args.build_dir.resolve(), args.output.resolve()
    need(args.resource_root.resolve() == Path('E:/v1_update').resolve(), 'Full original workspace resource scope required')
    need(not args.output.exists(), 'Fresh output directory required');args.output.mkdir(parents=True)
    try:
        identity = read(args.build_dir / 'candidate-receipt.json');X.unchanged(args.build_dir, identity)
        if args.phase in ('pilot','formal'):
            need(args.qualification is not None,'Completed qualification receipt required before pilot')
            q=read(args.qualification)
            need(q['status']=='passed' and q['weight_calls']==30 and q['model_smokes']==3 and q['dual_audit_routes']==3 and q['candidate_source']==identity['source'],'Pilot qualification scope/identity mismatch')
        if args.phase=='formal':formal_entry=formal_admission(args,identity)
        elif args.mode=='weight':need(args.repeats==(1 if args.phase=='pilot' else 0),'This stage permits only one functional call or first+one pilot call')
        need(sha(args.model) == S.MODEL_SHA, 'Model input SHA-256 mismatch')
        write(args.output / 'candidate-receipt.json', identity)
        child = args.output / 'result'
        env = {'HBFLAB_M4_PAGED':'0', 'LLAMA_GRAPH_REUSE_DISABLE':'1', 'HBFLAB_SAMPLED_TOKEN_ONLY':'1'}
        timeout, require_ggml = 120, False
        if args.mode in ('reference', 'model'):
            need(args.audit == 'summary', 'Full model runs use qualified summary mode only')
            profile = read(HERE / 'profile.json');profile['model_path'] = str(args.model.resolve())
            write(args.output / 'profile.json', profile)
            env.update(HBFLAB_SAMPLED_ROUTE=args.route, HBFLAB_SAMPLED_AUDIT=args.audit, HBFLAB_FIVE_PERF='1',
                HBFLAB_JOINT_KV='native', HBFLAB_SAMPLED_PREFILL='44', HBFLAB_SAMPLED_INVENTORY=str(HERE / 'inventory.json'),
                HBFLAB_SAMPLED_PROFILE=str(args.output / 'profile.json'))
            if args.mode == 'reference':
                need(args.route == 'native', 'Reference route');env['HBFLAB_FIVE_CAPTURE'] = str(args.output / 'fixtures')
            else: need(args.reference is not None and (args.reference/'model-acceptance.json').is_file(), 'Accepted native reference required')
            command = [args.build_dir/'bin/five_model.exe',args.model,ROOT/'experiments/v3-three-routes-sampled-001/prompt-44.txt',child,HERE/'inventory.json','4']
            timeout = 120 if args.mode == 'reference' else 1200;require_ggml=True
        elif args.mode == 'weight':
            need(args.fixture and args.fixture.is_file(), 'Frozen native fixture required')
            command = [args.build_dir/'bin/five_weight.exe',args.model,HERE/'inventory.json',args.fixture,child,args.route,str(args.repeats),args.audit]
            timeout = 900 if read(args.fixture)['tensor'] == 'output.weight' else 120;require_ggml=True
        elif args.mode == 'legacy-registration':
            child.mkdir()
            command = [args.build_dir/'bin/sampled_registration.exe',ROOT/'experiments/v3-three-routes-sampled-001/inventory.json',args.route,child/'summary.json']
        elif args.mode == 'diagnostic':
            command = [args.build_dir/'bin/sampled_reader_fixture.exe',HERE/'inventory.json',args.model,HERE/'diagnostic-plan.json',child,args.route,args.audit]
            timeout=120
        else: raise ValueError('Unknown mode')
        if args.process_wall_cap is not None:
            need(args.process_wall_cap>0,'Positive process wall cap required')
            timeout=min(timeout,args.process_wall_cap)
        need(X.directory_bytes(args.output.parent.parent)+33*1024**2<=2*1024**3,'Experiment output cap cannot admit another process')
        if args.phase in ('pilot','formal'):write(args.output/'environment-before.json',environment_probe())
        process = X.execute(args, command, 'result', child, extra_env=env, timeout=timeout, require_ggml=require_ggml, disk_cap=32*1024**2)
        if args.phase in ('pilot','formal'):write(args.output/'environment-after.json',environment_probe())
        need((args.output/'result.log').stat().st_size <= 1024**2, 'stdout/stderr budget')
        if args.mode in ('reference','model'):
            acceptance = validate_model(child, args.route, args.reference)
            if args.phase=='formal':
                timing=read(child/'summary.json');emissions=timing['emission_wall_ms']
                need(len(emissions)==4 and all(np.isfinite(emissions)) and all(a<b for a,b in zip([0]+emissions,emissions)),'Formal token timing coverage/order')
                need(abs(timing['ttft_wall_ms']-emissions[0])<1e-6 and abs(timing['tpot_wall_ms']-(emissions[-1]-emissions[0])/3)<1e-6 and abs(timing['generation_until_last_token_wall_ms']-emissions[-1])<1e-6,'Independent model timing reduction')
                acceptance['token_emission_ms']=emissions
            write(child/'model-acceptance.json',acceptance)
            if args.mode=='reference':
                need(args.historical_logits is not None and sha(args.historical_logits)=='942e5798c378ed9750caadcd8e802f26ee61024e12a7cfafd792ec908bce6835','Frozen native eight-step source missing or changed')
                native=(child/'logits.f32').read_bytes()
                need(native==args.historical_logits.read_bytes()[:len(native)],'Four-step reference differs from historical native prefix')
                acceptance['historical_native_prefix_bitwise']=True
                write(child/'model-acceptance.json',acceptance)
                need(len(list((args.output/'fixtures').glob('w*.json')))==10,'Ten original consumer fixtures missing')
                write(args.output/'fixtures/manifest.json',{'status':'pass','source_model':S.MODEL_SHA,'source_commit':identity['code_commit'],
                    'files':[S.identity(f) for f in sorted((args.output/'fixtures').iterdir()) if f.is_file()]})
        elif args.mode=='weight':
            s=read(child/'summary.json');calls=read(child/'calls.json');fixture=read(args.fixture)
            need(s['status']=='pass' and s['drained'] and s['borrowed_unchanged'] and s['calls']==1+args.repeats
                 and s['peak_resident_fragments']<=256 and all(c['native_bitwise_equal'] and c['end_ns']>c['begin_ns'] for c in calls), 'Consumer numerical/lifecycle scope')
            if args.phase=='formal':
                need(len(calls)==1+args.repeats and [c['index'] for c in calls]==list(range(len(calls))),'Formal complete call records')
                repeated=sum(c['end_ns']-c['begin_ns'] for c in calls[1:])/args.repeats
                need(abs(repeated-s['repeated_mean_ns'])<=1,'Independent repeated timing reduction')
            amount = fixture['weight']['bytes'] if fixture['op']!='GET_ROWS' else (44 if fixture['step']==0 else 1)*2048
            need(s['requested_bytes']==amount*(1+args.repeats),'Exact isolated consumer demand')
            if args.audit=='summary':reader_summary(child/'reader',args.route,amount*(1+args.repeats),((amount+511)//512)*(1+args.repeats))
            acceptance={'status':'pass','scenario_id':s['scenario_id'],'calls':s['calls'],'outputs_bitwise':True,'borrowed_unchanged':True,'drained':True,
                'first_call_ms':s['first_call_ns']/1e6,'repeated_mean_ms':s['repeated_mean_ns']/1e6,'initialization_ms':s['initialization_ns']/1e6}
        elif args.mode=='legacy-registration':
            summary=read(child/'summary.json')
            need(summary['status']=='pass' and summary['drained'] and summary['selected_ranges']==14 and summary['negative_checks']>=21
                 and summary['media_requests']==3 and summary['readonly_alignment']==32768 and summary['peak_resident_fragments']<=256,'Legacy registration regression')
            need('100% tests passed, 0 tests failed out of 608' in (args.build_dir/'core-tests.log').read_text(encoding='utf-8-sig'),'CPU core regression missing')
            acceptance={'status':'pass','legacy_registration':summary,'core_cpu_tests':608}
        else:
            summary=read(child/'summary.json');plan=read(HERE/'diagnostic-plan.json')
            need(summary['status']=='pass' and summary['drained'] and summary['selected_ranges']==5 and summary['negative_checks']==8,'Bounded fixture contract')
            if args.audit=='full':
                bindings={t['name']:t['address'] for t in read(child/'bindings.json')}
                coverage=validate_reader(child/'reader',plan['spans'],read(HERE/'inventory.json'),args.route,bindings)
                activity=X.gpu_activity(child)
                acceptance={'status':'pass','full_independent_reader_coverage':coverage,'gpu_activity':activity,'negative_checks':8}
            else:
                r=read(child/'reader/summary.json')
                need(r['all_runtime_lane_and_byte_checks'] and r['request_log_bytes']==r['lane_time_log_bytes']==0,'Summary raw exclusion')
                need(args.reference is not None,'Full diagnostic reference required')
                full=read(args.reference/'reader/summary.json')
                for key in ('requests','requested_bytes','payload_bytes','publication_bytes','launches','lane_checks','byte_checks','validation_d2h_bytes','memset_bytes'):
                    need(r[key]==full[key],'Full/summary runtime coverage differs: '+key)
                acceptance={'status':'pass','runtime_full_checks':r,'negative_checks':8,'full_summary_coverage_identical':True,'full_reference':str(args.reference)}
        X.unchanged(args.build_dir,identity)
        need(X.directory_bytes(args.output)<=32*1024**2,'Per-process output budget')
        write(args.output/'acceptance.json',{**acceptance,'mode':args.mode,'route':args.route,'audit':args.audit,'process_wall_seconds':process['wall_seconds'],
            'cpu_seconds':process['final_metrics']['cpu_seconds'],'output_bytes_before_acceptance':X.directory_bytes(args.output),
            'formal_sample':args.phase=='formal','sample_key':args.sample_key,'schedule_sequence':formal_entry['sequence'] if formal_entry else None,'independent_process_repeat':formal_entry['independent_process_repeat'] if formal_entry else None,'phase':args.phase,'process_wall_cap_seconds':timeout,'coordinator_cpu_seconds':time.process_time()-coordinator_start,'scope':'formal fixed matrix sample' if args.phase=='formal' else 'qualification or pilot; excluded from formal comparison'})
    except BaseException:
        write(args.output/'failure.json',{'status':'failed','error':traceback.format_exc()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['receipt','reference','model','weight','diagnostic','legacy-registration'])
    parser.add_argument('--build-dir',type=Path,required=True)
    for key in ('model','cuda-root','resource-root','output','fixture','reference','qualification','historical-logits'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--route',choices=['native','symbolic','ptx','cuda'],default='native')
    parser.add_argument('--audit',choices=['full','summary'],default='summary')
    parser.add_argument('--repeats',type=int,default=0)
    parser.add_argument('--process-wall-cap',type=int)
    parser.add_argument('--sample-key')
    parser.add_argument('--phase',choices=['qualification','pilot','formal'],default='qualification')
    args=parser.parse_args()
    if args.mode=='receipt':receipt(args.build_dir.resolve())
    else:run(args)
