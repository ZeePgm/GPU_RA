"""Direct candidate scientific runner: one guarded C++ process per immutable RUN.

Uses the unchanged full-root resource guard and OS observer. No nested Python
scientific runner, background controller, or change to historical validators.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import traceback

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
    need(r['lane_checks'] * 16 >= expected_bytes and r['all_runtime_lane_and_byte_checks'], 'Complete runtime lane checks')
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


def run(args):
    args.build_dir, args.output = args.build_dir.resolve(), args.output.resolve()
    need(args.resource_root.resolve() == Path('E:/v1_update').resolve(), 'Full original workspace resource scope required')
    need(not args.output.exists(), 'Fresh output directory required');args.output.mkdir(parents=True)
    try:
        identity = read(args.build_dir / 'candidate-receipt.json');X.unchanged(args.build_dir, identity)
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
        process = X.execute(args, command, 'result', child, extra_env=env, timeout=timeout, require_ggml=require_ggml, disk_cap=32*1024**2)
        need((args.output/'result.log').stat().st_size <= 1024**2, 'stdout/stderr budget')
        if args.mode in ('reference','model'):
            acceptance = validate_model(child, args.route, args.reference)
            write(child/'model-acceptance.json',acceptance)
            if args.mode=='reference':
                need(len(list((args.output/'fixtures').glob('w*.json')))==10,'Ten original consumer fixtures missing')
                write(args.output/'fixtures/manifest.json',{'status':'pass','source_model':S.MODEL_SHA,'source_commit':identity['code_commit'],
                    'files':[S.identity(f) for f in sorted((args.output/'fixtures').iterdir()) if f.is_file()]})
        elif args.mode=='weight':
            s=read(child/'summary.json');calls=read(child/'calls.json');fixture=read(args.fixture)
            need(s['status']=='pass' and s['drained'] and s['borrowed_unchanged'] and s['calls']==1+args.repeats
                 and s['peak_resident_fragments']<=256 and all(c['native_bitwise_equal'] and c['end_ns']>c['begin_ns'] for c in calls), 'Consumer numerical/lifecycle scope')
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
            'formal_sample':False,'scope':'qualification or pilot only; no formal matrix authorization'})
    except BaseException:
        write(args.output/'failure.json',{'status':'failed','error':traceback.format_exc()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['receipt','reference','model','weight','diagnostic','legacy-registration'])
    parser.add_argument('--build-dir',type=Path,required=True)
    for key in ('model','cuda-root','resource-root','output','fixture','reference'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--route',choices=['native','symbolic','ptx','cuda'],default='native')
    parser.add_argument('--audit',choices=['full','summary'],default='summary')
    parser.add_argument('--repeats',type=int,default=0)
    args=parser.parse_args()
    if args.mode=='receipt':receipt(args.build_dir.resolve())
    else:run(args)
