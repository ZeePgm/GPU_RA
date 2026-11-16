"""Sampled model, every consumer, native/paged KV and numerical contracts."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import stage1 as S
import sampled_run as X
from reader_validation import validate_reader


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def consumer_gpu_groups(folder, gates):
    activities = lines(Path(folder) / 'gpu-activity.jsonl')
    groups = []
    for step in range(8):
        kernels = sorted((r for r in activities if r['kind'] == 'kernel' and r['phase'] == step), key=lambda r: r['start_ns'])
        pre = [r for r in kernels if 'sampled_consumer_release' in r['name']]
        post = [r for r in kernels if 'sampled_consumer_complete' in r['name']]
        expected = [g for g in gates if g['step'] == step]
        S.require(len(pre) == len(post) == len(expected) == 14, 'GPU consumer marker coverage')
        for index, (begin, end, gate) in enumerate(zip(pre, post, expected)):
            S.require(begin['end_ns'] <= end['start_ns'], 'GPU consumer marker order')
            if index:
                S.require(post[index-1]['end_ns'] <= begin['start_ns'], 'GPU graph consumer intervals overlap')
            actual = [r for r in kernels if r['start_ns'] >= begin['end_ns'] and r['end_ns'] <= end['start_ns']
                      and 'sampled_consumer_' not in r['name']]
            S.require(actual, 'No actual GPU consumer kernel between markers')
            S.require(not any('sampled_window' in r['name'] or 'sampled_copy' in r['name'] for r in actual),
                      'Reader kernel overlaps original consumer interval')
            groups.append({'step': step, 'tensor': gate['tensor'], 'node': gate['node'],
                           'pre_end_ns': begin['end_ns'], 'post_start_ns': end['start_ns'],
                           'kernels': [{'name': r['name'], 'grid': r['grid'], 'block': r['block']} for r in actual],
                           'terminal_kernel': {'name': actual[-1]['name'], 'grid': actual[-1]['grid'], 'block': actual[-1]['block']}})
    return groups


def validate_kv(folder, inventory, input_tokens, kv):
    folder = Path(folder)
    length = input_tokens + 7
    release = S.read(folder / 'framework-release.json')
    S.require(release == {'status': 'pass', 'sequence': 0, 'empty': True}, 'Framework sequence release')
    if kv == 'native':
        positions = S.read(folder / 'native-kv-positions.json')
        S.require(positions == [{'step': step, 'min': 0, 'max': input_tokens + step - 1} for step in range(8)], 'Native KV position lifecycle')
        state = folder / 'native-kv-state.bin'
        S.require(0 < state.stat().st_size <= 128*1024**2, 'Native KV state serialization missing')
        return {'status': 'pass', 'mode': 'native', 'sequence_length': length, 'serialized_state': S.identity(state),
                'framework_released': True, 'paged_lifecycles': 'not used by native KV'}
    summary = S.read(folder / 'kv-summary.json')
    released = S.read(folder / 'kv-release.json')
    content = S.read(folder / 'kv-content.json')
    S.require(summary['steps'] == 8 and summary['length'] == length and summary['layers'] == 28 and summary['transactions'] == 8,
              'Paged KV transaction scope')
    S.require(summary['appended_layer_tokens'] == length*28 and summary['logical_pages'] == 14336 and
              summary['frame_bytes'] == summary['pinned_staging_bytes'] == 16384 and summary['blocks_per_bank'] == 65536 and
              summary['pages_per_block'] == 4, 'Paged KV fixed budgets')
    S.require(summary['cancellable_append'] and summary['paging_mode'] == 1 and summary['kv_write_mode'] == 'device',
              'Paged KV device/async configuration')
    S.require(summary['async_prepared'] == summary['async_retired'] and summary['staging_pages'] == 0 and
              summary['staging_budget_pages'] == 14336 and summary['peak_staging_pages'] <= 14336 and
              summary['pins'] == summary['dirty_frames'] == 0 and not any(released.values()), 'Paged KV resources not drained')
    S.require(all(summary[k] == 0 for k in ('prefetch_issued', 'prefetch_used', 'prefetch_cancelled')), 'Prefetch outside approved scope')
    expected_pages = ((length+1)//2)*28*2
    S.require(content['sequences']['0']['length'] == length and content['sequences']['0']['identity'] == 1 and
              content['allocated_pages'] == expected_pages and content['pins'] == content['active_leases'] == 0 and
              content['frame_bytes'] == 16384, 'Paged KV content/lease extent')
    S.require(S.read(folder / 'kv-versions.json') == [{'step': step, 'version': step+1} for step in range(8)], 'Paged KV version publication')
    count = 0
    for media in lines(folder / 'kv-media.jsonl'):
        S.require(inventory['kv_base'] <= media['logical'] and media['logical'] + media['bytes'] <= inventory['kv_base'] + inventory['kv_reserved_bytes'] and
                  media['end_ps'] >= media['arrival_ps'], 'KV media range/time dependency')
        count += 1
    S.require(count == summary['media_records'], 'KV media audit missing records')
    return {'status': 'pass', 'mode': 'modeled', 'sequence_length': length, 'sequences': content['sequences'],
            'media_records': count, 'all_release_counters_zero': True, 'versions': list(range(1, 9))}


def validate_model(folder, inventory, frozen, route, kv, reference_native=None, reference_same_kv=None):
    folder = Path(folder)
    environment = S.read(folder.parent / (folder.name + '.process.json'))['environment']
    required_environment = {'GGML_CUDA_DISABLE_GRAPHS': '1', 'GGML_CUDA_DISABLE_FUSION': '1',
                            'GGML_CUDA_CUBLAS_COMPUTE_TYPE': 'f32', 'HBFLAB_SAMPLED_ROUTE': route,
                            'HBFLAB_JOINT_KV': 'paged' if kv == 'modeled' else 'native',
                            'HBFLAB_SAMPLED_TOKEN_ONLY': '1', 'LLAMA_GRAPH_REUSE_DISABLE': '1',
                            'HBFLAB_KV_PAGING': 'async', 'HBFLAB_KV_STAGING_PAGES': '4',
                            'HBFLAB_KV_CANCELLABLE': '1', 'HBFLAB_KV_ASYNC_REPLICA': '1', 'HBFLAB_KV_WRITE_MODE': 'device'}
    S.require(all(environment.get(k) == v for k, v in required_environment.items()), 'Actual model environment differs from fixed graph/KV configuration')
    summary = S.read(folder / 'summary.json')
    adapter = S.read(folder / 'sampled-summary.json')
    S.require(summary['status'] == 'pass' and summary['purpose'] == 'sampled_model' and
              summary['prompt_tokens'] == frozen['prompt_tokens'] and summary['generated_tokens'] == frozen['generated_tokens'],
              'Frozen model token contract')
    S.require(summary['arithmetic'] == 'F32 MUL_MAT' and summary['context'] == 512 and summary['batch'] == summary['ubatch'] == 256
              and summary['threads'] == 8 and not summary['graphs'] and not summary['fusion'] and not summary['flash']
              and summary['embedding_buffer_override'] == 'CUDA0' and summary['token_only_graph'], 'Model/graph configuration drift')
    demand = S.demand_plan(inventory, folder, check_route_buffers=False)
    S.require(demand['consumer_count'] == 112 and not demand['runtime_alias_groups'], 'Sampled consumer/alias scope')
    S.require(adapter['status'] == 'pass' and adapter['route'] == route and adapter['kv'] == kv and adapter['steps'] == 8
              and adapter['consumers'] == adapter['post_consumers'] == 112 and adapter['borrowed_unchanged'], 'Adapter model summary')
    gates = lines(folder / 'sampled-gates.jsonl')
    post = lines(folder / 'sampled-consumers.jsonl')
    expected = {(d['step'], d['tensor']): d for d in frozen['demands']}
    S.require(Counter((g['step'], g['tensor']) for g in gates) == Counter(expected.keys()), 'Missing/extra gated consumers')
    S.require(Counter((g['step'], g['node']) for g in gates) == Counter((g['step'], g['node']) for g in post), 'Missing post-consumer dependency')
    post_map = {(g['step'], g['node']): g for g in post}
    spans = []
    for gate in gates:
        planned = expected[(gate['step'], gate['tensor'])]
        S.require(gate['ranges'] == planned['ranges'] and gate['op'] == planned['op'] and gate['route'] == route, 'Actual gate differs from frozen demand')
        after = post_map[(gate['step'], gate['node'])]
        S.require(0 < gate['gate_release_ns'] <= after['consumer_finished_ns'] and gate['gate_release_ns'] == after['gate_release_ns'], 'Consumer dependency clock')
        size = sum(r['bytes'] for r in planned['ranges'])
        payload = size if route in ('ptx', 'cuda') else 0
        requested = 0 if route == 'native' else size
        S.require(gate['requested_bytes'] == requested and gate['payload_bytes'] == gate['publication_bytes'] == payload,
                  'Route data work differs from declared scope')
        spans.extend({'tensor': gate['tensor'], 'offset': r['offset'], 'bytes': r['bytes']} for r in planned['ranges'])
    kernel_groups = consumer_gpu_groups(folder, gates)
    coverage = None
    if route != 'native':
        addresses = {t['name']: t['address'] for t in S.read(folder / 'loaded-tensors.json') if t['selected']}
        coverage = validate_reader(folder / 'readonly', spans, inventory, route, addresses)
        S.require(coverage['requested_bytes'] == frozen['model_demand_bytes'] == adapter['requested_bytes'] and
                  coverage['requests'] == adapter['requests'] == sum(g['requests'] for g in gates), 'Model request/coverage accounting')
        if route == 'ptx':
            ptx = S.read(folder / 'readonly/ptx-identities.json')
            S.require({p['unit'] for p in ptx} == {2, 16} and all(len(p[k]) == 64 for p in ptx for k in
                      ('source_sha256', 'rewritten_sha256', 'helper_sha256', 'artifact_sha256', 'cubin_sha256')), 'Actual PTX identity missing')
    else:
        S.require(adapter['requests'] == adapter['requested_bytes'] == adapter['payload_bytes'] == adapter['publication_bytes'] == 0, 'Native reference issued weight requests')
    if adapter['runtime'] is not None:
        runtime = adapter['runtime']
        S.require(runtime['drained'] and runtime['peak_resident_fragments'] <= 256 and runtime['requests'] == adapter['requests'] and
                  runtime['peak_inflight'] <= 64 and runtime['peak_active_launches'] <= 1 and runtime['audit_limit'] == 513 and
                  runtime['mailbox_slots'] == 2048 and runtime['launch_limit'] == 1 and runtime['audit_dropped'] == runtime['host_failed'] == 0 and
                  runtime['registered_globals'] == runtime['completed_globals'], 'Shared service resources/drain')
    kv_result = validate_kv(folder, inventory, len(frozen['prompt_tokens']), kv)
    if adapter['runtime'] is not None:
        S.require(adapter['runtime']['media_requests'] == (kv_result['media_records'] if kv == 'modeled' else 0), 'KV media/runtime request accounting')
    logits = np.fromfile(folder / 'logits.f32', dtype='<f4')
    S.require(logits.size == 8*151936 and np.isfinite(logits).all(), 'Complete logits missing/nonfinite')
    numerical = {'logits_sha256': S.sha(folder / 'logits.f32')}
    if reference_native:
        reference = np.fromfile(Path(reference_native) / 'logits.f32', dtype='<f4')
        diff = logits.astype('f8') - reference.astype('f8')
        maximum, rms = float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff*diff)))
        S.require((maximum == rms == 0) if kv == 'native' else (maximum <= .2 and rms <= .025), 'Original paged/native numerical contract')
        numerical.update(max_abs_vs_native=maximum, rms_vs_native=rms)
    if reference_same_kv:
        reference_same_kv = Path(reference_same_kv)
        S.require(S.sha(folder / 'logits.f32') == S.sha(reference_same_kv / 'logits.f32'), 'Same arithmetic/KV logits not bitwise')
        if kv == 'native':
            S.require(S.sha(folder / 'native-kv-state.bin') == S.sha(reference_same_kv / 'native-kv-state.bin'), 'Native KV complete state bytes differ')
        else:
            S.require(S.read(folder / 'kv-content.json')['sequences'] == S.read(reference_same_kv / 'kv-content.json')['sequences'], 'Modeled KV full/prefix bytes differ')
        reference_groups = S.read(reference_same_kv / 'model-acceptance.json')['gpu_consumer_groups']
        terminals = {(g['step'], g['tensor']): g['terminal_kernel'] for g in reference_groups}
        S.require(all(g['terminal_kernel'] == terminals[(g['step'], g['tensor'])] for g in kernel_groups), 'Original GPU consumer kernel identity differs from same-KV reference')
        numerical['same_KV_logits_and_state_bitwise'] = True
    result = {'status': 'pass', 'route': route, 'kv': kv, 'consumers': 112, 'independent_coverage': coverage,
              'kv_contract': kv_result, 'numerical': numerical, 'gpu_activity': X.gpu_activity(folder),
              'gpu_consumer_groups': kernel_groups, 'common_clock_qualified': False,
              'diagnostic_wall_ms': {k: summary[k] for k in ('initialization_wall_ms', 'ttft_wall_ms', 'tpot_wall_ms', 'final_audit_and_drain_wall_ms')}}
    S.write(folder / 'model-acceptance.json', result)
    return result
