"""Summarize sealed functional evidence and observed execution costs, without reruns."""
import argparse
import os
from datetime import datetime
from pathlib import Path
import stage1 as S


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('models', 'components', 'output'):
        ap.add_argument('--' + key, type=Path, required=True)
    args = ap.parse_args()
    S.require(not args.output.exists(), 'Use a fresh diagnostics output directory')
    args.output.mkdir(parents=True)
    models, components = S.read(args.models), S.read(args.components)
    S.require(models['status'] == components['status'] == 'pass' and models['model_processes'] == 16
              and components['full_tensor_coverage'], 'Full functional evidence missing')
    inventory = S.read(S.HERE / 'inventory.json')
    budget = S.read(S.HERE / 'budget.json')
    demand = S.read(S.HERE / 'demand-plan.json')['inputs']
    expected = {f'{key}-{route}-{kv}' for key in demand for route in ('native', 'symbolic', 'ptx', 'cuda') for kv in ('native', 'modeled')}
    S.require(set(models['routes']) == expected, 'Formal matrix has missing or extra processes')
    S.require(S.read(args.models.parent / 'candidate-receipt.json') == S.read(args.components.parent / 'candidate-receipt.json'),
              'Component/model candidate mismatch')
    inputs = {'models': S.identity(args.models), 'components': S.identity(args.components),
              'budget': S.identity(S.HERE / 'budget.json')}
    diagnostics, support = [], []
    for label, result in models['routes'].items():
        S.require(result['status'] == 'pass' and result['consumers'] == 112, 'Model contract not accepted')
        folder = args.models.parent / label
        adapter, resources = S.read(folder / 'sampled-summary.json'), S.read(folder / 'resources.json')
        reader_path = folder / 'readonly/summary.json'
        reader = S.read(reader_path) if reader_path.exists() else {}
        timing = result['diagnostic_wall_ms']
        diagnostics.append({'case': label, 'route': result['route'], 'kv': result['kv'],
            'initialization_wall_ms': timing['initialization_wall_ms'], 'TTFT_wall_ms': timing['ttft_wall_ms'],
            'TPOT_wall_ms': timing['tpot_wall_ms'], 'final_audit_and_drain_wall_ms': timing['final_audit_and_drain_wall_ms'],
            'KV_audit_wall_ms': adapter['kv_audit_wall_ms'], 'drain_wall_ms': adapter['drain_wall_ms'],
            'final_weight_hash_wall_ms': adapter['final_weight_hash_wall_ms'],
            'process_wall_seconds': result['process_wall_seconds'], 'child_CPU_seconds': result['cpu_seconds'],
            'GPU_activity_union_seconds': result['gpu_activity']['activity_union_seconds'],
            'backend_raw_model_now_ps': (adapter['runtime'] or {}).get('backend_model_now_ps'),
            'reader_wall_seconds': reader.get('read_wall_seconds', 0),
            'requests': adapter['requests'], 'requested_bytes': adapter['requested_bytes'],
            'payload_bytes': adapter['payload_bytes'], 'publication_bytes': adapter['publication_bytes'],
            'mirror_capacity_bytes': reader.get('mirror_capacity_bytes', 0), 'timing_capacity_bytes': reader.get('timing_capacity_bytes', 0),
            'validation_D2H_bytes': reader.get('validation_d2h_bytes', 0), 'memset_bytes': reader.get('memset_bytes', 0),
            'weight_hash_validation_bytes': adapter['hash_validation_bytes'],
            'GPU_activity_phases': result['gpu_activity']['phases'],
            'request_log_bytes': reader.get('request_log_bytes', 0), 'lane_time_log_bytes': reader.get('lane_time_log_bytes', 0),
            'output_bytes': result['output_bytes'], 'Host_peak_working_set_bytes': result['peak_working_set_bytes'],
            'GPU_free_memory_phase_samples': resources,
            'same_KV_logits_and_state_bitwise': result['numerical'].get('same_KV_logits_and_state_bitwise'),
            'max_abs_vs_native': result['numerical'].get('max_abs_vs_native'),
            'rms_vs_native': result['numerical'].get('rms_vs_native')})
    for tensor in (t for t in inventory['all_tensors'] if t['selected']):
        for route in ('symbolic', 'ptx', 'cuda'):
            fixture = components['routes'][route]['fixture']
            S.require(any(c['tensor'] == tensor['name'] and c['offset'] == 0 and c['bytes'] == tensor['bytes']
                          for c in fixture['calls']), 'Complete independent tensor extent missing')
            cases = []
            requested = 0
            for key in demand:
                for kv in ('native', 'modeled'):
                    label = f'{key}-{route}-{kv}'
                    result = models['routes'][label]
                    S.require(result['numerical']['same_KV_logits_and_state_bitwise'], 'Same-KV bitwise contract missing')
                    consumers = [c for c in result['gpu_consumer_groups'] if c['tensor'] == tensor['name']]
                    S.require(len(consumers) == 8, 'Per-tensor actual model consumer extent missing')
                    requested += sum(r['bytes'] for d in demand[key]['demands'] if d['tensor'] == tensor['name'] for r in d['ranges'])
                    cases.append(label)
            support.append({'tensor': tensor['name'], 'dtype': tensor['dtype'], 'physical_bytes': tensor['bytes'], 'route': route,
                'registered': True, 'component_complete_extent': True, 'model_cases': cases, 'model_consumers': 32,
                'model_requested_bytes': requested, 'model_payload_and_publication_bytes': 0 if route == 'symbolic' else requested,
                'status': 'pass', 'embedding_scope': 'actual token row occurrences in model; every row independently in fixture'
                    if tensor['name'] == 'token_embd.weight' else None})
    cost = models['cost']
    size = lambda root: sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    actual_output = size(args.models.parent) + size(args.components.parent)
    component_execution = S.read(args.components.parent.parent / 'execution.json')
    component_elapsed = (datetime.fromisoformat(component_execution['finished_utc']) -
                         datetime.fromisoformat(component_execution['started_utc'])).total_seconds()
    # No component-coordinator CPU counter was retained. Bound the whole component
    # interval by every logical CPU being busy, then add observed model CPU.
    component_cpu_upper = component_elapsed * os.cpu_count()
    cpu_upper = cost['child_cpu_seconds'] + cost['coordinator_cpu_seconds'] + component_cpu_upper
    wall = cost['elapsed_wall_seconds'] + component_elapsed
    gpu = cost['gpu_activity_seconds'] + sum(v['gpu_activity']['activity_union_seconds'] for v in components['routes'].values())
    S.require(all(v['output_bytes'] <= budget['process_output_bytes'] and
                  v['peak_working_set_bytes'] <= budget['process_host_working_set_bytes']
                  for v in [*models['routes'].values(), *components['routes'].values()]), 'Recorded process resource ceiling exceeded')
    S.require(all(v['process_wall_seconds'] < budget['model_process_wall_seconds'] for v in models['routes'].values()) and
              all(v['process_wall_seconds'] < budget['component_process_wall_seconds'] for v in components['routes'].values()), 'Recorded process wall ceiling exceeded')
    S.require(actual_output < budget['batch_output_bytes'] and wall < budget['batch_wall_seconds'] and
              gpu < budget['batch_gpu_activity_seconds'] and cpu_upper < budget['batch_cpu_seconds'], 'Final frozen batch budget exceeded')
    budget_audit = {'status': 'pass', 'actual_output_bytes_after_final_serialization': actual_output,
        'wall_seconds': wall, 'GPU_activity_union_seconds': gpu,
        'model_observed_child_and_coordinator_CPU_seconds': cost['child_cpu_seconds'] + cost['coordinator_cpu_seconds'],
        'component_execution_seconds': component_elapsed, 'logical_CPU_count_for_upper_bound': os.cpu_count(),
        'component_all_CPU_upper_bound_seconds': component_cpu_upper,
        'batch_scientific_CPU_upper_bound_seconds': cpu_upper,
        'CPU_bound_scope': 'Observed model C++/coordinator CPU plus conservative whole-machine component-interval upper bound; workflow controller outside science interval excluded',
        'budget': budget}
    S.write(args.output / 'budget-audit.json', budget_audit)
    S.write(args.output / 'diagnostics.json', {'schema': 1, 'inputs': inputs, 'cases': diagnostics, 'batch_cost': cost, 'budget_audit': budget_audit,
        'component_cost': {r: {k:v[k] for k in ('process_wall_seconds', 'cpu_seconds', 'peak_working_set_bytes', 'output_bytes')}
                           for r,v in components['routes'].items()},
        'measurement_scope': {'CPU': 'C++ process CPU plus model coordinator CPU; component coordinator not separately measured, conservatively bounded in budget-audit.json',
            'GPU_memory': 'phase samples plus bounded reader allocation identities; no continuous high-water measurement',
            'GPU_activity': 'CUPTI kernel/copy/memset interval union, no cross-epoch clock subtraction',
            'TTFT_TPOT': 'observed wall cost only; includes validation, logging and service overhead'},
        'common_clock_qualified': False, 'performance_verdict': 'insufficient_evidence',
        'reason': 'No validated common simulated timeline for native compute, readonly requests and paged KV; raw backend clock and CPU wall are separate diagnostics.'})
    S.write(args.output / 'support-matrix.json', {'schema': 1, 'inputs': inputs, 'cells': support,
        'functional_status': 'pass', 'physical_tensors': 14, 'routes': 3, 'unselected_native_tensors': 297,
        'formal_model_processes': 16, 'candidate_model_consumers': 1344, 'performance_verdict': 'insufficient_evidence'})
    report = ['# 三入口代表权重：功能验收与成本诊断', '',
        '功能验收通过：14个代表张量×3入口的完整组件、真实模型消费者及联合KV合同全部成立；正式16个模型进程均通过。', '',
        '性能结论为证据不足：尚无合格的共同模拟时钟。以下仅为带完整审计和日志的实测运行成本。', '',
        '| case | 初始化ms | TTFT墙钟ms | TPOT墙钟ms | drain ms | 全进程s | CPU s | GPU活动并集s | 输出GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in diagnostics:
        report.append(f"| {d['case']} | {d['initialization_wall_ms']:.2f} | {d['TTFT_wall_ms']:.2f} | {d['TPOT_wall_ms']:.2f} | {d['drain_wall_ms']:.2f} | {d['process_wall_seconds']:.3f} | {d['child_CPU_seconds']:.3f} | {d['GPU_activity_union_seconds']:.3f} | {d['output_bytes']/1024**3:.3f} |")
    report += ['', '逐张量支持与模型次数见support-matrix.json；请求/实际返回/发布、日志、显存阶段采样、Host峰值、原始后端时钟及测量范围见diagnostics.json。', '',
        'symbolic仅门控，PTX为实际改写模板读取，CUDA为显式读取；未选297个权重走原生路径。embedding模型访问按实际行及重复次数计入，全表覆盖仅属于独立夹具。', '',
        '两种KV模式下相同算术的完整logits与KV状态均逐字节一致；分页对native继续采用max_abs≤0.2、RMS≤0.025、精确token及原生命周期合同。没有追加统计重复或根据结果放宽阈值。', '']
    (args.output / 'DIAGNOSTICS.md').write_bytes('\n'.join(report).encode())
    S.write(args.output / 'acceptance.json', {'status': 'pass', 'scope': 'derived functional support matrix and observed costs; no new model sampling',
        'inputs': inputs, 'functional_status': 'pass', 'support_cells': len(support), 'model_processes': 16,
        'performance_verdict': 'insufficient_evidence', 'common_clock_qualified': False})


if __name__ == '__main__':
    main()
