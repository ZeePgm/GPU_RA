"""Four diagnostic references or the frozen sixteen-process sampled model matrix."""
import argparse
import copy
from pathlib import Path
import time
import traceback

import stage1 as S
import sampled_run as X
from model_validation import validate_model


def profile(model):
    base = S.read(S.ROOT / 'joint/configs/qwen3-0.6b.json')
    result = copy.deepcopy(base)
    result.update(model_path=str(model.resolve()), blocks_per_bank=65536)
    # These old two-tensor access descriptions do not describe this experiment.
    for key in ('coverage', 'kernel', 'access_plan'):
        result.pop(key)
    result['sampled_scope'] = {'inventory': S.identity(S.HERE / 'inventory.json'),
        'demand_plan': S.identity(S.HERE / 'demand-plan.json'),
        'selected_physical_tensors': 14, 'other_tensors': 'native',
        'embedding': 'actual token rows with occurrence; all rows only in independent fixture',
        'reader_chunks': {'symbolic': 262144, 'ptx': 32768, 'cuda': 262144},
        'readonly_alignment': 32768, 'KV_frames': 4, 'KV_pinned_slots': 4,
        'readonly_mailbox_slots': 2048, 'launch_slots': 1, 'runtime_audit_limit': 513,
        'inherited_profile_audit_budgets': 'retained, not the bounded readonly queue size'}
    S.require(result['numerical'] == base['numerical'], 'Original numerical policy changed')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['smoke', 'full'], required=True)
    for key in ('model', 'cuda-root', 'resource-root', 'build-dir', 'output'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--components', type=Path)
    args = ap.parse_args()
    args.output, args.build_dir = args.output.resolve(), args.build_dir.resolve()
    S.require(not args.output.exists(), 'Use a fresh output directory')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        receipt = S.read(args.build_dir / 'candidate-receipt.json')
        X.unchanged(args.build_dir, receipt)
        S.require(S.sha(args.model) == S.MODEL_SHA, 'Model identity mismatch')
        inventory = S.read(S.HERE / 'inventory.json')
        frozen = S.read(S.HERE / 'demand-plan.json')['inputs']
        budget = S.read(S.HERE / 'budget.json')
        S.write(args.output / 'candidate-receipt.json', receipt)
        S.write(args.output / 'profile.json', profile(args.model))
        S.write(args.output / 'budget.json', budget)
        prior = {'cpu_seconds': 0, 'wall_seconds': 0, 'gpu_seconds': 0, 'output_bytes': 0}
        if args.mode == 'full':
            S.require(args.components is not None, 'Full component acceptance required before full model matrix')
            component = S.read(args.components)
            S.require(component['status'] == 'pass' and component['full_tensor_coverage'], 'Full component contract missing')
            S.require(S.read(args.components.parent / 'candidate-receipt.json') == receipt, 'Component/model candidate identity differs')
            S.write(args.output / 'component-input-identity.json', S.identity(args.components))
            for item in component['routes'].values():
                prior['cpu_seconds'] += item['cpu_seconds']
                prior['wall_seconds'] += item['process_wall_seconds']
                prior['gpu_seconds'] += item['gpu_activity']['activity_union_seconds']
                prior['output_bytes'] += item['output_bytes']
        results, references = {}, {}
        cpu = gpu = 0.
        def check_budget():
            S.require(time.monotonic()-started + prior['wall_seconds'] < budget['batch_wall_seconds'], 'Batch wall budget exhausted')
            S.require(cpu + time.process_time() + prior['cpu_seconds'] < budget['batch_cpu_seconds'], 'Batch CPU budget exhausted')
            S.require(gpu + prior['gpu_seconds'] < budget['batch_gpu_activity_seconds'], 'Batch GPU activity budget exhausted')
            S.require(X.directory_bytes(args.output) + prior['output_bytes'] < budget['batch_output_bytes'], 'Batch output budget exhausted')
        cases = [(key, 'native', kv) for key in frozen for kv in ('native', 'modeled')]
        if args.mode == 'full':
            cases += [(key, route, kv) for key in frozen for route in ('symbolic', 'ptx', 'cuda') for kv in ('native', 'modeled')]
        for key, route, kv in cases:
            check_budget()
            X.unchanged(args.build_dir, receipt)
            label = f'{key}-{route}-{kv}'
            folder = args.output / label
            prompt = S.HERE / ('prompt-44.txt' if key == 'original44' else 'prompt-short.txt')
            process = X.execute(args, [args.build_dir / 'bin/sampled_inventory.exe', args.model, prompt,
                folder, S.HERE / 'inventory.json', '8'], label, folder, require_ggml=True,
                timeout=budget['model_process_wall_seconds'], disk_cap=budget['process_output_bytes'],
                extra_env={'HBFLAB_SAMPLED_ROUTE': route, 'HBFLAB_JOINT_KV': 'paged' if kv == 'modeled' else 'native',
                    'HBFLAB_SAMPLED_PREFILL': str(len(frozen[key]['prompt_tokens'])),
                    'HBFLAB_SAMPLED_INVENTORY': str(S.HERE / 'inventory.json'),
                    'HBFLAB_SAMPLED_PROFILE': str(args.output / 'profile.json'),
                    'HBFLAB_SAMPLED_TOKEN_ONLY': '1', 'LLAMA_GRAPH_REUSE_DISABLE': '1'})
            result = validate_model(folder, inventory, frozen[key], route, kv,
                reference_native=references.get((key, 'native')),
                reference_same_kv=references.get((key, kv)) if route != 'native' else None)
            result['process_wall_seconds'] = process['wall_seconds']
            result['cpu_seconds'] = process['final_metrics']['cpu_seconds']
            result['peak_working_set_bytes'] = max(m['peak_working_set_bytes'] or 0 for m in [*process['metrics'], process['final_metrics']])
            result['output_bytes'] = X.directory_bytes(folder)
            cpu += result['cpu_seconds']
            gpu += result['gpu_activity']['activity_union_seconds']
            results[label] = result
            if route == 'native':
                references[(key, kv)] = folder
            check_budget()
            S.write(args.output / 'progress.json', {'completed': list(results), 'planned': len(cases),
                'elapsed_wall_seconds': time.monotonic()-started, 'child_cpu_seconds': cpu, 'gpu_activity_seconds': gpu})
        X.unchanged(args.build_dir, receipt)
        S.require(len(results) == (16 if args.mode == 'full' else 4), 'Incomplete process matrix')
        S.write(args.output / 'acceptance.json', {'status': 'pass', 'mode': args.mode,
            'model_processes': len(results), 'routes': results, 'common_clock_qualified': False,
            'scope': 'complete representative-weight model and KV functional contracts' if args.mode == 'full'
                else 'diagnostic native-weight references only; excluded from formal matrix',
            'cost': {'elapsed_wall_seconds': time.monotonic()-started, 'child_cpu_seconds': cpu,
                     'coordinator_cpu_seconds': time.process_time(), 'gpu_activity_seconds': gpu,
                     'output_bytes': X.directory_bytes(args.output), 'prior_components': prior}})
    except BaseException:
        S.write(args.output / 'failure.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
