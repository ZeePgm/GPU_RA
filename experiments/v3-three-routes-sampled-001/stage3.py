"""One pilot or full independent component process per readonly route."""
import argparse
from pathlib import Path
import traceback
import sampled_run as X
import stage1 as S
from reader_validation import validate_reader


def plan(inventory, mode):
    selected = [t for t in inventory['all_tensors'] if t['selected']]
    spans = []
    def add(name, offset, size, kind):
        spans.append({'tensor': name, 'offset': offset, 'bytes': size, 'kind': kind})
    if mode == 'pilot':
        add('blk.10.attn_q_norm.weight', 0, 512, 'small_F32_vector')
        add('blk.10.attn_k.weight', 0, 2097152, 'one_complete_F16_matrix')
        for row in (0, 17, 17, 151935):
            add('token_embd.weight', row * 2048, 2048, 'actual_embedding_row_with_repeat')
        add('output.weight', 32768, 262144, 'bounded_output_fragment')
        add('blk.10.ffn_norm.weight', 4, 508, 'F32_nonzero_offset_tail')
        add('blk.10.ffn_gate.weight', 2, 1026, 'F16_nonzero_offset_tail')
    else:
        for tensor in selected:
            add(tensor['name'], 0, tensor['bytes'], 'all_embedding_rows_once' if tensor['name'] == 'token_embd.weight' else 'complete_tensor')
            element = 4 if tensor['dtype'] == 'F32' else 2
            add(tensor['name'], element, min(tensor['bytes'] - element, 1026 // element * element), 'typed_nonzero_tail')
        for size in (512, 3584, 4096, 4608, 261632, 262144, 262656):
            add('blk.10.attn_k.weight', 512, size, 'request_page_window_boundaries')
        for row in (17, 17, 151935, 0):
            add('token_embd.weight', row * 2048, 2048, 'repeat_row_and_buffer_reuse')
    return {'schema': 1, 'scope': 'pilot cost and selected boundaries' if mode == 'pilot' else 'independent complete tensor and all embedding row coverage', 'spans': spans}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['pilot', 'full'], required=True)
    for key in ('model', 'cuda-root', 'resource-root', 'build-dir', 'output'):
        ap.add_argument('--' + key, type=Path, required=True)
    args = ap.parse_args()
    args.output, args.build_dir = args.output.resolve(), args.build_dir.resolve()
    S.require(not args.output.exists(), 'Use a new output directory')
    args.output.mkdir(parents=True)
    try:
        receipt = S.read(args.build_dir / 'candidate-receipt.json')
        X.unchanged(args.build_dir, receipt)
        S.require(S.sha(args.model) == S.MODEL_SHA, 'Model identity mismatch')
        inventory = S.read(S.HERE / 'inventory.json')
        request_plan = plan(inventory, args.mode)
        S.write(args.output / 'request-plan.json', request_plan)
        S.write(args.output / 'candidate-receipt.json', receipt)
        results = {}
        for route in ('symbolic', 'ptx', 'cuda'):
            X.unchanged(args.build_dir, receipt)
            child = args.output / route
            process = X.execute(args, [args.build_dir / 'bin/sampled_reader_fixture.exe', S.HERE / 'inventory.json',
                               args.model, args.output / 'request-plan.json', child, route], route, child,
                               timeout=900 if args.mode == 'pilot' else 1800)
            bindings = S.read(child / 'bindings.json')
            coverage = validate_reader(child / 'reader', request_plan['spans'], inventory, route,
                                       {t['name']: t['address'] for t in bindings})
            fixture = S.read(child / 'summary.json')
            S.require(fixture['status'] == 'pass' and fixture['drained'] and fixture['negative_checks'] == 8
                      and fixture['selected_ranges'] == 14 and fixture['peak_resident_fragments'] <= 256, 'Component lifecycle/negative contract')
            gpu = X.gpu_activity(child)
            results[route] = {'fixture': fixture, 'independent_coverage': coverage, 'gpu_activity': gpu,
                              'process_wall_seconds': process['wall_seconds'], 'cpu_seconds': process['final_metrics']['cpu_seconds'],
                              'peak_working_set_bytes': max(m['peak_working_set_bytes'] or 0 for m in [*process['metrics'], process['final_metrics']]),
                              'peak_counter_available_after_exit': process['final_metrics']['memory_observed'],
                              'OS_observation_interval_seconds': 5,
                              'output_bytes': process['output_bytes']}
            S.write(args.output / (route + '-acceptance.json'), results[route])
        X.unchanged(args.build_dir, receipt)
        S.write(args.output / 'acceptance.json', {'status': 'pass', 'mode': args.mode, 'routes': results,
                'full_tensor_coverage': args.mode == 'full', 'model_contracts': 'pending stage4',
                'common_clock_qualified': False, 'scope': request_plan['scope']})
    except BaseException:
        S.write(args.output / 'failure.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
