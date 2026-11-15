"""Run fresh V3 F1/F2 correctness matrices against the unchanged contracts."""
import argparse
from pathlib import Path
import traceback
import runtime_common as R


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    R.add_arguments(parser)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', nargs='+', choices=['F1', 'F2'], default=['F1', 'F2'])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    R.require(args.scales in (['F1'], ['F1', 'F2']), 'F2 requires this fresh batch to pass F1 first')
    R.validate_inputs(args)
    output = args.output.resolve()
    R.require(not output.exists(), 'Use a fresh output directory')
    output.mkdir(parents=True)
    R.make_inputs(output / 'inputs', args.model)
    module = R.configure_correctness(output / 'inputs', output, args.build_dir, args.cuda_root, args.gpu_device)
    dependency_rows = R.identities(module, [Path(__file__)])
    R.write(output / 'dependencies.json', dependency_rows)
    R.write(output / 'protocol.json', {'schema': 1, 'created_utc': R.stamp(), 'kind': 'fresh correctness validation',
             'scales': args.scales, 'modes': R.ALL_MODES, 'outputs': 8, 'input_tokens': 44,
             'model_sha256': R.MODEL_SHA, 'prompt_sha256': R.PROMPT_SHA,
             'numerical_policy': 'Unchanged frozen V3 validator; same arithmetic bitwise, paged/native max_abs<=0.2 and RMS<=0.025',
             'source_build': str(args.build_dir.resolve()), 'statistical_performance_confirmation': False,
             'resource_root': str(args.resource_root.resolve())})
    if args.prepare_only:
        R.write(output / 'status.json', {'status': 'prepared', 'model_processes_executed': 0})
        print('Prepared', len(args.scales)*8, 'fresh correctness processes; none launched')
        return
    completed = 0
    try:
        for scale in args.scales:
            if scale == 'F2':
                R.require(R.read(output / 'cases/F1-acceptance.json')['status'] == 'pass', 'F1 not accepted')
            for label in R.ALL_MODES:
                R.verify_identities(dependency_rows)
                R.run_case(module, scale, label, args.resource_root)
                completed += 1
                R.write(output / 'status.json', {'status': 'running', 'completed': completed, 'utc': R.stamp()})
            # The unchanged comparison checks all eight paths and complete KV content.
            module.compare_scale(scale, 'cases')
            R.verify_identities(dependency_rows)
        R.validate_inputs(args)
        R.write(output / 'status.json', {'status': 'pass', 'completed': completed, 'utc': R.stamp(),
                'statistical_performance_confirmation': False})
    except BaseException:
        R.write(output / 'failure.json', {'utc': R.stamp(), 'completed': completed, 'traceback': traceback.format_exc()})
        R.write(output / 'status.json', {'status': 'failed', 'completed': completed, 'utc': R.stamp()})
        raise


if __name__ == '__main__':
    main()
