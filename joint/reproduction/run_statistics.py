"""Prepare or explicitly execute the frozen 142-process V3 timing design."""
import argparse
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import runtime_common as R

SCIENCE_FIELDS = ['scope', 'write_source', 'audit', 'primary_endpoint', 'secondary_endpoints',
                  'primary_contrasts', 'interval', 'AA_noise', 'gate', 'interaction', 'order',
                  'seed', 'processes', 'warmups_per_group_per_scale', 'blocks_per_scale',
                  'AA_pairs_per_scale', 'resource_policy', 'stopping', 'timeout_s']


def schedule():
    rng = random.Random(131113)
    rows, orders = [], {}
    for scale in ['F1', 'F2']:
        warm = list(R.MODES)
        rng.shuffle(warm)
        aa = [False, True] * 2
        rng.shuffle(aa)
        blocks = []
        for _ in range(4):
            order = list(R.MODES)
            rng.shuffle(order)
            blocks.extend([order, list(reversed(order))])
        rng.shuffle(blocks)
        orders[scale] = {'warmup': warm, 'AA_reverse': aa, 'blocks': blocks}
        rows += [{'scale': scale, 'stage': 'warmup', 'block': 0, 'mode': mode, 'replicate': 'w'} for mode in warm]
        for block, reverse in enumerate(aa):
            rows += [{'scale': scale, 'stage': 'AA', 'block': block, 'mode': 'joint-modeled', 'replicate': rep}
                     for rep in (['b', 'a'] if reverse else ['a', 'b'])]
        for block, modes in enumerate(blocks):
            rows += [{'scale': scale, 'stage': 'confirm', 'block': block, 'mode': mode, 'replicate': 'c'} for mode in modes]
    R.require(len(rows) == 142, 'Schedule size differs from frozen design')
    for index, row in enumerate(rows, 1):
        row['index'] = index
        row['id'] = f"{index:03d}-{row['scale']}-{row['stage']}-{row['block']:02d}-{row['replicate']}-{row['mode']}"
    return rows, orders


def bind(output, protocol):
    module = R.statistical_module()
    settings = protocol['runtime']
    R.validate_build(Path(settings['build_dir']), Path(settings['cuda_root']))
    module.ROOT, module.OUT, module.SEALED = Path(settings['resource_root']), output, output / 'inputs'
    module.C = R.configure_correctness(module.SEALED, output, Path(settings['build_dir']),
                                      Path(settings['cuda_root']), settings['gpu_device'])
    module.P = R.process_probe()
    references = sorted((module.SEALED / 'results/correctness-001').rglob('*'))
    reference_files = [path for path in references if path.is_file()]
    reference_receipts = [output / (scale + '-reference-acceptance.json') for scale in ['F1','F2']]
    reference_receipts += [output / 'gold-recomputed' / (scale + '-acceptance.json') for scale in ['F1','F2']]
    module.identities = lambda: R.identities(module.C, [*reference_files, *reference_receipts, Path(__file__),
                                  R.JOINT / 'statistical-batch/scripts/run.py', R.JOINT / 'statistical-batch/protocol.json'])
    module.probe = lambda allowed=None: R.probe(Path(settings['resource_root']), allowed)
    return module


def prepare(args):
    R.validate_inputs(args)
    output = args.output.resolve()
    R.require(not output.exists(), 'Use a fresh output directory')
    source = args.correctness_dir.resolve()
    cases = source / 'cases' if (source / 'cases').is_dir() else source
    for scale in ['F1', 'F2']:
        acceptance = R.read(cases / (scale + '-acceptance.json'))
        R.require(acceptance['status'] == 'pass' and acceptance['same_arithmetic_logits_bitwise'] and
                  acceptance['kv_content_sha256_equal'], 'Correctness reference not accepted: ' + scale)
        for row in acceptance['comparisons']:
            R.require(R.sha(cases / row['case'] / 'logits.f32') == row['logits_sha256'],
                      'Correctness reference logits changed after acceptance')
    output.mkdir(parents=True)
    try:
        R.make_inputs(output / 'inputs', args.model)
        validator = R.configure_correctness(output / 'inputs', output, args.build_dir, args.cuda_root, args.gpu_device)
        # Recompute all cross-path byte/KV/tolerance comparisons with unchanged function bodies.
        # Writes are redirected exclusively to this new prepare directory.
        original_out, original_write = validator.OUT, validator.write
        validator.OUT = cases.parent
        validator.write = lambda path, value: R.write(output / 'gold-recomputed' / Path(path).name, value)
        try:
            for scale in ['F1','F2']:
                validator.compare_scale(scale, cases.name)
                R.require(R.read(output / 'gold-recomputed' / (scale + '-acceptance.json')) ==
                          R.read(cases / (scale + '-acceptance.json')), 'Reference comparison drift: ' + scale)
        finally:
            validator.OUT, validator.write = original_out, original_write
        gold = output / 'inputs/results/correctness-001'
        for scale in ['F1', 'F2']:
            shutil.copy2(cases / (scale + '-acceptance.json'), output / (scale + '-reference-acceptance.json'))
            for label, (weights, kv, mode) in R.MODES.items():
                original = cases / (scale + '-' + label)
                validator.validate_case(original, scale, weights, kv, mode)
                summary = R.read(original / 'summary.json')
                R.require(not summary['observed'] and summary['sequence_count'] == 1 and not summary['scenario'],
                          'Unsupported reference workload')
                target = gold / (scale + '-' + label)
                target.mkdir(parents=True)
                for name in ['summary.json', 'logits.f32', 'kv-content.json']:
                    if (original / name).is_file():
                        shutil.copy2(original / name, target / name)
        rows, orders = schedule()
        # Reuse all frozen scientific settings, replace only path/identity/time/authorization wiring.
        old = R.read(R.JOINT / 'statistical-batch/protocol.json')
        protocol = {key: old[key] for key in SCIENCE_FIELDS}
        protocol.update(version=2, frozen_utc=R.stamp(), task_start_utc=R.stamp(),
                        authorization='New execution requires explicit run action; this prepare launched zero model processes.',
                        basis='Historical fixed 142-process design; newly bound inputs, references and source-built binaries.',
                        historical_protocol_sha256=R.sha(R.JOINT / 'statistical-batch/protocol.json'),
                        modes=R.MODES, schedule=rows, orders=orders,
                        sealed_source=str(R.JOINT / 'source'), model_sha256=R.MODEL_SHA,
                        runner_sha256=R.sha(__file__),
                        runtime={'model': str(args.model.resolve()), 'cuda_root': str(args.cuda_root.resolve()),
                                 'build_dir': str(args.build_dir.resolve()), 'gpu_device': args.gpu_device,
                                 'resource_root': str(args.resource_root.resolve())})
        module = bind(output, protocol)
        protocol['dependencies'] = module.identities()
        R.write(output / 'protocol.json', protocol)
        for name in ['raw', 'reports', 'provenance']:
            (output / name).mkdir(exist_ok=True)
        module.verify_identity(protocol)
        snapshot = module.probe()
        R.write(output / 'provenance/prepare-resource-preflight.json', snapshot)
        R.require(not snapshot['conflicts'], 'Resource conflict during protocol preflight')
        R.require(protocol['schedule'] == old['schedule'] and protocol['orders'] == old['orders'],
                  'New schedule differs from historical frozen order')
        for row in rows:
            module.numerical(gold / (row['scale'] + '-' + row['mode']), row['scale'], row['mode'])
        R.write(output / 'prepare-validation.json', {'status': 'pass', 'processes': 142,
                 'model_processes_executed': 0, 'schedule_matches_frozen_design': True,
                 'protocol_sha256': R.sha(output / 'protocol.json'), 'cross_path_gold_recomputed': True,
                 'reference_groups_validated': 14, 'reference_schedule_numerical_checks': len(rows),
                 'new_dependencies_frozen': len(protocol['dependencies']),
                 'source_reference': str(cases), 'host_resource_visibility': 'pass'})
        R.write(output / 'status.json', {'state': 'prepared', 'completed': 0, 'total': 142, 'utc': R.stamp()})
        print('Prepared fixed 142-process protocol and 14 reference groups; zero model processes launched')
    except BaseException:
        R.write(output / 'failure.json', {'utc': R.stamp(), 'stage': 'prepare', 'traceback': traceback.format_exc()})
        raise


def verify_frozen_protocol(output, protocol):
    receipt = R.read(output / 'prepare-validation.json')
    R.require(receipt['status'] == 'pass' and receipt['protocol_sha256'] == R.sha(output / 'protocol.json'),
              'Protocol changed after successful prepare')
    R.require(receipt['model_processes_executed'] == 0 and receipt['cross_path_gold_recomputed'],
              'Prepare receipt lacks required reference validation')
    old = R.read(R.JOINT / 'statistical-batch/protocol.json')
    R.require(protocol['historical_protocol_sha256'] == R.sha(R.JOINT / 'statistical-batch/protocol.json'),
              'Frozen historical protocol identity drift')
    R.require(all(protocol[key] == old[key] for key in SCIENCE_FIELDS), 'Scientific protocol settings changed')
    expected_rows, expected_orders = schedule()
    R.require(protocol['schedule'] == expected_rows and protocol['orders'] == expected_orders,
              'Fixed 142-process schedule/order changed')
    R.require({key: tuple(value) for key,value in protocol['modes'].items()} == R.MODES,
              'Protocol group assignment changed')
    R.require(protocol['model_sha256'] == R.MODEL_SHA and protocol['runner_sha256'] == R.sha(__file__),
              'Protocol model/coordinator identity changed')


def execute(args):
    output = args.output.resolve()
    protocol = R.read(output / 'protocol.json')
    verify_frozen_protocol(output, protocol)
    module = bind(output, protocol)
    module.verify_identity(protocol)
    R.require(not (output / 'run-start.json').exists() and not (output / 'samples.json').exists(),
              'This fixed batch cannot be restarted or samples replaced; prepare a new directory')
    R.require(not any((output / 'raw').iterdir()), 'Raw directory already contains a model attempt')
    R.require(R.read(output / 'prepare-validation.json')['status'] == 'pass', 'Prepare validation missing')
    started = R.stamp()
    R.write(output / 'run-start.json', {'utc': started, 'protocol_sha256': R.sha(output / 'protocol.json'),
                                      'fixed_process_budget': 142})
    samples, results = [], {}
    start = time.monotonic()
    try:
        initial = module.probe()
        R.write(output / 'provenance/host-processes-before.json', module.P.processes())
        R.require(not initial['conflicts'], 'Resource conflict before fixed batch')
        module.gpu_snapshot('gpu-before-F1.csv')
        for row in protocol['schedule']:
            if row['scale'] == 'F2' and 'F1' not in results:
                results['F1'] = module.analyze('F1', samples)
                module.verify_identity(protocol)
                module.gpu_snapshot('gpu-before-F2.csv')
            sample = module.run_one(row, protocol, len(samples))
            samples.append(sample)
            R.write(output / 'samples.json', samples)
            R.write(output / 'status.json', {'state': 'running', 'completed': len(samples), 'total': 142,
                     'last_pass': row['id'], 'utc': R.stamp(), 'run_elapsed_s': time.monotonic()-start})
        results['F2'] = module.analyze('F2', samples)
        module.verify_identity(protocol)
        R.validate_build(Path(protocol['runtime']['build_dir']), Path(protocol['runtime']['cuda_root']))
        module.gpu_snapshot('gpu-after.csv')
        R.write(output / 'provenance/host-processes-after.json', module.P.processes())
        reporting_protocol = dict(protocol, task_start_utc=started)
        module.report(reporting_protocol, samples, results)
        module.event('complete', accepted=len(samples), run_elapsed_s=time.monotonic()-start)
    except BaseException:
        R.write(output / 'failure.json', {'utc': R.stamp(), 'stage': 'run', 'completed': len(samples),
                                       'traceback': traceback.format_exc()})
        R.write(output / 'status.json', {'state': 'failed', 'completed': len(samples), 'total': 142, 'utc': R.stamp()})
        raise


def check(args):
    output = args.output.resolve()
    protocol = R.read(output / 'protocol.json')
    verify_frozen_protocol(output, protocol)
    module = bind(output, protocol)
    module.verify_identity(protocol)
    R.require(protocol['schedule'] == schedule()[0], 'Schedule drift')
    snap = module.probe()
    R.require(not snap['conflicts'], 'Host resource conflict')
    print('PASS: frozen input/binary/helper/reference identities, 142-row schedule and host resource preflight')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('prepare', help='Freeze protocol and validate references; launches zero models')
    R.add_arguments(command)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--correctness-dir', type=Path, required=True,
                         help='Fresh full F1/F2 correctness output, or its cases directory')
    for name in ['run', 'check']:
        command = sub.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': execute, 'check': check}[args.action](args)


if __name__ == '__main__':
    main()
