"""Candidate registration and unchanged historical F1/F2 scientific regressions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import stage1 as S
R = S.R


def run_fixture(args, command, name, expected_exit=0, timeout=600):
    command = list(map(str, command))
    before = R.probe(args.resource_root)
    S.require(not before['conflicts'], 'Resource conflict before fixture: ' + repr(before))
    frozen = R.correctness_module()
    frozen.BIN, frozen.CUDA = args.build_dir / 'bin', args.cuda_root
    record = {'command': command, 'resource_probes': [before], 'expected_exit': expected_exit}
    start = time.monotonic()
    print('START', name, flush=True)
    with (args.output / (name + '.log')).open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(list(map(str, command)), stdout=log, stderr=log, env=frozen.environment())
        try:
            next_probe = 0
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed >= next_probe:
                    snap = R.probe(args.resource_root, proc.pid)
                    record['resource_probes'].append(snap)
                    S.require(not snap['conflicts'], 'Resource conflict during fixture: ' + repr(snap))
                    next_probe = elapsed + 5
                S.require(elapsed < timeout, 'Fixture timeout: ' + name)
                time.sleep(.1)
        except BaseException:
            record['error'] = traceback.format_exc()
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            raise
        finally:
            record.update(returncode=proc.poll(), wall_seconds=time.monotonic() - start)
            S.write(args.output / (name + '.process.json'), record)
    S.require(proc.returncode == expected_exit, 'Fixture exit mismatch: ' + name)
    print('PASS', name, record['wall_seconds'], flush=True)


def unchanged(build, receipt):
    S.require(S.source_identity() == receipt['source'], 'Candidate source drift')
    S.require(S.audit_build(build) == receipt['build'], 'Candidate build drift')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--part', choices=['registration', 'regression'], required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--cuda-root', type=Path, required=True)
    ap.add_argument('--resource-root', type=Path, required=True)
    ap.add_argument('--build-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.build_dir = args.build_dir.resolve()
    args.output = args.output.resolve()
    S.require(not args.output.exists(), 'Use a fresh output directory')
    args.output.mkdir(parents=True)
    receipt = S.read(args.build_dir / 'candidate-receipt.json')
    try:
        unchanged(args.build_dir, receipt)
        S.require(S.sha(args.model) == S.MODEL_SHA, 'Model input identity mismatch')
        S.write(args.output / 'candidate-receipt.json', receipt)
        core = args.build_dir / 'core-tests.log'
        S.require('100% tests passed, 0 tests failed out of 608' in core.read_text(encoding='utf-8-sig'), 'Complete CPU regression missing')
        S.write(args.output / 'cpu-regression.json', {'status': 'pass', 'tests': 608, 'log': S.identity(core)})
        if args.part == 'registration':
            inventory_path = S.HERE / 'inventory.json'
            inventory = S.read(inventory_path)
            for route in ('symbolic', 'ptx', 'cuda'):
                report = args.output / (route + '-registration.json')
                run_fixture(args, [args.build_dir / 'bin/sampled_registration.exe', inventory_path, route, report], route + '-registration')
                result = S.read(report)
                S.require(result['status'] == 'pass' and result['drained'] and result['selected_ranges'] == 14 and
                          result['selected_bytes'] == inventory['selected_bytes'] and result['negative_checks'] >= 21 and
                          result['kv_base'] == inventory['kv_base'] and result['kv_bytes'] == inventory['kv_reserved_bytes'] and
                          result['media_requests'] == 3 and result['peak_resident_fragments'] <= 256 and
                          result['readonly_alignment'] == 32768, 'Registration contract failed: ' + route)
            run_fixture(args, [args.build_dir / 'bin/joint_b2_fixture.exe', args.output / 'b2-fixture.json', 'normal'], 'b2-fixture')
            b2 = S.read(args.output / 'b2-fixture.json')
            S.require(b2['status'] == 'pass' and b2['drained'] and len(b2['cases']) == 28 and
                      b2['peak_resident_fragments'] <= 256 and b2['refill_overlap'] > 0, 'B2 fixture regression')
            S.require(b2['checks'] == sum(r['bytes'] // 16 for r in b2['cases']) and
                      b2['requests'] == sum(r['bytes'] // 512 for r in b2['cases']), 'B2 byte/request checks')
            run_fixture(args, [args.build_dir / 'bin/joint_b2_fixture.exe', args.output / 'b2-negative.json', 'negative'], 'b2-negative', 23)
            S.require(not (args.output / 'b2-negative.json').exists(), 'Negative B2 produced success')
            run_fixture(args, [args.build_dir / 'bin/joint_media_gpu.exe'], 'shared-media')
            S.require('PASS media-only shared runtime, sparse base, read/write/read, final drain' in
                      (args.output / 'shared-media.log').read_text(encoding='utf-8-sig'), 'Shared media regression')
            S.write(args.output / 'acceptance.json', {'status': 'pass', 'stage': '2 registration', 'routes': ['symbolic', 'ptx', 'cuda'],
                    'selected_ranges': 14, 'core_tests': 608, 'B2_cases': 28, 'B2_checks': b2['checks'],
                    'B2_requests': b2['requests'], 'shared_media': 'pass',
                    'scope': 'Common registration, configuration, resource and drain contracts; route reads and model integration pending'})
        else:
            R.make_inputs(args.output / 'inputs', args.model)
            module = R.configure_correctness(args.output / 'inputs', args.output, args.build_dir, args.cuda_root)
            for scale in ('F1', 'F2'):
                if scale == 'F2':
                    S.require(S.read(args.output / 'cases/F1-acceptance.json')['status'] == 'pass', 'F1 not accepted')
                for label in R.ALL_MODES:
                    unchanged(args.build_dir, receipt)
                    R.run_case(module, scale, label, args.resource_root)
                module.compare_scale(scale, 'cases')
            S.write(args.output / 'acceptance.json', {'status': 'pass', 'stage': '2 historical regression', 'model_processes': 16,
                    'scales': ['F1', 'F2'], 'validator': S.identity(S.ROOT / 'joint/scripts/run_correctness.py'),
                    'scope': 'Unchanged numerical, byte, KV, dependency and drain contracts; not a new sampled-route model verdict'})
        unchanged(args.build_dir, receipt)
    except BaseException:
        S.write(args.output / 'failure.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
