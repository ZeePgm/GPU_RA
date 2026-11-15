"""Audit fresh V3 compiled source paths, tests, fixtures and binary identities."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from verify_source import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--skip-tests', action='store_true')
    args = parser.parse_args()
    joint, build, reports = args.joint_root.resolve(), args.build_root.resolve(), args.reports.resolve()
    source = (joint/'source').resolve()
    configuration_file = reports/'build-configuration.json'
    configuration = json.loads(configuration_file.read_text(encoding='utf-8-sig')) if configuration_file.is_file() else {
        'source': str(source), 'build_root': str(build), 'python': str(Path(sys.executable).resolve()),
        'configuration_record': 'Initial wrapper invocation; final parameterized wrapper will separately revalidate.'}
    if Path(configuration['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Audit interpreter does not match selected Python')
    identities = []
    for kind in ['core', 'model']:
        rows = json.loads((build/kind/'compile_commands.json').read_text(encoding='utf-8'))
        outside, generated, expanded_sources = [], [], set()
        for row in rows:
            path = Path(row['file']).resolve()
            if path.is_relative_to(source):
                expanded_sources.add(path.relative_to(source).as_posix())
                continue
            if path.is_relative_to(build/kind) and 'Unity' in path.parts and path.name.startswith('unity_') and path.suffix == '.cxx':
                included = re.findall(r'^#include "([^"]+)"', path.read_text(encoding='utf-8'), re.M)
                if not included:
                    raise ValueError('Generated unity unit has no sealed includes: '+str(path))
                for child in included:
                    original = Path(child).resolve()
                    if not original.is_relative_to(source) or not original.is_file():
                        raise ValueError('Unity source outside sealed snapshot: '+str(original))
                    expanded_sources.add(original.relative_to(source).as_posix())
                generated.append({'file': path.relative_to(build).as_posix(),
                                  'sealed_includes': [Path(child).resolve().relative_to(source).as_posix() for child in included]})
            else:
                outside.append(str(path))
        if outside:
            raise ValueError('Compiled sources outside sealed snapshot: '+repr(outside))
        ninja = (build/kind/'build.ninja').read_text(encoding='utf-8')
        bad = re.findall(r'[^\r\n]*(?:build-r2-final|build-r3-current|20261112-dual-line-r1[/\\]build)[^\r\n]*', ninja)
        if bad:
            raise ValueError('Historical library/build dependency remains: '+repr(bad))
        identities.append({'project': kind, 'compile_source_records': len(rows), 'source_paths_outside_sealed_tree': outside,
                           'historical_build_dependencies': bad, 'generated_unity_units': generated,
                           'expanded_sealed_source_files': sorted(expanded_sources),
                           'scope': 'Declared compile commands; Unity inputs are expanded back to sealed source files.'})
        cache = (build/kind/'CMakeCache.txt').read_text(encoding='utf-8')
        selected_python = re.search(r'^Python3_EXECUTABLE:[^=]+=(.*)$', cache, re.M)
        identities[-1]['cmake_python'] = selected_python[1] if selected_python else None
        if configuration_file.is_file() and (not selected_python or Path(selected_python[1]).resolve() != Path(configuration['python']).resolve()):
            raise ValueError('CMake Python does not match selected interpreter')
    evidence = {}
    if not args.skip_tests:
        text = (reports/'ctest-core.log').read_text(encoding='utf-8-sig')
        matched = re.search(r'100% tests passed, 0 tests failed out of (\d+)', text)
        if not matched:
            raise ValueError('Core test completion missing')
        if int(matched[1]) != 608:
            raise ValueError('Expected complete 608-test CPU core suite')
        fixture = json.loads((reports/'b2-fixture.json').read_text())
        assert fixture['status'] == 'pass' and fixture['drained'] and len(fixture['cases']) == 28
        assert fixture['peak_resident_fragments'] <= 256 and fixture['refill_overlap'] > 0
        assert fixture['checks'] == sum(row['bytes']//16 for row in fixture['cases'])
        assert fixture['requests'] == sum(row['bytes']//512 for row in fixture['cases'])
        assert all(row['requests'] == row['bytes']//512 and row['K'] == 262144 for row in fixture['cases'])
        assert {(row['W'], row['bytes']) for row in fixture['cases']} == {
            (w, size) for w in (4096,8192,16384,32768) for size in (512,3584,4096,4608,261632,262144,262656)}
        assert 'PASS media-only shared runtime, sparse base, read/write/read, final drain' in (reports/'media-only.log').read_text()
        assert 'terminal status 1' in (reports/'b2-negative.log').read_text()
        assert not (reports/'b2-negative.json').exists()
        evidence = {'cpu_core_tests_passed': int(matched[1]), 'B2_GPU_cases': 28, 'B2_checks': fixture['checks'],
                    'B2_range_negative_exit': 23, 'B2_refill_overlap': fixture['refill_overlap'],
                    'shared_media_read_write_read_drain': 'pass'}
    binaries = []
    bindir = build/'model/bin'
    required = ['realgpu_llama.exe', 'joint_b2_fixture.exe', 'joint_media_gpu.exe', 'ggml-cuda.dll', 'llama.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml.dll']
    for name in required:
        if not (bindir/name).is_file():
            raise ValueError('Required new artifact missing: '+name)
    for p in sorted({*(bindir/name for name in required), *bindir.glob('*.dll')}):
        with p.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        binaries.append({'path': str(p.relative_to(build)), 'bytes': p.stat().st_size, 'sha256': digest})
    historical_source = verify(joint)
    if historical_source['status'] != 'pass':
        raise ValueError('Historical source changed during fresh build')
    result = {'status': 'pass', 'historical_source': historical_source, 'source_builds': identities,
              'tests': evidence, 'new_binaries': binaries, 'old_builds_used': False,
              'build_configuration': configuration,
              'tests_executed': not args.skip_tests,
              'final_parameterized_entry': configuration_file.is_file(),
              'build_root': str(build), 'source_root': str(source),
              'model_GPU_experiments_executed_by_build_entry': False}
    identity_file = joint/'source-bundle/manifest.json'
    result['historical_source_manifest'] = {'path': str(identity_file), 'bytes': identity_file.stat().st_size,
        'sha256': hashlib.sha256(identity_file.read_bytes()).hexdigest()}
    (reports/'build-acceptance.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if configuration_file.is_file():
        (build/'build-acceptance.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
