"""Combine verified P0-B model and boundary evidence without pooling their outcomes."""
import argparse
import csv
import json
from pathlib import Path
from xml.etree import ElementTree

from run import BUILD, MODES, ROOT, SOURCE, check_outputs, read, require, sha, verify_build, write
from run_llama_integrated import rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    parser.add_argument('--component-build', type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    identity = read(batch / 'build-identity.json')
    require(verify_build(BUILD) == identity, 'Model build identity changed')
    for source in identity['sources']:
        relative = Path(source['path']).relative_to(SOURCE)
        require(sha(ROOT / relative) == source['sha256'], f'Workspace source differs: {relative}')
    matrix = read(batch / 'matrix-result.json')
    expected = {f'{r}-{length}-{n}' for r in 'ABC' for length in ('short', 'long') for n in (32, 64)}
    require({c['name'] for c in matrix['cells']} == expected and len(matrix['cells']) == len(expected),
            'Incomplete model grid')
    table = []
    model_identities = {}
    for cell in matrix['cells']:
        row = dict(case=cell['name'], status=cell['status'], evidence=cell['evidence'],
                   prompt_tokens='', ubatch='', cross_ubatch='', cross_page='',
                   requested_tokens=cell['tokens'], cross_page_rows='')
        if cell['status'] == 'pass':
            folder = Path(cell['folder'])
            result = check_outputs(folder, cell['tokens'])
            profile = read(folder / 'profile.json')
            model = Path(profile['model_path'])
            conversion = Path(profile['conversion'])
            if model not in model_identities:
                recorded = json.loads(conversion.read_text(encoding='utf-8-sig'))
                require(sha(model) == recorded['gguf_sha256'] == profile['gguf_sha256'] and
                        recorded['llama_commit'] == profile['framework_commit'] and
                        recorded['source']['revision'] == profile['revision'],
                        f'Model or conversion identity changed: {model}')
                model_identities[model] = dict(model=str(model), gguf_sha256=recorded['gguf_sha256'],
                                               conversion=str(conversion), conversion_sha256=sha(conversion))
            require(model_identities[model]['gguf_sha256'] == profile['gguf_sha256'] and
                    model_identities[model]['conversion'] == str(conversion),
                    f'Conflicting model identity: {model}')
            native_text = (folder / 'native/generated.txt').read_bytes()
            for mode in MODES:
                mode_folder = folder / mode
                tokens = read(mode_folder / 'summary.json')['generated_tokens']
                steps = rows(mode_folder / 'steps.jsonl')
                require([(step['step'], step['token']) for step in steps] == list(enumerate(tokens)),
                        f'Incomplete token delivery: {cell["name"]}/{mode}')
                require((mode_folder / 'generated.txt').read_bytes() == native_text,
                        f'Decoded output differs: {cell["name"]}/{mode}')
            cross_page_rows = result['comparisons']['modeled']['kv']['cross_page_rows']
            crossing_possible = any((i * profile['row_bytes']) % 4096 + profile['row_bytes'] > 4096
                                    for i in range(profile['block_tokens']))
            row.update(prompt_tokens=result['prompt_tokens'], ubatch=profile['ubatch'],
                       cross_ubatch='pass' if result['prompt_tokens'] > profile['ubatch'] else
                       'fail' if cell['length'] == 'long' else 'outside_short_case',
                       cross_page_rows=cross_page_rows,
                       cross_page='pass' if cross_page_rows > 0 else
                       'fail' if crossing_possible else 'unsupported')
        table.append(row)

    cells = {c['name']: c for c in matrix['cells']}
    for length in ('short', 'long'):
        first = cells[f'A-{length}-32']
        second = cells[f'A-{length}-64']
        if first['status'] != 'pass' or second['status'] != 'pass':
            continue
        for mode in MODES:
            early = Path(first['folder']) / mode
            extended = Path(second['folder']) / mode
            logits = (early / 'logits.f32').read_bytes()
            with (extended / 'logits.f32').open('rb') as file:
                require(file.read(len(logits)) == logits, f'A-{length} logits prefix changed: {mode}')
            s32, s64 = read(early / 'summary.json'), read(extended / 'summary.json')
            require(s64['prompt_tokens'] == s32['prompt_tokens'] and
                    s64['generated_tokens'][:32] == s32['generated_tokens'],
                    f'A-{length} token prefix changed: {mode}')

    test_file = batch / 'components.xml'
    tests = []
    for test in ElementTree.parse(test_file).iter('testcase'):
        tests.append(dict(case=test.attrib['name'], evidence='synthetic_component_test',
                          status='fail' if len(test.findall('failure')) or len(test.findall('error')) else
                          'unsupported' if len(test.findall('skipped')) else 'pass'))
    expected_tests = {'lab_capacity_failure', 'lab_capacity_kv_failure', 'lab_capacity_kv_capacity',
                      'lab_capacity_kv_cross_page', 'lab_kv_transaction_failure', 'lab_kv_transaction_abort',
                      'lab_incremental_failure_drain', 'lab_incremental_partial_initialization',
                      'host_completion_dependency_cancel', 'host_completion_timeout_retry'}
    require({t['case'] for t in tests} == expected_tests and len(tests) == len(expected_tests),
            'Component set incomplete')
    component_build = args.component_build.resolve()
    executables = [component_build / f'{n}.exe' for n in
                   ('hbflab_capacity_contract', 'hbflab_kv_transaction', 'hbflab_incremental_bridge')]
    executables.append(component_build / 'eventlab/hbf_eventlab_host_completion_tests.exe')
    boundary = read(batch / 'boundaries/results.json')
    require(len(boundary) == 14 and len({c['case'] for c in boundary}) == 14, 'Boundary set incomplete')
    with (batch / 'checklist.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    failed = (any('fail' in (c['status'], c['cross_page'], c['cross_ubatch']) for c in table)
              or any(c['status'] == 'fail' for c in tests + boundary))
    write(batch / 'acceptance.json', dict(
        status='fail' if failed else 'pass', backend=identity['backend'],
        model_build_identity_sha256=sha(batch / 'build-identity.json'),
        sealed_source_records_matched_to_workspace=len(identity['sources']),
        model_identities=list(model_identities.values()),
        matrix=table, boundaries=boundary, components=tests,
        component_build=dict(cache_sha256=sha(component_build / 'CMakeCache.txt'),
                             executables=[dict(path=str(p), sha256=sha(p)) for p in executables],
                             junit_sha256=sha(test_file))))


if __name__ == '__main__':
    main()
