"""Recheck restored V3 migration records without executing a model, DLL or GPU API."""
import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path, PureWindowsPath
import random
from types import SimpleNamespace
import types
import numpy as np
import replay_historical as H

MODEL_SHA = H.MODEL_SHA
PROMPT_SHA = H.PROMPT_SHA
MODES = {**H.MODES, 'joint-host': H.MODES['joint-modeled']}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return H.read(path)


def sha(path):
    return H.sha(path)


def original_to_restored(root, original):
    parts = PureWindowsPath(original).parts
    require('joint' in parts, 'Frozen dependency outside the archived joint subtree: ' + original)
    suffix = parts[parts.index('joint'):]
    require('..' not in suffix, 'Unsafe frozen dependency path')
    return root.joinpath(*suffix)


def windows_key(path):
    return PureWindowsPath(str(path)).as_posix().lower()


def verify_rows(root, rows):
    seen = set()
    for row in rows:
        require(row['path'] not in seen, 'Duplicate dependency record')
        seen.add(row['path'])
        path = original_to_restored(root, row['path'])
        require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Archived dependency mismatch: ' + row['path'])
    return len(rows)


def functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in nodes} == set(names), 'Missing frozen function body')
    module = types.ModuleType('fresh_offline_selected_functions')
    module.__dict__.update(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


def validate_build_receipt(root):
    build = root / 'joint/build-reproduction-host'
    receipt = read(build / 'build-acceptance.json')
    require(receipt['status'] == 'pass' and receipt['tests_executed'] and receipt['final_parameterized_entry'],
            'Final parameterized build was not accepted')
    require(receipt['historical_source']['status'] == 'pass' and receipt['historical_source']['source_records'] == 4276,
            'Historical source provenance mismatch')
    original_manifest = receipt['historical_source_manifest']
    manifest = root / 'joint/source-bundle/manifest.json'
    require(manifest.stat().st_size == original_manifest['bytes'] and sha(manifest) == original_manifest['sha256'],
            'Build source manifest mismatch')
    require(len(read(manifest)['files']) == 4276, 'Source snapshot membership mismatch')
    for project in receipt['source_builds']:
        require(not project['source_paths_outside_sealed_tree'] and not project['historical_build_dependencies'],
                'Build imports old sources/libraries')
    tests = receipt['tests']
    require(tests['cpu_core_tests_passed'] == 608 and tests['B2_GPU_cases'] == 28 and
            tests['B2_range_negative_exit'] == 23 and tests['shared_media_read_write_read_drain'] == 'pass',
            'Component qualification incomplete')
    original_base = PureWindowsPath(receipt['build_root'])
    rows, dlls, exe = [], {}, None
    for row in receipt['new_binaries']:
        relative = PureWindowsPath(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid build artifact path')
        restored = build.joinpath(*relative.parts)
        require(restored.stat().st_size == row['bytes'] and sha(restored) == row['sha256'],
                'Archived fresh artifact mismatch: ' + row['path'])
        original = original_base / relative
        mapped = {**row, 'original_path': str(original), 'restored_path': restored}
        rows.append(mapped)
        if relative.suffix.lower() == '.dll':
            dlls[relative.name.lower()] = mapped
        if relative.name.lower() == 'realgpu_llama.exe':
            exe = mapped
    require(len(rows) == 8 and len(dlls) == 5 and exe is not None, 'Expected three fresh executables and five DLLs')
    return receipt, rows, dlls, exe


def profile_contract(inputs, original_profile):
    profile = read(inputs / 'configs/qwen3-0.6b.json')
    require(profile['gguf_sha256'] == MODEL_SHA and sha(inputs / 'configs/prompt.txt') == PROMPT_SHA,
            'Fresh model/prompt identity differs')
    keep = lambda value: {key:item for key,item in value.items() if key not in ('model_path','conversion')}
    require(keep(profile) == keep(original_profile), 'Fresh profile changes semantics or budgets')
    require(profile['numerical']['max_abs'] == .2 and profile['numerical']['rms'] == .025,
            'Numerical tolerance changed')
    require(read(inputs / 'configs/conversion.json')['gguf_sha256'] == MODEL_SHA, 'Conversion identity mismatch')
    return profile


def module_checks(record, rows, dlls, exe):
    require(windows_key(record['command'][0]) == windows_key(exe['original_path']), 'Model command uses another executable')
    require(len(rows) == len(dlls), 'Missing loaded DLL identities')
    loaded = {windows_key(path) for path in record['modules']}
    expected = set(dlls)
    for row in rows:
        name = PureWindowsPath(row['path']).name.lower()
        require(name in expected, 'Unexpected/duplicate recorded model DLL')
        expected.remove(name)
        artifact = dlls[name]
        require(windows_key(row['path']) == windows_key(artifact['original_path']) and
                row['sha256'] == artifact['sha256'] and windows_key(row['path']) in loaded,
                'Loaded model DLL path/hash differs from fresh build')
    require(not expected, 'Required DLL not observed')


def check_process(record, scale, label, folder, root, profile, exe):
    weights, kv, mode = MODES[label]
    H.environment(record, scale, weights, kv, 'host' if label == 'joint-host' else 'device')
    command = record['command']
    require(command[4:7] == ['8','0',mode], 'Different output count/observation/mode command')
    require(windows_key(command[1]) == windows_key(profile['model_path']), 'Model argument differs from profile')
    require(original_to_restored(root, command[2]) == folder.parents[1] / 'inputs/configs/prompt.txt', 'Prompt command path mismatch')
    require(original_to_restored(root, command[3]) == folder, 'Output command path mismatch')
    require(original_to_restored(root, command[7]) == folder.parents[1] / 'inputs/configs/qwen3-0.6b.json', 'Profile command path mismatch')
    probes = record['resource_probes']
    require(probes and all(not row['conflicts'] and row['visible_processes'] > 4 for row in probes),
            'Successful fresh model overlaps a recorded resource conflict')
    require(datetime.fromisoformat(record['end_utc']) > datetime.fromisoformat(record['start_utc']),
            'Invalid fresh process interval')
    require(windows_key(command[0]) == windows_key(exe['original_path']), 'Fresh executable path mismatch')


def compare_historical(folder, old_folder):
    new_summary, old_summary = read(folder / 'summary.json'), read(old_folder / 'summary.json')
    require(new_summary['prompt_tokens'] == old_summary['prompt_tokens'] and
            new_summary['generated_tokens'] == old_summary['generated_tokens'], 'Historical token IDs changed')
    left = np.fromfile(folder / 'logits.f32', dtype='<f4')
    right = np.fromfile(old_folder / 'logits.f32', dtype='<f4')
    require(left.shape == right.shape == (1215488,) and np.isfinite(left).all() and np.isfinite(right).all(),
            'Historical/fresh logits shape or finiteness differs')
    diff = left.astype('f8') - right.astype('f8')
    result = {'case': folder.name, 'bitwise_equal': sha(folder / 'logits.f32') == sha(old_folder / 'logits.f32'),
              'fresh_logits_sha256': sha(folder / 'logits.f32'), 'historical_logits_sha256': sha(old_folder / 'logits.f32'),
              'max_abs_difference': float(np.max(np.abs(diff))), 'rms_difference': float(np.sqrt(np.mean(diff*diff))),
              'token_IDs_equal': True}
    if (folder / 'kv-content.json').is_file():
        result['historical_KV_hashes_equal'] = read(folder / 'kv-content.json')['sequences'] == read(old_folder / 'kv-content.json')['sequences']
    return result


def fresh_success(root, historical, out, dlls, exe, original_profile):
    batch = root / 'joint/output/correctness-new-002'
    status = read(batch / 'status.json')
    require(status['status'] == 'pass' and status['completed'] == 16 and not status['statistical_performance_confirmation'],
            'Fresh full correctness batch not accepted')
    require(not (batch / 'failure.json').exists(), 'Successful batch has a failure record')
    protocol = read(batch / 'protocol.json')
    require(protocol['scales'] == ['F1','F2'] and protocol['outputs'] == 8 and protocol['input_tokens'] == 44,
            'Fresh correctness protocol scope changed')
    require({key:tuple(value) for key,value in protocol['modes'].items()} == MODES,
            'Fresh correctness groups changed')
    deps = verify_rows(root, read(batch / 'dependencies.json'))
    profile = profile_contract(batch / 'inputs', original_profile)
    script = root / 'joint/scripts/run_correctness.py'
    require(sha(script) == '3e93f076b1add9426309de741c062e7fd44b1721961663304ae89726d04ba318',
            'Original correctness validator changed')
    C = functions(script, ['validate_case','compare_scale'],
                  {'R':batch/'inputs','OUT':batch,'np':np,'json':json,'read':read,'sha':sha,
                   'write':lambda path,value: H.write(out/'correctness-recomputed'/Path(path).name,value)})
    processes, comparisons, streams, modules, probes = [], [], Counter(), 0, 0
    for scale in ['F1','F2']:
        for label, (weights, kv, mode) in MODES.items():
            name = scale + '-' + label
            folder = batch / 'cases' / name
            record = read(batch / 'cases' / (name + '.process.json'))
            check_process(record, scale, label, folder, root, profile, exe)
            rows = read(batch / 'cases' / (name + '.modules.json'))
            module_checks(record, rows, dlls, exe)
            modules += len(rows)
            probes += len(record['resource_probes'])
            C.validate_case(folder, scale, weights, kv, mode)
            streams.update(H.raw_streams(folder, scale, weights, kv, mode))
            comparisons.append(compare_historical(folder, historical / 'results/correctness-001' / name))
            processes.append(record)
            print('Fresh raw model rechecked', len(processes), '/ 16', name, flush=True)
        C.compare_scale(scale, 'cases')
        require(read(out/'correctness-recomputed'/(scale+'-acceptance.json')) ==
                read(batch/'cases'/(scale+'-acceptance.json')), 'Fresh cross-path acceptance differs')
    require(len(processes) == 16 and modules == 80, 'Fresh process/module count mismatch')
    for left, right in zip(processes, processes[1:]):
        require(datetime.fromisoformat(left['end_utc']) <= datetime.fromisoformat(right['start_utc']),
                'Fresh model processes overlap or scale/order was changed')
    H.write(out / 'historical-logits-comparison.json', comparisons)
    return {'status':'pass','model_processes':16,'frozen_dependency_records':deps,'loaded_module_hashes':modules,
            'resource_probes':probes,'nonoverlapping_adjacent_processes':15,'raw_streams':dict(streams),
            'historical_bitwise_logits_matches':sum(row['bitwise_equal'] for row in comparisons),
            'historical_KV_hash_matches':sum(row.get('historical_KV_hashes_equal',False) for row in comparisons),
            'historical_difference_cases':[row for row in comparisons if not row['bitwise_equal']],
            'cross_path_original_acceptance_recomputed':True}


def excluded_attempt(root):
    batch = root / 'joint/output/correctness-new-001'
    status, failure = read(batch / 'status.json'), read(batch / 'failure.json')
    require(status['status'] == 'failed' and status['completed'] == failure['completed'] == 2,
            'First attempt exclusion accounting differs')
    require(not (batch / 'cases/F1-acceptance.json').exists() and not (batch / 'cases/F2-acceptance.json').exists(),
            'Failed attempt incorrectly declares full correctness acceptance')
    record = read(batch / 'cases/F1-readonly-modeled.process.json')
    conflicts = [row for probe in record['resource_probes'] for row in probe['conflicts']]
    require(record['error'] and record['returncode'] != 0 and 'Resource conflict' in failure['traceback'],
            'Failed attempt no longer records the resource guard stop')
    require({row['pid'] for row in conflicts} == {32920,37536} and
            all(row['name'].lower() == 'python.exe' for row in conflicts), 'Recorded Python conflict identity differs')
    require(not any((batch / 'cases').glob('F2-*.process.json')), 'Failed batch proceeded into F2')
    return {'status':'excluded','completed_prefix_processes':2,'failed_case':'F1-readonly-modeled',
            'reason':'OS resource guard found two auxiliary Python processes; no complete acceptance',
            'conflicting_PIDs':sorted({row['pid'] for row in conflicts}),
            'mixed_into_successful_16_process_batch':False}


def statistical_prepare(root, out, original_profile):
    batch = root / 'joint/output/statistics-prepare-001'
    proto, prepared = read(batch/'protocol.json'), read(batch/'prepare-validation.json')
    require(sha(batch/'protocol.json') == prepared['protocol_sha256'] and prepared['status'] == 'pass',
            'Prepared statistics protocol hash mismatch')
    require(prepared['model_processes_executed'] == 0 and prepared['cross_path_gold_recomputed'] and
            prepared['reference_groups_validated'] == 14 and prepared['reference_schedule_numerical_checks'] == 142,
            'Statistics preparation evidence incomplete')
    raw = batch/'raw'
    # The lossless file archive does not materialize empty directories.
    require(not (batch/'samples.json').exists() and not (batch/'run-start.json').exists() and
            (not raw.exists() or not any(raw.iterdir())), 'Prepared batch contains unreported model attempts/samples')
    require(read(batch/'status.json')['state'] == 'prepared', 'Prepared statistics status changed')
    count = verify_rows(root, proto['dependencies'])
    profile_contract(batch/'inputs', original_profile)
    path = root/'joint/reproduction/run_statistics.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    science = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and
                   any(isinstance(target,ast.Name) and target.id=='SCIENCE_FIELDS' for target in node.targets))
    proxy = SimpleNamespace(read=read,sha=sha,require=require,JOINT=root/'joint',MODES=H.MODES,MODEL_SHA=MODEL_SHA)
    S = functions(path,['schedule','verify_frozen_protocol'],
                  {'R':proxy,'random':random,'SCIENCE_FIELDS':science,'__file__':str(path)})
    S.verify_frozen_protocol(batch,proto)
    require(len(proto['schedule']) == 142 and Counter(row['stage'] for row in proto['schedule']) ==
            {'warmup':14,'AA':16,'confirm':112}, 'Prepared process budget/stage assignment differs')
    preflight = read(batch/'provenance/prepare-resource-preflight.json')
    require(preflight['visible_processes'] > 4 and not preflight['conflicts'], 'Statistics resource preflight failed')
    gold_count, gold_files = 0, 0
    fresh = root/'joint/output/correctness-new-002/cases'
    gold = batch/'inputs/results/correctness-001'
    for scale in ['F1','F2']:
        require(read(batch/'gold-recomputed'/(scale+'-acceptance.json')) == read(fresh/(scale+'-acceptance.json')),
                'Prepare gold comparison differs from fresh correctness acceptance')
        for mode in H.MODES:
            name = scale+'-'+mode
            for filename in ['summary.json','logits.f32','kv-content.json']:
                source = fresh/name/filename
                if source.is_file():
                    require(sha(gold/name/filename) == sha(source), 'Prepare copied gold bytes differ: '+name+'/'+filename)
                    gold_files += 1
            gold_count += 1
    return {'status':'pass','model_processes_executed':0,'fixed_schedule_rows':142,
            'scientific_settings_and_orders_rechecked':True,'frozen_dependency_records':count,
            'gold_groups':gold_count,'gold_files_byte_exact_to_fresh_correctness':gold_files,
            'prepared_protocol_sha256':sha(batch/'protocol.json'),'sample_or_attempt_records_present':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Clean restored root containing joint/')
    parser.add_argument('--historical-root',type=Path,required=True,help='Restored root containing original integrated/')
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    root,historical,out=H.extended(args.root),H.extended(args.historical_root),H.extended(args.report_dir)
    require(not out.exists(), 'Use a fresh report directory')
    out.mkdir(parents=True)
    old=historical/'integrated/20261112-dual-line-r1'
    profile=read(old/'configs/qwen3-0.6b.json')
    require(profile['gguf_sha256']==MODEL_SHA and sha(old/'configs/prompt.txt')==PROMPT_SHA,
            'Historical comparison input identity differs')
    build, artifacts, dlls, exe = validate_build_receipt(root)
    success=fresh_success(root,old,out,dlls,exe,profile)
    excluded=excluded_attempt(root)
    prepared=statistical_prepare(root,out,profile)
    report={'status':'pass' if success['historical_bitwise_logits_matches']==16 else 'historical_bitwise_difference',
            'model_or_GPU_executed':False,'new_build_artifact_hashes_verified':len(artifacts),
            'fresh_correctness':success,'failed_attempt':excluded,'statistical_prepare':prepared,
            'limits':['No GGUF acquisition, original OS enumeration, model execution or DLL loading occurs during replay.',
                      'Original streams export consumer maxima per warp and in-process byte/lane counters, not every lane timestamp; KV content is recorded by hashes rather than exported raw KV bytes. Checks rederive only exported evidence.',
                      'Build/component execution is represented by the archived acceptance receipt; the eight actual fresh executable/DLL bytes are independently hashed here.'],
            'offline_parser_sha256':sha(Path(__file__).with_name('replay_historical.py'))}
    H.write(out/'acceptance.json',report)
    require(report['status']=='pass', 'Fresh acceptance passed but historical bitwise differences were recorded; inspect report')
    print('PASS: fresh 16-process correctness, excluded attempt and zero-model statistics prepare',flush=True)


if __name__=='__main__':
    main()
