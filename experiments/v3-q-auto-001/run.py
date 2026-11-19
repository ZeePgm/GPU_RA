"""Qualify one registered Q weight on native prefill and decode with the V3 guard."""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAMPLED = ROOT / 'experiments/v3-three-routes-sampled-001'
FIVE = ROOT / 'experiments/v3-route-perf-five-001'
sys.path.insert(0, str(SAMPLED))
import sampled_run as X
import stage1 as S

SITE_COUNT = 16384
UNITS = ('convert', 'mmvf')
PROMPT = SAMPLED / 'prompt-44.txt'
read, write, require, sha = S.read, S.write, S.require, S.sha


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def build_identity(build):
    """Lock objects with the PTX site metadata, loaded binaries and toolchain."""
    units = {}
    for unit in UNITS:
        path = build / 'q-auto-ptx' / unit / 'manifest.json'
        manifest = read(path)
        require(manifest['schema'] == 1 and manifest['unit'] == unit, 'PTX manifest ' + unit)
        required_files = {'native_ptx', 'rewritten_ptx', 'cubin', 'object'}
        require(set(manifest['artifacts']) >= required_files, 'PTX provenance ' + unit)
        src = Path(manifest['source']['path']).resolve()
        require(src.is_relative_to((ROOT/'joint/source').resolve()) and src.name == unit + '.cu',
                'Candidate compiler source path ' + unit)
        for artifact in [manifest['source'], manifest['header'], manifest['compiler'], *manifest['artifacts'].values()]:
            file = Path(artifact['path'])
            require(file.stat().st_size == artifact['bytes'] and sha(file) == artifact['sha256'],
                    'PTX provenance changed: ' + str(file))
        require(manifest['sites'] and
                manifest['artifacts']['native_ptx']['sha256'] != manifest['artifacts']['rewritten_ptx']['sha256'],
                'No PTX modification ' + unit)
        units[unit] = S.identity(path)
    return {'commit': git('rev-parse', 'HEAD'), 'source_tree': git('rev-parse', 'HEAD:joint/source'),
            'build': S.audit_build(build), 'translation_units': units}


def select_weight(cfg, output):
    inventory = read(FIVE / 'inventory.json')
    tensor = cfg['weight']
    for row in inventory['all_tensors']:
        row['selected'] = row['name'] == tensor
    chosen = [row for row in inventory['all_tensors'] if row['selected']]
    require(len(chosen) == 1, 'Expected one Q weight')
    q = chosen[0]
    require(q['dtype'] == 'F16' and q['shape'] == cfg['shape'] and q['bytes'] == cfg['bytes'] and
            q['logical_base'] == cfg['logical_base'] and q['expected_op'] == 'MUL_MAT' and
            q['expected_source_slot'] == 0, 'Native Q tensor layout mismatch')
    inventory.update(selected_count=1, unselected_count=len(inventory['all_tensors'])-1,
                     selected_bytes=q['bytes'], scope='Only the registered Q weight is selected')
    write(output, inventory)


def site_inventory(receipt):
    by_id = {}
    for index, unit in enumerate(UNITS):
        manifest = read(Path(receipt['translation_units'][unit]['path']))
        lines = Path(manifest['artifacts']['native_ptx']['path']).read_text(encoding='utf-8').splitlines()
        for row in manifest['sites']:
            n = row['site']
            require(n not in by_id and index * SITE_COUNT <= n < (index + 1) * SITE_COUNT and
                    row['entry'] and row['width'] > 0 and
                    1 <= row['line'] <= len(lines) and lines[row['line'] - 1].strip() == row['instruction'],
                    'Unbound compiler site: ' + str(n))
            by_id[n] = {'unit': unit, **row}
    return by_id


def trace_coverage(folder, step, record, known_sites, weight_bytes):
    """Recompute counts from both independent device arrays and original PTX widths."""
    words = weight_bytes // 2
    expected_counts = f'counts-{step}.u32'
    expected_sites = f'site-{step}.u64'
    require(record['counts_file'] == expected_counts and record['sites_file'] == expected_sites,
            'Incorrect counter evidence filenames')
    count_file, site_file = folder / expected_counts, folder / expected_sites
    require(count_file.stat().st_size == words*4 and site_file.stat().st_size == 2*SITE_COUNT*8,
            'Incomplete dynamic counter arrays')
    counts = np.fromfile(count_file, dtype='<u4')
    sites = np.fromfile(site_file, dtype='<u8')
    untouched = int(np.count_nonzero(counts == 0))
    repeated = int(np.count_nonzero(counts > 1))
    total = int(counts.sum(dtype=np.uint64))
    require((record['words'], record['untouched_words'], record['multiply_read_words'], record['total_word_reads'])
            == (words, untouched, repeated, total), 'Host and raw word counters disagree')
    require(untouched == 0, 'Some registered Q halves were never read in step ' + str(step))
    device = record['device_summary']
    require(device['partial_or_misaligned'] == 0, 'Partial or misaligned Q access')
    hits = []
    for site_id in np.flatnonzero(sites):
        i = int(site_id)
        require(i in known_sites, 'Dynamic site absent from original PTX manifests: ' + str(i))
        hits.append({**known_sites[i], 'dynamic_reads': int(sites[i])})
    byte_sum = 0
    for unit, fields in zip(UNITS, [('convert_loads','convert_bytes'), ('decode_loads','decode_bytes')]):
        rows = [row for row in hits if row['unit'] == unit]
        loads = sum(row['dynamic_reads'] for row in rows)
        size = sum(row['width']*row['dynamic_reads'] for row in rows)
        require((loads, size) == (device[fields[0]], device[fields[1]]), 'GPU site/width cross-check ' + unit)
        require((loads > 0) == (unit == UNITS[step]), 'Unexpected active native Q consumer ' + unit)
        byte_sum += size
    require(byte_sum == 2*total, 'Word counters and weighted instruction counts disagree')
    return {'phase': record['phase'], 'words': words, 'untouched_words': untouched,
            'multiply_read_words': repeated, 'total_word_reads': total, 'matched_bytes': byte_sum,
            'minimum_word_reads': int(counts.min()), 'maximum_word_reads': int(counts.max()),
            'device_summary': device, 'executed_sites': hits,
            'counts_sha256': sha(count_file), 'sites_sha256': sha(site_file)}


def measure(args):
    cfg = read(args.config)['parameters']
    out = args.output
    require(args.resource_root.resolve() == Path('E:/v1_update').resolve(), 'Expected full outer resource root')
    require(args.config.resolve() == (HERE/'config.json').resolve(), 'Use the registered configuration')
    out.mkdir(parents=True, exist_ok=False)
    receipt = read(args.receipt)
    require(receipt == build_identity(args.build_dir), 'Candidate build does not match its frozen receipt')
    sites = site_inventory(receipt)
    require(sha(args.model) == S.MODEL_SHA and sha(PROMPT) == S.PROMPT_SHA, 'Model or prompt changed')
    selection = out / 'selection.json'
    select_weight(cfg, selection)
    profile = read(FIVE / 'profile.json')
    profile.pop('sampled_scope')
    profile['model_path'] = str(args.model)
    write(out / 'profile.json', profile)
    env = {'HBFLAB_JOINT_KV':'native', 'HBFLAB_M4_PAGED':'0', 'LLAMA_GRAPH_REUSE_DISABLE':'1',
           'HBFLAB_SAMPLED_TOKEN_ONLY':'1', 'HBFLAB_SAMPLED_AUDIT':'summary', 'HBFLAB_SAMPLED_ROUTE':'native',
           'HBFLAB_SAMPLED_PROFILE':str(out/'profile.json'),
           'HBFLAB_SYMBOLIC_CONFIG':str(ROOT/'joint/source/experiments/gpu_ffn/H0-512MiB.ini')}
    expected_prompt = read(FIVE / 'input-spec.json')['prompt_tokens']
    require(len(expected_prompt) == cfg['prompt_length'], 'Historical prompt token count')
    evidence = {}
    for mode in cfg['modes']:
        child = out / mode
        command = [args.build_dir/'bin/five_model.exe', args.model, PROMPT, child, selection,
                   str(len(cfg['output_tokens']))]
        X.execute(args, command, mode, child, extra_env={**env, 'HBFLAB_AUTO_Q_MODE':mode},
                  require_ggml=True, timeout=cfg['process_timeout_seconds'], disk_cap=cfg['process_disk_limit_bytes'])
        model, q = read(child/'summary.json'), read(child/'q-auto.json')
        require(model['status'] == 'pass' and model['generated_tokens'] == cfg['output_tokens'] and
                model['prompt_tokens'] == expected_prompt and model['generated_steps'] == model['consumer_records'] == 2 and
                model['selected_tensors'] == 1 and model['kv'] == 'native F16', 'Model/consumer scope ' + mode)
        selected = [row for row in read(child/'loaded-tensors.json') if row['selected']]
        require(len(selected) == 1 and selected[0]['name'] == cfg['weight'] and
                selected[0]['address'] == q['device_address'] == selected[0]['storage_address'] and
                selected[0]['cuda_memory_type'] == 2 and selected[0]['cuda_device'] == q['cuda_device'] == 0,
                'Registered address must belong to the loaded Q GPU allocation')
        route = read(child/'route-summary.json')
        require(route['drained'] and route['steps'] == route['consumers'] == 2 and
                route['requested_bytes'] == route['range_or_scalar_calls'] == 0, 'Unexpected modeled weight service')
        require(q['finished'] and q['mode'] == mode and q['tensor'] == cfg['weight'] and q['bytes'] == cfg['bytes'] and
                q['logical_base'] == cfg['logical_base'] and q['registrations'] == 1 and
                q['device_registrations'] == int(mode == 'trace') and len(q['steps']) == 2, 'Registration scope ' + mode)
        vector_bytes = cfg['vocabulary'] * 4
        logits = (child/'logits.f32').read_bytes()
        require(len(logits) == 2 * vector_bytes, 'Two complete F32 logit vectors required')
        phases = []
        for step, row in enumerate(q['steps']):
            phase, columns = [('prefill', cfg['prompt_length']), ('decode', 1)][step]
            require(row['step'] == step and row['phase'] == phase and row['weight_slot'] == 0 and
                    row['op'] == 'MUL_MAT' and row['consumer'] and row['activation_columns'] == columns and
                    row['projection_values'] == cfg['shape'][1] * columns and
                    row['projection_file'] == f'q-output-{step}.f32', 'Q consumer phase/slot/shape')
            file = child / f'q-output-{step}.f32'
            require(file.stat().st_size == 4 * cfg['shape'][1] * columns and
                    sha(file) == cfg['projection_sha256'][step], 'Native reference Q output ' + phase)
            digest = hashlib.sha256(logits[step*vector_bytes:(step+1)*vector_bytes]).hexdigest()
            require(digest == cfg['logit_vector_sha256'][step], 'Native reference complete logit vector ' + phase)
            item = {'phase': phase, 'graph_consumer': row['consumer'], 'projection_sha256': sha(file),
                    'logit_vector_sha256': digest}
            if mode == 'trace':
                item['coverage'] = trace_coverage(child, step, row, sites, cfg['bytes'])
            phases.append(item)
        positions = read(child/'native-kv-positions.json')
        require(positions == [{'step':i, 'min':0, 'max':cfg['prompt_length']+i-1} for i in range(2)],
                'KV positions ' + mode)
        require(read(child/'framework-release.json') == {'status':'pass','sequence':0,'empty':True} and
                (child/'native-kv-state.bin').stat().st_size > 0, 'KV snapshot or sequence release ' + mode)
        evidence[mode] = {'phases':phases, 'output_tokens':model['generated_tokens'], 'kv_positions':positions,
                          'kv_sha256':sha(child/'native-kv-state.bin'), 'registrations':q['registrations'],
                          'device_registrations':q['device_registrations']}
    for name in ['logits.f32','native-kv-state.bin','native-kv-positions.json','framework-release.json',
                 'q-output-0.f32','q-output-1.f32']:
        require((out/'native'/name).read_bytes() == (out/'trace'/name).read_bytes(), 'Native/trace bytes: ' + name)
    require(receipt == build_identity(args.build_dir), 'Source/build changed while running')
    write(out/'acceptance.json', {'status':'passed', 'code_commit':receipt['commit'], 'source_tree':receipt['source_tree'],
          'build_receipt_sha256':sha(args.receipt), 'configuration_sha256':sha(args.config), 'modes':evidence,
          'timing_service_used':False})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    receipt = sub.add_parser('receipt')
    measure_parser = sub.add_parser('measure')
    for p in [receipt, measure_parser]:
        p.add_argument('--build-dir', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
    for flag in ['cuda-root','model','receipt','config','resource-root']:
        measure_parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'receipt':
        require(not args.output.exists() and not git('status','--porcelain'), 'Receipt requires a new file and a clean commit')
        write(args.output, build_identity(args.build_dir))
    else:
        measure(args)


if __name__ == '__main__':
    main()
