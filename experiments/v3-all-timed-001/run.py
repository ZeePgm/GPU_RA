"""Run both original Qwen3 graphs and verify every registered weight and demanded page."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import math
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('five_timed', ROOT/'experiments/v3-five-timed-001/run.py')
five = importlib.util.module_from_spec(spec)
spec.loader.exec_module(five)
S, X = five.S, five.X
require, read, write, sha = S.require, S.read, S.write, S.sha
PAGE, WORDS, UNITS = five.PAGE, five.WORDS, five.UNITS


def full_scope(inventory, cfg):
    entries = inventory['all_tensors']
    expected = {}
    for exemplar, role in S.EXPECTED.items():
        names = [exemplar.replace('blk.10.', f'blk.{i}.') for i in range(cfg['layers'])] if exemplar.startswith('blk.') else [exemplar]
        expected.update(dict.fromkeys(names, role))
    require(len(entries) == len(expected) == cfg['selected_tensors'] and
            {t['name'] for t in entries} == expected.keys() and inventory['file_alias_groups'] == [],
            'Frozen model scope or GGUF aliases differ')
    for t in entries:
        category, op, slot, shape, dtype = expected[t['name']]
        require(t['shape'] == shape and t['dtype'] == dtype and
                t['bytes'] == math.prod(shape)*(2 if dtype == 'F16' else 4), 'Invalid stored tensor ' + t['name'])
        t.update(selected=True, category=category, expected_op=op, expected_source_slot=slot)
    total = sum(t['bytes'] for t in entries)
    require(total == cfg['weight_bytes'], 'Incorrect total original tensor bytes')
    inventory.update(selected_count=len(entries), unselected_count=0, selected_bytes=total, scope='all original Qwen3 weights')
    return sorted(entries, key=lambda t: t['logical_base'])


def dimensions(t, tokens):
    name = t['name']
    n = len(tokens)
    if name in ('output.weight', 'output_norm.weight') or name.startswith('blk.27.ffn_'):
        n = 1
    width = t['shape'][0]
    if name == 'token_embd.weight':
        return [n,1,1,1], [width,n,1,1]
    if t['expected_op'] == 'MUL_MAT':
        return [width,n,1,1], [t['shape'][1],n,1,1]
    if name.endswith(('attn_q_norm.weight', 'attn_k_norm.weight')):
        heads = 16 if name.endswith('attn_q_norm.weight') else 8
        return [width,heads,n,1], [width,heads,n,1]
    return [width,n,1,1], [width,n,1,1]


def registered_ranges(child, model, entries):
    loaded = read(child/'loaded-tensors.json')
    by_name = {t['name']: t for t in loaded}
    ranges = model['ranges']
    require(len(loaded) == len(by_name) == len(ranges) == len(entries) and not model['aliases'],
            'Incomplete original storage inventory or unexpected alias')
    spans = []
    for ri, (ref, r) in enumerate(zip(entries, ranges)):
        name = ref['name']
        t = by_name[name]
        shape = ref['shape'] + [1]*(4-len(ref['shape']))
        strides = [int(np.prod(shape[:i], dtype=np.int64))*(2 if ref['dtype'] == 'F16' else 4) for i in range(4)]
        require(t['selected'] is True and t['cuda_memory_type'] == 2 and t['cuda_device'] == 0 and t['contiguous'] and
                t['dtype'].upper() == ref['dtype'] and t['shape'] == shape and t['strides'] == strides and
                t['view_offset'] == 0 and r['base'] == t['storage_address'] == t['address'] > 0 and
                r['bytes'] == t['bytes'] == ref['bytes'] and r['logical'] == ref['logical_base'] and
                r['physical_id'] == ri and r['name'] == name and
                r['retain_words'] == (ref['dtype'] == 'F32' or name == 'token_embd.weight'), 'Storage layout ' + name)
        spans.append((r['base'], r['base'] + r['bytes']))
    spans.sort()
    require(all(a[1] <= b[0] for a,b in zip(spans,spans[1:])), 'Unexpected overlap in original GPU allocations')
    return ranges, by_name


def graph_outputs(child, entries, loaded, steps):
    phases = read(child/'all-consumers.json')['phases']
    require([p['step'] for p in phases] == [0,1], 'Wrong graph phases')
    outputs = []
    for phase, tokens in zip(phases, steps):
        rows = phase['entries']
        index = {r['name']:r for r in rows}
        require(len(rows) == len(index) == len(entries), 'Duplicate/missing graph consumers')
        hashes = {}
        for ri, ref in enumerate(entries):
            name = ref['name']
            row = index[name]
            inp, out = dimensions(ref, tokens)
            require(row['node'] and row['after'] is True and row['op'] == ref['expected_op'] and
                    row['slot'] == ref['expected_source_slot'] and row['physical_id'] == ri and
                    row['source_offset'] == 0 and row['source_shape'] == row['weight_shape'] == loaded[name]['shape'] and
                    row['weight_strides'] == loaded[name]['strides'] and row['input_shape'] == inp and
                    row['output_shape'] == out and row['output_type'].upper() == 'F32' and
                    re.fullmatch('[0-9a-f]{64}', row['output_sha256']), 'Graph source, shape or F32 output ' + name)
            require(row.get('gather_rows') == (tokens if name == 'token_embd.weight' else None), 'Graph token gather ' + name)
            hashes[name] = row['output_sha256']
        outputs.append(hashes)
    return outputs


def sparse_sites(child, step, n_ranges, sites):
    path = child/f'sites-{step}.u64'
    size = path.stat().st_size
    require(0 < size <= 32*n_ranges*sum(map(len,sites)) and size % 32 == 0, 'Invalid sparse site file')
    rows = np.fromfile(path, dtype='<u8').reshape(-1,4)
    active = defaultdict(list)
    keys = set()
    for ri, unit, sid, count in map(lambda row: map(int,row), rows):
        key = (ri,unit,sid)
        require(ri < n_ranges and unit < len(UNITS) and sid in sites[unit] and count > 0 and key not in keys,
                'Invalid or duplicated dynamic PTX site ' + str(key))
        keys.add(key)
        active[ri].append({'unit':UNITS[unit], **sites[unit][sid], 'dynamic_reads':count})
    require(len(active) == n_ranges, 'Weight missing dynamic PTX sites')
    return active


def demand_pages(ranges, counts):
    base = 0
    for r in ranges:
        pages = (r['bytes']+PAGE-1)//PAGE
        if r['name'] == 'token_embd.weight':
            require(all(0 <= t < r['bytes']//2048 for t in counts), 'Token gather outside embedding')
            local = np.array(sorted({t//2 for t in counts}), dtype=np.uint64)
        else:
            local = np.arange(pages,dtype=np.uint64)
        yield r, local, base
        base += pages


def check_phase(child, step, entries, ranges, tokens, sites, cfg):
    counts = Counter(tokens)
    demands = list(demand_pages(ranges, counts))
    n = sum(len(pages) for _,pages,_ in demands)
    require(n == cfg['expected_demand_pages'][step], 'Independent demand page count')
    media = five.load_matrix(child, f'media-{step}.u64', '<u8', (n,10))
    audit = five.load_matrix(child, f'audit-{step}.u64', '<u8', (n,5))
    coverage = five.load_matrix(child, f'coverage-{step}.u32', '<u4', (n,WORDS//32))
    retained = sum(len(pages) for r,pages,_ in demands if r['retain_words'])
    repeats = five.load_matrix(child, f'repeats-{step}.u32', '<u4', (retained,WORDS))
    active = sparse_sites(child,step,len(ranges),sites)
    index, logical, gpu, requested, begins, ends, delays, sectors, seq, status = media.T
    ready, earliest, latest, loads, sizes = audit.T
    max_u64 = np.iinfo(np.uint64).max
    duration = ends-begins
    require(np.all(requested > 0) and np.all(ends > begins) and np.all(delays == duration//1000 + (duration%1000 != 0)) and
            np.all(requested <= max_u64-delays) and np.all(sectors == PAGE//64) and np.all(status == 2) and
            np.all(ready == gpu) and np.all(earliest >= requested+delays) and np.all(latest >= earliest) and
            np.all((loads > 0) & (loads <= sizes//2)), 'Invalid page receipt or early/missing native resume')
    require(np.array_equal(np.sort(seq),np.arange(1,n+1)), 'Service sequence is incomplete or duplicated')
    rows = []
    start = rep_start = 0
    for ri, (ref, (r, local, base)) in enumerate(zip(entries,demands)):
        name = ref['name']
        stop = start+len(local)
        sl = slice(start,stop)
        require(np.array_equal(index[sl],base+local) and np.array_equal(logical[sl],r['logical']+PAGE*local) and
                np.array_equal(gpu[sl],r['base']+PAGE*local), 'Unexpected demanded page ' + name)
        inp, out = dimensions(ref,tokens)
        if r['retain_words']:
            actual = repeats[rep_start:rep_start+len(local)]
            target = np.zeros_like(actual)
            if name == 'token_embd.weight':
                unit = 'getrows'
                for i,page in enumerate(local):
                    target[i,:1024] = counts[2*int(page)]
                    target[i,1024:] = counts[2*int(page)+1]
            else:
                unit = 'binbcast'
                require(len(local) == 1, 'Unexpected norm extent')
                target[0,:r['bytes']//2] = math.prod(out[1:])
            bits = np.unpackbits(coverage[sl].view('u1'),axis=1,bitorder='little')
            require(np.array_equal(actual,target) and np.array_equal(bits,target != 0), 'Word multiplicity or padding ' + name)
            unique = 2*int(np.count_nonzero(target))
            wanted_bytes = 2*target.sum(axis=1,dtype=np.uint64)
            rep_start += len(local)
        else:
            unit = 'convert' if inp[1] > 1 else 'mmvf'
            require(r['bytes'] % PAGE == 0 and np.all(coverage[sl] == 0xffffffff), 'Missing matrix halfword ' + name)
            unique = r['bytes']
            wanted_bytes = np.full(len(local),PAGE,dtype=np.uint64)
        require(np.array_equal(sizes[sl],wanted_bytes), 'Native demand bytes per page ' + name)
        hits = active[ri]
        actual_loads, actual_bytes = int(loads[sl].sum()), int(sizes[sl].sum())
        require(all(h['unit'] == unit for h in hits) and sum(h['dynamic_reads'] for h in hits) == actual_loads and
                sum(h['dynamic_reads']*h['width'] for h in hits) == actual_bytes, 'Native PTX and per-page counters disagree ' + name)
        rows.append({'weight':name, 'pages':len(local), 'unique_bytes':unique,'matched_bytes':actual_bytes,
                     'loads':actual_loads,'sites':hits,'min_resume_margin_ns':int(np.min(earliest[sl]-requested[sl]-delays[sl]))})
        start = stop
    require(start == n and rep_start == retained and sum(r['unique_bytes'] for r in rows) == cfg['expected_unique_bytes'][step] and
            sum(r['matched_bytes'] for r in rows) == cfg['expected_dynamic_bytes'][step], 'Unexpected total demand')
    return {'phase':['prefill','decode'][step], 'ranges':rows, 'pages':n, 'sectors':int(sectors.sum()),
            'earliest_media_ps':int(begins.min()),'latest_media_ps':int(ends.max()),
            'minimum_resume_margin_ns':int(np.min(earliest-requested-delays)),'missing_timestamps':0,'early_resumes':0}


def inspect(child, mode, entries, inputs, cfg, gold, sites):
    result = read(child/'summary.json')
    media = read(child/'all-media.json')
    timed = mode == 'timed'
    tokens, prompt = cfg['output_tokens'], inputs['prompt_tokens']
    require(result['status'] == 'pass' and result['generated_tokens'] == tokens and result['prompt_tokens'] == prompt and
            result['generated_steps'] == 2 and result['selected_tensors'] == len(entries) and result['capture_fixtures'] == 10 and
            result['consumer_records'] == 2*len(entries) and result['kv'] == 'native F16', 'Model scope ' + mode)
    tiles = math.ceil(cfg['vocabulary']/cfg['output_tile_rows'])
    require(media['mode'] == mode and media['registrations'] == (len(entries) if timed else 0) and
            media['output_tile_rows'] == cfg['output_tile_rows'] and media['output_tile_launches'] == [tiles,tiles] and
            media['input_tile_bytes'] == cfg['input_tile_bytes'] and media['media_finished'] is timed, 'Media scope ' + mode)
    ranges, loaded = registered_ranges(child,media,entries)
    route = read(child/'route-summary.json')
    require(route['drained'] and route['steps'] == 2 and route['consumers'] == 2*len(entries) and
            route['requested_bytes'] == route['range_or_scalar_calls'] == 0, 'Unexpected alternative route ' + mode)
    steps = [prompt,tokens[:1]]
    hashes = graph_outputs(child,entries,loaded,steps)
    backups = five.check_fixtures(child,gold)
    for meta in (child/'fixtures').glob('*.json'):
        row = read(meta)
        require(hashes[row['step']][row['tensor']] == backups[row['output']['file']], 'Graph output disagrees with full fixture')
    block = cfg['vocabulary']*4
    raw = (child/'logits.f32').read_bytes()
    digests = [hashlib.sha256(raw[i:i+block]).hexdigest() for i in (0,block)]
    require(len(raw) == 2*block and digests == cfg['logit_vector_sha256'] and
            [x['output.weight'] for x in hashes] == digests, 'Native and frozen complete logit vectors ' + mode)
    positions = read(child/'native-kv-positions.json')
    kv = sha(child/'native-kv-state.bin')
    require(positions == [{'step':i,'min':0,'max':len(prompt)+i-1} for i in range(2)] and
            read(child/'framework-release.json') == {'status':'pass','sequence':0,'empty':True} and
            kv == cfg['native_kv_state_sha256'], 'Native KV content or lifecycle ' + mode)
    report = {'registrations':media['registrations'],'graph_outputs':hashes,'frozen_output_sha256':backups,
              'tokens':tokens,'logit_vector_sha256':digests,'kv_sha256':kv,'kv_positions':positions,
              'output_tile_launches':media['output_tile_launches'],'media_finished':media['media_finished']}
    if timed:
        report['phases'] = [check_phase(child,i,entries,ranges,ids,sites,cfg) for i,ids in enumerate(steps)]
        require(report['phases'][1]['earliest_media_ps'] >= report['phases'][0]['latest_media_ps'], 'Media clock regressed')
    return report


def measure(args):
    out = args.output
    cfg = read(args.config)['parameters']
    require(args.resource_root.resolve() == Path('E:/v1_update').resolve() and
            args.config.resolve() == (HERE/'config.json').resolve(), 'Expected full workspace and committed experiment config')
    out.mkdir(parents=True,exist_ok=False)
    receipt = read(args.receipt)
    require(receipt == five.build_identity(args.build_dir), 'Source, PTX or binaries differ from frozen build')
    sites = five.index_sites(receipt)
    inputs = read(five.FIVE/'input-spec.json')
    source_inventory = ROOT/cfg['inventory']
    require(sha(args.model) == S.MODEL_SHA and sha(five.PROMPT) == S.PROMPT_SHA and
            sha(args.reference) == inputs['captured_fixture_manifest']['sha256'] and
            sha(source_inventory) == cfg['inventory_sha256'], 'Frozen model, prompt, reference or inventory identity')
    manifest = read(args.reference)
    gold = {Path(row['path']).name:row for row in manifest['files']}
    require(manifest['status'] == 'pass' and len(gold) == len(manifest['files']) == 30 and
            len(inputs['prompt_tokens']) == cfg['prompt_length'], 'Frozen references or input length')
    inventory = read(source_inventory)
    entries = full_scope(inventory,cfg)
    selected = out/'selection.json'
    write(selected,inventory)
    profile = read(five.FIVE/'profile.json')
    profile.pop('sampled_scope')
    profile['model_path'] = str(args.model)
    write(out/'profile.json',profile)
    env = {'HBFLAB_JOINT_KV':'native','HBFLAB_M4_PAGED':'0','LLAMA_GRAPH_REUSE_DISABLE':'1',
           'HBFLAB_SAMPLED_TOKEN_ONLY':'1','HBFLAB_SAMPLED_AUDIT':'summary','HBFLAB_SAMPLED_ROUTE':'native',
           'HBFLAB_SAMPLED_PROFILE':str(out/'profile.json'),
           'HBFLAB_SYMBOLIC_CONFIG':str(ROOT/'joint/source/experiments/gpu_ffn/H0-512MiB.ini')}
    require(cfg['modes'] == ['native','timed'], 'Native and timed arms required')
    evidence = {}
    for mode in cfg['modes']:
        child = out/mode
        cmd = [args.build_dir/'bin/five_model.exe',args.model,five.PROMPT,child,selected,str(len(cfg['output_tokens']))]
        X.execute(args,cmd,mode,child,extra_env={**env,'HBFLAB_ALL_MEDIA':mode,'HBFLAB_FIVE_MEDIA':mode,
                  'HBFLAB_FIVE_CAPTURE':str((child/'fixtures').resolve())},require_ggml=True,
                  timeout=cfg['process_timeout_seconds'],disk_cap=cfg['process_disk_limit_bytes'])
        require(X.directory_bytes(child) <= cfg['process_disk_limit_bytes'], 'Final process output exceeds budget')
        evidence[mode] = inspect(child,mode,entries,inputs,cfg,gold,sites)
    require(evidence['native']['graph_outputs'] == evidence['timed']['graph_outputs'], 'Native/timed graph output differs')
    for file in ('logits.f32','native-kv-state.bin','native-kv-positions.json','framework-release.json'):
        require((out/'native'/file).read_bytes() == (out/'timed'/file).read_bytes(), 'Native/timed bytes ' + file)
    require(receipt == five.build_identity(args.build_dir), 'Build changed across samples')
    write(out/'acceptance.json',{'status':'passed','code_commit':receipt['commit'],'source_tree':receipt['source_tree'],
          'build_receipt_sha256':sha(args.receipt),'fixture_manifest_sha256':sha(args.reference),
          'configuration_sha256':sha(args.config),'selection_sha256':sha(selected),'modes':evidence,'media_page_bytes':PAGE})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    make, run = sub.add_parser('receipt'), sub.add_parser('measure')
    for command in (make,run):
        command.add_argument('--build-dir',type=Path,required=True)
        command.add_argument('--output',type=Path,required=True)
    for flag in ('cuda-root','model','receipt','config','reference','resource-root'):
        run.add_argument('--'+flag,type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'receipt':
        require(not args.output.exists() and not five.git('status','--porcelain'), 'Receipt requires a clean commit and new file')
        write(args.output,five.build_identity(args.build_dir))
    else:
        measure(args)


if __name__ == '__main__':
    main()
