"""Direct native/timed model experiment and independent five-weight acceptance."""
import argparse
from collections import Counter
import hashlib
import sys
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIVE = ROOT / 'experiments/v3-route-perf-five-001'
SAMPLED = ROOT / 'experiments/v3-three-routes-sampled-001'
PROMPT = SAMPLED / 'prompt-44.txt'
sys.path.insert(0, str(SAMPLED))
import sampled_run as X
import stage1 as S

read, write, sha, require = S.read, S.write, S.sha, S.require
UNITS = ('convert', 'mmvf', 'binbcast', 'getrows')
SITE_COUNT = 32768
PAGE = 4096
WORDS = PAGE // 2


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def build_identity(build):
    records = {}
    for unit in UNITS:
        manifest_path = build / 'q-auto-ptx' / unit / 'manifest.json'
        m = read(manifest_path)
        require(m['schema'] == 1 and m['unit'] == unit and m['sites'], 'Missing PTX sites for ' + unit)
        artifacts = m['artifacts']
        require({'native_ptx','rewritten_ptx','cubin','object'} <= artifacts.keys() and
                artifacts['native_ptx']['sha256'] != artifacts['rewritten_ptx']['sha256'], 'Missing rebuilt PTX ' + unit)
        source = Path(m['source']['path']).resolve()
        require(source.is_relative_to((ROOT/'joint/source').resolve()) and source.name == unit+'.cu',
                'Wrong candidate source: ' + str(source))
        for row in [v for v in m.values() if isinstance(v, dict) and 'path' in v and 'sha256' in v] + list(artifacts.values()):
            path = Path(row['path'])
            require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Compiler identity: ' + str(path))
        records[unit] = S.identity(manifest_path)
    return {'commit': git('rev-parse','HEAD'), 'source_tree': git('rev-parse','HEAD:joint/source'),
            'build': S.audit_build(build), 'translation_units': records}


def index_sites(receipt):
    indices = []
    for unit in UNITS:
        m = read(Path(receipt['translation_units'][unit]['path']))
        lines = Path(m['artifacts']['native_ptx']['path']).read_text(encoding='utf-8').splitlines()
        entries = {}
        for row in m['sites']:
            n = row['site']
            require(0 <= n < SITE_COUNT and n not in entries and row['width'] > 0 and row['entry'] and
                    1 <= row['line'] <= len(lines) and lines[row['line']-1].strip() == row['instruction'],
                    f'Invalid source location for {unit}/{n}')
            entries[n] = row
        indices.append(entries)
    return indices


def load_matrix(folder, filename, dtype, shape):
    file = folder / filename
    require(file.stat().st_size == int(np.prod(shape))*np.dtype(dtype).itemsize, 'Incomplete array ' + filename)
    return np.fromfile(file, dtype=dtype).reshape(shape)


def check_fixtures(child, gold):
    captures = child / 'fixtures'
    require({f.name for f in captures.iterdir() if f.is_file()} == set(gold), 'Incomplete ten native fixtures')
    output_hashes = {}
    for name, row in gold.items():
        path = captures/name
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Frozen native fixture ' + name)
        if name.endswith('-output.bin'):
            output_hashes[name] = row['sha256']
    require(len(output_hashes) == 10, 'Expected ten native outputs')
    return output_hashes


def registered_ranges(folder, run, inventory, weights):
    selected = {t['name']:t for t in read(folder/'loaded-tensors.json') if t['selected']}
    entries = sorted((t for t in inventory['all_tensors'] if t['selected']), key=lambda t:t['logical_base'])
    require(len(entries) == len(run['ranges']) == len(selected) == 5, 'Expected five registered tensors')
    require(set(selected) == {w['name'] for w in weights}, 'Missing registered weight')
    for item, ref in zip(run['ranges'], entries):
        t = selected[ref['name']]
        require(item['name'] == ref['name'] and item['base'] == t['address'] == t['storage_address'] and
                item['bytes'] == t['bytes'] == ref['bytes'] and item['logical'] == ref['logical_base'] and
                t['cuda_memory_type'] == 2 and t['cuda_device'] == 0 and t['contiguous'] and
                item['retain_words'] == (ref['name'] in {'token_embd.weight','blk.10.ffn_norm.weight'}),
                'Registered address/layout mismatch ' + ref['name'])
    return run['ranges']


def demands(ranges, tokens):
    """Derive pages from tensor layouts and real token IDs, independently of GPU counters."""
    demands = []
    base = 0
    embedding_rows = Counter(tokens)
    for r in ranges:
        size = r['bytes']
        require(size > 0 and size % PAGE == 0, 'Page-aligned tensor extent required')
        all_pages = size//PAGE
        if r['name'] == 'token_embd.weight':
            pages = np.array(sorted({row//2 for row in embedding_rows}), dtype=np.int64)
            require(np.all((0 <= pages) & (pages < all_pages)), 'Embedding row bounds')
        else:
            pages = np.arange(all_pages)
        demands.append((r, pages, base))
        base += all_pages
    return demands


def check_phase(folder, step, ranges, tokens, sites, cfg):
    expected = demands(ranges, tokens)
    n = sum(len(pages) for _,pages,_ in expected)
    require(n == cfg['expected_demand_pages'][step], 'Reference demand differs from the frozen page count')
    media = load_matrix(folder, f'media-{step}.u64', '<u8', (n,10))
    audit = load_matrix(folder, f'audit-{step}.u64', '<u8', (n,5))
    coverage = load_matrix(folder, f'coverage-{step}.u32', '<u4', (n,WORDS//32))
    retained = sum(len(pages) for r,pages,_ in expected if r['retain_words'])
    repeats = load_matrix(folder, f'repeats-{step}.u32', '<u4', (retained,WORDS))
    seen_sites = load_matrix(folder, f'sites-{step}.u64', '<u8', (len(ranges),len(UNITS),SITE_COUNT))
    index, logical, gpu, requested, begins, ends, delays, sectors, seq, status = media.T
    ready, earliest, latest, loads, sizes = audit.T
    require(np.all(requested > 0) and np.all(ends > begins) and np.all(delays == (ends-begins+999)//1000) and
            np.all(requested < np.iinfo(np.uint64).max-delays) and np.all(sectors == PAGE//64) and
            np.all(status == 2) and np.all(ready == gpu) and np.all(earliest >= requested+delays) and
            np.all(latest >= earliest) and np.all(loads > 0), 'Unready page, invalid receipt, or early/missing GPU resume')
    require(np.array_equal(np.sort(seq), np.arange(1,n+1)), 'Incomplete demand queue')

    rows = []
    start = 0
    rep_start = 0
    counts = Counter(tokens)
    for ri,(r,local,global_base) in enumerate(expected):
        name = r['name']
        stop = start + len(local)
        sl = slice(start,stop)
        require(np.array_equal(index[sl],global_base+local) and np.array_equal(logical[sl],r['logical']+PAGE*local) and
                np.array_equal(gpu[sl],r['base']+PAGE*local), 'Wrong demanded page mapping ' + name)
        masks = coverage[sl]
        if r['retain_words']:
            actual = repeats[rep_start:rep_start+len(local)]
            ref = np.zeros(actual.shape, dtype='<u4')
            if name == 'token_embd.weight':
                for row_index,page in enumerate(local):
                    ref[row_index,:1024] = counts[2*int(page)]
                    ref[row_index,1024:] = counts[2*int(page)+1]
            else:
                ref.fill(len(tokens))
            require(np.array_equal(actual,ref), 'Incorrect repetitions for ' + name)
            bits = np.unpackbits(masks.view('u1'),axis=1,bitorder='little')
            require(np.array_equal(bits,ref != 0), 'Incorrect word coverage ' + name)
            wanted_bytes = 2*ref.sum(axis=1, dtype=np.uint64)
            unique_bytes = 2*int(np.count_nonzero(ref))
            rep_start += len(local)
        else:
            require(np.all(masks == 0xffffffff), 'Incomplete whole-matrix coverage ' + name)
            wanted_bytes = np.full(len(local),PAGE,dtype=np.uint64)
            unique_bytes = len(local)*PAGE
        require(np.array_equal(sizes[sl],wanted_bytes), 'Incorrect GPU byte demand per page ' + name)
        hits = []
        for ui, sid in np.argwhere(seen_sites[ri] != 0):
            site = int(sid)
            require(site in sites[ui], f'Unbound measured PTX site {UNITS[ui]}/{site}')
            hits.append({'unit':UNITS[ui], **sites[ui][site], 'dynamic_reads':int(seen_sites[ri,ui,site])})
        actual_loads = int(loads[sl].sum())
        actual_bytes = int(sizes[sl].sum())
        require(sum(h['dynamic_reads'] for h in hits) == actual_loads and
                sum(h['dynamic_reads']*h['width'] for h in hits) == actual_bytes,
                'GPU per-site and per-page totals disagree ' + name)
        rows.append({'weight':name,'pages':len(local),'unique_bytes':unique_bytes,'matched_bytes':actual_bytes,
                     'loads':actual_loads,'sites':hits,'min_resume_margin_ns':int(np.min(earliest[sl]-requested[sl]-delays[sl]))})
        start = stop
    require(start == n and rep_start == retained, 'Coverage inventory mismatch')
    return {'phase':['prefill','decode'][step], 'ranges':rows, 'pages':n, 'sectors':int(sectors.sum()),
            'earliest_media_ps':int(begins.min()),'latest_media_ps':int(ends.max()),
            'minimum_resume_margin_ns':int(np.min(earliest-requested-delays)), 'missing_timestamps':0,'early_resumes':0}


def inspect(child, mode, inventory, spec, cfg, gold, sites):
    model = read(child/'summary.json')
    run = read(child/'five-media.json')
    timed = mode == 'timed'
    tokens = cfg['output_tokens']
    prompt = spec['prompt_tokens']
    require(model['status'] == 'pass' and model['generated_tokens'] == tokens and model['prompt_tokens'] == prompt and
            model['generated_steps'] == 2 and model['consumer_records'] == model['capture_fixtures'] == 10 and
            model['selected_tensors'] == 5 and model['kv'] == 'native F16', 'Model scope ' + mode)
    tiles = (cfg['vocabulary'] + cfg['output_tile_rows'] - 1) // cfg['output_tile_rows']
    require(run['mode'] == mode and run['registrations'] == (5 if timed else 0) and
            run['output_tile_rows'] == cfg['output_tile_rows'] and run['output_tile_launches'] == [tiles, tiles]
            and run['media_finished'] is timed,
            'Wrong registration, tiling, or unfinished service ' + mode)
    ranges = registered_ranges(child, run, inventory, spec['weights'])
    route = read(child/'route-summary.json')
    require(route['drained'] and route['steps'] == 2 and route['consumers'] == 10 and
            route['requested_bytes'] == route['range_or_scalar_calls'] == 0, 'Another weight route was used')
    output_hashes = check_fixtures(child,gold)
    block = cfg['vocabulary'] * 4
    values = (child/'logits.f32').read_bytes()
    require(len(values) == 2*block and [hashlib.sha256(values[i:i+block]).hexdigest() for i in (0,block)]
            == cfg['logit_vector_sha256'],'Two complete frozen logit vectors ' + mode)
    positions = read(child/'native-kv-positions.json')
    require(positions == [{'step':i,'min':0,'max':len(prompt)+i-1} for i in range(2)] and
            read(child/'framework-release.json') == {'status':'pass','sequence':0,'empty':True} and
            sha(child/'native-kv-state.bin') == cfg['native_kv_state_sha256'], 'Native KV content or lifecycle ' + mode)
    item = {'registrations':run['registrations'],'output_sha256':output_hashes, 'tokens':tokens,
            'kv_sha256':sha(child/'native-kv-state.bin'), 'kv_positions':positions,
            'logit_vector_sha256':cfg['logit_vector_sha256'],'media_finished':run['media_finished'],
            'output_tile_launches':run['output_tile_launches'],'output_tile_rows':run['output_tile_rows']}
    if timed:
        item['phases'] = [check_phase(child, step, ranges, input_ids, sites, cfg)
                          for step,input_ids in enumerate([prompt,tokens[:1]])]
        require(item['phases'][1]['earliest_media_ps'] >= item['phases'][0]['latest_media_ps'],
                'Shared media clock regressed across phases')
    return item


def measure(args):
    out = args.output
    cfg = read(args.config)['parameters']
    require(args.resource_root.resolve() == Path('E:/v1_update').resolve() and
            args.config.resolve() == (HERE/'config.json').resolve(), 'Use the full resource root and registered config')
    out.mkdir(parents=True, exist_ok=False)
    receipt = read(args.receipt)
    require(receipt == build_identity(args.build_dir),'Source, toolchain, PTX or binary differs from frozen build')
    sites = index_sites(receipt)
    spec = read(FIVE/'input-spec.json')
    require(sha(args.model) == S.MODEL_SHA and sha(PROMPT) == S.PROMPT_SHA and
            sha(args.reference) == spec['captured_fixture_manifest']['sha256'], 'Frozen model/prompt/fixtures identity')
    manifest = read(args.reference)
    gold = {Path(row['path']).name:row for row in manifest['files']}
    require(manifest['status'] == 'pass' and len(gold) == len(manifest['files']) == 30 and
            len(spec['prompt_tokens']) == cfg['prompt_length'], 'Historical fixture inventory')
    inv_path = ROOT/cfg['inventory']
    inventory = read(inv_path)
    profile = read(FIVE/'profile.json')
    profile.pop('sampled_scope')
    profile['model_path'] = str(args.model)
    write(out/'profile.json',profile)
    env = {'HBFLAB_JOINT_KV':'native','HBFLAB_M4_PAGED':'0','LLAMA_GRAPH_REUSE_DISABLE':'1',
           'HBFLAB_SAMPLED_TOKEN_ONLY':'1','HBFLAB_SAMPLED_AUDIT':'summary','HBFLAB_SAMPLED_ROUTE':'native',
           'HBFLAB_SAMPLED_PROFILE':str(out/'profile.json'),
           'HBFLAB_SYMBOLIC_CONFIG':str(ROOT/'joint/source/experiments/gpu_ffn/H0-512MiB.ini')}
    require(cfg['modes'] == ['native','timed'], 'Native and timed arms both required')
    evidence = {}
    for mode in cfg['modes']:
        child = out/mode
        cmd = [args.build_dir/'bin/five_model.exe',args.model,PROMPT,child,inv_path,str(len(cfg['output_tokens']))]
        X.execute(args,cmd,mode,child,extra_env={**env,'HBFLAB_FIVE_MEDIA':mode,
                  'HBFLAB_FIVE_CAPTURE':str((child/'fixtures').resolve())},require_ggml=True,
                  timeout=cfg['process_timeout_seconds'],disk_cap=cfg['process_disk_limit_bytes'])
        require(X.directory_bytes(child) <= cfg['process_disk_limit_bytes'], 'Final model output exceeds disk limit')
        evidence[mode] = inspect(child,mode,inventory,spec,cfg,gold,sites)
    for file in ['logits.f32','native-kv-state.bin','native-kv-positions.json','framework-release.json']:
        require((out/'native'/file).read_bytes() == (out/'timed'/file).read_bytes(), 'Native/timed bytes ' + file)
    require(receipt == build_identity(args.build_dir),'Build changed across phases')
    write(out/'acceptance.json',{'status':'passed','code_commit':receipt['commit'],'source_tree':receipt['source_tree'],
          'build_receipt_sha256':sha(args.receipt),'fixture_manifest_sha256':sha(args.reference),
          'configuration_sha256':sha(args.config),'modes':evidence,'media_page_bytes':PAGE})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    make = sub.add_parser('receipt')
    run = sub.add_parser('measure')
    for p in (make,run):
        p.add_argument('--build-dir',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
    for flag in ['cuda-root','model','receipt','config','reference','resource-root']:
        run.add_argument('--'+flag,type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'receipt':
        require(not args.output.exists() and not git('status','--porcelain'), 'Receipt requires a clean commit and new file')
        write(args.output,build_identity(args.build_dir))
    else:
        measure(args)


if __name__ == '__main__':
    main()
