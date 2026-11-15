"""Offline rechecks of restored historical GPU evidence; never launches a model."""
import argparse
import ast
from collections import Counter
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import types
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def historical_ffn():
    source = (HERE / 'evidence_validators/ffn_run.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Import)):
            keep.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module not in ('platform_environment', 'integrated_build_identity'):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'STAGE_LAYERS' for t in node.targets):
            keep.append(node)
    module = types.ModuleType('historical_ffn_audit')
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'historical_ffn_run.py', 'exec'), module.__dict__)
    return module


def check_v1(root, out, ffn):
    base = root / 'output/gpu-ffn/20261105-new-backend-f1-f3'
    results = []
    for stage in ('F1', 'F2', 'F3'):
        p = base / stage
        index = read(p / 'artifact-index.json')
        for row in index['files']:
            f = p / row['path'].replace('\\', '/')
            assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256'], str(f)
        table = read(p / 'compute-table.json')
        header = {k: v for k, v in table['header'].items() if k not in ('frozen_G1', 'unit', 'semantics')}
        rebuilt = ffn.capture_table(p / 'native', header)
        assert rebuilt == table, stage + ' native CUPTI table'
        # The historical auditor writes paired-tokens.csv; redirect just that file to the report directory.
        original_open = Path.open
        def redirected_open(path, *args, **kwargs):
            if path == p / 'paired-tokens.csv' and args and args[0] == 'w':
                return original_open(out / (stage + '-paired-tokens.csv'), *args, **kwargs)
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', redirected_open):
            result = ffn.audit(p, read(p / 'regions.json'), table)
        prior = read(p / 'result.json')
        for key in ('stage','coverage','G2','G3','total_wait_ps','compute_ps','online_audit','closed_and_drained'):
            assert result[key] == prior[key], stage + ' reconstructed ' + key
        namespace = {'ffn': ffn, 'copy': copy, 'patch': patch, 'Path': Path}
        source = ast.parse((HERE / 'evidence_validators/ffn_validate.py').read_text(encoding='utf-8-sig'))
        methods = [n for n in source.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=methods,type_ignores=[]), 'historical_ffn_validate.py','exec'), namespace)
        with patch.object(Path, 'open', redirected_open):
            negatives = namespace['negative_checks'](p, read(p / 'regions.json'), table)
        results.append({'stage':stage,'raw_artifact_hashes':len(index['files']),
                        'native_table_rebuilt':True,'reconstructed':result,
                        'corrupt_record_rejections':negatives['cases']})
        print(stage, 'raw capture/table/coverage/logits/dependency/clock audit PASS', flush=True)
    return {'status':'pass','stages':results}


def check_v2(root, out, ffn):
    source_path = HERE / 'evidence_validators/readonly_compare_analyze.py'
    source = source_path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    nodes, skipped = [], []
    # Keep the historical analyzer's substantive raw-data checks unchanged.
    # Build/model identity conditions belong to reconstructing binaries and input acquisition,
    # and are checked separately; they must not require the original workstation in offline replay.
    for node in tree.body:
        text = ast.get_source_segment(source, node) or ''
        if text.startswith('R=') or text.startswith('O=') or text.startswith('sys.path.insert') or text == 'import run as f':
            continue
        if any(marker in text for marker in ('candidate-identity.json', 'dependency-identity.json', 'preflight-inputs.json', "profile=read(", "'model drift'")):
            skipped.append(text.splitlines()[0])
            continue
        nodes.append(node)
    saved_write = ffn.write
    captured = {}
    def report_write(path, value):
        captured['comparison'] = value
        saved_write(out / 'V2-comparison-recomputed.json', value)
    ffn.write = report_write
    namespace = {'R':root,'O':root/'output/readonly-f-compare-a-20261108','f':ffn}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source_path),'exec'),namespace)
    finally:
        ffn.write = saved_write
    got = captured['comparison']
    prior = read(root/'output/readonly-f-compare-a-20261108/comparison.json')
    assert got['cases'] == prior['cases'], 'Recomputed F1/F2 metrics differ'
    assert len(got['excluded_runs']) == len(prior['excluded_runs']) == 2
    print('V2 same-batch raw logits/inputs/CUPTI/batches/clocks/failure-prefix audit PASS', flush=True)
    return {'status':'pass','comparison':'V2-comparison-recomputed.json',
            'excluded_identity_checks':skipped,
            'identity_policy':'Bundle verifies archived original bytes. Fresh-build and acquired GGUF identity are separate reproduction steps.'}


def check_r_series(root):
    logits_pairs = 0
    for rel, left, right in [('r1-20261107/comparison','old','new'),('r2-20261107/final-models','ptx','cuda_scalar')]:
        folder = root/'output'/rel
        for case in ('qwen-short','qwen-long','qwen-1.7b','smol-short'):
            for mode in ('native','bypass','modeled'):
                a = folder/(case+'-'+left)/mode/'logits.f32'
                b = folder/(case+'-'+right)/mode/'logits.f32'
                assert sha(a) == sha(b), str(a)
                logits_pairs += 1
    base = root/'output/r3-20261107'
    qualified = 0
    for file in (base/'qualification-final').glob('*.result.json'):
        d = read(file)
        if d.get('status') != 'pass':
            continue
        cfg = d['config']
        actual = Counter()
        for parent in d.get('parents',[]):
            actual.update(range(parent['offset'],parent['offset']+parent['bytes']))
        expected = Counter()
        mask = cfg.get('mask',0xffffffff)
        for i in range(cfg['lanes']):
            lane = i % 32
            if not (mask >> lane) & 1:
                continue
            index = (i//32)*32 + (mask & ((1 << lane)-1)).bit_count()
            if cfg.get('layout') == 'duplicate': index = (i//32)*32
            if cfg.get('layout') == 'reverse': index = (i//32)*32+31-lane
            if cfg.get('layout') == 'hole' and lane == 5: index += 1
            offset = cfg.get('shift',0)+index*cfg.get('stride',1)*cfg['width']
            expected.update(range(offset,offset+cfg['width']))
        assert actual == expected and d['dependency_checks'] == d['logical_accesses'], str(file)
        qualified += 1
    assert qualified == 112
    performance = base/'performance'
    pairs = read(performance/'pairs.json')
    for row in pairs:
        for flag, key in [(False,'scalar_ns'),(True,'merged_ns')]:
            p = performance / ('%s-%s-%s-%s.result.json' % (row['route'],row['case'],row['pair'],flag))
            d = read(p)
            assert d['status'] == 'pass' and d['drained']
            assert statistics.mean(x['wall_ns'] for x in d['samples']) == row[key]
        assert row['scalar_ns']-row['merged_ns'] == row['difference_ns']
    for saved in read(performance/'analysis.json'):
        selected=[p for p in pairs if p['route']==saved['route'] and p['case']==saved['case']]
        diffs=sorted(p['difference_ns'] for p in selected)
        n=len(diffs); alpha=.025 if saved['case']=='aligned' else .05
        k=max([k for k in range(1,n//2+1) if 2*sum(math.comb(n,j) for j in range(k))/2**n<=alpha] or [0])
        interval=[diffs[k-1],diffs[n-k]] if k else [None,None]
        assert statistics.median(diffs)==saved['paired_median_difference_ns'] and interval==saved['median_interval_ns']
        assert sum(x>0 for x in diffs)==saved['positive_pairs']
    assert len(pairs)==160
    print('R series: 24 full-logits pairs, 112 byte-coverage cases, 160 paired raw timings/statistics PASS', flush=True)
    return {'status':'pass','R1_R2_bitwise_logits_pairs':logits_pairs,'R3_coverage_cases':qualified,
            'R3_reconstructed_raw_timing_pairs':len(pairs),'R3_statistics_recomputed':10,
            'scope':'Archive preserves all R1/R2/down/R3/context raw evidence. This recheck covers model logits identity, R3 qualification coverage/dependencies and formal R3 timing statistics; it does not re-execute GPU correctness or rederive every R1/R2/down/context acceptance rule.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Root containing restored output/ tree')
    p.add_argument('--version',choices=['v1','v2'],required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    args=p.parse_args()
    root,out=args.root.resolve(),args.report_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    ffn=historical_ffn()
    report={'status':'pass','version':args.version,'gpu_executed':False,'v1':check_v1(root,out,ffn)}
    if args.version=='v2':
        report['v2']=check_v2(root,out,ffn)
        report['r_series']=check_r_series(root)
    (out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS; offline acceptance:',out/'acceptance.json')


if __name__=='__main__':
    main()
