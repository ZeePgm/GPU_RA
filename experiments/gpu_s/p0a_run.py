"""P0-A fixed-dose qualification and fresh/reused-context crossover experiment."""
import argparse
import copy
import random
import subprocess
import time
import shutil
import math
from statistics import stdev
from scipy.stats import t
from datetime import datetime, timezone
from pathlib import Path
from common import read, write, rows, sha
from study import ROOT, invoke
from p0a_qualification import freeze, qualify, adjudicate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', required=True)
    parser.add_argument('--qualification-only', action='store_true')
    parser.add_argument('--calibration-pairs',type=int,default=64)
    args = parser.parse_args()
    out = ROOT / args.study
    out.mkdir(parents=True, exist_ok=False)
    base = ROOT/'output/gpu-s/20261101-s8-002/G3'
    binary = ROOT/'build-s3/bin/gpu_a1.exe'
    cfg = read(base/'configs/native-verify.json')
    cfg['model_path'] = str(ROOT/'models/Qwen3-0.6B-F16.gguf')
    cfg['compute_table_path'] = str(base/'compute-table.json')
    identities = rows(base/'G3-timing-1/gates.jsonl')
    # Dose and uncertainty fixed before any new observations; no data-dependent enlargement.
    contract = dict(target_ns=100_000_000, relative_margin=.1, alpha=.05,
                    endpoints=16, warmup_runs=4, pairs=30, order_seed=4102,
                    protocol='fresh_process', binary_sha256=sha(binary),
                    calibration_pairs=args.calibration_pairs, created_utc=datetime.now(timezone.utc).isoformat(),
                    endpoint='token ID deliverable minus request entry; first gate only',
                    budget_policy='After independent calibration: max(30, ceil((t_29 * largest endpoint noise SD / 5 ms)^2)), capped at 256; freeze before formal; inconclusive retained; no optional stopping')
    write(out/'frozen-contract.json', dict(contract=contract, sha256=freeze(contract)))

    source=out/'source';source.mkdir()
    for file in Path(__file__).parent.glob('p0a*.py'): shutil.copyfile(file,source/file.name)
    for name in ['native_a1.cpp','a1_segments.hpp']: shutil.copyfile(Path(__file__).parent/name,source/name)
    shutil.copyfile(binary,source/binary.name)
    write(source/'manifest.json',[dict(file=p.name,sha256=sha(p)) for p in source.iterdir()])

    def frequency(name):
        r = subprocess.run(['nvidia-smi','--query-gpu=uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv'], capture_output=True, text=True, check=True)
        write(out/'frequency'/f'{name}.json', dict(utc=datetime.now(timezone.utc).isoformat(), data=r.stdout))

    def command(name, mode='zero', lean=True, verification=False):
        c = copy.deepcopy(cfg)
        c['a1'] = dict(mode=mode, lean=lean, session_config=str(base/'effective-H0.ini'), gates=[dict(node_id=g['node_id'], step=g['step'], wait_ns=contract['target_ns'] if k==0 else 0) for k,g in enumerate(identities)])
        write(out/'configs'/f'{name}.json', c)
        return [str(x) for x in [cfg['model_path'], base/'prompt.txt', out/name, 8, 'verify' if verification else 'timing', out/'configs'/f'{name}.json']]

    def run(name, **kw):
        frequency(name+'-before')
        invoke([binary,*command(name, **kw)], out/'logs'/name)
        frequency(name+'-after')
        print(name, 'completed', flush=True)

    def batch(name, requests):
        write(out/f'{name}-requests.json', [command(n, **kw) for n,kw in requests])
        frequency(name+'-before')
        invoke([binary,'--requests',out/f'{name}-requests.json'], out/'logs'/name)
        frequency(name+'-after')
        print(name, 'completed', flush=True)

    for i in range(4): run(f'warmup-{i}')
    run('full-audit', lean=False, verification=True)
    run('lean-audit', verification=True)
    run('fresh-reference')
    batch('reuse-check', [('reuse-before',{}),('reuse-interposed',dict(mode='inject')),('reuse-after',{})])
    reference = out/'fresh-reference'
    baseline = read(reference/'a1-clocks.json')
    checks = {}
    for name in ['full-audit','lean-audit','reuse-before','reuse-interposed','reuse-after']:
        d = out/name
        clocks = read(d/'a1-clocks.json')
        checks[name] = dict(logits_equal=(d/'logits.f32').read_bytes()==(reference/'logits.f32').read_bytes(),
                            access_equal=clocks['access_identity_fnv1a64']==baseline['access_identity_fnv1a64'],
                            tokens_equal=[x['token_id'] for x in clocks['tokens']]==[x['token_id'] for x in baseline['tokens']],
                            reset=read(d/'summary.json')['reset'])
    full, lean = out/'full-audit', out/'lean-audit'
    checks['snapshot_parity'] = all((lean/r['file']).read_bytes()==(full/r['file']).read_bytes() for r in rows(full/'snapshots.jsonl'))
    write(out/'identity-checks.json', checks)
    if not checks['snapshot_parity'] or not all(v['logits_equal'] and v['access_equal'] and v['tokens_equal'] for v in checks.values() if isinstance(v,dict)):
        raise RuntimeError('Mode/reuse equivalence failed')
    invoke([binary,'--calibrate',contract['target_ns'],64,out/'injector.json'],out/'logs/injector')
    overshoot = [r['overshoot_ns'] for r in read(out/'injector.json')]

    def endpoints(name):
        r=read(out/name/'a1-clocks.json')
        entry=next(b['ns'] for b in r['boundaries'] if b['event']=='request_entry')
        return [b['ns']-entry for b in r['boundaries'] if b['event']=='token_id_deliverable']

    reports={}
    for protocol in ['fresh_process','reuse_context']:
        c = dict(contract, protocol=protocol)
        write(out/f'{protocol}-contract.json', dict(contract=c,sha256=freeze(c)))
        requests=[]
        noise_orders=[['a','b'] if i%2==0 else ['b','a'] for i in range(args.calibration_pairs)]
        random.Random(4103).shuffle(noise_orders)
        write(out/f'{protocol}-noise-orders.json',noise_orders)
        for i,order in enumerate(noise_orders):
            for arm in order: requests.append((f'{protocol}-noise-{i}-{arm}',{}))
        if protocol=='reuse_context':
            batch(protocol+'-noise', [(f'{protocol}-noise-warmup-{i}',{}) for i in range(4)]+requests)
        else:
            for n,kw in requests: run(n,**kw)
        noises=[[b-a for a,b in zip(endpoints(f'{protocol}-noise-{i}-a'),endpoints(f'{protocol}-noise-{i}-b'))] for i in range(args.calibration_pairs)]
        evidence=dict.fromkeys(['frozen_before_calibration','independent_calibration','math_equal','access_identity_equal','frequency_recorded','warmup_fixed','crossover_order_fixed','kv_reset_verified','cache_reset_verified','backend_reset_verified'],True)
        qualification=[qualify(c,[r[k] for r in noises],overshoot,evidence) for k in range(8)]
        write(out/f'{protocol}-qualification.json',qualification)
        reports[protocol]={'qualification':qualification,'formal_status':'not_started'}
        if args.qualification_only or any(r['status']!='qualified' for r in qualification): continue
        sd=max(stdev([r[k] for r in noises]) for k in range(8))
        c['pairs']=min(256,max(30,math.ceil((float(t.ppf(1-c['alpha']/(2*c['endpoints']),29))*sd/5_000_000)**2)))
        write(out/f'{protocol}-formal-contract.json',dict(contract=c,sha256=freeze(c)))
        orders=[['zero','inject'] if i%2==0 else ['inject','zero'] for i in range(c['pairs'])]
        random.Random(c['order_seed']).shuffle(orders)
        write(out/f'{protocol}-orders.json',orders)
        requests=[(f'{protocol}-formal-{i}-{arm}',dict(mode=arm)) for i,order in enumerate(orders) for arm in order]
        if protocol=='reuse_context': batch(protocol+'-formal',[(f'{protocol}-formal-warmup-{i}',{}) for i in range(4)]+requests)
        else:
            for n,kw in requests: run(n,**kw)
        errors=[[b-a-c['target_ns'] for a,b in zip(endpoints(f'{protocol}-formal-{i}-zero'),endpoints(f'{protocol}-formal-{i}-inject'))] for i in range(c['pairs'])]
        results=[adjudicate(c,freeze(c),[r[k] for r in errors]) for k in range(8)]
        reports[protocol].update(formal_status='completed',results=results,errors_ns=errors)
        write(out/'result.json', reports)
    write(out/'result.json',reports)

if __name__=='__main__': main()
