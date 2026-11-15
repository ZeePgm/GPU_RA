"""Frozen exploratory R2 operator protocol; independent process is the sample unit."""
import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from run_llama_integrated import environment,read,write
from setup_llama_baseline import sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',type=Path,default=ROOT/'build-r2-down-20261107')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build=args.build.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    exe=build/'readonly_r2_down.exe';env=environment(build)
    widths=[1536,3072,6144]
    boundary=[2,62,64,66,510,512,514,2050,16384]
    source_files=[*Path(__file__).parent.glob('*.cpp'),*Path(__file__).parent.glob('*.cu'),Path(__file__),
                  Path(__file__).parent/'CMakeLists.txt',ROOT/'adapters/llama_cpp/integrated_norm_explicit.cu',
                  ROOT/'adapters/llama_cpp/integrated_weight_mul.cu',ROOT/'include/hbflab/readonly_access.hpp',ROOT/'src/ptx.cpp',ROOT/'adapters/llama_cpp/integrated_down_scalar.cu',
                  ROOT/'tools/run_llama_integrated.py',ROOT/'tools/platform_environment.py']
    artifacts=[exe,build/'CMakeCache.txt',build/'norm.ptx',build/'down.ptx',ROOT/'build-r1-20261107/release-identity.json',ROOT/'build-r1-20261107/realgpu_runtime.lib',ROOT/'build-r1-20261107/eventlab/hbf_eventlab.lib']
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.current.sm','--format=csv'],capture_output=True,text=True,check=True)
    protocol={'status':'frozen','sample_unit':'fresh independent process; mean of 3 measured launches after one warmup',
              'primary':'wall_ns from launch API entry through synchronization',
              'secondary':['submit_ns','stream_interval_ns','process_cpu_ns','module_setup_ns','first_launch_ns'],
              'widths':widths,'boundary_widths':boundary,'pilot_aa_pairs_per_strategy':4,
              'paired_ab_sessions_per_condition':12,'repeats_per_process':3,
              'sample_budget_basis':'fixed exploratory budget; no predeclared application delta_min, no default-policy selection',
              'interval':'paired mean difference; 20000 bootstrap resamples of process-pairs; nominal 95% Bonferroni-adjusted exploratory intervals over 6 conditions',
              'decision':'report estimates and uncertainty; no equivalence or speedup qualification without delta_min and P0 precision qualification',
              'random_seed':20261107,'tokens_correctness':[1,2],'data_patterns':['dyadic','non_dyadic_trigonometric'],'performance_tokens':1,'scope':'same row zero, scalar half2 4B requests, same 256-thread reduction; default model warp route unchanged','gpu_start':hardware.stdout,
              'environment':{key:env.get(key) for key in ['CUDA_PATH','CUDA_VISIBLE_DEVICES']},
              'source_files':[{'path':str(p),'sha256':sha(p)} for p in source_files],
              'artifacts':[{'path':str(p),'sha256':sha(p)} for p in artifacts]}
    write(out/'protocol.json',protocol)
    rng=random.Random(protocol['random_seed']);records=[]

    def run(name,strategy,mode,width,timed=False,tokens=1,pattern=0):
        path=out/(name+'.json')
        command=[str(exe),strategy,mode,str(width),'3','1' if timed else '0',str(build/'down.ptx'),str(tokens),str(path),str(pattern)]
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(command,env=env,stdout=log,stderr=log,check=True,timeout=120)
        result=read(path)
        assert result['status']=='pass' and result['drained']
        records.append(path)
        return result

    correctness=[]
    for width in widths+boundary:
      for tokens in [1,2]:
       for pattern in [0,1]:
        results={};tag=f'{width}-t{tokens}-p{pattern}'
        for strategy in ['ptx','cuda']:
            for mode in ['native','bypass','modeled']:
                results[strategy,mode]=run(f'check-{strategy}-{mode}-{tag}',strategy,mode,width,mode=='modeled',tokens,pattern)
        hashes={r['output_sha256'] for r in results.values()}
        assert len(hashes)==1,'arithmetic changed'
        assert len({r['weight_sha256'] for r in results.values()})==1
        expected=sorted([[i*4,4,1,1] for t in range(tokens) for i in range(width//2)])
        for strategy in ['ptx','cuda']:
            assert results[strategy,'modeled']['parents']==expected
        replay=out/f'replay-{tag}.json'
        subprocess.run([str(exe),'replay',str(out/f'check-ptx-modeled-{tag}.json'),str(out/f'check-cuda-modeled-{tag}.json'),str(replay)],env=env,check=True,timeout=120)
        correctness.append({'ffn':width,'tokens':tokens,'pattern':pattern,'status':'pass','exact_outputs':True,'exact_parents':True,'fixed_arrival':read(replay)})
        print('correctness',tag,'passed',flush=True)
    write(out/'correctness.json',{'status':'pass','cases':correctness})
    failures=[]
    for strategy in ['ptx','cuda']:
      for mode in ['fault','invalid_range']:
       for tokens in [1,2]:
        path=out/f'negative-{strategy}-{mode}-{tokens}.json'
        cmd=[str(exe),strategy,mode,'1536','1','0',str(build/'down.ptx'),str(tokens),str(path)]
        with path.with_suffix('.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=log,check=True,timeout=30)
        value=read(path);assert value['status']=='expected_failure' and value['cuda_drained']
        failures.append(value)
    write(out/'failures.json',{'status':'pass','cases':failures})
    pilots=[]
    for strategy in ['ptx','cuda']:
        for i in range(4):
            runs=[run(f'aa-{strategy}-{i}-{j}',strategy,'modeled',3072) for j in range(2)]
            pilots.append({'strategy':strategy,'pair':i,'difference_ns':statistics.mean(s['wall_ns'] for s in runs[0]['samples'])-statistics.mean(s['wall_ns'] for s in runs[1]['samples'])})
    write(out/'aa-pilot.json',{'pairs':pilots,'qualification':'exploratory; delta_min not specified'})
    conditions=[(width,mode) for width in widths for mode in ['bypass','modeled']]
    schedule=[]
    # Balance order within each condition, randomize both pair order and conditions.
    for width,mode in conditions:
        orders=[['ptx','cuda']]*6+[['cuda','ptx']]*6
        rng.shuffle(orders)
        schedule.extend((width,mode,i,order) for i,order in enumerate(orders))
    rng.shuffle(schedule)
    write(out/'schedule.json',schedule)
    pairs=[]
    for width,mode,index,order in schedule:
        results={strategy:run(f'ab-{width}-{mode}-{index}-{strategy}',strategy,mode,width) for strategy in order}
        pair={'width':width,'mode':mode,'index':index,'order':order,
              'ptx':{key:statistics.mean(s[key] for s in results['ptx']['samples']) for key in results['ptx']['samples'][0]},
              'cuda':{key:statistics.mean(s[key] for s in results['cuda']['samples']) for key in results['cuda']['samples'][0]}}
        pairs.append(pair);write(out/'pairs.json',pairs)
        print('pair',width,mode,index,'passed',flush=True)
    summaries=[];randoms=np.random.default_rng(20261107)
    for width,mode in conditions:
        selected=[p for p in pairs if p['width']==width and p['mode']==mode]
        differences=np.array([p['ptx']['wall_ns']-p['cuda']['wall_ns'] for p in selected])
        bootstrap=differences[randoms.integers(0,len(differences),(20000,len(differences)))].mean(axis=1)
        quantile=.05/(2*len(conditions));interval=np.quantile(bootstrap,[quantile,1-quantile])
        summaries.append({'width':width,'mode':mode,'process_pairs':len(selected),
                          'ptx_mean_wall_ns':statistics.mean(p['ptx']['wall_ns'] for p in selected),
                          'cuda_mean_wall_ns':statistics.mean(p['cuda']['wall_ns'] for p in selected),
                          'ptx_minus_cuda_mean_ns':float(differences.mean()),'simultaneous_interval_ns':interval.tolist(),
                          'order_means_ns':{str(order):statistics.mean(p['ptx']['wall_ns']-p['cuda']['wall_ns'] for p in selected if p['order'][0]==order) for order in ['ptx','cuda']}})
    # Verify that sampled binaries/source inputs did not change mid-study.
    for entry in protocol['source_files']+protocol['artifacts']:
        assert sha(Path(entry['path']))==entry['sha256'],'study input changed'
    write(out/'result.json',{'down_correctness':'pass','fixed_arrival':'pass','performance':'exploratory',
                           'down':'supported_scalar_half2','failures':'pass','conditions':summaries,
                           'normal_sessions':len(records),'correctness_cases':len(correctness),'independent_ab_pairs':len(pairs)})
    write(out/'manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]})


if __name__=='__main__':
    main()
