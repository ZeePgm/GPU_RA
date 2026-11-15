"""Run independent old/new model comparisons for the R1 management refactor."""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from run_llama_integrated import environment, read, rows, validate, write
from setup_llama_baseline import ROOT, sha


def run_model(build, out, profile, prompt):
    out.mkdir(parents=True)
    (out/'profile.json').write_bytes(profile.read_bytes())
    (out/'prompt.txt').write_bytes(prompt.read_bytes())
    config=read(profile)
    env=environment(build)
    for mode in ('native','bypass','modeled'):
        command=[str(build/'bin/realgpu_llama.exe'),config['model_path'],str(out/'prompt.txt'),
                 str(out/mode),'2','1' if mode=='modeled' else '0',mode,str(out/'profile.json')]
        print(out.name,mode,flush=True)
        with (out/(mode+'.stdout.log')).open('w') as stdout, (out/(mode+'.stderr.log')).open('w') as stderr:
            subprocess.run(command,cwd=build/'bin',env=env,stdout=stdout,stderr=stderr,check=True,timeout=3600)
    return validate(out)


def logical_trace(folder):
    host=defaultdict(list)
    for event in rows(folder/'modeled/hbf-host.jsonl'):
        host[(event['step'],event['launch'])].append(event)
    result=[]
    for filename,strategy in [('ptx-calls.jsonl','ptx_scalar'),('hbf-calls.jsonl','cuda_warp')]:
        for call in rows(folder/'modeled'/filename):
            base=call['weight'] if strategy=='ptx_scalar' else call['address']
            key=(strategy,call['step'],call.get('row',call.get('row_base')),call.get('rows',1))
            trace=host.pop((call['step'],call['launch']))
            # One host record is one parent access. Keep each parent's span and
            # sector signature; producer/ticket IDs and real addresses can differ.
            parents=sorted((e['address']-base,e['bytes'],e['sectors'],e['span_count']) for e in trace)
            result.append((key,parents))
    if host:
        raise RuntimeError('unmatched host accesses')
    return sorted(result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    builds=[args.baseline.resolve(),args.candidate.resolve()]
    cases=[('qwen-short','qwen3-0.6b','prompt.txt'),
           ('qwen-long','qwen3-0.6b','prompt-integrated-qwen-long.txt'),
           ('smol-short','smollm2-135m','prompt-integrated-smol-short.txt'),
           ('qwen-1.7b','qwen3-1.7b','prompt.txt')]
    write(out/'protocol.json',{'cases':cases,'tokens':2,'acceptance':'existing model validator plus exact old/new logits, logical parents and publication records',
          'artifacts':[{ 'path':str(p),'sha256':sha(p)} for b in builds for p in sorted((b/'bin').glob('*')) if p.is_file()]})
    results=[]
    for case,model,prompt in cases:
        folders=[]
        for label,build in zip(('old','new'),builds):
            folder=out/(case+'-'+label)
            run_model(build,folder,ROOT/'configs/integrated'/f'{model}.json',ROOT/'adapters/llama_cpp'/prompt)
            folders.append(folder)
        old,new=folders
        for mode in ('native','bypass','modeled'):
            if (old/mode/'logits.f32').read_bytes()!=(new/mode/'logits.f32').read_bytes():
                raise RuntimeError('old/new logits changed: '+case+' '+mode)
            for filename in ('summary.json',):
                a,b=read(old/mode/filename),read(new/mode/filename)
                if a['generated_tokens']!=b['generated_tokens'] or a['prompt_tokens']!=b['prompt_tokens']:
                    raise RuntimeError('old/new tokens changed')
        if logical_trace(old)!=logical_trace(new):
            raise RuntimeError('old/new parent access multiset changed: '+case)
        if rows(old/'modeled/hbf-steps.jsonl')!=rows(new/'modeled/hbf-steps.jsonl'):
            raise RuntimeError('graph coverage changed')
        strategies=read(new/'modeled/readonly-summary.json')['strategies']
        summary=read(new/'modeled/hbf-summary.json')
        if sum(e['requests'] for e in strategies)!=summary['selected_requests']:
            raise RuntimeError('shared strategy audit mismatch')
        results.append({'case':case,'status':'pass','strategies':strategies,
                        'selected_requests':summary['selected_requests'],'steps':summary['steps'],
                        'old_model_time_ps':read(old/'modeled/hbf-summary.json')['runtime']['model_now_ps'],
                        'new_model_time_ps':summary['runtime']['model_now_ps']})
        write(out/'result.json',{'status':'running','cases':results})
    # Final backend failure must fail closed in both binaries.
    failures=[]
    for label,build in zip(('old','new'),builds):
        profile=ROOT/'configs/integrated/qwen3-0.6b.json'
        folder=out/('fault-'+label)
        command=[str(build/'bin/realgpu_llama.exe'),read(profile)['model_path'],str(ROOT/'adapters/llama_cpp/prompt.txt'),str(folder),'2','0','fault',str(profile)]
        r=subprocess.run(command,cwd=build/'bin',env=environment(build),capture_output=True,timeout=180)
        (out/('fault-'+label+'.log')).write_bytes(r.stdout+r.stderr)
        if r.returncode==0 or (folder/'summary.json').exists():
            raise RuntimeError('fault did not fail closed')
        failures.append({'binary':label,'returncode':r.returncode,'success_summary':False})
    write(out/'result.json',{'status':'pass','cases':results,'faults':failures})
    write(out/'manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]})


if __name__=='__main__':
    main()
