"""Compare actual model norm strategies under the frozen scalar contract."""
import argparse
import os
import subprocess
from pathlib import Path
from run_llama_integrated import ROOT,environment,read,rows,write
from run_readonly_r1 import logical_trace,run_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build=args.build.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    os.environ['HBFLAB_KV_WRITE_MODE']='host'
    os.environ.pop('HBFLAB_STORAGE_CONFIG',None)
    cases=[('qwen-short','qwen3-0.6b','prompt.txt'),('qwen-long','qwen3-0.6b','prompt-integrated-qwen-long.txt'),
           ('smol-short','smollm2-135m','prompt-integrated-smol-short.txt'),('qwen-1.7b','qwen3-1.7b','prompt.txt')]
    write(out/'protocol.json',{'cases':cases,'strategies':['ptx','cuda_scalar'],'tokens':2,'kv_write_mode':'host',
                             'contract':'exact strategy logits and parents, existing native tolerances'})
    results=[]
    for case,model,prompt in cases:
        folders=[]
        for strategy in ['ptx','cuda_scalar']:
            os.environ['HBFLAB_R2_NORM_STRATEGY']=strategy
            folder=out/(case+'-'+strategy)
            run_model(build,folder,ROOT/'configs/integrated'/f'{model}.json',ROOT/'adapters/llama_cpp'/prompt)
            folders.append(folder)
        a,b=folders
        for mode in ['native','bypass','modeled']:
            assert (a/mode/'logits.f32').read_bytes()==(b/mode/'logits.f32').read_bytes(),'strategy arithmetic changed'
            assert read(a/mode/'summary.json')['generated_tokens']==read(b/mode/'summary.json')['generated_tokens']
        assert logical_trace(a)==logical_trace(b),'strategy parent coverage changed'
        assert rows(a/'modeled/hbf-steps.jsonl')==rows(b/'modeled/hbf-steps.jsonl')
        cuda_summary=read(b/'modeled/readonly-summary.json')['strategies']
        assert cuda_summary[0]['strategy']=='cuda_scalar'
        kernels={}
        for strategy,folder in zip(['ptx','cuda_scalar'],folders):
            selected=[k for k in rows(folder/'modeled/kernels.jsonl') if 'm6_weight_mul' in k['name']]
            assert all(k['grid']==[2,1,1] and k['block']==[128,1,1] for k in selected)
            kernels[strategy]={'count':len(selected),'cupti_sum_kernel_ns':sum(k['end_ns']-k['start_ns'] for k in selected)}
        results.append({'case':case,'status':'pass','selected_requests':read(b/'modeled/hbf-summary.json')['selected_requests'],
                        'norm_requests':cuda_summary[0]['requests'],'audited_kernel_durations':kernels})
        write(out/'result.json',{'status':'running','cases':results})
    failures=[]
    for strategy in ['ptx','cuda_scalar','unsupported']:
        os.environ['HBFLAB_R2_NORM_STRATEGY']=strategy
        profile=ROOT/'configs/integrated/qwen3-0.6b.json';folder=out/('fault-'+strategy)
        mode='modeled' if strategy=='unsupported' else 'fault'
        cmd=[str(build/'bin/realgpu_llama.exe'),read(profile)['model_path'],str(ROOT/'adapters/llama_cpp/prompt.txt'),str(folder),'2','0',mode,str(profile)]
        r=subprocess.run(cmd,cwd=build/'bin',env=environment(build),capture_output=True,timeout=180)
        log=r.stdout+r.stderr;(out/('fault-'+strategy+'.log')).write_bytes(log)
        assert r.returncode!=0 and not (folder/'summary.json').exists()
        expected=b'unsupported norm access strategy' if strategy=='unsupported' else b'shared read terminal failure'
        assert expected in log,'wrong failure cause'
        failures.append({'strategy':strategy,'returncode':r.returncode,'expected_failure':expected.decode()})
    write(out/'result.json',{'status':'pass','cases':results,'failures':failures})


if __name__=='__main__':
    main()
