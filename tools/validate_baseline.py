"""Validate the active backend, model operators and A1/A2/A3 execution paths."""
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from integrated_build_identity import verify_build
from platform_environment import runtime_environment,cache_value

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--build-dir',type=Path,default=ROOT/'build-portability-model')
    ap.add_argument('--core-build',type=Path,default=ROOT/'build-cuda')
    args=ap.parse_args();out=args.output.resolve();assert not out.exists(),'choose a fresh output directory';out.mkdir(parents=True)
    identity=verify_build(args.build_dir);env=runtime_environment(args.build_dir,ROOT)
    env.update(GGML_CUDA_CUBLAS_COMPUTE_TYPE='f32',GGML_CUDA_DISABLE_FUSION='1',NVIDIA_TF32_OVERRIDE='0')
    py=sys.executable
    def run(name,argv,timeout=1200):
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,argv)),stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,timeout=timeout)
        if p.returncode:raise RuntimeError(name+' failed; see '+str(out/(name+'.log')))
        print(name+' PASS',flush=True)
    cmake=cache_value(args.core_build,'CMAKE_COMMAND');assert cmake
    ctest=Path(cmake).with_name('ctest.exe' if os.name=='nt' else 'ctest')
    run('core',[ctest,'--test-dir',args.core_build,'--output-on-failure','-j','1','--output-junit',out/'core.xml'])
    run('operators',[py,ROOT/'tools/check_llama_integrated.py','--build-dir',args.build_dir,'--profiles-dir',ROOT/'configs/integrated','--output',out/'operators'])
    run('python-contract',[py,ROOT/'tests/integrated_validation_contract.py'])
    prompt=out/'prompt.txt';prompt.write_text('A computer stores data in memory. '*6+'Explain this.',encoding='utf-8')
    for name in ['smollm2-135m','qwen3-0.6b','qwen3-1.7b']:
        run('model-'+name,[py,ROOT/'tools/run_llama_integrated.py','--profile',ROOT/'configs/integrated'/(name+'.json'),'--build-dir',args.build_dir,'--prompt',prompt,'--tokens','2','--output',out/('model-'+name)])
    run('a2',[py,ROOT/'experiments/gpu_a2/run.py','--study',out/'a2'])
    run('a3',[py,ROOT/'experiments/gpu_a3/run.py','--study',out/'a3'])
    run('a3-analysis',[py,ROOT/'experiments/gpu_a3/analyze.py','--study',out/'a3'])
    assert read(out/'a3/result.json')['response_model_status']=='pass'
    p=ROOT/'output/gpu-s/20261101-s8-002/G3';cfg=read(p/'configs/native-verify.json')
    cfg.update(model_path=str(ROOT/'models/Qwen3-0.6B-F16.gguf'),conversion=str(ROOT/'models/Qwen3-0.6B-conversion.json'),compute_table_path=str(p/'compute-table.json'))
    gates=[{'node_id':r['node_id'],'step':r['step'],'wait_ns':100000} for r in map(json.loads,(p/'G3-timing-1/gates.jsonl').read_text().splitlines())]
    clocks={}
    for mode in ['zero','model','inject']:
        c=dict(cfg);c['a1']={'mode':mode,'gates':gates,'lean':True,'session_config':str(p/'effective-H0.ini')}
        config=out/('a1-'+mode+'.json');write(config,c);dest=out/('a1-'+mode)
        run('a1-'+mode,[ROOT/'build-s3/bin/gpu_a1.exe',c['model_path'],p/'prompt.txt',dest,'8','timing',config])
        clocks[mode]=read(dest/'a1-clocks.json')
        assert (dest/'logits.f32').read_bytes()==(p/'G3-timing-1/logits.f32').read_bytes(),'A1 logits differ from preserved baseline'
    for i,(z,m,j) in enumerate(zip(clocks['zero']['tokens'],clocks['model']['tokens'],clocks['inject']['tokens'])):
        assert m['model_wait_ps']==sum(g['wait_ns'] for g in gates if g['step']<=i)*1000
        assert z['model_wait_ps']==j['model_wait_ps']==0
        assert z['model_compute_ps']==m['model_compute_ps']==j['model_compute_ps']
    assert verify_build(args.build_dir)==identity
    write(out/'validation.json',{'status':'pass','backend':identity['backend'],'operators':len(read(out/'operators/validation.json')['cases']),'python_contracts':5,'models':['smollm2-135m','qwen3-0.6b','qwen3-1.7b'],'generated_tokens_per_model':2,'a1':{'modes':3,'tokens_per_mode':8,'logits_exact_to_historical_baseline':True,'clock_contracts':True,'statistical_equivalence_rerun':False},'a2':read(out/'a2/result.json'),'a3_counts':read(out/'a3/counts-result.json'),'a3_response_status':read(out/'a3/result.json')['response_model_status']})
    print('Active baseline validation PASS',flush=True)

if __name__=='__main__':main()
