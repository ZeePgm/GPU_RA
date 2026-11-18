"""Bounded correctness runs. Use the existing V3 process and resource observer."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIVE = ROOT / 'experiments/v3-route-perf-five-001'
SAMPLED = ROOT / 'experiments/v3-three-routes-sampled-001'
sys.path.insert(0, str(SAMPLED))
import sampled_run as X
import stage1 as S

sys.path.insert(0, str(ROOT / 'experiments/gpu_ffn'))
import run as F

read, write, require, sha = S.read, S.write, S.require, S.sha
PROMPT = SAMPLED / 'prompt-44.txt'
H0 = ROOT / 'joint/source/experiments/gpu_ffn/H0-512MiB.ini'


def run(args):
    case, out = args.case, args.output
    exe = args.build_dir / 'bin'
    require(args.resource_root.resolve() == Path('E:/v1_update').resolve(), 'Full resource root required')
    out.mkdir(parents=True, exist_ok=False)
    require(sha(args.model) == S.MODEL_SHA and sha(PROMPT) == S.PROMPT_SHA, 'Model or prompt identity')
    receipt = read(args.receipt)
    require(receipt['commit'] == subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), 'Build commit')
    require(S.audit_build(args.build_dir) == receipt['build'] and sha(args.build_dir/'down.ptx') == receipt['down_ptx'], 'Build identity')
    profile = read(FIVE/'profile.json')
    profile.pop('sampled_scope')
    profile['model_path'] = str(args.model)
    write(out/'profile.json',profile)
    env = {'HBFLAB_JOINT_KV':'native','HBFLAB_M4_PAGED':'0','LLAMA_GRAPH_REUSE_DISABLE':'1',
           'HBFLAB_SAMPLED_TOKEN_ONLY':'1','HBFLAB_SAMPLED_ROUTE':case,'HBFLAB_SAMPLED_AUDIT':'summary',
           'HBFLAB_SAMPLED_PROFILE':str(out/'profile.json'),'HBFLAB_SYMBOLIC_CONFIG':str(H0)}

    def execute(name, cmd, require_ggml=True, overrides=None):
        return X.execute(args, cmd, name, out/name, extra_env=env if overrides is None else overrides,
                         timeout=300, require_ggml=require_ggml, disk_cap=128*1024**2)

    if case == 'second-f':
        layout = F.model_layout(args.model)
        regions = F.select_regions(layout,'F1')
        write(out/'layout-all-ffn.json', layout)
        cfg = {'model_path':str(args.model),'gguf_sha256':S.MODEL_SHA,'context':512,'batch':256,'ubatch':256,
               'capture_compute':True,'fixed_segment_ps':0,'hbf':{'enabled':False}}
        def dispatch(name):
            path=out/(name+'-config.json')
            write(path,cfg)
            execute(name,[exe/'symbolic_f.exe',args.model,PROMPT,out/name,'1','timing',path],
                    overrides={'LLAMA_GRAPH_REUSE_DISABLE':'1','HBFLAB_M4_PAGED':'0'})
        dispatch('native')
        header = {'model_sha256':S.MODEL_SHA,'prompt_sha256':S.PROMPT_SHA,'binary_sha256':sha(exe/'symbolic_f.exe'),
                  'native_config_sha256':sha(out/'native-config.json'),'native_events_sha256':sha(out/'native/events.jsonl')}
        table = F.capture_table(out/'native',header)
        write(out/'compute-table.json',table)
        cfg.update(capture_compute=False,compute_table_path=str(out/'compute-table.json'),expected_compute_header=table['header'],
                   hbf={'enabled':True,'regions':regions,'effective_config_path':str(H0)})
        dispatch('F1')
        result=F.audit(out,regions,table)
        write(out/'acceptance.json',{'case':case,'coverage':result['coverage'],'online_audit':result['online_audit'],
              'clock_conserved':result['all_conservation_residuals_zero'],'logits_bitwise':result['native_logits_bitwise_equal'],
              'drained':result['closed_and_drained'],'status':result['status']})
        return

    frozen=read(FIVE/'input-spec.json')
    ref=args.reference/'result'
    for file,key in [('logits.f32','full_logits'),('native-kv-state.bin','kv_identity')]:
        require(sha(ref/file)==frozen['new_native_reference'][key]['sha256'],'Frozen reference '+file)
    # Run one selected norm through the real model, including prefill and decode.
    inventory=read(FIVE/'inventory.json')
    for t in inventory['all_tensors']:
        t['selected']=t['name']=='blk.10.ffn_norm.weight'
    write(out/'selection.json',inventory)
    execute('model',[exe/'five_model.exe',args.model,PROMPT,out/'model',out/'selection.json','4'])
    for name in ('logits.f32','native-kv-state.bin','native-kv-positions.json','framework-release.json'):
        require(sha(out/'model'/name)==sha(ref/name),'Model reference mismatch '+name)
    got,expected=read(out/'model/summary.json'),read(ref/'summary.json')
    require(got['generated_tokens']==expected['generated_tokens'] and got['prompt_tokens']==expected['prompt_tokens']
            and got['selected_tensors']==1 and got['consumer_records']==4,'Tokens or consumer coverage')
    result={'case':case,'model_tokens':got['generated_tokens'],'consumers':4,'logits_bitwise':True,'kv_bitwise':True}
    if case=='ptx':
        r=read(out/'model/ptx-summary.json')
        require(r['rows']==47 and r['modeled_requests']==47*1024 and r['strategy']=='ptx_scalar','Model arithmetic hook')
        execute('down',[exe/'r2_down.exe','ptx','modeled','3072','1','0',args.build_dir/'down.ptx','1',out/'down.json','1'],False)
        down=read(out/'down.json')
        require(down['status']=='pass' and down['drained'] and down['submitted']==3072 and
                down['registered_globals']==down['completed_globals'],'R2 row-zero coverage')
        result.update(norm_requests=r['modeled_requests'],down_row=0,down_requests=down['submitted'])
    fixture_dir=args.reference/'fixtures'
    manifest=fixture_dir/'manifest.json'
    require(sha(manifest)==frozen['captured_fixture_manifest']['sha256'],'Fixture manifest identity')
    index={Path(row['path']).name:row['sha256'] for row in read(manifest)['files']}
    for scenario in (['w1-d','w3-d'] if case=='ptx' else ['w3-d','w4-d']):
        fixture=fixture_dir/(scenario+'.json')
        require(sha(fixture)==index[fixture.name],'Fixture metadata identity')
        meta=read(fixture)
        for item in [*meta['inputs'],meta['output']]:
            require(sha(fixture_dir/item['file'])==index[item['file']],'Fixture binary identity')
        call=[exe/'five_weight.exe',args.model,FIVE/'inventory.json',fixture,out/scenario,case,'0','summary']
        if case=='ptx' and scenario=='w3-d':
            try:
                execute(scenario,call)
            except ValueError:
                proc=read(out/(scenario+'.process.json'))
                require(proc['returncode']==1 and 'R2 model PTX supports one contiguous F32 norm' in
                        (out/(scenario+'.log')).read_text(encoding='utf8') and not (out/scenario/'summary.json').exists(),
                        'Wrong full down rejection')
                result['full_down_rejected']=True
                continue
            raise ValueError('Full down unexpectedly accepted by norm entry')
        execute(scenario,call)
        child=out/scenario
        require(read(child/'calls.json')[0]['native_bitwise_equal'] and sha(child/'call-0.bin')==index[meta['output']['file']],
                'Component differs from frozen native')
        size=meta['weight']['bytes'] if scenario!='w4-d' else meta['weight']['strides'][1]
        if case=='symbolic':
            s=read(child/'session.json')
            require(len(s['accesses'])==1 and s['read_globals']==(size+4095)//4096 and
                    s['online_audit']['read_fragments']==size//64 and s['host_drained'] and s['device_drained'],'F range accounting')
            if scenario=='w4-d':
                row=int.from_bytes((fixture_dir/meta['inputs'][0]['file']).read_bytes(),'little',signed=True)
                require(s['accesses'][0]['global_address']==s['initialization'][0]['global_address']+row*size,'Gather row')
        if case=='cuda':
            b=read(child/'b2-summary.json')
            require(b['requests']==size//512 and b['launches']==(size+262143)//262144 and
                    b['publications']==1 and b['lane_checks']==size//16 and b['drained'],'B2 publication/accounting')
        result[scenario]={'output_bitwise':True,'bytes':size}
    result['status']='pass'
    write(out/'acceptance.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=['second-f','symbolic','ptx','cuda'],required=True)
    for name in ('build-dir','cuda-root','model','reference','resource-root','output','receipt'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
