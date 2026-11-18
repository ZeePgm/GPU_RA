"""One full output projection through the restored executable under the reduced plan."""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIVE = ROOT / 'experiments/v3-route-perf-five-001'
sys.path.insert(0, str(FIVE))
import workflow as W

X, S = W.X, W.S
read, write, require, sha = W.read, W.write, W.need, W.sha


def check_build(args, cfg):
    require(sha(args.receipt) == cfg['receipt_sha256'], 'Restored build receipt identity')
    receipt = read(args.receipt)
    def tree(rev):
        return subprocess.check_output(['git', 'rev-parse', rev+':joint/source'], cwd=ROOT).strip()
    require(tree('HEAD') == tree(receipt['commit']), 'Candidate source differs from the accepted build')
    for key in ['binaries','metadata','toolchains']:
        for row in receipt['build'][key]:
            require(sha(Path(row['path'])) == row['sha256'], 'Build input or artifact changed '+row['path'])
    binaries = {Path(row['path']).resolve() for row in receipt['build']['binaries']}
    require(binaries == {p.resolve() for p in (args.build_dir/'bin').iterdir()
                         if p.suffix.lower() in ('.dll','.exe')}, 'Runtime directory mismatch')
    for name, digest in receipt['ptx'].items():
        require(sha(args.build_dir/name) == digest, 'PTX identity')


def run(a):
    out, case, route = a.output, a.case, a.route
    cfg = read(a.config)['parameters']
    name = f'qual-{case}-{route}-001'
    require(out.parent.name == name and name in cfg['qualification_run_ids'], 'Unregistered case')
    require(a.resource_root.resolve() == Path('E:/v1_update').resolve(), 'Full resource root')
    caps = read(FIVE/'config.json')['parameters']['budgets']
    require(shutil.disk_usage(out.parent).free >= 16*1024**3 and
            X.directory_bytes(out.parent.parent)+32*1024**2 <= 2*1024**3, 'Disk budget')
    start_cpu = time.process_time()
    out.mkdir(exist_ok=False)
    fixture = a.fixtures/(case+'.json')
    spec = read(FIVE/'input-spec.json')
    manifest = a.fixtures/'manifest.json'
    require(sha(a.model)==S.MODEL_SHA and sha(manifest)==spec['captured_fixture_manifest']['sha256'], 'Frozen inputs')
    index = {Path(f['path']).name:f['sha256'] for f in read(manifest)['files']}
    require(sha(fixture)==index[fixture.name], 'Fixture identity')
    meta = read(fixture)
    for item in [*meta['inputs'], meta['output']]:
        require(sha(a.fixtures/item['file']) == index[item['file']], 'Fixture payload')
    require(meta['weight']['shape']==[1024,151936,1,1] and meta['tensor']=='output.weight', 'Full projection')
    check_build(a,cfg)
    profile = read(FIVE/'profile.json')
    del profile['sampled_scope']
    profile['model_path'] = str(a.model)
    write(out/'profile.json',profile)
    env = {'HBFLAB_JOINT_KV':'native','HBFLAB_M4_PAGED':'0','LLAMA_GRAPH_REUSE_DISABLE':'1',
           'HBFLAB_SAMPLED_TOKEN_ONLY':'1','HBFLAB_SAMPLED_AUDIT':'summary','HBFLAB_SAMPLED_ROUTE':route,
           'HBFLAB_SAMPLED_PROFILE':str(out/'profile.json'),
           'HBFLAB_SYMBOLIC_CONFIG':str(ROOT/'joint/source/experiments/gpu_ffn/H0-512MiB.ini')}
    before = W.environment_probe()
    write(out/'environment-before.json',before)
    child = out/'result'
    cmd = [a.build_dir/'bin/five_weight.exe',a.model,FIVE/'inventory.json',fixture,child,route,'0','summary']
    proc = X.execute(a,cmd,'result',child,extra_env=env,timeout=caps['process_wall_seconds']['output_weight'],
                     require_ggml=True,disk_cap=32*1024**2)
    after = W.environment_probe()
    write(out/'environment-after.json',after)
    require(W.environment_identity(before)==W.environment_identity(after), 'Environment changed')
    got, calls, stats = read(child/'summary.json'), read(child/'calls.json'), read(child/'route-summary.json')
    size = meta['weight']['bytes']
    require(len(calls)==1 and calls[0]['native_bitwise_equal'] and
            sha(child/'call-0.bin')==index[meta['output']['file']], 'Full native projection output')
    require(got['status']=='pass' and got['borrowed_unchanged'] and got['calls']==1 and got['repeated_calls']==0 and
            stats['consumers']==1 and stats['steps']==1 and stats['drained'] and stats['requested_bytes']==size,
            'Restored consumer coverage')
    require(got['first_call_ns']==calls[0]['end_ns']-calls[0]['begin_ns']>0, 'Completion timing')
    if route=='cuda':
        b=read(child/'b2-summary.json')
        require(b['drained'] and b['bytes']==size and b['requests']==size//512 and b['lane_checks']==size//16 and
                b['launches']==(size+262143)//262144 and b['publications']==1,'B2 coverage')
    else:
        g=read(child/'session.json')
        require(len(g['accesses'])==1 and g['read_globals']==(size+4095)//4096 and
                g['online_audit']['read_fragments']==size//64 and g['host_drained'] and g['device_drained'],'F coverage')
    require(X.directory_bytes(out)<32*1024**2 and (out/'result.log').stat().st_size<=1024**2, 'Run output budget')
    write(out/'acceptance.json',dict(status='pass',phase='qualification',formal_sample=False,case=case,route=route,
          rows=151936,weight_bytes=size,logits_bitwise_equal=True,first_call_ns=got['first_call_ns'],
          init_ns=got['initialization_ns'],first_result_ns=got['first_result_ns'],wall_seconds=proc['wall_seconds'],
          process_cpu_seconds=proc['final_metrics']['cpu_seconds'],coordinator_cpu_seconds=time.process_time()-start_cpu))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--route',required=True,choices=['symbolic','cuda'])
    p.add_argument('--case',required=True,choices=['w5-p','w5-d'])
    for key in ['build-dir','cuda-root','model','fixtures','receipt','config','resource-root','output']:
        p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
