"""Single-call correctness checks with frozen model captures and the V3 resource observer."""
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

read, write, require, sha = S.read, S.write, S.require, S.sha
PROMPT = SAMPLED / 'prompt-44.txt'
H0 = ROOT / 'joint/source/experiments/gpu_ffn/H0-512MiB.ini'


def build_identity(build):
    return {'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
            'build': S.audit_build(build), 'ptx': {name: sha(build / name) for name in
            ('weight_kernel.ptx','down.ptx','r2_half_to_float.ptx')}}


def projection_window(folder, fixture, inventory, frozen):
    """Use the same nonzero row offset in the original GGUF, HBF layout and native output."""
    folder.mkdir()
    rows = read(HERE / 'config.json')['parameters']['output_projection_last_rows']
    weight, output = fixture['weight'], fixture['output']
    entry = next(t for t in inventory['all_tensors'] if t['name'] == fixture['tensor'])
    total = weight['shape'][1]
    require(entry['shape'] == weight['shape'][:2] and output['shape'] == [total,1,1,1], 'Projection shape')
    first = total - rows
    stride = weight['strides'][1]
    require(first > 0 and first*stride % 32768 == 0 and rows*stride % 32768 == 0, 'Group alignment')
    entry.update(file_offset=entry['file_offset']+first*stride, logical_base=entry['logical_base']+first*stride,
                 bytes=rows*stride, reserved_bytes=rows*stride, shape=[weight['shape'][0],rows])
    weight.update(shape=[weight['shape'][0],rows,1,1], strides=[2,stride,rows*stride,rows*stride], bytes=rows*stride)
    binary = (frozen / output['file']).read_bytes()
    (folder / output['file']).write_bytes(binary[first*4:total*4])
    output.update(shape=[rows,1,1,1], strides=[4,rows*4,rows*4,rows*4], bytes=rows*4)
    for item in fixture['inputs']:
        (folder / item['file']).write_bytes((frozen / item['file']).read_bytes())
    write(folder / 'fixture.json', fixture)
    write(folder / 'inventory.json', inventory)
    return folder / 'inventory.json', folder / 'fixture.json', [first,total-1]


def run(args):
    out = args.output
    require(args.resource_root.resolve() == Path('E:/v1_update').resolve(), 'Full resource root required')
    out.mkdir(parents=True, exist_ok=False)
    require(build_identity(args.build_dir) == read(args.receipt), 'Build/source identity')
    require(sha(args.model) == S.MODEL_SHA and sha(PROMPT) == S.PROMPT_SHA, 'Model or prompt identity')
    spec = read(FIVE / 'input-spec.json')
    ref = args.reference / 'fixtures'
    require(sha(ref / 'manifest.json') == spec['captured_fixture_manifest']['sha256'], 'Frozen fixture manifest')
    index = {Path(f['path']).name:f['sha256'] for f in read(ref / 'manifest.json')['files']}
    profile = read(FIVE / 'profile.json')
    profile.pop('sampled_scope')
    profile['model_path'] = str(args.model)
    write(out / 'profile.json', profile)
    env = {'HBFLAB_JOINT_KV':'native','HBFLAB_M4_PAGED':'0','LLAMA_GRAPH_REUSE_DISABLE':'1',
           'HBFLAB_SAMPLED_TOKEN_ONLY':'1','HBFLAB_SAMPLED_AUDIT':'summary',
           'HBFLAB_SAMPLED_PROFILE':str(out/'profile.json'),'HBFLAB_SYMBOLIC_CONFIG':str(H0)}
    route = args.route
    def execute(name, binary, args_list, selected_route):
        X.execute(args,[args.build_dir/'bin'/binary,*args_list],name,out/name,
                  extra_env={**env,'HBFLAB_SAMPLED_ROUTE':selected_route},
                  timeout=600, require_ggml=True, disk_cap=128*1024**2)

    results = []
    for weight in [2,3,5,4,1]:
        for phase in ['d','p']:
            name = f'w{weight}-{phase}'
            fixture = ref / (name+'.json')
            require(sha(fixture)==index[fixture.name], 'Fixture identity '+name)
            meta = read(fixture)
            for item in [*meta['inputs'], meta['output']]:
                require(sha(ref/item['file'])==index[item['file']], 'Fixture blob '+item['file'])
            inv = FIVE/'inventory.json'
            rows = None
            if weight == 5:
                inv,fixture,rows = projection_window(out/(name+'-window'), meta, read(inv), ref)
            result = out/name
            execute(name,'five_weight.exe',[args.model,inv,fixture,result,route,'0','summary'],route)
            expected = fixture.parent/meta['output']['file']
            require(sha(result/'call-0.bin')==sha(expected) and read(result/'calls.json')[0]['native_bitwise_equal'],
                    'Native output '+name)
            summary = read(result/'summary.json')
            stats = read(result/'route-summary.json')
            require(summary['borrowed_unchanged'] and summary['calls']==stats['steps']==stats['consumers']==1
                    and stats['drained'], 'Component coverage')
            size = meta['weight']['bytes']
            nrows = meta['inputs'][0]['shape'][0] if weight==4 else 1
            demand = nrows*meta['weight']['strides'][1] if weight==4 else size
            blocks = nrows
            if route=='ptx':
                if weight==1:
                    demand *= meta['inputs'][0]['shape'][1]
                kind = ('r2_norm' if weight==1 else 'get_rows' if weight==4 else
                        'cublas_f16_to_f32' if meta['inputs'][0]['shape'][1]>16 else 'r2_dot')
                require(stats['scalar_requests_by_consumer']=={kind:demand//4} and
                        stats['range_or_scalar_calls']==demand//4, 'R2 scalar consumer coverage')
            elif route=='cuda':
                b2 = read(result/'b2-summary.json')
                require(b2['drained'] and b2['requests']==demand//512 and b2['bytes']==demand and
                        b2['publications']==blocks and b2['lane_checks']==demand//16 and
                        b2['launches']==blocks*((demand//blocks+262143)//262144), 'B2 coverage')
            else:
                f = read(result/'session.json')
                require(len(f['accesses'])==blocks and f['online_audit']['read_fragments']==demand//64 and
                        f['host_drained'] and f['device_drained'], 'F coverage')
                if weight==4:
                    import struct
                    ids = [row for (row,) in struct.iter_unpack('<i',(ref/meta['inputs'][0]['file']).read_bytes())]
                    base = f['initialization'][0]['global_address']
                    require([v['global_address'] for v in f['accesses']]==
                            [base+v*meta['weight']['strides'][1] for v in ids], 'F embedding row order')
            require(stats['requested_bytes']==demand, 'Selected demand')
            results.append(dict(case=name,tensor=meta['tensor'],output_bytes=expected.stat().st_size,
                                output_sha256=sha(expected),requested_bytes=demand,projection_rows=rows))
            write(out/'components.json',results)

    inv = read(FIVE/'inventory.json')
    for entry in inv['all_tensors']:
        if entry['name']=='output.weight':
            entry['selected']=False
    require(sum(t['selected'] for t in inv['all_tensors'])==4, 'Joint selection')
    write(out/'selection.json',inv)
    for name,selected in [('native-model','native'),('joint-model',route)]:
        execute(name,'five_model.exe',[args.model,PROMPT,out/name,out/'selection.json','1'],selected)
    native, joint = out/'native-model',out/'joint-model'
    expected = args.reference/'result'/'logits.f32'
    require(sha(expected)==spec['new_native_reference']['full_logits']['sha256'], 'Frozen full logits identity')
    logits = (joint/'logits.f32').read_bytes()
    require(len(logits)==151936*4 and logits==expected.read_bytes()[:len(logits)], 'Frozen first full logit vector')
    for name in ['logits.f32','native-kv-state.bin','native-kv-positions.json','framework-release.json']:
        require(sha(joint/name)==sha(native/name), 'Model equality '+name)
    model = read(joint/'summary.json')
    require(model['consumer_records']==model['selected_tensors']==4 and model['generated_tokens']==
            read(native/'summary.json')['generated_tokens'] and len(model['prompt_tokens'])==44, 'Model coverage')
    write(out/'acceptance.json',dict(status='pass',route=route,cases=results,model={
        'four_joint_weights':True,'logits_sha256':sha(joint/'logits.f32'),'kv_sha256':sha(joint/'native-kv-state.bin'),
        'token':model['generated_tokens']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--route',choices=['symbolic','ptx','cuda'],required=True)
    for opt in ['build-dir','cuda-root','reference','model','receipt','output','resource-root']:
        parser.add_argument('--'+opt,type=Path,required=True)
    run(parser.parse_args())
