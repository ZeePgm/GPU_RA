"""Recheck raw P0-A artifacts independently; do not trust runner evidence flags."""
import argparse
from pathlib import Path
from common import read, write, rows, sha
from p0a_qualification import freeze, adjudicate
from p0a_timeline import report


def validate(study):
    frozen=read(study/'frozen-contract.json')
    assert freeze(frozen['contract'])==frozen['sha256']
    reference=study/'fresh-reference'
    ref=read(reference/'a1-clocks.json')
    expected=(reference/'logits.f32').read_bytes()
    records=[]
    for path in sorted(study.glob('*/a1-clocks.json')):
        r=read(path);folder=path.parent
        assert (folder/'logits.f32').read_bytes()==expected,folder
        assert r['access_identity_fnv1a64']==ref['access_identity_fnv1a64'],folder
        assert [(x['token_id'],x['session_tick_ps']) for x in r['tokens']]==[(x['token_id'],x['session_tick_ps']) for x in ref['tokens']],folder
        boundaries=r['boundaries']
        assert all(x['clock']=='steady_clock' for x in boundaries)
        assert [x['ns'] for x in boundaries]==sorted(x['ns'] for x in boundaries)
        for event in ['token_id_deliverable','output_returned']:
            assert len([x for x in boundaries if x['event']==event])==8
        assert [(g['node_id'],g['step'],g['target_ns']) for g in r['gates']]==[(g['node_id'],g['step'],g['target_ns']) for g in ref['gates']]
        reset=read(folder/'summary.json')['reset']
        assert reset['kv_max_position']==-1 and reset['kv_data_cleared'] and reset['backend_tick']==0
        if reset['context_reused']: assert reset['unrelated_prefix_before_reset']
        write(folder/'p0a-timeline.json',report(folder))
        records.append(dict(run=folder.name,logits_sha256=sha(folder/'logits.f32'),access_identity=r['access_identity_fnv1a64'],reset=reset))
    a,b=study/'full-audit',study/'lean-audit'
    snapshots=rows(a/'snapshots.jsonl')
    assert len(snapshots)>0 and len(snapshots)==len(rows(b/'snapshots.jsonl'))
    assert all((a/r['file']).read_bytes()==(b/r['file']).read_bytes() for r in snapshots)
    formal=read(study/'result.json')
    for protocol,result in formal.items():
        if result['formal_status']!='completed': continue
        c=read(study/f'{protocol}-formal-contract.json')
        assert all(x['status']=='qualified' for x in read(study/f'{protocol}-qualification.json'))
        def endpoint(name):
            r=read(study/name/'a1-clocks.json')
            entry=next(b['ns'] for b in r['boundaries'] if b['event']=='request_entry')
            return [b['ns']-entry for b in r['boundaries'] if b['event']=='token_id_deliverable']
        actual=[[b-a-c['contract']['target_ns'] for a,b in zip(endpoint(f'{protocol}-formal-{i}-zero'),endpoint(f'{protocol}-formal-{i}-inject'))] for i in range(c['contract']['pairs'])]
        assert actual==result['errors_ns']
        assert [adjudicate(c['contract'],c['sha256'],[r[k] for r in result['errors_ns']]) for k in range(8)]==result['results']
    result=dict(status='pass',runs=len(records),snapshot_arrays=len(snapshots),records=records,
                scope='Artifact integrity and mathematical/access equivalence; statistical conclusions remain in result.json')
    write(study/'validation.json',result)
    print('validated',len(records),'runs;',len(snapshots),'snapshot arrays')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);args=p.parse_args();validate(args.study)
