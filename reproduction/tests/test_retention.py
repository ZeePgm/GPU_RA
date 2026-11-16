"""CPU-only approved-deletion tests, with disposable sealed runs outside real worktrees."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

import test_experiment_workflow as T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    T.initialize(args.output_root)
    T.LOGS.mkdir(parents=True)
    T.REPO.mkdir()
    T.DATA.mkdir(parents=True)
    for rel in ('reproduction/experiment.py', 'reproduction/retention.py', 'reproduction/run_job.cs', 'reproduction/data_paths.py'):
        target = T.REPO / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(T.SOURCE / rel, target)
    (T.REPO / '.gitignore').write_bytes(b'data.local.json\n__pycache__/\n*.pyc\n')
    (T.REPO / '.gitattributes').write_bytes(b'* -text\n')
    (T.REPO / 'VERSION_MAP.md').write_bytes(b'CPU retention fixture\n')
    (T.REPO / 'DATASETS.json').write_bytes(b'{"schema":1,"datasets":[]}\n')
    T.git(T.REPO, 'init', '-b', 'main')
    for key, value in [('user.name','Retention fixture'), ('user.email','fixture@example.invalid'), ('core.longpaths','true'),
                       ('commit.gpgsign','false'), ('core.hooksPath',str(T.BASE/'no-hooks'))]:
        T.git(T.REPO, 'config', key, value)
    T.commit(T.REPO, 'Fixture baseline')
    T.git(T.REPO, 'branch', 'codex/v2-readonly')
    _, result = T.execute('new', T.cli(T.REPO, 'new', '--id', T.EXPERIMENT, '--version', 'v2', '--title', 'CPU retention',
        '--purpose', 'No scientific claim', '--worktree-root', T.WORKTREES, '--data-root', T.DATA), T.REPO)
    root = Path(json.loads(result)['worktree'])
    T.commit(root, 'Fixture source A')
    code = ('import os;from pathlib import Path;p=Path(os.environ["GPU_RA_RUN_DIR"]);'
            '(p/"readonly").mkdir(parents=True);'
            '[(p/"readonly"/n).write_bytes(n.encode()*1000) for n in ("lane-times.bin","requests.bin","weight-requests.jsonl","chunks.jsonl")];'
            '(p/"logits.f32").write_bytes(b"necessary reference");(p/"acceptance.json").write_bytes(b"{}")')
    info = T.start(root, 'good', code)
    T.launch(root, 'good')
    T.finish(root, 'good')
    T.commit(root, 'Fixture report B')
    T.verify(root, 'good')
    run = Path(info['run_directory'])
    evidence = root/'evidence'/T.EXPERIMENT/'good'
    original = {p:p.read_bytes() for p in (evidence/'manifest.json', root/'reports'/T.EXPERIMENT/'good.json',
        root/'reports'/T.EXPERIMENT/'good.md', T.HOME/'seals'/T.EXPERIMENT/'good'/'seal.json')}
    command = [T.PYTHON, root/'reproduction/retention.py']
    common = ['--id', T.EXPERIMENT, '--run','good','--data-root',T.DATA]
    T.execute('missing-approval', [*command,'plan',*common], root, 'nonzero')
    T.execute('empty-approval', [*command,'plan',*common,'--approval-text',' '], root, 'nonzero')
    T.execute('approved-plan', [*command,'plan',*common,'--approval-text','Explicit approval for disposable CPU test traces only.'], root)
    p = json.loads((evidence/'retention-plan.json').read_text(encoding='utf-8'))
    assert len(p['remove']) == 4 and not any(r['path'].endswith('logits.f32') for r in p['remove'])
    T.execute('uncommitted-plan', [*command,'apply',*common], root, 'nonzero')
    T.commit(root, 'Approved exact plan C')
    sys.path.insert(0,str(T.SOURCE/'reproduction'))
    import retention as R
    for path in ('../escape/requests.bin','data/../escape/requests.bin','data/model/logits.f32','/data/x/requests.bin','data/x:stream/requests.bin'):
        try:
            R.trace_path(run,path)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe path accepted: '+path)
    T.CHECKS.append({'case':'reference-and-traversal-guards','status':'pass'})
    # A real NTFS junction must be rejected even when it points at another fixture directory.
    dest=T.BASE/'outside run';dest.mkdir();(dest/'requests.bin').write_bytes(b'untouched')
    link=run/'data'/'junction'
    ps=T.BASE/'make-junction.ps1';ps.write_bytes(b'param([string]$link,[string]$dest)\nNew-Item -ItemType Junction -Path $link -Target $dest | Out-Null\n')
    T.execute('make-fixture-junction',['pwsh','-NoProfile','-File',ps,str(link),str(dest)],root)
    try:
        R.trace_path(run,'data/junction/requests.bin')
    except ValueError:
        pass
    else:
        raise AssertionError('Junction escape accepted')
    link.rmdir()  # unlink this checked fixture junction itself; never recurse into its target
    assert (dest/'requests.bin').read_bytes()==b'untouched'
    T.CHECKS.append({'case':'junction-refusal','status':'pass'})
    # Simulate interruption after one file, and prove the exact approved plan can resume.
    original_unlink=Path.unlink
    calls=[]
    def interrupted(path,*args,**kwargs):
        calls.append(path)
        if len(calls)==2:
            raise PermissionError('simulated interruption')
        return original_unlink(path,*args,**kwargs)
    with patch.object(R.E,'repository_root',return_value=root), patch.object(Path,'unlink',new=interrupted):
        try:
            R.apply(SimpleNamespace(id=T.EXPERIMENT,run='good',data_root=T.DATA))
        except PermissionError:
            pass
        else:
            raise AssertionError('Interruption not exercised')
    assert not (evidence/'retention-result.json').exists()
    T.execute('resume-approved-cleanup',[*command,'apply',*common],root)
    assert (run/'data/logits.f32').read_bytes()==b'necessary reference'
    for path,value in original.items():
        assert path.read_bytes()==value
    T.verify(root,'good','full-verify-still-rejects-missing-traces','nonzero')
    verify=[*T.cli(root,'verify',*common),'--retained']
    _,out=T.execute('retained-precommit',verify,root)
    assert json.loads(out)['retention_committed'] is False
    T.commit(root,'Cleanup result D')
    _,out=T.execute('retained-committed',verify,root)
    result=json.loads(out)
    assert result['report_committed'] and result['retention_committed'] and not result['full_payload_present'] and not result['full_raw_trace_recheck_available']
    target=run/'data/logits.f32';saved=target.read_bytes();target.write_bytes(b'corrupted')
    T.execute('corrupt-reference-rejected',verify,root,'nonzero');target.write_bytes(saved)
    target.unlink()
    T.execute('missing-unapproved-reference-rejected',verify,root,'nonzero');target.write_bytes(saved)
    extra=run/'data/extra.json';extra.write_bytes(b'{}')
    T.execute('extra-data-rejected',verify,root,'nonzero');extra.unlink()
    result_path=evidence/'retention-result.json';saved=result_path.read_bytes();result_path.write_bytes(b'{}')
    T.execute('tampered-cleanup-receipt-rejected',verify,root,'nonzero');result_path.write_bytes(saved)
    T.execute('repeated-cleanup-rejected',[*command,'apply',*common],root,'nonzero')
    T.execute('final-retained-check',verify,root)
    T.SUMMARY.update(status='pass',models_or_GPU_executed=0,real_experiment_data_deleted=0)
    T.save()
    print(json.dumps({'status':'pass','checks':len(T.CHECKS),'fixture':str(T.BASE),'models_or_GPU_executed':0}))


if __name__=='__main__':
    main()
