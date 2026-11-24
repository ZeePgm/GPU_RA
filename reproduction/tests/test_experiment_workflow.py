"""Portable CPU regression for the GPU_RA experiment workflow.

Usage (Windows x64, PowerShell 7, Git and Python 3.12+):
    python reproduction/tests/test_experiment_workflow.py --output-root <external-directory>

SOURCE is the checkout containing this script. It is queried read-only to reject
output under any registered source worktree. Each invocation creates a unique
output-root/e2e-ID directory and a new fixture Git repository. Only Git refs in
that newly created fixture are modified; real repository refs and existing data
are untouched. No GPU model is launched. Source workflow files are copied into
the fixture before its 43 end-to-end and refusal checks run.

Logs, the fixture, and acceptance.json remain in the unique directory on both
success and failure. The input/output paths intentionally contain spaces and an
apostrophe to exercise Windows launch quoting. No cleanup or publication occurs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


SOURCE = Path(__file__).resolve().parents[2]
BASE = REPO = WORKTREES = HOME = DATA = LOGS = None
PYTHON = str(Path(sys.executable).resolve())
# Prevent inherited Git routing/config injection from redirecting fixture writes.
ENV = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
ENV.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
EXPERIMENT = 'fixture-v2'
CHECKS = []
SEQUENCE = 0
SUMMARY = {'status': 'not-started', 'checks': CHECKS}


def initialize(output_root):
    global BASE, REPO, WORKTREES, HOME, DATA, LOGS
    probe = subprocess.run(['git', '-c', 'core.longpaths=true', '-C', str(SOURCE),
                            'rev-parse', '--show-toplevel'], capture_output=True, env=ENV)
    if probe.returncode or Path(probe.stdout.decode('utf-8').strip()).resolve() != SOURCE:
        raise ValueError('Run this regression from a GPU_RA source checkout.')
    result = subprocess.run(['git', '-c', 'core.longpaths=true', '-C', str(SOURCE),
                             'worktree', 'list', '--porcelain', '-z'],
                            capture_output=True, env=ENV)
    if result.returncode:
        raise ValueError('Cannot inspect registered source worktrees.')
    trees = [Path(row[9:]).resolve() for row in result.stdout.decode('utf-8').split('\0')
             if row.startswith('worktree ')]
    output_root = Path(output_root).expanduser().resolve()
    if any(output_root.is_relative_to(tree) for tree in trees):
        raise ValueError('Output root must be outside every registered source worktree: '
                         + str(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    BASE = output_root / ('e2e-' + uuid.uuid4().hex[:10])
    if BASE.exists() or BASE.is_symlink():
        raise FileExistsError('Use a fresh fixture directory: ' + str(BASE))
    REPO = BASE / 'fixture source'
    WORKTREES = BASE / 'work trees'
    HOME = BASE / "external data's home"
    DATA = HOME / 'bundles'
    LOGS = BASE / 'logs'
    SUMMARY.update(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                   fixture=str(BASE), source_checkout=str(SOURCE),
                   output_root=str(output_root), models_or_GPU_executed=0,
                   source_worktrees_checked=[str(tree) for tree in trees])


def save():
    (BASE / 'acceptance.json').write_text(json.dumps(SUMMARY, indent=2) + '\n', encoding='utf-8')


def execute(label, cmd, cwd, expected=0, contains=None):
    global SEQUENCE
    SEQUENCE += 1
    cmd = [str(x) for x in cmd]
    result = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, timeout=60)
    log = LOGS / ('%03d-' % SEQUENCE + label + '.log')
    log.write_bytes(b'COMMAND ' + json.dumps(cmd).encode() + b'\nSTDOUT\n' + result.stdout
                    + b'\nSTDERR\n' + result.stderr)
    text = (result.stdout + result.stderr).decode('utf-8', errors='replace')
    good = result.returncode == expected if isinstance(expected, int) else result.returncode != 0
    if contains is not None:
        good = good and contains in text
    CHECKS.append({'case': label, 'status': 'pass' if good else 'fail', 'returncode': result.returncode,
                   'expected': expected, 'log': str(log),
                   'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
    save()
    if not good:
        raise AssertionError(label + ': ' + text[-9000:])
    print('PASS', label, flush=True)
    return result, result.stdout.decode('utf-8', errors='replace')


def git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], env=ENV, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr.decode('utf-8', errors='replace'))
    return result.stdout.decode('utf-8', errors='replace').strip()


def commit(root, title):
    git(root, 'add', '--all')
    git(root, 'commit', '-m', title)
    return git(root, 'rev-parse', 'HEAD')


def cli(root, action, *args):
    return [PYTHON, root / 'reproduction/experiment.py', action, *args]


def start(root, run, code, inputs=()):
    args = ['--id', EXPERIMENT, '--run', run, '--data-root', DATA]
    for item in inputs:
        args.extend(['--input', item])
    _, out = execute('prepare-' + run, cli(root, 'start', *args, '--', PYTHON, '-c', code), root)
    info = json.loads(out)
    assert info['status'] == 'prepared' and not Path(info['output_directory']).exists()
    return info


def launch(root, run, expected=0):
    return execute('launch-' + run, ['pwsh', '-NoProfile', '-File', HOME / 'runs' / EXPERIMENT / run / 'launch.ps1'], root, expected)


def finish(root, run, verdict='passed', expected=0):
    return execute('finish-' + run + '-' + verdict,
                   cli(root, 'finish', '--id', EXPERIMENT, '--run', run, '--data-root', DATA,
                       '--verdict', verdict, '--conclusion', 'CPU fixture conclusion; no scientific claim.'), root, expected)


def verify(root, run, label=None, expected=0, contains=None):
    result, out = execute(label or 'verify-' + run,
                          cli(root, 'verify', '--id', EXPERIMENT, '--run', run, '--data-root', DATA),
                          root, expected, contains)
    return json.loads(out) if result.returncode == 0 else None


def mutate_reject(root, path, content, label):
    previous = path.read_bytes()
    try:
        path.write_bytes(content)
        verify(root, 'good', label, expected='nonzero')
    finally:
        path.write_bytes(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-root', type=Path, required=True,
                        help='External directory for a unique fixture, complete logs and receipt')
    args = parser.parse_args()
    try:
        initialize(args.output_root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for relative in ['reproduction/experiment.py', 'reproduction/run_job.cs', 'reproduction/data_paths.py']:
        assert (SOURCE / relative).is_file(), relative
    LOGS.mkdir(parents=True)
    REPO.mkdir()
    DATA.mkdir(parents=True)
    SUMMARY['source_files'] = []
    for relative in ['reproduction/experiment.py', 'reproduction/run_job.cs', 'reproduction/data_paths.py']:
        original = SOURCE / relative
        target = REPO / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        SUMMARY['source_files'].append({'path': relative, 'sha256': hashlib.sha256(original.read_bytes()).hexdigest()})
    (REPO / '.gitignore').write_text('data.local.json\n__pycache__/\n*.pyc\n', encoding='utf-8')
    (REPO / 'VERSION_MAP.md').write_text('Isolated workflow fixture\n', encoding='utf-8')
    (REPO / 'DATASETS.json').write_text('{"schema":1,"datasets":[]}\n', encoding='utf-8')
    (REPO / 'version.txt').write_text('v1\n', encoding='utf-8')
    git(REPO, 'init', '-b', 'main')
    for key, value in [('user.name', 'GPU_RA workflow fixture'), ('user.email', 'fixture@example.invalid'),
                       ('commit.gpgsign', 'false'), ('core.longpaths', 'true'), ('core.hooksPath', str(BASE / 'empty-hooks'))]:
        git(REPO, 'config', key, value)
    commit(REPO, 'Fixture V1')
    git(REPO, 'checkout', '-b', 'codex/v2-readonly')
    (REPO / 'version.txt').write_text('v2\n', encoding='utf-8')
    (REPO / 'provenance').mkdir()
    old_delivery = REPO / 'provenance/delivery-manifest.json'
    current_delivery = REPO / 'provenance/project-layout-manifest.json'
    old_delivery.write_bytes(b'{"version_family":"historical"}\n')
    current_delivery.write_bytes(b'{"version_family":"v2"}\n')
    v2_base = commit(REPO, 'Fixture V2')
    git(REPO, 'checkout', 'main')
    git(REPO, 'checkout', '-b', 'codex/v3-joint')
    (REPO / 'version.txt').write_text('v3\n', encoding='utf-8')
    commit(REPO, 'Fixture V3')
    git(REPO, 'checkout', 'main')
    save()
    common_new = ['--version', 'v2', '--title', 'CPU workflow fixture', '--purpose', 'Test bookkeeping only',
                  '--worktree-root', WORKTREES, '--data-root', DATA]
    execute('invalid-ID', cli(REPO, 'new', '--id', '../escape', *common_new), REPO, 'nonzero', 'safe lowercase ID')
    execute('reserved-ID', cli(REPO, 'new', '--id', 'con', *common_new), REPO, 'nonzero', 'safe lowercase ID')
    _, text = execute('new-from-real-V2-base', cli(REPO, 'new', '--id', EXPERIMENT, *common_new), REPO)
    new = json.loads(text)
    root = Path(new['worktree'])
    assert new['base_commit'] == v2_base and (root / 'version.txt').read_text().strip() == 'v2'
    meta = json.loads((root / 'experiments' / EXPERIMENT / 'experiment.json').read_bytes())
    assert [row['path'] for row in meta['baseline_manifests']] == [
        'provenance/project-layout-manifest.json', 'DATASETS.json']
    current = subprocess.check_output([
        'git', '-C', str(root), 'show', v2_base + ':provenance/project-layout-manifest.json'])
    assert meta['baseline_manifests'][0]['sha256'] == hashlib.sha256(current).hexdigest()
    assert git(root, 'rev-parse', 'HEAD') == v2_base
    assert git(root, 'branch', '--show-current') == 'codex/exp-' + EXPERIMENT
    assert git(root, 'status', '--porcelain')
    _, duplicate = execute('duplicate-new', cli(REPO, 'new', '--id', EXPERIMENT, *common_new), REPO, 'nonzero', 'already exists')
    good_code = 'import os,json;from pathlib import Path;p=Path(os.environ["GPU_RA_RUN_DIR"]);p.mkdir(parents=True);(p/"result.json").write_text(json.dumps({"result":42,"commit":os.environ["GPU_RA_CODE_COMMIT"]}),encoding="utf-8");print("fixture stdout")'
    execute('uncommitted-source-rejected', cli(root, 'start', '--id', EXPERIMENT, '--run', 'uncommitted',
            '--data-root', DATA, '--', PYTHON, '-c', good_code), root, 'nonzero', 'not clean')
    assert not (HOME / 'runs' / EXPERIMENT / 'uncommitted').exists()
    code_a = commit(root, 'Commit experiment source and config A')
    incoming = BASE / "source's input.txt"
    incoming.write_bytes(b'immutable fixture input\n')
    info = start(root, 'good', good_code, [incoming])
    launch(root, 'good')
    run_dir = Path(info['run_directory'])
    record = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    assert record['status'] == 'completed' and record['exit_code'] == 0 and record['code_commit'] == code_a
    assert record['identity_unchanged_after'] is True
    frozen_run = (run_dir / 'run.json').read_bytes()
    finish(root, 'good')
    assert (run_dir / 'run.json').read_bytes() == frozen_run
    assert git(root, 'rev-parse', 'HEAD') == code_a
    report = json.loads((root / 'reports' / EXPERIMENT / 'good.json').read_text(encoding='utf-8'))
    assert report['code_commit'] == code_a
    code_b = commit(root, 'Commit report B')
    proof = verify(root, 'good', 'verify-after-report-commit-B')
    assert proof['code_commit'] == code_a and proof['current_commit'] == code_b and code_a != code_b
    assert proof['report_commit'] == code_b
    SUMMARY.update(experiment_root=str(root), code_commit_A=code_a, report_commit_B=code_b)
    execute('duplicate-run-rejected', cli(root, 'start', '--id', EXPERIMENT, '--run', 'good', '--data-root', DATA,
            '--', PYTHON, '-c', good_code), root, 'nonzero', 'sealed')
    finish(root, 'good', expected='nonzero')
    launch(root, 'good', expected='nonzero')
    assert (run_dir / 'run.json').read_bytes() == frozen_run
    output = run_dir / 'data/result.json'
    mutate_reject(root, output, b'changed fixture result', 'tampered-raw-data-rejected')
    additional = run_dir / 'data/unlisted.txt'
    additional.write_bytes(b'extra')
    try:
        verify(root, 'good', 'additional-raw-file-rejected', 'nonzero', 'additional files')
    finally:
        additional.unlink()
    cas = HOME / record['inputs'][0]['storage_path']
    mutate_reject(root, cas, b'corrupt CAS', 'tampered-CAS-rejected')
    mutate_reject(root, run_dir / 'config.json', b'{}', 'tampered-config-snapshot-rejected')
    mutate_reject(root, root / 'reports' / EXPERIMENT / 'good.json', b'{}', 'tampered-report-rejected')
    mutate_reject(root, root / 'reports' / EXPERIMENT / 'good.md', b'changed', 'tampered-markdown-rejected')
    manifest_path = root / 'evidence' / EXPERIMENT / 'good/manifest.json'
    mutate_reject(root, manifest_path, b'{}', 'tampered-manifest-rejected')
    mutate_reject(root, root / 'DATASETS.json', b'{"experiment_runs":[]}', 'tampered-catalog-rejected')
    original_input = incoming.read_bytes()
    incoming.write_bytes(b'later original input content')
    verify(root, 'good', 'sealed-verification-uses-CAS-not-mutable-origin')
    incoming.write_bytes(original_input)
    verify(root, 'good', 'verify-restored-fixture-integrity')
    dirty = root / 'untracked-fixture.txt'
    dirty.write_text('source dirty\n', encoding='utf-8')
    try:
        execute('untracked-source-rejected', cli(root, 'start', '--id', EXPERIMENT, '--run', 'dirty', '--data-root', DATA,
                '--', PYTHON, '-c', good_code), root, 'nonzero', 'not clean')
    finally:
        dirty.unlink()
    start(root, 'failed', 'import sys;print("intentional failure");sys.exit(7)')
    launch(root, 'failed', 'nonzero')
    failed = json.loads((HOME / 'runs' / EXPERIMENT / 'failed/run.json').read_text(encoding='utf-8'))
    assert failed['status'] == 'failed' and failed['exit_code'] == 7
    finish(root, 'failed', 'passed', 'nonzero')
    finish(root, 'failed', 'failed')
    commit(root, 'Commit retained failed run report')
    verify(root, 'failed')
    drift_code = 'from pathlib import Path;Path("drift-fixture.txt").write_text("source drift",encoding="utf-8")'
    start(root, 'drift', drift_code)
    launch(root, 'drift', 'nonzero')
    drifting = json.loads((HOME / 'runs' / EXPERIMENT / 'drift/run.json').read_text(encoding='utf-8'))
    assert drifting['status'] == 'identity_drift' and drifting['exit_code'] == 0
    finish(root, 'drift', 'passed', 'nonzero')
    (root / 'drift-fixture.txt').unlink()
    finish(root, 'drift', 'inconclusive')
    commit(root, 'Commit retained identity drift report')
    verify(root, 'drift')
    running_code = 'import os,time;from pathlib import Path;p=Path(os.environ["GPU_RA_RUN_DIR"]);p.mkdir(parents=True);(p/"ready").write_text("ready");time.sleep(20)'
    running = start(root, 'cancelled', running_code)
    cancel_dir = Path(running['run_directory'])
    live_log = LOGS / 'live-cancelled-launch.log'
    with live_log.open('wb') as stream:
        controller = subprocess.Popen(['pwsh', '-NoProfile', '-File', str(cancel_dir / 'launch.ps1')], cwd=root, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while not (cancel_dir / 'data/ready').exists() and time.monotonic() < deadline and controller.poll() is None:
                time.sleep(.05)
            assert (cancel_dir / 'data/ready').is_file()
            finish(root, 'cancelled', 'inconclusive', 'nonzero')
            cancel_env = dict(ENV, FIXTURE_CANCEL=str(cancel_dir / 'cancel.request'))
            subprocess.run(['pwsh', '-NoProfile', '-Command', '[IO.File]::WriteAllText($env:FIXTURE_CANCEL,"fixture cancellation")'], env=cancel_env, check=True, timeout=10)
            assert controller.wait(timeout=20) != 0
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait(timeout=10)
    cancelled = json.loads((cancel_dir / 'run.json').read_text(encoding='utf-8'))
    assert cancelled['status'] == 'interrupted'
    execution = json.loads((cancel_dir / 'execution.json').read_text(encoding='utf-8'))
    assert execution['drain_confirmed'] and execution['active_processes_at_end'] == 0
    finish(root, 'cancelled', 'inconclusive')
    commit(root, 'Commit retained cancelled run report')
    verify(root, 'cancelled')
    valid_adapter = (root / 'reproduction/run_job.cs').read_bytes()
    (root / 'reproduction/run_job.cs').write_text('Deliberately invalid C sharp fixture.\n', encoding='utf-8')
    commit(root, 'Commit deliberately invalid fixture adapter')
    start(root, 'compiler-failed', good_code)
    launch(root, 'compiler-failed', 'nonzero')
    compiler_failed = json.loads((HOME / 'runs' / EXPERIMENT / 'compiler-failed/run.json').read_text(encoding='utf-8'))
    assert compiler_failed['status'] == 'launch_failed'
    execution = json.loads((HOME / 'runs' / EXPERIMENT / 'compiler-failed/execution.json').read_text(encoding='utf-8'))
    assert execution['launched'] is False and execution['drain_confirmed'] is True
    finish(root, 'compiler-failed', 'passed', 'nonzero')
    finish(root, 'compiler-failed', 'failed')
    (root / 'reproduction/run_job.cs').write_bytes(valid_adapter)
    commit(root, 'Commit compiler failure report and restore adapter')
    verify(root, 'compiler-failed')
    final_proof = verify(root, 'good', 'verify-original-A-B-after-later-report-commits')
    assert final_proof['code_commit'] == code_a and final_proof['report_commit'] == code_b
    assert final_proof['current_commit'] != code_b
    assert git(root, 'status', '--porcelain') == ''
    assert git(REPO, 'branch', '--show-current') == 'main'
    SUMMARY.update(status='pass', completed_utc=datetime.now(timezone.utc).isoformat(), check_count=len(CHECKS),
                   final_worktree_clean=True, version_base_selection='v2 actual tree',
                   data_outside_source_worktrees=True, real_repository_refs_changed=False,
                   failed_and_interrupted_records_retained=True)
    save()
    print(json.dumps({'status': 'pass', 'checks': len(CHECKS), 'receipt': str(BASE / 'acceptance.json')}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if BASE is not None and BASE.exists():
            SUMMARY.update(status='fail', error=repr(error), stopped_utc=datetime.now(timezone.utc).isoformat())
            save()
        raise
