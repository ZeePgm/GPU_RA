"""GPU_RA experiment worktrees, immutable run identities, external data and reports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import uuid

from data_paths import _configured_root, repository_root

VERSIONS = {'v1': 'main', 'v2': 'codex/v2-readonly', 'v3': 'codex/v3-joint'}
TERMINAL = {'completed', 'failed', 'interrupted', 'launch_failed', 'identity_drift'}
ENV_KEYS = ('PATH', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_MODULE_LOADING',
            'OMP_NUM_THREADS', 'OMP_WAIT_POLICY', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED')


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular file: ' + str(path))
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while hashing: ' + str(path))
    return {'bytes': after.st_size, 'sha256': digest}


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        write_new(temp, data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def git(root, *args, check=True):
    result = subprocess.run(['git', '-c', 'core.longpaths=true', '-C', str(root), *args],
                            capture_output=True, env=dict(os.environ, GIT_NO_REPLACE_OBJECTS='1'))
    if check and result.returncode:
        raise ValueError('Git failed: ' + result.stderr.decode('utf-8', errors='replace').strip())
    return result


def git_text(root, *args):
    return git(root, *args).stdout.decode('utf-8').strip()


def segment(value):
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {p + str(i) for p in ('COM', 'LPT') for i in range(1, 10)}
    if (not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}', value)
            or value.endswith('.') or value.split('.')[0].upper() in reserved):
        raise ValueError('Use a safe lowercase ID (1-64 letters/digits/dot/underscore/hyphen): ' + value)
    return value


def child(root, relative):
    name = PurePosixPath(str(relative).replace('\\', '/'))
    if name.is_absolute() or not name.parts or '..' in name.parts or ':' in str(name):
        raise ValueError('Unsafe relative path: ' + str(relative))
    result = Path(root).joinpath(*name.parts).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('Path leaves its root: ' + str(relative))
    return result


def worktrees(root):
    rows = git(root, 'worktree', 'list', '--porcelain', '-z').stdout.decode('utf-8')
    return [Path(line[9:]).resolve() for line in rows.split('\0') if line.startswith('worktree ')]


def external(root, path):
    path = Path(path).expanduser().resolve()
    if any(path.is_relative_to(tree) for tree in worktrees(root)):
        raise ValueError('Experiment data/worktree must be outside existing source worktrees: ' + str(path))
    return path


def data_home(root, option):
    bundles = external(root, _configured_root(option))
    home = bundles.parent
    external(root, home / 'runs')
    external(root, home / 'inputs')
    external(root, home / 'seals')
    return bundles, home


def require_clean(root):
    status = git(root, 'status', '--porcelain=v1', '--untracked-files=all').stdout
    if status:
        raise ValueError('Commit source, scripts and configuration first; this worktree is not clean.')


def locations(root, experiment_id, run_id):
    experiment_id, run_id = segment(experiment_id), segment(run_id)
    return {'experiment': root / 'experiments' / experiment_id / 'experiment.json',
            'config': root / 'experiments' / experiment_id / 'config.json',
            'manifest': root / 'evidence' / experiment_id / run_id / 'manifest.json',
            'report': root / 'reports' / experiment_id / (run_id + '.json'),
            'markdown': root / 'reports' / experiment_id / (run_id + '.md')}


def descriptor(root, experiment_id):
    path = root / 'experiments' / segment(experiment_id) / 'experiment.json'
    meta = load(path)
    if (meta.get('schema') != 1 or meta.get('experiment_id') != experiment_id
            or meta.get('version') not in VERSIONS or meta.get('branch') != 'codex/exp-' + experiment_id
            or meta.get('base_ref') != VERSIONS[meta['version']]
            or not re.fullmatch('[0-9a-f]{40}', meta.get('base_commit', ''))):
        raise ValueError('Invalid experiment descriptor')
    return meta


def git_file(root, commit, relative):
    return git(root, 'show', commit + ':' + relative).stdout


def new(args):
    root = repository_root()
    require_clean(root)
    experiment_id = segment(args.id)
    if not args.title.strip() or not args.purpose.strip():
        raise ValueError('Experiment title and purpose must be nonempty')
    bundles, home = data_home(root, args.data_root)
    ref = VERSIONS[args.version]
    result = git(root, 'rev-parse', '--verify', 'refs/heads/' + ref, check=False)
    if result.returncode:
        result = git(root, 'rev-parse', '--verify', 'refs/remotes/origin/' + ref, check=False)
    if result.returncode:
        raise ValueError('Version branch unavailable locally; fetch ' + ref + ' before creating an experiment.')
    base = result.stdout.decode().strip()
    branch = 'codex/exp-' + experiment_id
    if git(root, 'show-ref', '--verify', '--quiet', 'refs/heads/' + branch, check=False).returncode == 0:
        raise ValueError('Experiment branch already exists: ' + branch)
    if git(root, 'cat-file', '-e', base + ':experiments/' + experiment_id + '/experiment.json', check=False).returncode == 0:
        raise ValueError('Experiment ID already exists in the selected base')
    common = Path(git_text(root, 'rev-parse', '--path-format=absolute', '--git-common-dir'))
    default_parent = common.parent.parent / ('experiments' if common.parent.name == 'main' else common.parent.name + '-experiments')
    parent = external(root, args.worktree_root or default_parent)
    target = parent / experiment_id
    data_locations = [bundles, home / 'runs', home / 'inputs', home / 'seals']
    if (target.exists() or target.is_symlink()
            or any(target.is_relative_to(path) or path.is_relative_to(target) for path in data_locations)):
        raise ValueError('Worktree target exists or contains the configured data root: ' + str(target))
    baseline = []
    checkout_manifest = 'provenance/project-layout-manifest.json'
    if git(root, 'cat-file', '-e', base + ':' + checkout_manifest, check=False).returncode:
        checkout_manifest = 'provenance/delivery-manifest.json'
    for rel in (checkout_manifest, 'DATASETS.json', 'joint/provenance/final-source-build-manifest.json'):
        probe = git(root, 'show', base + ':' + rel, check=False)
        if probe.returncode == 0:
            baseline.append({'path': rel, 'bytes': len(probe.stdout), 'sha256': sha_bytes(probe.stdout)})
    parent.mkdir(parents=True, exist_ok=True)
    git(root, 'worktree', 'add', '-b', branch, str(target), base)
    if os.name == 'nt':
        git(target, 'config', 'core.longpaths', 'true')
    if git(target, 'check-ignore', '-q', 'data.local.json', check=False).returncode:
        raise ValueError('Base must ignore data.local.json; worktree preserved for inspection: ' + str(target))
    meta = {'schema': 1, 'experiment_id': experiment_id, 'title': args.title, 'purpose': args.purpose,
            'version': args.version, 'base_ref': ref, 'base_commit': base, 'branch': branch,
            'source_prefix': 'joint/source' if args.version == 'v3' else '.',
            'created_utc': now(), 'baseline_manifests': baseline}
    write_new(target / 'experiments' / experiment_id / 'experiment.json', encoded(meta))
    write_new(target / 'experiments' / experiment_id / 'config.json', encoded(
        {'schema': 1, 'experiment_id': experiment_id, 'parameters': {}, 'acceptance': {}}))
    write_new(target / 'reports' / experiment_id / 'README.md',
              ('# ' + args.title + '\n\n' + args.purpose + '\n\nRun reports are generated by reproduction/experiment.py finish.\n').encode('utf-8'))
    write_atomic(target / 'data.local.json', encoded({'data_root': str(bundles)}))
    print(json.dumps({'status': 'created', 'worktree': str(target), 'branch': branch, 'base_commit': base,
                      'source_directory': str(target / meta['source_prefix']),
                      'next': 'Edit source/protocol/config, commit them, then invoke start from this worktree.'}, ensure_ascii=False))


def archive_input(home, path):
    original = Path(path).expanduser()
    info = identity(original)
    original = original.resolve()
    destination = home / 'inputs' / info['sha256']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if identity(destination) != info:
            raise ValueError('Existing immutable input object is corrupt: ' + str(destination))
    else:
        lock = destination.with_suffix('.lock')
        lock_stream = lock.open('x')
        try:
            temp = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                if destination.exists():
                    if identity(destination) != info:
                        raise ValueError('Conflicting input object')
                else:
                    with original.open('rb') as src, temp.open('xb') as out:
                        shutil.copyfileobj(src, out, 1024 * 1024)
                    if identity(temp) != info:
                        raise ValueError('Input changed while copying: ' + str(original))
                    os.replace(temp, destination)
            finally:
                temp.unlink(missing_ok=True)
        finally:
            lock_stream.close()
            lock.unlink(missing_ok=True)
    return {'original_path': str(original), 'storage_path': 'inputs/' + info['sha256'], **info}


def resource_scope(root):
    common = Path(git_text(root, 'rev-parse', '--path-format=absolute', '--git-common-dir'))
    for outer in common.parent.parents:
        if (outer / '.git').exists():
            return outer.resolve()
    return common.parent.parent.resolve()


def ps_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def launcher(root, record, run_dir, bundles):
    values = {
        'PYTHON': ps_string(sys.executable), 'SCRIPT': ps_string(root / 'reproduction/experiment.py'),
        'ID': ps_string(record['experiment_id']), 'RUN': ps_string(record['run_id']),
        'DATA_ROOT': ps_string(bundles), 'RUN_DIR': ps_string(run_dir),
        'OUTPUT': ps_string(run_dir / 'data'), 'RECEIPT': ps_string(run_dir / 'run.json'),
        'EXECUTION': ps_string(run_dir / 'execution.json'), 'CWD': ps_string(root),
        'RESOURCE': ps_string(record['resource_root']), 'JOB_SOURCE': ps_string(root / 'reproduction/run_job.cs'),
        'EXE': ps_string(record['executable']['path']),
        'ARGUMENTS': ps_string(subprocess.list2cmdline(record['command'][1:])),
        'TOKEN': ps_string(record['launch_token']), 'COMMAND_SHA': ps_string(record['command_sha256']),
        'COMMIT': ps_string(record['code_commit']), 'INPUTS': ps_string(json.dumps(record['inputs'])),
        'STDOUT': ps_string(run_dir / 'stdout.log'), 'STDERR': ps_string(run_dir / 'stderr.log'),
        'CANCEL': ps_string(run_dir / 'cancel.request'),
    }
    template = r"""# Generated by GPU_RA experiment.py. Execute only after start has exited.
# PowerShell 7 / Windows x64. No Python controller remains during model execution.
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -lt 7 -or -not $IsWindows) { throw 'Windows x64 and PowerShell 7 are required.' }
& @@PYTHON@@ @@SCRIPT@@ _begin --id @@ID@@ --run @@RUN@@ --data-root @@DATA_ROOT@@ --token @@TOKEN@@ --command-sha @@COMMAND_SHA@@
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:GPU_RA_RUN_DIR = @@OUTPUT@@
$env:GPU_RA_RUN_RECORD = @@RECEIPT@@
$env:GPU_RA_EXPERIMENT_ID = @@ID@@
$env:GPU_RA_RUN_ID = @@RUN@@
$env:GPU_RA_CODE_COMMIT = @@COMMIT@@
$env:GPU_RA_DATA_ROOT = @@DATA_ROOT@@
$env:GPU_RA_RESOURCE_ROOT = @@RESOURCE@@
$env:GPU_RA_INPUTS_JSON = @@INPUTS@@
$execution = @{schema=1; launch_token=@@TOKEN@@; command_sha256=@@COMMAND_SHA@@; controller_pid=$PID; started_utc=[DateTime]::UtcNow.ToString('o'); exit_code=1; launched=$false; drain_confirmed=$false; cancelled=$false}
$rc = 1
$runInvoked = $false
try {
    Add-Type -Path @@JOB_SOURCE@@
    $runInvoked = $true
    $result = [GPU_RA.RunJob]::Run(@@EXE@@, @@ARGUMENTS@@, @@CWD@@, @@STDOUT@@, @@STDERR@@, @@CANCEL@@)
    $execution.launched = $true
    $execution.process_id = $result.ProcessId
    $execution.total_processes = $result.TotalProcesses
    $execution.active_processes_at_end = $result.ActiveProcessesAtEnd
    $execution.cancelled = $result.Cancelled
    $rc = $result.ExitCode
} catch {
    $execution.error = $_.Exception.ToString()
    $rc = 1
} finally {
    if (-not $runInvoked) { $execution.drain_confirmed = $true }
    elseif ('GPU_RA.RunJob' -as [type]) { $execution.drain_confirmed = [GPU_RA.RunJob]::LastDrainConfirmed }
    $execution.exit_code = $rc
    $execution.finished_utc = [DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText(@@EXECUTION@@, ($execution | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
}
if (-not $execution.drain_confirmed) {
    Write-Error 'Child-tree exit could not be confirmed. Preserve this incomplete run; do not seal it as passed.'
    exit 1
}
& @@PYTHON@@ @@SCRIPT@@ _complete --id @@ID@@ --run @@RUN@@ --data-root @@DATA_ROOT@@ --token @@TOKEN@@
$completeRc = $LASTEXITCODE
if ($completeRc -ne 0) { exit $completeRc }
exit $rc
"""
    return re.sub(r'@@([A-Z_]+)@@', lambda match: values[match[1]], template).encode('utf-8')


def start(args):
    root = repository_root()
    meta = descriptor(root, args.id)
    paths = locations(root, args.id, args.run)
    require_clean(root)
    branch = git_text(root, 'branch', '--show-current')
    if branch != meta['branch']:
        raise ValueError('Run from the experiment branch: ' + meta['branch'])
    commit = git_text(root, 'rev-parse', 'HEAD')
    git(root, 'merge-base', '--is-ancestor', meta['base_commit'], commit)
    for key in ('experiment', 'config'):
        rel = paths[key].relative_to(root).as_posix()
        if git_file(root, commit, rel) != paths[key].read_bytes():
            raise ValueError('Experiment configuration must be committed: ' + rel)
    command = list(args.command)
    if command and command[0] == '--':
        command.pop(0)
    if not command:
        raise ValueError('start requires -- COMMAND ARGUMENTS')
    if not (root / 'reproduction/run_job.cs').is_file():
        raise ValueError('The selected code commit lacks the Windows execution adapter')
    bundles, home = data_home(root, args.data_root)
    run_dir = external(root, home / 'runs' / args.id / args.run)
    if (home / 'seals' / args.id / args.run).exists():
        raise ValueError('Run ID is already sealed; use a new ID')
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {'schema': 1, 'experiment_id': args.id, 'run_id': args.run, 'version': meta['version'],
              'base_ref': meta['base_ref'], 'base_commit': meta['base_commit'], 'branch': branch,
              'code_commit': commit, 'code_tree': git_text(root, 'rev-parse', 'HEAD^{tree}'),
              'status': 'preparing', 'registered_utc': now(), 'command': command, 'command_template': command[:], 'cwd': str(root),
              'command_sha256': sha_bytes(encoded(command)), 'launch_token': uuid.uuid4().hex,
              'resource_root': str(resource_scope(root)), 'output_directory': 'data', 'working_tree_clean_before': True,
              'host': {'platform': platform.platform(), 'python': sys.version, 'python_executable': sys.executable},
              'environment': {key: os.environ[key] for key in ENV_KEYS if key in os.environ}, 'inputs': []}
    receipt_path = run_dir / 'run.json'
    write_new(receipt_path, encoded(record))
    try:
        for key in ('experiment', 'config'):
            raw = paths[key].read_bytes()
            write_new(run_dir / (key + '.json'), raw)
            record[key + '_identity'] = {'path': paths[key].relative_to(root).as_posix(), 'bytes': len(raw), 'sha256': sha_bytes(raw)}
        for path in args.input:
            record['inputs'].append(archive_input(home, path))
        replacements = {'{run_dir}': str(run_dir / 'data'), '{resource_root}': record['resource_root'],
                        '{config}': str(paths['config']), '{run_record}': str(receipt_path)}
        replacements.update({'{input' + str(i) + '}': str(home / item['storage_path'])
                             for i, item in enumerate(record['inputs'])})
        command = [replacements.get(value, value) for value in command]
        record['command'] = command
        record['command_sha256'] = sha_bytes(encoded(command))
        executable = shutil.which(command[0])
        if executable is None and Path(command[0]).is_file():
            executable = str(Path(command[0]).resolve())
        if executable is None:
            raise FileNotFoundError('Command executable not found: ' + command[0])
        record['executable'] = {'path': str(Path(executable).resolve()), **identity(Path(executable))}
        write_new(run_dir / 'launch.ps1', launcher(root, record, run_dir, bundles))
        record['launcher_identity'] = identity(run_dir / 'launch.ps1')
        record['status'] = 'prepared'
        write_atomic(receipt_path, encoded(record))
    except Exception as error:
        record.update(status='launch_failed', error=str(error), exit_code=1, finished_utc=now())
        write_atomic(receipt_path, encoded(record))
        raise
    print(json.dumps({'status': 'prepared', 'code_commit': commit, 'run_directory': str(run_dir),
                      'output_directory': str(run_dir / 'data'), 'launcher': str(run_dir / 'launch.ps1'),
                      'execute_after_start_exits': ['pwsh', '-NoProfile', '-File', str(run_dir / 'launch.ps1')]}, ensure_ascii=False))


def internal(args):
    root = repository_root()
    _, home = data_home(root, args.data_root)
    run_dir = external(root, home / 'runs' / segment(args.id) / segment(args.run))
    path = run_dir / 'run.json'
    record = load(path)
    if record['experiment_id'] != args.id or record['run_id'] != args.run or record['launch_token'] != args.token:
        raise ValueError('Execution token/ID mismatch')
    if args.action == '_begin':
        if record['status'] != 'prepared':
            raise ValueError('Run is not prepared; launching a run twice is forbidden')
        write_new(run_dir / 'launch.lock', encoded({'token': args.token, 'claimed_utc': now()}))
        try:
            require_clean(root)
            if git_text(root, 'rev-parse', 'HEAD') != record['code_commit'] or git_text(root, 'branch', '--show-current') != record['branch']:
                raise ValueError('Code commit or branch changed since registration')
            if args.command_sha != record['command_sha256'] or sha_bytes(encoded(record['command'])) != record['command_sha256']:
                raise ValueError('Execution command changed since registration')
            if identity(run_dir / 'launch.ps1') != record['launcher_identity']:
                raise ValueError('Generated launcher changed')
            meta = descriptor(root, args.id)
            run_identity(root, meta, record, run_dir, home, terminal=False)
            for row in record['inputs']:
                if identity(Path(row['original_path'])) != {k: row[k] for k in ('bytes', 'sha256')}:
                    raise ValueError('Input changed since registration')
            if identity(Path(record['executable']['path'])) != {k: record['executable'][k] for k in ('bytes', 'sha256')}:
                raise ValueError('Executable changed since registration')
            record.update(status='running', started_utc=now(),
                          runtime_environment={key: os.environ[key] for key in ENV_KEYS if key in os.environ})
        except Exception as error:
            record.update(status='launch_failed', error=str(error), exit_code=1, finished_utc=now())
            write_atomic(path, encoded(record))
            raise
        write_atomic(path, encoded(record))
        return 0
    if record['status'] != 'running':
        raise ValueError('Only a running execution may be completed')
    execution = load(run_dir / 'execution.json')
    if (execution.get('launch_token') != args.token or execution.get('command_sha256') != record['command_sha256']
            or not execution.get('drain_confirmed') or execution.get('active_processes_at_end', 0) != 0):
        raise ValueError('Execution identity or complete child-tree drain was not confirmed')
    exit_code = execution['exit_code']
    status = 'interrupted' if execution.get('cancelled') else ('completed' if exit_code == 0 else 'failed')
    if not execution.get('launched'):
        status = 'launch_failed'
    record.update(status=status, exit_code=exit_code, finished_utc=execution['finished_utc'], execution=identity(run_dir / 'execution.json'))
    try:
        drift = (git_text(root, 'rev-parse', 'HEAD') != record['code_commit']
                 or git_text(root, 'branch', '--show-current') != record['branch']
                 or bool(git(root, 'status', '--porcelain=v1', '--untracked-files=all').stdout))
        inputs_changed = any(identity(Path(row['original_path'])) != {k: row[k] for k in ('bytes', 'sha256')} for row in record['inputs'])
        executable_changed = identity(Path(record['executable']['path'])) != {k: record['executable'][k] for k in ('bytes', 'sha256')}
        record['identity_unchanged_after'] = not (drift or inputs_changed or executable_changed)
        if not record['identity_unchanged_after']:
            record['status'] = 'identity_drift'
    except Exception as error:
        record.update(status='identity_drift', identity_unchanged_after=False, error=str(error))
    write_atomic(path, encoded(record))
    return 0 if record['status'] == 'completed' else 1


def scan_run(run_dir):
    rows = []
    for path in sorted(run_dir.rglob('*')):
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()) or not path.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError('Symlinks are not permitted in run data: ' + str(path))
        if path.is_file():
            rows.append({'path': path.relative_to(run_dir).as_posix(), **identity(path)})
        elif not path.is_dir():
            raise ValueError('Unsupported run-data object: ' + str(path))
    return rows


def run_identity(root, meta, record, run_dir, home, terminal=True):
    if (record.get('schema') != 1 or record.get('experiment_id') != meta['experiment_id']
            or record.get('version') != meta['version'] or record.get('base_commit') != meta['base_commit']
            or record.get('branch') != meta['branch'] or record.get('status') not in (TERMINAL if terminal else TERMINAL | {'prepared', 'running'})
            or not re.fullmatch('[0-9a-f]{40}', record.get('code_commit', ''))):
        raise ValueError('Run identity/status does not match this experiment')
    commit = record['code_commit']
    if git_text(root, 'rev-parse', commit + '^{tree}') != record['code_tree']:
        raise ValueError('Code tree does not match the recorded commit')
    git(root, 'merge-base', '--is-ancestor', meta['base_commit'], commit)
    for key in ('experiment', 'config'):
        info = record.get(key + '_identity')
        if info is None:
            if record['status'] in ('launch_failed', 'interrupted'):
                continue
            raise ValueError('Missing committed configuration identity')
        expected_path = 'experiments/' + meta['experiment_id'] + '/' + key + '.json'
        if info['path'] != expected_path:
            raise ValueError('Unexpected configuration path')
        original = git_file(root, commit, info['path'])
        if len(original) != info['bytes'] or sha_bytes(original) != info['sha256']:
            raise ValueError('Recorded config differs from code commit')
        if identity(run_dir / (key + '.json')) != {k: info[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Runtime configuration snapshot changed')
    for row in record['inputs']:
        if row['storage_path'] != 'inputs/' + row['sha256']:
            raise ValueError('Invalid content-addressed input path')
        if identity(child(home, row['storage_path'])) != {k: row[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Archived input changed')


def finish(args):
    if not args.conclusion.strip():
        raise ValueError('A report requires a nonempty conclusion')
    root = repository_root()
    meta = descriptor(root, args.id)
    paths = locations(root, args.id, args.run)
    _, home = data_home(root, args.data_root)
    run_dir = external(root, home / 'runs' / args.id / args.run)
    record = load(run_dir / 'run.json')
    if record.get('run_id') != args.run:
        raise ValueError('Run ID mismatch')
    run_identity(root, meta, record, run_dir, home)
    if args.verdict == 'passed' and (record['status'] != 'completed' or record['exit_code'] != 0
                                    or not record.get('identity_unchanged_after')):
        raise ValueError('A failed/interrupted/drifting run cannot be recorded as passed')
    if git_text(root, 'branch', '--show-current') != meta['branch']:
        raise ValueError('Generate reports in the corresponding experiment branch')
    catalog_path = root / 'DATASETS.json'
    catalog = load(catalog_path)
    entries = catalog.setdefault('experiment_runs', [])
    if any(row.get('experiment_id') == args.id and row.get('run_id') == args.run for row in entries):
        raise ValueError('Run is already indexed; do not replace reports')
    seal_dir = external(root, home / 'seals' / args.id / args.run)
    for path in [paths['manifest'], paths['report'], paths['markdown'], seal_dir]:
        if path.exists() or path.is_symlink():
            raise ValueError('Refusing to replace an existing report/seal: ' + str(path))
    files = scan_run(run_dir)
    manifest = {'schema': 1, 'kind': 'external-experiment-run', 'experiment_id': args.id, 'run_id': args.run,
                'version': meta['version'], 'code_commit': record['code_commit'], 'code_tree': record['code_tree'],
                'base_commit': record['base_commit'], 'storage': {'relative_to': 'data_root.parent', 'path': 'runs/' + args.id + '/' + args.run},
                'files': files, 'file_count': len(files), 'total_bytes': sum(r['bytes'] for r in files),
                'input_objects': [{k: r[k] for k in ('storage_path', 'bytes', 'sha256')} for r in record['inputs']]}
    manifest_bytes = encoded(manifest)
    manifest_hash = sha_bytes(manifest_bytes)
    report = {'schema': 1, 'experiment_id': args.id, 'run_id': args.run, 'version': meta['version'],
              'code_commit': record['code_commit'], 'code_tree': record['code_tree'], 'base_commit': record['base_commit'],
              'run_status': record['status'], 'exit_code': record['exit_code'], 'verdict': args.verdict,
              'conclusion': args.conclusion, 'manifest': paths['manifest'].relative_to(root).as_posix(),
              'manifest_sha256': manifest_hash, 'created_utc': now(),
              'verification_scope': 'Run identity and data integrity; scientific acceptance is supplied by the experiment author.'}
    report_bytes = encoded(report)
    markdown = ('# ' + meta['title'] + ' — ' + args.run + '\n\n'
                + '- Code commit A: `' + record['code_commit'] + '`\n'
                + '- Base: `' + meta['base_ref'] + '` / `' + record['base_commit'] + '`\n'
                + '- Run: `' + args.id + '/' + args.run + '`\n'
                + '- Process status / exit: `' + record['status'] + '` / `' + str(record['exit_code']) + '`\n'
                + '- Author verdict: `' + args.verdict + '`\n'
                + '- Data manifest SHA-256: `' + manifest_hash + '`\n\n'
                + args.conclusion + '\n\nTraceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.\n').encode('utf-8')
    write_new(paths['manifest'], manifest_bytes)
    write_new(paths['report'], report_bytes)
    write_new(paths['markdown'], markdown)
    seal_dir.mkdir(parents=True, exist_ok=False)
    write_new(seal_dir / 'manifest.json', manifest_bytes)
    write_new(seal_dir / 'seal.json', encoded({'schema': 1, 'experiment_id': args.id, 'run_id': args.run,
              'code_commit': record['code_commit'], 'manifest_sha256': manifest_hash,
              'report_sha256': sha_bytes(report_bytes), 'markdown_sha256': sha_bytes(markdown)}))
    entries.append({'experiment_id': args.id, 'run_id': args.run, 'code_commit': record['code_commit'],
                    'manifest': {'path': report['manifest'], 'bytes': len(manifest_bytes), 'sha256': manifest_hash},
                    'report': {'path': paths['report'].relative_to(root).as_posix(), 'bytes': len(report_bytes), 'sha256': sha_bytes(report_bytes)},
                    'storage': manifest['storage']})
    write_atomic(catalog_path, encoded(catalog))
    print(json.dumps({'status': 'sealed', 'code_commit': record['code_commit'], 'manifest': report['manifest'],
                      'report': paths['report'].relative_to(root).as_posix(),
                      'next': 'Review scientific results, then commit DATASETS.json, evidence and reports as report commit B.'}))


def verify(args):
    root = repository_root()
    meta = descriptor(root, args.id)
    paths = locations(root, args.id, args.run)
    _, home = data_home(root, args.data_root)
    run_dir = external(root, home / 'runs' / args.id / args.run)
    record = load(run_dir / 'run.json')
    if record.get('run_id') != args.run:
        raise ValueError('Run ID mismatch')
    run_identity(root, meta, record, run_dir, home)
    manifest_bytes, report_bytes, markdown = paths['manifest'].read_bytes(), paths['report'].read_bytes(), paths['markdown'].read_bytes()
    manifest, report = json.loads(manifest_bytes), json.loads(report_bytes)
    expected_inputs = [{k: r[k] for k in ('storage_path', 'bytes', 'sha256')} for r in record['inputs']]
    if manifest.get('input_objects') != expected_inputs:
        raise ValueError('Manifest input identities do not match the run')
    seal_dir = home / 'seals' / args.id / args.run
    seal = load(seal_dir / 'seal.json')
    if (seal['manifest_sha256'] != sha_bytes(manifest_bytes) or seal['report_sha256'] != sha_bytes(report_bytes)
            or seal['markdown_sha256'] != sha_bytes(markdown) or (seal_dir / 'manifest.json').read_bytes() != manifest_bytes):
        raise ValueError('Report or manifest differs from the write-once local seal')
    expected_storage = {'relative_to': 'data_root.parent', 'path': 'runs/' + args.id + '/' + args.run}
    if manifest['storage'] != expected_storage or manifest['experiment_id'] != args.id or manifest['run_id'] != args.run:
        raise ValueError('Manifest storage/ID mismatch')
    rows = scan_run(run_dir)
    retention = None
    if getattr(args, 'retained', False):
        from retention import verify_retained
        retention = verify_retained(root, home, paths, manifest, report, rows)
    elif rows != manifest['files'] or len(rows) != manifest['file_count'] or sum(r['bytes'] for r in rows) != manifest['total_bytes']:
        raise ValueError('Run data changed, is missing, or contains additional files')
    for key in ('code_commit', 'code_tree', 'base_commit'):
        if manifest[key] != record[key] or report[key] != record[key]:
            raise ValueError('Report code reference differs from the actual run: ' + key)
    if (report['manifest_sha256'] != sha_bytes(manifest_bytes) or report['manifest'] != paths['manifest'].relative_to(root).as_posix()
            or report['experiment_id'] != args.id or report['run_id'] != args.run
            or report['run_status'] != record['status'] or report['exit_code'] != record['exit_code']):
        raise ValueError('Report/run association changed')
    entries = [row for row in load(root / 'DATASETS.json').get('experiment_runs', [])
               if row.get('experiment_id') == args.id and row.get('run_id') == args.run]
    if (len(entries) != 1 or entries[0]['code_commit'] != record['code_commit'] or entries[0]['storage'] != expected_storage
            or entries[0]['manifest'] != {'path': report['manifest'], 'bytes': len(manifest_bytes), 'sha256': sha_bytes(manifest_bytes)}
            or entries[0]['report'] != {'path': paths['report'].relative_to(root).as_posix(), 'bytes': len(report_bytes), 'sha256': sha_bytes(report_bytes)}):
        raise ValueError('Dataset index does not match the sealed report/manifest')
    if report['verdict'] == 'passed' and (record['status'] != 'completed' or record['exit_code'] != 0 or not record.get('identity_unchanged_after')):
        raise ValueError('Invalid passed verdict for unsuccessful execution')
    committed = git(root, 'show', 'HEAD:' + paths['report'].relative_to(root).as_posix(), check=False)
    report_commit = None
    if committed.returncode == 0 and committed.stdout == report_bytes:
        report_commit = git_text(root, 'log', '-1', '--format=%H', '--', paths['report'].relative_to(root).as_posix())
    print(json.dumps({'status': 'pass', 'experiment_id': args.id, 'run_id': args.run,
                      'code_commit': record['code_commit'], 'current_commit': git_text(root, 'rev-parse', 'HEAD'),
                      'report_commit': report_commit, 'report_committed': report_commit is not None,
                      'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
                      **(retention or {'mode': 'full', 'full_payload_present': True}),
                      'scope': ('Retained evidence integrity after explicitly approved trace deletion; full raw trace recheck unavailable.'
                                if retention else 'Traceability and data integrity; scientific verdict is recorded separately.')}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('_begin', '_complete'):
        private = argparse.ArgumentParser()
        private.add_argument('action', choices=('_begin', '_complete'))
        private.add_argument('--id', required=True)
        private.add_argument('--run', required=True)
        private.add_argument('--data-root', type=Path, required=True)
        private.add_argument('--token', required=True)
        private.add_argument('--command-sha')
        try:
            return internal(private.parse_args())
        except (ValueError, FileNotFoundError, FileExistsError, KeyError, OSError) as error:
            print('ERROR: ' + str(error), file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('new')
    p.add_argument('--id', required=True)
    p.add_argument('--version', choices=VERSIONS, required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--purpose', required=True)
    p.add_argument('--worktree-root', type=Path)
    p.add_argument('--data-root', type=Path)
    for action in ('start', 'finish', 'verify'):
        p = commands.add_parser(action)
        p.add_argument('--id', required=True)
        p.add_argument('--run', required=True)
        p.add_argument('--data-root', type=Path)
        if action == 'start':
            p.add_argument('--input', type=Path, action='append', default=[])
            p.add_argument('command', nargs=argparse.REMAINDER)
        elif action == 'finish':
            p.add_argument('--verdict', choices=('passed', 'failed', 'inconclusive'), required=True)
            p.add_argument('--conclusion', required=True)
        elif action == 'verify':
            p.add_argument('--retained', action='store_true', help='Verify retained evidence and explicit cleanup receipts; does not claim complete original traces')
    args = parser.parse_args()
    try:
        return {'new': new, 'start': start, 'finish': finish, 'verify': verify}[args.action](args) or 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, OSError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
