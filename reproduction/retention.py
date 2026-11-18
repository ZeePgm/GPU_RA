"""Explicitly approved trace retirement; original reports, manifests and seals stay unchanged."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import experiment as E

TRACE_NAMES = {'lane-times.bin', 'requests.bin', 'weight-requests.jsonl', 'gpu-activity.jsonl', 'chunks.jsonl', 'kv-media.jsonl'}


def require(value, message):
    if not value:
        raise ValueError(message)


def names(paths):
    return paths['manifest'].parent / 'retention-plan.json', paths['manifest'].parent / 'retention-result.json'


def committed(root, path, content, revision='HEAD'):
    relative = path.relative_to(root).as_posix()
    data = E.git(root, 'show', revision + ':' + relative, check=False)
    if data.returncode or data.stdout != content:
        return None
    return E.git_text(root, 'log', '-1', '--format=%H', revision, '--', relative)


def trace_name(relative, subprocess_logs=False):
    parts = PurePosixPath(relative).parts
    if subprocess_logs:
        require(len(parts) == 2 and parts[0] == 'data' and PurePosixPath(parts[1]).suffix == '.log',
                'Not a top-level child process log: ' + relative)
    else:
        require(len(parts) >= 3 and parts[0] == 'data' and parts[-1] in TRACE_NAMES, 'Not an eligible raw trace: ' + relative)
    require(not PurePosixPath(relative).is_absolute() and '..' not in parts and ':' not in relative and '\\' not in relative
            and '\x00' not in relative and PurePosixPath(relative).as_posix() == relative, 'Unsafe trace path')
    return parts


def trace_path(run_dir, relative, subprocess_logs=False):
    parts = trace_name(relative, subprocess_logs)
    # Reject every reparse point before resolving; never unlink through a junction.
    base = Path(run_dir).absolute()
    target = base.joinpath(*parts)
    for path in [target, *target.parents]:
        try:
            info = path.lstat()
            require(not path.is_symlink() and not (getattr(info, 'st_file_attributes', 0) & 0x400), 'Reparse point in trace path')
        except FileNotFoundError:
            pass
        if path == base:
            break
    resolved = target.resolve()
    require(resolved != base.resolve() and resolved.is_relative_to(base.resolve()), 'Trace leaves resolved run root')
    return resolved


def checked_plan(root, paths, manifest, report):
    plan_path, _ = names(paths)
    raw = plan_path.read_bytes()
    p = json.loads(raw)
    require(p.get('schema') == 1 and p.get('kind') == 'approved-trace-retention' and
            p.get('experiment_id') == manifest['experiment_id'] and p.get('run_id') == manifest['run_id'], 'Retention plan identity')
    require(report['verdict'] == 'passed', 'Only accepted runs are eligible; failed/interrupted evidence stays intact')
    require(p['manifest_sha256'] == E.identity(paths['manifest'])['sha256'] and
            p['report_sha256'] == E.identity(paths['report'])['sha256'], 'Retention plan original identity')
    approval = p.get('approval', {})
    require(approval.get('source') == 'explicit_user_message' and approval.get('text', '').strip(), 'Explicit user approval record required')
    proof = p.get('full_verification_before_cleanup', {})
    require(proof.get('status') == 'pass' and proof.get('report_committed') and
            proof.get('code_commit') == manifest['code_commit'] and proof.get('files') == manifest['file_count'] and
            proof.get('bytes') == manifest['total_bytes'], 'Complete pre-cleanup verification evidence missing')
    original = {r['path']: r for r in manifest['files']}
    removed = p.get('remove', [])
    require(removed and len({r['path'] for r in removed}) == len(removed), 'Empty/duplicate removal list')
    for row in removed:
        require(original.get(row['path']) == row, 'Removal row differs from original sealed manifest')
        subprocess_logs = p.get('scope') == 'subprocess-logs'
        trace_name(row['path'], subprocess_logs)
        if subprocess_logs:
            require(row['path'][:-4] + '.process.json' in original, 'Missing sealed child process record')
    require(p['remove_bytes'] == sum(r['bytes'] for r in removed), 'Removal byte count')
    keep = [r for r in manifest['files'] if r['path'] not in {r['path'] for r in removed}]
    require(p['retained_inventory_sha256'] == E.sha_bytes(E.encoded(keep)), 'Retained inventory identity')
    return p, raw, keep


def context(args):
    root = E.repository_root()
    meta = E.descriptor(root, args.id)
    paths = E.locations(root, args.id, args.run)
    _, home = E.data_home(root, args.data_root)
    spelling = home / 'runs' / args.id / args.run
    for parent in [spelling, *spelling.parents]:
        require(not parent.is_symlink() and not (getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400), 'Reparse point in run root')
        if parent == home:
            break
    run_dir = E.external(root, spelling)
    manifest, report = E.load(paths['manifest']), E.load(paths['report'])
    return root, home, paths, run_dir, manifest, report


def full_verify(args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        E.verify(SimpleNamespace(id=args.id, run=args.run, data_root=args.data_root, retained=False))
    result = json.loads(out.getvalue())
    require(result['report_committed'], 'Commit report B before planning or applying retirement')
    return result


def prepare(args):
    root, home, paths, run_dir, manifest, report = context(args)
    require(args.approval_text.strip(), 'Nonempty approval text required; an agent must obtain actual user approval first')
    plan_path, result_path = names(paths)
    require(not plan_path.exists() and not result_path.exists(), 'A retention record already exists')
    verification = full_verify(args)
    requested = getattr(args, 'subprocess_log', None)
    if requested:
        originals = {r['path']: r for r in manifest['files']}
        require(len(requested) == len(set(requested)), 'Duplicate child log selection')
        for relative in requested:
            trace_name(relative, True)
            require(relative in originals and relative[:-4] + '.process.json' in originals,
                    'Child log or matching process record missing from sealed manifest: ' + relative)
        removed = [originals[path] for path in sorted(requested)]
    else:
        removed = [r for r in manifest['files'] if r['path'].startswith('data/') and Path(r['path']).name in TRACE_NAMES]
    for row in removed:
        trace_path(run_dir, row['path'], bool(requested))
    keep = [r for r in manifest['files'] if r not in removed]
    plan = {'schema': 1, 'kind': 'approved-trace-retention', 'experiment_id': args.id, 'run_id': args.run,
            'created_utc': E.now(), 'approval': {'source': 'explicit_user_message', 'text': args.approval_text},
            'manifest_sha256': E.identity(paths['manifest'])['sha256'], 'report_sha256': E.identity(paths['report'])['sha256'],
            'full_verification_before_cleanup': verification, 'remove': removed,
            'remove_bytes': sum(r['bytes'] for r in removed), 'retained_inventory_sha256': E.sha_bytes(E.encoded(keep)),
            'scope_after_cleanup': 'Retained references and reports remain verifiable; complete raw trace replay/recalculation is unavailable.'}
    if requested:
        plan['scope'] = 'subprocess-logs'
        plan['scope_after_cleanup'] = 'Only the listed child process text logs are retired; their sealed process records remain.'
    checked = E.encoded(plan)
    E.write_new(plan_path, checked)
    checked_plan(root, paths, manifest, report)
    print(json.dumps({'status': 'planned', 'files': len(removed), 'bytes': plan['remove_bytes'],
                      'plan': str(plan_path), 'next': 'Commit the approved exact plan before apply.'}))


def durable_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def apply(args):
    root, home, paths, run_dir, manifest, report = context(args)
    E.require_clean(root)
    code_commit = E.git_text(root, 'rev-parse', 'HEAD')
    tool = {}
    for filename in ('retention.py', 'experiment.py', 'data_paths.py'):
        tool[filename] = E.identity(Path(__file__).parent / filename)['sha256']
        require(tool[filename] == E.sha_bytes(E.git_file(root, code_commit, 'reproduction/' + filename)), 'Cleanup implementation differs from committed source')
    plan_path, result_path = names(paths)
    require(not result_path.exists(), 'Cleanup is already recorded; do not repeat it')
    p, raw, keep = checked_plan(root, paths, manifest, report)
    plan_commit = committed(root, plan_path, raw)
    require(plan_commit, 'Approved exact removal plan must be committed before deletion')
    directory = E.external(root, home / 'retentions' / args.id / args.run)
    anchor, start, journal = directory / 'plan.json', directory / 'start.json', directory / 'events.jsonl'
    if anchor.exists():
        require(anchor.read_bytes() == raw, 'External retention anchor differs')
    else:
        durable_new(anchor, raw)
    if not start.exists():
        full_verify(args)
        durable_new(start, E.encoded({'plan_sha256': E.sha_bytes(raw), 'plan_commit': plan_commit, 'started_utc': E.now()}))
    started = E.load(start)
    require(started['plan_sha256'] == E.sha_bytes(raw) and committed(root, plan_path, raw, started['plan_commit']), 'Cleanup start identity')
    attempted = set()
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            require(item['path'] in {r['path'] for r in p['remove']}, 'Journal has an unapproved path')
            if item['event'] == 'before_unlink':
                attempted.add(item['path'])
    current = E.scan_run(run_dir)
    originals = {r['path']: r for r in manifest['files']}
    require(all(originals.get(r['path']) == r for r in current), 'Unexpected or modified retained/run file')
    missing = set(originals) - {r['path'] for r in current}
    require(missing <= attempted, 'Missing file without a recorded deletion intent')
    def event(kind, row):
        with journal.open('ab') as stream:
            stream.write(json.dumps({'utc': E.now(), 'event': kind, **row}).encode('utf-8') + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
    for row in p['remove']:
        target = trace_path(run_dir, row['path'], p.get('scope') == 'subprocess-logs')
        if target.exists():
            require(E.identity(target) == {k: row[k] for k in ('bytes', 'sha256')}, 'Trace changed before deletion')
            event('before_unlink', row)
            target.unlink()  # a regular file only, within the checked absolute run root
            event('unlinked', row)
    after = E.scan_run(run_dir)
    require(after == keep, 'Retained files differ from original sealed identities')
    result = {'schema': 1, 'kind': 'completed-trace-retention', 'experiment_id': args.id, 'run_id': args.run,
              'status': 'completed', 'completed_utc': E.now(), 'plan_sha256': E.sha_bytes(raw),
              'plan_commit': started['plan_commit'], 'cleanup_code_commit': code_commit, 'cleanup_tool_sha256': tool,
              'removed_files': len(p['remove']), 'removed_bytes': p['remove_bytes'],
              'retained_files': len(keep), 'retained_bytes': sum(r['bytes'] for r in keep),
              'retained_inventory_sha256': E.sha_bytes(E.encoded(after)), 'full_payload_present': False,
              'full_raw_trace_recheck_available': False, 'original_manifest_unchanged': True}
    external_result = directory / 'result.json'
    if external_result.exists():
        result = E.load(external_result)
        require(result['retained_inventory_sha256'] == E.sha_bytes(E.encoded(after)) and result['plan_sha256'] == E.sha_bytes(raw), 'External completion differs')
    else:
        durable_new(external_result, E.encoded(result))
    E.write_new(result_path, external_result.read_bytes())
    print(json.dumps(result))


def verify_retained(root, home, paths, manifest, report, rows):
    p, raw, keep = checked_plan(root, paths, manifest, report)
    plan_path, result_path = names(paths)
    result_raw = result_path.read_bytes()
    result = json.loads(result_raw)
    directory = E.external(root, home / 'retentions' / manifest['experiment_id'] / manifest['run_id'])
    require((directory / 'plan.json').read_bytes() == raw and (directory / 'result.json').read_bytes() == result_raw, 'Retention external anchor mismatch')
    require(result.get('status') == 'completed' and result.get('kind') == 'completed-trace-retention' and
            result.get('schema') == 1 and result['experiment_id'] == manifest['experiment_id'] and result['run_id'] == manifest['run_id'], 'Retention completion identity')
    require(committed(root, plan_path, raw, result['plan_commit']), 'Removal plan was not committed as recorded')
    for filename in ('retention.py', 'experiment.py', 'data_paths.py'):
        require(result['cleanup_tool_sha256'][filename] == E.sha_bytes(E.git_file(root, result['cleanup_code_commit'], 'reproduction/' + filename)),
                'Cleanup implementation identity changed')
    require(result['plan_sha256'] == E.sha_bytes(raw) and result['removed_files'] == len(p['remove']) and
            result['removed_bytes'] == p['remove_bytes'] and rows == keep and result['retained_files'] == len(keep) and
            result['retained_bytes'] == sum(r['bytes'] for r in keep) and
            result['retained_inventory_sha256'] == E.sha_bytes(E.encoded(rows)), 'Retained evidence changed or unauthorized files are missing/present')
    require(result['full_payload_present'] is False and result['full_raw_trace_recheck_available'] is False and result['original_manifest_unchanged'] is True,
            'Invalid complete-data claim after retirement')
    return {'mode': 'retained', 'full_payload_present': False, 'full_raw_trace_recheck_available': False,
            'removed_files': len(p['remove']), 'removed_bytes': p['remove_bytes'],
            'original_files': manifest['file_count'], 'original_bytes': manifest['total_bytes'],
            'retention_commit': committed(root, result_path, result_raw),
            'retention_committed': committed(root, result_path, result_raw) is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('plan', 'apply'):
        p = commands.add_parser(action)
        p.add_argument('--id', required=True)
        p.add_argument('--run', required=True)
        p.add_argument('--data-root', type=Path)
        if action == 'plan':
            p.add_argument('--approval-text', required=True)
            p.add_argument('--subprocess-log', nargs='+', metavar='data/NAME.log',
                           help='Remove only these exact child logs with matching sealed process records')
    args = parser.parse_args()
    try:
        {'plan': prepare, 'apply': apply}[args.action](args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, OSError) as error:
        print('ERROR: ' + str(error), file=E.sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
