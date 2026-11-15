"""Portable bindings for the unchanged, frozen V3 correctness contracts."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import types
from datetime import datetime, timezone

JOINT = Path(__file__).resolve().parents[1]
REPOSITORY = JOINT.parent
MODEL_SHA = '74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2'
PROFILE_SHA = '5bac49b46ed3ab989dfa8a34a2d3f02e9fe1e9cf4d686f7170b5138e26eae5da'
PROMPT_SHA = 'ce4fa40becfa4a191064fbda15721985a953ae231557ea6f6f8f7b669e0d1a74'
VALIDATOR_SHA = '3e93f076b1add9426309de741c062e7fd44b1721961663304ae89726d04ba318'
MODES = {
    'native': ('native', 'native', 'native'),
    'readonly-bypass': ('b2', 'native', 'bypass'),
    'readonly-modeled': ('b2', 'native', 'modeled'),
    'kv-bypass': ('native', 'paged', 'bypass'),
    'kv-modeled': ('native', 'paged', 'modeled'),
    'joint-bypass': ('b2', 'paged', 'bypass'),
    'joint-modeled': ('b2', 'paged', 'modeled'),
}
ALL_MODES = {**MODES, 'joint-host': ('b2', 'paged', 'modeled')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correctness_module():
    path = JOINT / 'scripts/run_correctness.py'
    require(sha(path) == VALIDATOR_SHA, 'Frozen correctness validator changed')
    return import_file('gpu_ra_v3_frozen_correctness', path)


def statistical_module():
    """Load original function bodies without their obsolete path/import wiring."""
    path = JOINT / 'statistical-batch/scripts/run.py'
    require(sha(path) == 'c0d719dc2ef9a62ef26b4f2db4b2e4c9d223f5d5fd72ca3cd42cfdefd624e46a',
            'Frozen statistical runner changed')
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    module = types.ModuleType('gpu_ra_v3_frozen_statistics')
    module.__file__ = str(Path(__file__).parent / 'run_statistics.py')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    module.MODES = MODES
    return module


def add_arguments(parser):
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cuda-root', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=JOINT / 'build-reproduction/model')
    parser.add_argument('--gpu-device', default='0')
    parser.add_argument('--resource-root', type=Path, default=REPOSITORY.parent,
                        help='OS process scope, normally contains sibling version worktrees')


def validate_inputs(args, require_build=True):
    require(os.name == 'nt', 'V3 live execution and host-resource validation require Windows')
    require(args.model.is_file() and sha(args.model) == MODEL_SHA, 'GGUF SHA-256 differs from frozen V3')
    require(args.model.stat().st_size == 1509347712, 'GGUF byte size differs from frozen V3')
    require(sha(JOINT / 'configs/qwen3-0.6b.json') == PROFILE_SHA, 'Frozen profile changed')
    require(sha(JOINT / 'configs/prompt.txt') == PROMPT_SHA, 'Original 249-byte CRLF prompt changed')
    require((args.cuda_root / 'bin').is_dir(), 'CUDA root must contain bin')
    require((args.cuda_root / 'lib').is_dir(), 'CUDA root must contain lib/CUPTI runtime')
    if require_build:
        require((args.build_dir / 'bin/realgpu_llama.exe').is_file(), 'Build realgpu_llama from this V3 source first')
        validate_build(args.build_dir, args.cuda_root)
    correctness_module()


def long_path(path):
    path = Path(path).resolve()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        path = Path('\\\\?\\' + str(path))
    return path


def validate_source():
    manifest = read(JOINT / 'source-bundle/manifest.json')
    rows = manifest['files']
    require(len(rows) == 4276 and len({row['path'] for row in rows}) == 4276, 'V3 source membership changed')
    for row in rows:
        path = long_path(JOINT / row['path'])
        require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Historical V3 source drift: ' + row['path'])
    return {'status': 'pass', 'source_records': len(rows)}


def validate_build(build_dir, cuda_root):
    build_dir = build_dir.resolve()
    base = build_dir.parent
    record_path = base / 'build-acceptance.json'
    require(record_path.is_file(), 'Source-build acceptance missing: ' + str(record_path))
    record = read(record_path)
    require(record['status'] == 'pass' and record['tests_executed'] and record.get('final_parameterized_entry') is True,
            'Final parameterized source build/component verification did not pass')
    require(Path(record['build_root']).resolve() == base, 'Build-root provenance mismatch')
    require(Path(record['source_root']).resolve() == (JOINT / 'source').resolve(), 'Build used another source tree')
    require(record['historical_source']['status'] == 'pass' and record['historical_source']['source_records'] == 4276,
            'Build source provenance incomplete')
    frozen = record['historical_source_manifest']
    source_manifest = JOINT / 'source-bundle/manifest.json'
    require(frozen['bytes'] == source_manifest.stat().st_size and frozen['sha256'] == sha(source_manifest),
            'Source manifest differs from accepted build')
    require(Path(record['build_configuration']['cuda_root']).resolve() == cuda_root.resolve(),
            'CUDA runtime root differs from the verified source build')
    for project in record['source_builds']:
        require(not project['source_paths_outside_sealed_tree'] and not project['historical_build_dependencies'],
                'Build imports sources or libraries outside the restored closure')
    artifacts = {}
    for row in record['new_binaries']:
        path = (base / row['path']).resolve()
        require(path.is_relative_to(base), 'Build artifact escapes its build root')
        require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Fresh build artifact drift: ' + str(path))
        artifacts[path] = row
    expected = [build_dir / 'bin/realgpu_llama.exe', *list((build_dir / 'bin').glob('*.dll'))]
    require(all(path.resolve() in artifacts for path in expected), 'Build receipt does not cover model executable/DLLs')
    return validate_source()


def make_inputs(destination, model):
    configs = destination / 'configs'
    configs.mkdir(parents=True, exist_ok=False)
    base = read(JOINT / 'configs/qwen3-0.6b.json')
    profile = dict(base)
    conversion = REPOSITORY / 'provenance/Qwen3-0.6B-conversion.json'
    require(conversion.is_file(), 'Pinned model conversion record missing')
    require(read(conversion)['gguf_sha256'] == MODEL_SHA, 'Conversion identity changed')
    shutil.copy2(conversion, configs / 'conversion.json')
    profile['model_path'] = str(model.resolve())
    profile['conversion'] = str((configs / 'conversion.json').resolve())
    require({k:v for k,v in profile.items() if k not in ('model_path','conversion')} ==
            {k:v for k,v in base.items() if k not in ('model_path','conversion')}, 'Profile semantics changed')
    write(configs / 'qwen3-0.6b.json', profile)
    shutil.copy2(JOINT / 'configs/prompt.txt', configs / 'prompt.txt')
    return profile


def configure_correctness(inputs, output, build_dir, cuda_root, gpu_device='0'):
    module = correctness_module()
    module.R, module.OUT = inputs.resolve(), output.resolve()
    module.BIN, module.CUDA = (build_dir / 'bin').resolve(), cuda_root.resolve()
    original_env = module.environment
    def environment():
        env = original_env()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
        return env
    module.environment = environment
    return module


def process_probe():
    return import_file('gpu_ra_v3_process_probe', Path(__file__).parent / 'process_probe.py')


def probe(resource_root, child=None):
    rows = process_probe().processes()
    require(any(p['pid'] == 4 and p['name'] == 'System' for p in rows), 'Host process visibility missing')
    allowed = {os.getpid(), os.getppid(), child}
    root = str(resource_root.resolve()).replace('\\', '/').lower().rstrip('/') + '/'
    names = {'cl.exe', 'nvcc.exe', 'ninja.exe', 'realgpu_llama.exe', 'gpu_ffn.exe', 'f_candidate.exe', 'f_clock.exe'}
    conflicts = [p for p in rows if p['pid'] not in allowed and
                 (p['name'].lower() in names or
                  (p['path'].replace('\\', '/').lower().startswith(root) and p['name'].lower().endswith('.exe')))]
    return {'utc': stamp(), 'visible_processes': len(rows), 'conflicts': conflicts}


def identities(module, extra=()):
    paths = [*module.BIN.glob('*.exe'), *module.BIN.glob('*.dll'),
             JOINT / 'configs/qwen3-0.6b.json', JOINT / 'configs/prompt.txt',
             JOINT / 'scripts/run_correctness.py', JOINT / 'source-bundle/manifest.json',
             module.BIN.parents[1] / 'build-acceptance.json',
             module.R / 'configs/qwen3-0.6b.json', module.R / 'configs/prompt.txt',
             module.R / 'configs/conversion.json', *extra]
    paths += [Path(__file__), Path(__file__).parent / 'process_probe.py']
    return [{'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(paths))]


def verify_identities(items):
    for row in items:
        path = Path(row['path'])
        require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Dependency drift: ' + row['path'])


def run_case(module, scale, label, resource_root, timeout=3600):
    weights, kv, mode = ALL_MODES[label]
    output = module.OUT / 'cases'
    output.mkdir(exist_ok=True)
    name = f'{scale}-{label}'
    folder, record_path = output / name, output / (name + '.process.json')
    require(not folder.exists() and not record_path.exists(), 'Refuse overwrite: ' + name)
    before = probe(resource_root)
    require(not before['conflicts'], 'Resource conflict before model launch: ' + repr(before))
    env = module.environment()
    env.update(HBFLAB_JOINT_SCALE=scale, HBFLAB_JOINT_WEIGHTS=weights, HBFLAB_JOINT_KV=kv,
               HBFLAB_KV_WRITE_MODE='host' if label == 'joint-host' else 'device')
    profile = read(module.R / 'configs/qwen3-0.6b.json')
    command = [str(module.BIN / 'realgpu_llama.exe'), profile['model_path'],
               str(module.R / 'configs/prompt.txt'), str(folder), '8', '0', mode,
               str(module.R / 'configs/qwen3-0.6b.json')]
    start, started = time.monotonic(), stamp()
    observed, probes, failure = [], [before], None
    print('START', name, started, flush=True)
    with (output / (name + '.log')).open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        next_probe = 0
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed >= next_probe:
                    snap = probe(resource_root, proc.pid)
                    probes.append(snap)
                    require(not snap['conflicts'], 'Resource conflict: ' + repr(snap))
                    next_probe = elapsed + 5
                if not observed and elapsed >= .3:
                    paths = module.modules(proc.pid)
                    if any(Path(path).name.lower() == 'ggml-cuda.dll' for path in paths):
                        observed = paths
                if elapsed > timeout:
                    raise TimeoutError(name)
                time.sleep(.1)
        except BaseException:
            failure = traceback.format_exc()
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
    data = {'start_utc': started, 'end_utc': stamp(), 'wall_s': time.monotonic()-start,
            'returncode': proc.returncode, 'error': failure, 'command': command,
            'environment': {k:v for k,v in env.items() if k.startswith(('HBFLAB_', 'GGML_', 'CUDA_'))},
            'modules': observed, 'resource_probes': probes}
    write(record_path, data)
    require(failure is None and proc.returncode == 0, f'{name}: {failure or proc.returncode}')
    dlls = {Path(path).name.lower(): Path(path) for path in observed}
    modules = []
    for path in module.BIN.glob('*.dll'):
        if path.name.lower().startswith(('ggml', 'llama')):
            require(path.name.lower() in dlls and dlls[path.name.lower()].resolve() == path.resolve(),
                    'Wrong or missing loaded DLL: ' + str(path))
            modules.append({'path': str(path.resolve()), 'sha256': sha(path)})
    write(output / (name + '.modules.json'), modules)
    module.validate_case(folder, scale, weights, kv, mode)
    print('PASS', name, round(data['wall_s'], 3), 'seconds', flush=True)
    return data
