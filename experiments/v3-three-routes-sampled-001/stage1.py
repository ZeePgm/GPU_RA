"""Independent GGUF demand inventory and native GPU graph observation for stage 1."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'joint/source/third_party/llama.cpp-integrated/gguf-py'))
sys.path.insert(0, str(ROOT / 'joint/reproduction'))
from gguf import GGUFReader
import runtime_common as R

MODEL_SHA = '74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2'
PROMPT_SHA = 'ce4fa40becfa4a191064fbda15721985a953ae231557ea6f6f8f7b669e0d1a74'
EXPECTED = {
    'token_embd.weight': ('embedding', 'GET_ROWS', 0, [1024, 151936], 'F16'),
    'output.weight': ('output', 'MUL_MAT', 0, [1024, 151936], 'F16'),
    'output_norm.weight': ('norm', 'MUL', 1, [1024], 'F32'),
}
for role, shape in [('attn_q', [1024, 2048]), ('attn_k', [1024, 1024]),
                    ('attn_v', [1024, 1024]), ('attn_output', [2048, 1024]),
                    ('ffn_gate', [1024, 3072]), ('ffn_up', [1024, 3072]), ('ffn_down', [3072, 1024])]:
    EXPECTED[f'blk.10.{role}.weight'] = ('ffn' if role.startswith('ffn') else 'attention', 'MUL_MAT', 0, shape, 'F16')
for role, width in [('attn_norm', 1024), ('ffn_norm', 1024), ('attn_q_norm', 128), ('attn_k_norm', 128)]:
    EXPECTED[f'blk.10.{role}.weight'] = ('norm', 'MUL', 1, [width], 'F32')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with io_path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def io_path(path):
    path = Path(path).resolve()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def identity(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': io_path(path).stat().st_size, 'sha256': sha(path)}


def inventory(model):
    require(sha(model) == MODEL_SHA, 'Model identity mismatch')
    reader = GGUFReader(str(model))
    require(reader.get_field('general.architecture').contents() == 'qwen3', 'Architecture mismatch')
    rows, logical = [], 0
    for t in sorted(reader.tensors, key=lambda t: t.name):
        shape = [int(x) for x in t.shape]
        selected = t.name in EXPECTED
        if selected:
            category, op, slot, expected_shape, dtype = EXPECTED[t.name]
            require(shape == expected_shape and t.tensor_type.name == dtype, 'Selected layout mismatch: ' + t.name)
        else:
            category, op, slot = 'unselected_native', None, None
        padded = (int(t.n_bytes) + 4095) // 4096 * 4096
        rows.append({'name': t.name, 'shape': shape, 'dtype': t.tensor_type.name, 'bytes': int(t.n_bytes),
                     'file_offset': int(t.data_offset), 'logical_base': logical, 'reserved_bytes': padded,
                     'selected': selected, 'category': category, 'expected_op': op, 'expected_source_slot': slot})
        logical += padded
    require(sum(t['selected'] for t in rows) == 14, 'Selected name inventory incomplete')
    selected = [t for t in rows if t['selected']]
    file_spans = sorted((t['file_offset'], t['file_offset'] + t['bytes'], t['name']) for t in selected)
    require(all(a[1] <= b[0] for a, b in zip(file_spans, file_spans[1:])), 'Selected file storage overlaps')
    return {'schema': 1, 'model': identity(model), 'all_tensors': rows,
            'selected_count': 14, 'unselected_count': len(rows) - 14,
            'selected_bytes': sum(t['bytes'] for t in selected), 'logical_weight_layout_bytes': logical,
            'file_alias_groups': [], 'kv_base': logical, 'kv_reserved_bytes': 14336 * 4096,
            'media_geometry': {'pages_per_block': 4, 'blocks_per_bank': 65536, 'banks': 2,
                               'capacity_bytes': 2 * 1024**3, 'old_capacity_bytes': 1024**3},
            'scope': 'GGUF file inventory; loaded aliases and graph consumers validated independently'}


def source_identity():
    paths = subprocess.check_output(['git', 'ls-files', '-z', 'joint/source'], cwd=ROOT).decode().split('\0')
    return [identity(ROOT / name) for name in paths if name]


def audit_build(build):
    require((build / 'compile_commands.json').is_file(), 'Compile source inventory missing')
    compiled = set()
    source = (ROOT / 'joint/source').resolve()
    for row in read(build / 'compile_commands.json'):
        path = Path(row['file']).resolve()
        if path.is_relative_to(source):
            compiled.add(path.relative_to(ROOT).as_posix())
        elif path.is_relative_to(build) and 'Unity' in path.parts:
            children = re.findall(r'^#include "([^"]+)"', path.read_text(encoding='utf-8'), re.M)
            require(bool(children), 'Empty generated unity source')
            for child in children:
                original = Path(child).resolve()
                require(original.is_relative_to(source), 'Unity imports foreign source: ' + child)
                compiled.add(original.relative_to(ROOT).as_posix())
        else:
            raise ValueError('Foreign compiled source: ' + str(path))
    ninja = (build / 'build.ninja').read_text(encoding='utf-8')
    require(not any(s in ninja for s in ('build-r2-final', 'build-r3-current', 'build-reproduction-host')), 'Historical binary dependency')
    return {'compiled_source_paths': sorted(compiled),
            'binaries': [identity(p) for p in sorted((build / 'bin').iterdir()) if p.suffix.lower() in ('.exe', '.dll')],
            'historical_verifiers_modified': False}


def launch_probe(args, build, selection, prompt, label):
    output = args.output / label
    before = R.probe(args.resource_root)
    require(not before['conflicts'], 'Resource conflict before probe: ' + repr(before))
    frozen = R.correctness_module()
    frozen.BIN, frozen.CUDA = build / 'bin', args.cuda_root
    env = frozen.environment()
    env.update(HBFLAB_M4_PAGED='0', LLAMA_GRAPH_REUSE_DISABLE='1')
    command = [str(build / 'bin/sampled_inventory.exe'), str(args.model), str(prompt), str(output), str(selection), '8']
    record = {'command': command, 'prompt': identity(prompt), 'resource_probes': [before], 'loaded_modules': []}
    start = time.monotonic()
    with (args.output / (label + '.log')).open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            next_probe = 0
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed >= next_probe:
                    snap = R.probe(args.resource_root, proc.pid)
                    record['resource_probes'].append(snap)
                    require(not snap['conflicts'], 'Resource conflict during probe: ' + repr(snap))
                    next_probe = elapsed + 5
                if not record['loaded_modules'] and elapsed >= .2:
                    modules = frozen.modules(proc.pid)
                    if any(Path(p).name.lower() == 'ggml-cuda.dll' for p in modules):
                        record['loaded_modules'] = modules
                require(elapsed < 180, 'Inventory model process timeout')
                time.sleep(.1)
        except BaseException:
            record['error'] = traceback.format_exc()
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            raise
        finally:
            record.update(exit_code=proc.poll(), wall_seconds=time.monotonic() - start)
            write(args.output / (label + '.process.json'), record)
    require(proc.returncode == 0, 'Inventory process failed: ' + label)
    loaded = {Path(p).name.lower(): Path(p).resolve() for p in record['loaded_modules']}
    require(all(loaded.get(p.name.lower()) == p.resolve() for p in (build / 'bin').glob('*.dll')
                if p.name.lower().startswith(('ggml', 'llama'))), 'Loaded module path mismatch')
    return output


def demand_plan(selection, case):
    loaded = read(case / 'loaded-tensors.json')
    actual = {t['name']: t for t in loaded}
    expected = {t['name']: t for t in selection['all_tensors'] if t['selected']}
    selected = {n: actual[n] for n in expected}
    for name, t in selected.items():
        require(t['contiguous'] and t['cuda_memory_type'] == 2 and t['cuda_device'] == 0, 'Unsupported loaded layout: ' + name)
        require(t['shape'] == expected[name]['shape'] + [1] * (4 - len(expected[name]['shape'])), 'Loaded shape differs: ' + name)
        require(t['dtype'].upper() == expected[name]['dtype'], 'Loaded dtype differs: ' + name)
    spans = sorted((t['address'], t['address'] + t['bytes'], n) for n, t in selected.items())
    aliases = []
    for a, b in zip(spans, spans[1:]):
        if a[1] > b[0]:
            require(a[:2] == b[:2], 'Partial loaded alias unsupported')
            aliases.append([a[2], b[2]])
    summary = read(case / 'summary.json')
    rows = [json.loads(line) for line in (case / 'consumers.jsonl').read_text(encoding='utf-8').splitlines()]
    require(Counter((r['step'], r['weight']) for r in rows) == Counter((step, name) for step in range(8) for name in expected),
            'Consumer multiplicity mismatch; graph path needs explicit support')
    demands = []
    for step in range(8):
        input_tokens = summary['prompt_tokens'] if step == 0 else [summary['generated_tokens'][step - 1]]
        require(summary['steps'][step]['input_tokens'] == input_tokens, 'Generation input chain mismatch')
        for name, tensor in expected.items():
            row = next(r for r in rows if r['step'] == step and r['weight'] == name)
            require(row['op'] == tensor['expected_op'] and row['source_slot'] == tensor['expected_source_slot'],
                    'Consumer operation/slot mismatch: ' + name)
            if name == 'token_embd.weight':
                ranges = [{'row': token, 'occurrence': i, 'offset': token * 2048, 'bytes': 2048} for i, token in enumerate(input_tokens)]
                require(row['indices'] == input_tokens, 'Gather indices do not match actual model inputs')
            else:
                ranges = [{'offset': 0, 'bytes': tensor['bytes']}]
            require(row['ranges'] == ranges, 'Observed range differs from independent demand: ' + name)
            for span in ranges:
                require(0 <= span['offset'] and span['offset'] + span['bytes'] <= tensor['bytes'], 'Demand outside tensor')
            demands.append({'step': step, 'phase': 'prefill' if step == 0 else 'decode', 'tensor': name,
                            'op': tensor['expected_op'], 'source_slot': tensor['expected_source_slot'],
                            'logical_base': tensor['logical_base'], 'ranges': ranges, 'calls': 1,
                            'node_name': row['node']['name'], 'node_shape': row['node']['shape'],
                            'registration_bytes': tensor['bytes'], 'fixture_coverage': 'pending_stage3',
                            'model_demand_bytes': sum(r['bytes'] for r in ranges), 'gate_coverage': 'pending_stage2'})
    resource = read(case / 'resources.json')
    require(len(resource['route_buffers']) == 3 and all(r['allocation_check'] == 'pass' for r in resource['route_buffers']), 'Route allocation check incomplete')
    return {'schema': 1, 'status': 'pass', 'prompt_tokens': summary['prompt_tokens'], 'generated_tokens': summary['generated_tokens'],
            'demands': demands, 'runtime_alias_groups': aliases, 'consumer_count': len(rows),
            'model_demand_bytes': sum(d['model_demand_bytes'] for d in demands),
            'embedding_rows': sum(len(d['ranges']) for d in demands if d['tensor'] == 'token_embd.weight'),
            'embedding_unique_rows': len({r['row'] for d in demands if d['tensor'] == 'token_embd.weight' for r in d['ranges']}),
            'unsupported_observed_layouts': [], 'resources': resource,
            'gate_status': 'Inventory observes native consumers; installation and dependency proof are later stage gates'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--cuda-root', type=Path, required=True)
    ap.add_argument('--resource-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output = args.output.resolve()
    require(not args.output.exists(), 'Use a new output directory')
    args.output.mkdir(parents=True)
    try:
        selection = inventory(args.model)
        require(sha(HERE / 'prompt-44.txt') == PROMPT_SHA, 'Original CRLF input identity changed')
        write(args.output / 'selection.json', selection)
        source = source_identity()
        write(args.output / 'candidate-source.json', source)
        build = args.output / 'build'
        with (args.output / 'build.log').open('w', encoding='utf-8') as log:
            subprocess.run(['pwsh', '-NoProfile', '-File', str(HERE / 'build_candidate.ps1'),
                            '-CudaRoot', str(args.cuda_root), '-BuildRoot', str(build), '-Python', sys.executable],
                           stdout=log, stderr=log, check=True, timeout=1200)
        write(args.output / 'candidate-build.json', audit_build(build))
        plans = {}
        for label, prompt in [('original44', HERE / 'prompt-44.txt'), ('short', HERE / 'prompt-short.txt')]:
            case = launch_probe(args, build, args.output / 'selection.json', prompt, label)
            plans[label] = demand_plan(selection, case)
            write(args.output / (label + '-demand.json'), plans[label])
        require(len(plans['original44']['prompt_tokens']) == 44, 'Original token count changed')
        require(1 < len(plans['short']['prompt_tokens']) < 44, 'Short prompt token scope')
        require(set(plans['short']['prompt_tokens']) - set(plans['original44']['prompt_tokens']), 'Short input adds no embedding rows')
        require(source == source_identity(), 'Candidate source identity changed during inventory')
        total = selection['logical_weight_layout_bytes'] + selection['kv_reserved_bytes']
        require(total <= selection['media_geometry']['capacity_bytes'], 'Complete logical layout does not fit')
        write(args.output / 'stage1-acceptance.json', {
            'status': 'pass', 'stage': 1, 'selected_names': 14, 'unselected_native_names': selection['unselected_count'],
            'model_sha256': MODEL_SHA, 'inputs': {label: {'prompt_tokens': p['prompt_tokens'], 'generated_tokens': p['generated_tokens'],
                'consumer_count': p['consumer_count'], 'model_demand_bytes': p['model_demand_bytes'],
                'embedding_rows': p['embedding_rows'], 'embedding_unique_rows': p['embedding_unique_rows']} for label, p in plans.items()},
            'resources': {'selected_bytes': selection['selected_bytes'], 'logical_layout_and_kv_bytes': total,
                'media_capacity_bytes': selection['media_geometry']['capacity_bytes'], 'media_free_bytes': selection['media_geometry']['capacity_bytes'] - total,
                'registered_range_limit': 16, 'selected_physical_ranges': 14, 'host_resident_fragments': 256,
                'gpu_kv_frames': 4, 'pinned_slots': 4, 'gpu_mirror_peak_bytes': 262144, 'gpu_validation_peak_bytes': 262144,
                'host_verification_buffers_bytes': 2 * 262144, 'runtime_cost_budget': 'pilot_required_before_full_matrix'},
            'observed_gates': 'not installed; required GET_ROWS/MUL/MUL_MAT mapped for stage2',
            'numerical_contract': {'same_arithmetic_logits': 'bitwise', 'paged_vs_native_max_abs': .2, 'paged_vs_native_rms': .025, 'tokens': 'exact'},
            'scope': 'Stage1 inventory and demand only; no route correctness or performance verdict'})
        print(json.dumps(read(args.output / 'stage1-acceptance.json'), ensure_ascii=False), flush=True)
    except BaseException:
        write(args.output / 'stage1-failure.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
