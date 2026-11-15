"""Run the frozen R3 complete-FFN candidate with explicit portable inputs.

Defaults reproduce the historical 44-input/8-output F1/F2 contract. A shorter
--outputs value is an entry-point smoke test, recorded separately from that run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = '74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2'
PROMPT_SHA256 = 'ce4fa40becfa4a191064fbda15721985a953ae231557ea6f6f8f7b669e0d1a74'


def sha(path):
    path = Path(path).absolute()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        value = str(path)
        path = Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def union_ns(intervals):
    total = 0
    left = right = None
    for a, b in sorted(intervals):
        if left is None:
            left, right = a, b
        elif a <= right:
            right = max(right, b)
        else:
            total += right - left
            left, right = a, b
    return total + (right - left if left is not None else 0)


def clock_config(capture, layers, outputs, ini):
    events = [json.loads(line) for line in (capture / 'gpu.jsonl').read_text().splitlines()]
    marks = [json.loads(line) for line in (capture / 'batches.jsonl').read_text().splitlines()]
    actions = []
    regions = []
    for layer in layers:
        for index, name in enumerate(('gate', 'up', 'down')):
            regions.append(dict(tensor=f'blk.{layer}.ffn_{name}.weight',
                                global_base=(layer * 3 + index) * 6291456,
                                valid_bytes=6291456))
    for step in range(outputs):
        gpu = sorted((v for v in events if v['step'] == step), key=lambda v: v['start'])
        batches = [v for v in marks if v['step'] == step]
        cursor = 0
        normal = []
        previous_end = 0
        for pos, event in enumerate(gpu):
            earlier_end = previous_end
            previous_end = max(previous_end, event['end'])
            assert event['end'] > event['start']
            if 'read_plan' in event.get('name', ''):
                if normal:
                    actions.append(dict(kind='compute', ps=1000 * union_ns(normal)))
                    normal.clear()
                mark = batches[cursor]
                cursor += 1
                assert mark['at'] <= event['start'], 'read kernel precedes its batch marker'
                assert cursor == len(batches) or event['end'] <= batches[cursor]['at'], 'read kernel extends beyond the next batch marker'
                assert earlier_end <= event['start'], 'read kernel overlaps prior GPU work'
                assert pos + 1 == len(gpu) or gpu[pos + 1]['start'] >= event['end'], 'read kernel overlaps next GPU work'
                actions.extend((dict(kind='read', tensor=mark['tensor'], offset=mark['offset'], bytes=4096),
                                dict(kind='compute', ps=1000 * (event['end'] - event['start']))))
            else:
                normal.append((event['start'], event['end']))
        if normal:
            actions.append(dict(kind='compute', ps=1000 * union_ns(normal)))
        assert cursor == len(batches) == len(layers) * 18 * 1024**2 // 4096
        for region in regions:
            offsets = [v['offset'] for v in batches if v['tensor'] == region['tensor']]
            assert offsets == list(range(0, region['valid_bytes'], 4096)), 'per-step full matrix coverage'
        actions.append(dict(kind='token', step=step))
    assert sum(v['ps'] for v in actions if v['kind'] == 'compute') == 1000 * union_ns((v['start'], v['end']) for v in events), 'GPU time conservation'
    return dict(ini=str(ini), regions=regions, chunk=512, actions=actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--cuda-root', required=True, type=Path)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build-v2')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scale', choices=('F1', 'F2', 'both'), default='both')
    parser.add_argument('--mode', choices=('native', 'capture', 'live', 'all'), default='all')
    parser.add_argument('--outputs', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--clock', action='store_true', help='Replay this newly collected GPU capture through the source-built common clock')
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    cuda = args.cuda_root.resolve(strict=True)
    build = args.build_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; use a fresh directory')
    if sha(model) != MODEL_SHA256:
        parser.error('Model SHA-256 differs from frozen Qwen3-0.6B F16 input')
    prompt = ROOT / 'reproduce/inputs/original-prompt.txt'
    ini = ROOT / 'reproduce/inputs/H0-512MiB.ini'
    assert sha(prompt) == PROMPT_SHA256 and len(prompt.read_bytes()) == 249
    historical = load(ROOT / 'reproduce/inputs/historical-native.json')
    exe = build / 'bin/f_candidate.exe'
    if not exe.is_file():
        parser.error(f'Build first: {exe} is missing')
    if args.clock and args.mode not in ('all', 'capture'):
        parser.error('--clock requires --mode capture or all')
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith(('HBFLAB_', 'GGML_')):
            del environment[key]
    environment.update(HBFLAB_M4_PAGED='0', GGML_CUDA_DISABLE_FUSION='1',
                       GGML_CUDA_CUBLAS_COMPUTE_TYPE='f32', GGML_BACKEND_PATH=str(build / 'bin'))
    runtime_directories = [build / 'bin'] + [cuda / sub for sub in (
        'bin', 'bin/x64', 'lib', 'lib/x64', 'lib64', 'extras/CUPTI/lib64', 'extras/CUPTI/lib/x64')]
    environment['PATH'] = os.pathsep.join([str(p) for p in runtime_directories if p.is_dir()] + [environment.get('PATH', '')])
    output.mkdir(parents=True)
    summary = dict(status='running', model_sha256=MODEL_SHA256, prompt_sha256=PROMPT_SHA256,
                   candidate_sha256=sha(exe), outputs=args.outputs,
                   scope='historical 44/8 contract' if args.outputs == 8 else 'short entry-point smoke; not a new historical 44/8 sample',
                   kv='native F16; paging and later KV optimizations disabled', runs=[])
    binary_paths = sorted(set((build / 'bin').glob('*.dll')) | {exe})
    if args.clock:
        binary_paths.append(build / 'bin/f_clock.exe')
    summary['build_binary_sha256'] = {p.name: sha(p) for p in binary_paths}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))

    def invoke(binary, config, name, result_file=False):
        config_path = output / (name + '.config.json')
        config_path.write_text(json.dumps(config, indent=2))
        result = output / (name + '.json' if result_file else name)
        started = time.perf_counter()
        with (output / (name + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run([str(binary), str(config_path), str(result)], env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        return result, time.perf_counter() - started

    modes = ('native', 'capture', 'live') if args.mode == 'all' else (args.mode,)
    scales = ('F1', 'F2') if args.scale == 'both' else (args.scale,)
    for scale in scales:
        layers = [10] if scale == 'F1' else [0, 10, 20, 27]
        reference_logits = None
        reference_tokens = None
        for mode in modes:
            config = dict(model=str(model), prompt=str(prompt), layers=layers, outputs=args.outputs,
                          route='native' if mode == 'native' else 'cuda', merge=True,
                          bypass=mode == 'capture', verify=mode == 'live')
            result, wall = invoke(exe, config, f'{scale}-{mode}')
            values = load(result / 'result.json')
            assert values['status'] == 'pass' and values['drained']
            assert values['prompt_tokens'] == 44 and len(values['tokens']) == args.outputs
            input_record = load(result / 'input.json')
            assert input_record['prompt_bytes'] == 249 and len(input_record['tokens']) == 44
            assert input_record['tokens'] == historical['prompt_ids'], 'historical input token IDs differ'
            assert values['tokens'] == historical['raw_tokens'][:args.outputs], 'historical output token IDs differ'
            logits = sha(result / 'logits.f32')
            if args.outputs == 8:
                assert logits == historical['logits_sha256'], 'historical complete logits differ'
            if reference_logits is None:
                reference_logits, reference_tokens = logits, values['tokens']
            assert logits == reference_logits and values['tokens'] == reference_tokens, 'complete logits/token mismatch'
            expected_bytes = len(layers) * 18 * 1024**2 * args.outputs
            if mode != 'native':
                assert values['covered_bytes'] == expected_bytes
                assert values['requests'] == (expected_bytes // 512 if mode == 'live' else 0)
                assert values['dependency_checks'] == (expected_bytes // 16 if mode == 'live' else 0)
            row = dict(scale=scale, mode=mode, process_wall_s=wall, logits_sha256=logits,
                       requests=values['requests'], covered_bytes=values['covered_bytes'])
            if args.clock and mode == 'capture':
                clock, clock_wall = invoke(build / 'bin/f_clock.exe', clock_config(result, layers, args.outputs, ini),
                                           f'{scale}-clock', result_file=True)
                clock_values = load(clock)
                assert clock_values['status'] == 'pass' and clock_values['bytes'] == expected_bytes
                assert len(clock_values['tokens']) == args.outputs
                row.update(clock=clock.name, clock_wall_s=clock_wall)
            summary['runs'].append(row)
            (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    summary['status'] = 'pass'
    assert all(sha(p) == summary['build_binary_sha256'][p.name] for p in binary_paths), 'build binary drift'
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
