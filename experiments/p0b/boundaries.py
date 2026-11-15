"""Record boundary outcomes against the same sealed runner as the P0-B matrix."""
import argparse
import copy
from pathlib import Path
import subprocess

from run import BUILD, PRIOR, MODES, read, require, sha, verify_build, write
from run_llama_integrated import environment
from platform_environment import executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    identity = verify_build(BUILD)
    require(identity == read(batch / 'build-identity.json'), 'Boundary build differs from matrix')
    out = batch / 'boundaries'
    out.mkdir(exist_ok=False)
    exe = executable(BUILD / 'bin', 'realgpu_llama')
    env = environment(BUILD)
    base_path = PRIOR / 'evidence/A-short-32/profile.json'
    base = read(base_path)
    short = PRIOR / 'evidence/A-short-32/prompt.txt'
    eos = out / 'eos-prompt.txt'
    eos.write_bytes(short.read_bytes().replace(
        b'Explain how a computer uses memory in about one hundred words.',
        b'Reply with exactly one word: Yes.'))
    cases = []
    for mode in MODES:
        for count in (0, 1):
            cases.append((f'{mode}-output-{count}', mode, count, base_path, short,
                          'token_count must be 2..448', 'synthetic_runner_input', 'unsupported'))
    frame = copy.deepcopy(base)
    frame['frames'] = frame['minimum_frames'] - 1
    frame_path = out / 'insufficient-frames.json'
    write(frame_path, frame)
    cases.append(('frame-budget', 'modeled', 2, frame_path, short,
                  'M6 insufficient page/frame budget', 'synthetic_runner_input', 'pass'))
    for resource in ('host', 'gpu'):
        profile = copy.deepcopy(base)
        profile['budgets'][f'{resource}_required_bytes'] = 2**60
        profile_path = out / f'insufficient-{resource}.json'
        write(profile_path, profile)
        diagnostic = ('M6 host memory budget unavailable' if resource == 'host'
                      else 'M6 GPU weight/scratch budget unavailable')
        cases.append((f'{resource}-budget', 'modeled', 2, profile_path, short, diagnostic,
                      'synthetic_runner_input', 'pass'))
    for mode in MODES:
        cases.append((f'{mode}-natural-eos', mode, 32, base_path, eos,
                      'Early EOG', 'real_model_generation', 'unsupported'))
    write(out / 'predeclared.json', [
        dict(case=name, mode=mode, requested_tokens=count, profile_sha256=sha(profile),
             prompt_sha256=sha(prompt), expected_diagnostic=diagnostic, evidence=evidence)
        for name, mode, count, profile, prompt, diagnostic, evidence, _ in cases])
    results = []
    for name, mode, count, profile, prompt, diagnostic, evidence, intended in cases:
        target = out / name
        argv = [str(exe), base['model_path'], str(prompt), str(target), str(count),
                '1' if mode == 'modeled' else '0', mode, str(profile)]
        run = subprocess.run(argv, env=env, cwd=BUILD / 'bin', capture_output=True, timeout=3600)
        (out / f'{name}.stdout.log').write_bytes(run.stdout)
        (out / f'{name}.stderr.log').write_bytes(run.stderr)
        success = (target / 'summary.json').is_file()
        matched = run.returncode == 1 and diagnostic.encode() in run.stderr and not success
        results.append(dict(case=name, mode=mode, status=intended if matched else 'fail',
                            evidence=evidence, actual_trigger=matched, returncode=run.returncode,
                            successful_output=success, expected_diagnostic=diagnostic,
                            in_process_release_audit='unavailable' if evidence == 'real_model_generation'
                            else 'preflight rejection before model allocation', argv=argv))
        write(out / 'results.json', results)
        print(name, results[-1]['status'], flush=True)
    for name in ('cooperative-model-cancel', 'cooperative-model-timeout'):
        results.append(dict(case=name, status='unsupported', evidence='runner_contract_inspection',
                            actual_trigger=False, reason='No cooperative termination input or release record'))
    require(verify_build(BUILD) == identity, 'Build changed during boundaries')
    write(out / 'results.json', results)


if __name__ == '__main__':
    main()
