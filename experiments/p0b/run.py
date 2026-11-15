"""Extend the verified nine-case model matrix using its sealed build."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'output/matrix-20261104'
SOURCE = PRIOR / 'source'
BUILD = ROOT / 'build-matrix-20261104'
MODES = ('native', 'bypass', 'modeled')
sys.path.insert(0, str(SOURCE / 'tools'))
from integrated_build_identity import verify_build
from integrated_profile import derive
from run_llama_integrated import read, write, require
from setup_llama_baseline import sha


def check_outputs(folder, tokens):
    validation = read(folder / 'validation.json')
    require(validation['status'] == 'pass' and validation['tokens'] == tokens,
            f'Invalid complete-run validation: {folder}')
    for mode in MODES:
        result = read(folder / mode / 'summary.json')
        require(result['status'] == 'success' and len(result['generated_tokens']) == tokens,
                f'Incomplete output: {folder}/{mode}')
        require((folder / mode / 'logits.f32').stat().st_size == tokens * result['vocab_size'] * 4,
                f'Incomplete logits: {folder}/{mode}')
    return validation


def mapped_reference(case, manifest, old_plan):
    old = PRIOR / 'evidence' / case
    view = PRIOR / 'reference-view' / case
    original = Path(manifest['original'])
    require(sha(original / 'manifest.json') == manifest['original_manifest_sha256'],
            f'Original manifest changed: {case}')
    for record in manifest['verified_original_files']:
        file = Path(record['path'])
        require(sha(file) == record['sha256'], f'Historical reference changed: {file}')
        if file.name != 'profile.json':
            require(sha(view / file.relative_to(original)) == record['sha256'],
                    f'Mapped reference changed: {file}')
    original_profile = read(original / 'profile.json')
    for field, mapping in manifest['changed_profile_fields'].items():
        require(field in ('conversion', 'model_path') and original_profile[field] == mapping['old'],
                f'Invalid identity mapping: {case}/{field}')
        original_profile[field] = mapping['new']
    require(original_profile == read(view / 'profile.json') == read(old / 'profile.json'),
            f'Mapped profile differs: {case}')
    require(sha(old / 'profile.json') == old_plan['profile_sha256'] and
            sha(old / 'prompt.txt') == old_plan['prompt_sha256'] == sha(view / 'prompt.txt'),
            f'Frozen input changed: {case}')
    for mode in MODES:
        require(sha(old / mode / 'logits.f32') == sha(view / mode / 'logits.f32'),
                f'Same-mode historical logits mismatch: {case}/{mode}')
        for field in ('prompt_tokens', 'generated_tokens'):
            require(read(old / mode / 'summary.json')[field] == read(view / mode / 'summary.json')[field],
                    f'Same-mode historical tokens mismatch: {case}/{mode}')
    records = {Path(r['path']): r for r in read(old / 'manifest.json')['files']
               if Path(r['path']).is_relative_to(old)}
    actual = {file.resolve() for file in old.rglob('*') if file.is_file() and file.name != 'manifest.json'}
    require(actual == records.keys(), f'Case evidence membership changed: {case}')
    for file, record in records.items():
        require(file.stat().st_size == record['bytes'] and sha(file) == record['sha256'],
                f'Case evidence changed: {file}')
    return old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = verify_build(BUILD)
    require(identity == read(PRIOR / 'evidence/build-identity.json'), 'Prior build identity mismatch')
    write(out / 'build-identity.json', identity)
    mapped = {row['case']: row for row in read(PRIOR / 'reference-provenance.json')}
    prior_plan = {row['name']: row for row in read(PRIOR / 'evidence/predeclared.json')['cases']}
    cases = []
    for role in ('A', 'B', 'C'):
        for length in ('short', 'long'):
            for tokens in (32, 64):
                name = f'{role}-{length}-{tokens}'
                cases.append(dict(name=name, role=role, length=length, tokens=tokens,
                                  evidence='prior_sealed_run' if name in mapped else 'new_run'))
    write(out / 'predeclared.json', dict(cases=cases, build_identity_sha256=sha(out / 'build-identity.json'),
                                         historical_provenance_sha256=sha(PRIOR / 'reference-provenance.json')))
    results = {}
    for case in cases:
        if case['evidence'] != 'prior_sealed_run':
            continue
        name = case['name']
        old = mapped_reference(name, mapped[name], prior_plan[name])
        require(read(old / 'build-identity.json') == identity, f'Run build mismatch: {name}')
        result = check_outputs(old, case['tokens'])
        results[name] = dict(case, status='pass', folder=str(old), prompt_tokens=result['prompt_tokens'],
                             cross_page_rows=result['comparisons']['modeled']['kv']['cross_page_rows'])
        print(f'Verified {name}', flush=True)
    write(out / 'progress.json', results)
    for case in cases:
        if case['evidence'] != 'new_run':
            continue
        name = case['name']
        base = read(PRIOR / 'evidence/A-short-32/profile.json')
        ubatch = 64 if case['length'] == 'long' else 256
        profile = derive(Path(base['model_path']), Path(base['conversion']), ubatch=ubatch)
        require(profile['numerical'] == base['numerical'], 'Original numerical contract changed')
        profile_path = out / f'{name}.profile.json'
        write(profile_path, profile)
        prompt = (PRIOR / 'evidence/A-short-prompt.txt' if case['length'] == 'short'
                  else PRIOR / 'evidence/B-long-prompt.txt')
        folder = out / name
        command = [sys.executable, str(SOURCE / 'tools/run_llama_integrated.py'), '--build-dir',
                   str(BUILD), '--profile', str(profile_path), '--prompt', str(prompt),
                   '--tokens', str(case['tokens']), '--output', str(folder)]
        write(out / f'{name}.input.json', dict(command=command, profile_sha256=sha(profile_path),
                                                prompt_sha256=sha(prompt), numerical=profile['numerical']))
        print(f'Running {name}', flush=True)
        with (out / f'{name}.runner.log').open('w', encoding='utf-8') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = dict(case, folder=str(folder), returncode=run.returncode, status='fail')
        if run.returncode == 0:
            checked = check_outputs(folder, case['tokens'])
            require(case['length'] != 'long' or checked['prompt_tokens'] > ubatch, 'Long input must cross ubatch')
            result.update(status='pass', prompt_tokens=checked['prompt_tokens'],
                          cross_page_rows=checked['comparisons']['modeled']['kv']['cross_page_rows'])
        results[name] = result
        write(out / 'progress.json', results)
        print(f"{name}: {result['status']}", flush=True)
    require(verify_build(BUILD) == identity, 'Build changed during matrix')
    write(out / 'matrix-result.json', dict(status='pass' if all(r['status'] == 'pass' for r in results.values()) else 'fail',
                                          cells=[results[c['name']] for c in cases]))


if __name__ == '__main__':
    main()
