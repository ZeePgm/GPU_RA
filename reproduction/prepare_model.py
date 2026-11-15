"""Acquire pinned HF inputs and convert the historical Qwen3-0.6B F16 model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True,
                   help='Directory named Qwen3-0.6B containing the pinned HF files')
    p.add_argument('--output', type=Path, help='Fresh .gguf path; omit to verify HF input files only')
    p.add_argument('--download', action='store_true', help='Download missing pinned HF files (about 1.5 GB)')
    a = p.parse_args()
    source = json.loads((ROOT/'provenance/Qwen3-0.6B-conversion.json').read_text(encoding='utf-8'))
    if a.input_dir.name != 'Qwen3-0.6B':
        p.error('Use an input directory named Qwen3-0.6B to preserve converter naming')
    if a.output and a.output.exists():
        p.error('Use a fresh GGUF output path')
    for row in source['source']['files']:
        dest = a.input_dir/row['name']
        if not dest.is_file():
            if not a.download:
                p.error('Missing pinned input: '+str(dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name+'.download')
            if partial.exists():
                p.error('Previous partial download exists: '+str(partial))
            url = ('https://huggingface.co/'+source['source']['repo']+'/resolve/'+
                   source['source']['revision']+'/'+row['name'])
            urllib.request.urlretrieve(url, partial)
            if partial.stat().st_size != row['bytes'] or digest(partial) != row['sha256']:
                raise ValueError('Downloaded input identity mismatch: '+str(partial))
            partial.rename(dest)
        if dest.stat().st_size != row['bytes'] or digest(dest) != row['sha256']:
            raise ValueError('Pinned HF input identity mismatch: '+str(dest))
    print('PASS', len(source['source']['files']), 'pinned HF input files')
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        conversion_env = dict(os.environ)
        conversion_env.pop('NO_LOCAL_GGUF', None)
        subprocess.run([sys.executable, str(ROOT/'third_party/llama.cpp-integrated/convert_hf_to_gguf.py'),
                        str(a.input_dir.resolve()), '--outtype', 'f16', '--outfile', str(a.output.resolve())],
                       cwd=ROOT, env=conversion_env, check=True)
        if a.output.stat().st_size != source['gguf_bytes'] or digest(a.output) != source['gguf_sha256']:
            raise ValueError('Converted GGUF differs from historical input; retained for diagnosis')
        print('PASS historical GGUF byte identity:', source['gguf_sha256'])


if __name__ == '__main__':
    main()
