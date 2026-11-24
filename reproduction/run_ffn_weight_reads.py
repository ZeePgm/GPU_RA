"""Run FFN weight reads on the GPU with consumer-side HBF timing."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = '74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2'
PROMPT_SHA256 = 'ce4fa40becfa4a191064fbda15721985a953ae231557ea6f6f8f7b669e0d1a74'
COVERAGE_STAGES = {
    'single-layer': ['F1'],
    'four-layers': ['F1', 'F2'],
    'all-layers': ['F1', 'F3'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--build-root', type=Path, default=ROOT/'build-reproduction')
    p.add_argument('--cuda-root', type=Path)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--coverage', choices=COVERAGE_STAGES,
                           help='Choose the weight coverage. Each run starts with the single-layer reference.')
    selection.add_argument('--stages', nargs='+', choices=['F1', 'F2', 'F3'],
                           help='Select ordered historical stages starting with F1. Default: F1 F2 F3.')
    p.add_argument('--tokens', type=int, choices=range(1, 9), default=8)
    p.add_argument('--timeout', type=float, default=21600)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    stages = COVERAGE_STAGES[a.coverage] if a.coverage else a.stages or ['F1', 'F2', 'F3']
    if stages != sorted(set(stages)) or stages[0] != 'F1':
        p.error('Stages must be unique and ordered, starting with F1')
    if not a.model.is_file():
        p.error('Model must be an existing Qwen3-0.6B F16 GGUF')
    with a.model.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != MODEL_SHA256:
        p.error('Model SHA-256 differs from the historical V1 input')
    prompt = ROOT/'experiments/gpu_ffn/prompt.txt'
    if hashlib.sha256(prompt.read_bytes()).hexdigest() != PROMPT_SHA256:
        p.error('Original 249-byte CRLF prompt identity changed')
    if a.output.exists():
        p.error('Use a fresh output directory')
    if a.timeout <= 0:
        p.error('Timeout must be positive')
    output = a.output.resolve()
    for stage in stages:
        cmd = [sys.executable, str(ROOT/'experiments/gpu_ffn/run.py'),
               '--stage', stage, '--model', str(a.model.resolve()),
               '--output', str(output/stage), '--build-dir', str(a.build_root.resolve()/'ffn'),
               '--tokens', str(a.tokens), '--timeout', str(a.timeout)]
        if stage != 'F1':
            cmd += ['--reference', str(output/'F1')]
        if a.cuda_root:
            cmd += ['--cuda-root', str(a.cuda_root.resolve())]
        if a.prepare_only:
            cmd += ['--prepare-only']
        subprocess.run(cmd, cwd=ROOT, check=True)
        if not a.prepare_only:
            subprocess.run([sys.executable, str(ROOT/'experiments/gpu_ffn/validate.py'),
                            '--output', str(output/stage), '--negative-checks'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
