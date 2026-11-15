"""Verify the recovered V1 source against the historical second-F manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if os.name == 'nt' and not str(ROOT).startswith('\\\\?\\'):
    ROOT = Path('\\\\?\\'+str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    historical = ROOT/'provenance/historical-f-build-identity.json'
    identity = json.loads(historical.read_text(encoding='utf-8'))
    missing, changed = [], []
    for name, expected in identity['sources'].items():
        path = ROOT/Path(name.replace('\\', '/'))
        if not path.is_file():
            missing.append(name)
            continue
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            changed.append(name)
    report = {'status': 'pass' if not missing and not changed else 'fail',
              'scope': 'Historical second-F source bytes only; newly rebuilt binaries have separate identities',
              'historical_manifest_sha256': hashlib.sha256(historical.read_bytes()).hexdigest(),
              'source_records': len(identity['sources']), 'missing': missing, 'changed': changed}
    if args.output:
        if args.output.exists():
            parser.error('Use a fresh report path')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if missing or changed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
