"""Read-only validation of the complete sealed V3 source snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def long_path(path):
    path = path.resolve()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        text = str(path)
        path = Path('\\\\?\\UNC\\' + text[2:] if text.startswith('\\\\') else '\\\\?\\' + text)
    return path


def verify(joint):
    joint = long_path(joint)
    manifest = json.loads((joint/'provenance/final-source-build-manifest.json').read_text(encoding='utf-8'))
    expected = [row for row in manifest if row['path'].replace('\\', '/').startswith('source/')]
    missing, changed = [], []
    for row in expected:
        p = joint/row['path']
        if not p.is_file():
            missing.append(row['path'])
            continue
        with p.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if p.stat().st_size != row['bytes'] or digest != row['sha256']:
            changed.append(row['path'])
    if len(expected) != 4276:
        raise ValueError('Unexpected historical V3 source inventory size')
    return {'status': 'pass' if not missing and not changed else 'fail',
            'source_records': len(expected), 'missing': missing, 'changed': changed,
            'scope': 'Historical sealed source bytes; fresh build identities are recorded separately.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify(args.joint_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if report['status'] != 'pass':
        raise SystemExit(1)
