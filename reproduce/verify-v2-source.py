"""Verify restored historical model/core/runtime sources without old binaries."""
import hashlib
import json
import os
from pathlib import Path


def extended(path):
    path = path.absolute()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        value = str(path)
        return Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)
    return path


ROOT = extended(Path(__file__).resolve().parents[1])


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    source = json.loads((ROOT / 'provenance/v2-r2-source-origin.json').read_text())
    added = json.loads((ROOT / 'provenance/v2-added-source-origin.json').read_text())
    checks = []
    for row in source['files']:
        path = ROOT / row['path']
        assert path.is_file() and sha(path) == row['sha256'], row['path']
        checks.append(row['path'])
    for row in added['files']:
        relative = row['path']
        # Only the historical dependency wiring changes; retain its exact old
        # bytes separately, while replacing hardcoded libraries with targets.
        if relative == 'experiments/readonly_f_compare/CMakeLists.txt':
            relative = 'provenance/historical/readonly_f_compare-CMakeLists.txt'
        # AGENTS is a workspace instruction document, not an implementation
        # input. Its historical bytes remain auditable while the root document
        # can describe the current experiment workflow. The source manifest and
        # its expected SHA stay unchanged; provenance records this relocation.
        elif relative == 'AGENTS.md':
            relative = 'provenance/historical/AGENTS.md'
        path = ROOT / relative
        assert path.is_file() and sha(path) == row['sha256'], relative
        checks.append(relative)
    assert sha(ROOT / 'reproduce/inputs/original-prompt.txt') == 'ce4fa40becfa4a191064fbda15721985a953ae231557ea6f6f8f7b669e0d1a74'
    print(json.dumps(dict(status='pass', model_source_records=len(source['files']),
                          comparison_and_R3_records=len(added['files']), checked_records=len(checks),
                          historical_instruction_document='provenance/historical/AGENTS.md',
                          old_binaries_required=False), indent=2))


if __name__ == '__main__':
    main()
