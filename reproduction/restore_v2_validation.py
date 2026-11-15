"""Verify or restore the external V2 fresh F1 ZIP using its tracked identity."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from data_paths import repository_root, require_external_directory, resolve_payload

BLOCK = 1024 * 1024
ARCHIVE_PATH = 'provenance/clean-validation/raw-f1-entry-validation.zip'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def member_path(root, name):
    """Reject traversal, drive/UNC names and Windows-ambiguous ZIP members."""
    relative = str(name).replace('\\', '/')
    path = PurePosixPath(relative)
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{p}{i}' for p in ('COM', 'LPT') for i in range(1, 10)}
    if (not path.parts or path.is_absolute() or ':' in relative or '\x00' in relative
            or any(p in ('', '.', '..') or p.endswith((' ', '.'))
                   or p.upper().split('.')[0] in reserved for p in relative.split('/'))):
        raise ValueError('Unsafe ZIP member path: ' + repr(name))
    result = root.joinpath(*path.parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('ZIP member leaves its destination: ' + name)
    return result


def verify_or_restore(archive, expected, destination=None):
    archive = Path(archive)
    if archive.stat().st_size != expected['bytes'] or sha(archive) != expected['sha256']:
        raise ValueError('V2 validation ZIP does not match tracked size/SHA-256')
    destination = require_external_directory(destination) if destination is not None else None
    # Verification uses an imaginary external destination solely for path validation.
    path_root = destination or archive.parent
    files = []
    seen = set()
    total = 0
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) != expected['files']:
            raise ValueError('Unexpected V2 validation ZIP member count')
        for entry in infos:
            target = member_path(path_root, entry.filename)
            key = entry.filename.replace('\\', '/').casefold()
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir() or mode not in (0, stat.S_IFREG) or key in seen:
                raise ValueError('Duplicate, directory or special ZIP member: ' + entry.filename)
            seen.add(key)
            files.append((entry, target))
        # Read every stream to EOF (including ZIP CRC) and check all conflicts before writing.
        for entry, target in files:
            with bundle.open(entry) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            total += entry.file_size
            if destination is not None and (target.exists() or target.is_symlink()):
                if not target.is_file() or target.stat().st_size != entry.file_size or sha(target) != digest:
                    raise ValueError('Refusing to overwrite differing file: ' + str(target))
        restored = 0
        if destination is not None:
            for entry, target in files:
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, BLOCK)
                restored += 1
    return {'status': 'pass', 'archive': ARCHIVE_PATH, 'verified_paths': len(files),
            'uncompressed_bytes': total, 'restored': destination is not None,
            'newly_written_paths': restored if destination is not None else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, help='Overrides GPU_RA_DATA_ROOT and data.local.json')
    parser.add_argument('--dest', type=Path, help='New/restored data directory outside this checkout')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and args.dest is None:
        parser.error('Use --verify-only or specify --dest')
    root = repository_root()
    metadata_path = root / 'provenance/clean-validation/validation.json'
    expected = json.loads(metadata_path.read_text(encoding='utf-8-sig'))['raw_archive']
    if expected['path'] != ARCHIVE_PATH:
        raise ValueError('Unexpected tracked V2 validation archive path')
    archive = resolve_payload(metadata_path.parent, Path(ARCHIVE_PATH).name, args.data_root)
    report = verify_or_restore(archive, expected, None if args.verify_only else args.dest)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
