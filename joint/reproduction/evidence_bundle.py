"""Lossless, content-deduplicated evidence bundles; Python standard library only."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reproduction'))
from data_paths import require_external_directory, resolve_payload

BLOCK = 1024 * 1024


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def safe(root, relative):
    p = PurePosixPath(relative.replace('\\', '/'))
    if p.is_absolute() or '..' in p.parts or not p.parts or ':' in str(p):
        raise ValueError('Unsafe evidence path: ' + relative)
    dest = root.joinpath(*p.parts).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError('Path leaves destination: ' + relative)
    return dest


class ShardWriter:
    def __init__(self, directory, limit):
        self.directory, self.limit = directory, limit
        self.stream = None
        self.size = 0
        self.parts = []

    def write(self, data):
        total = len(data)
        while data:
            if self.stream is None or self.size == self.limit:
                if self.stream:
                    self.stream.close()
                self.path = self.directory / ('part-%04d.gzpart' % len(self.parts))
                self.stream = self.path.open('wb')
                self.parts.append(self.path)
                self.size = 0
            n = min(len(data), self.limit - self.size)
            self.stream.write(data[:n])
            self.size += n
            data = data[n:]
        return total

    def flush(self):
        if self.stream:
            self.stream.flush()

    def close(self):
        if self.stream:
            self.stream.close()


class ShardReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open('rb')
            n = self.current.readinto(buffer)
            if n:
                return n
            self.current.close()
            self.current = None

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def pack(args):
    root, out = args.source_root.resolve(), args.bundle.resolve()
    require_external_directory(out)
    if out.exists():
        raise ValueError('Use a fresh bundle directory')
    files, omissions = [], []
    for rel in args.paths:
        p = safe(root, rel)
        if not p.exists():
            raise FileNotFoundError(p)
        for f in sorted(p.rglob('*')) if p.is_dir() else [p]:
            if not f.is_file():
                continue
            r = f.relative_to(root).as_posix()
            if '__pycache__' in f.parts or f.suffix.lower() == '.pyc':
                omissions.append({'path': r, 'reason': 'regenerable Python bytecode'})
                continue
            if f.is_symlink() or f.suffix.lower() == '.gguf':
                raise ValueError('Unexpected symlink or model GGUF: ' + r)
            files.append({'path': r, 'bytes': f.stat().st_size, 'sha256': sha(f)})
    files = sorted({f['path']: f for f in files}.values(), key=lambda f: f['path'])
    blobs = {}
    for row in files:
        blobs.setdefault(row['sha256'], row)
    out.mkdir(parents=True)
    writer = ShardWriter(out, args.shard_mib * 1024**2)
    try:
        with gzip.GzipFile(fileobj=writer, mode='wb', compresslevel=6, mtime=0, filename='') as gz:
            with tarfile.open(fileobj=gz, mode='w|') as tar:
                for digest, row in sorted(blobs.items()):
                    entry = tarfile.TarInfo('blobs/' + digest)
                    entry.size, entry.mtime, entry.mode = row['bytes'], 0, 0o644
                    with safe(root, row['path']).open('rb') as stream:
                        tar.addfile(entry, stream)
    finally:
        writer.close()
    manifest = {'schema_version': 1, 'name': args.name,
                'format': 'SHA-256 deduplicated tar stream, gzip, byte-split parts',
                'lossless': True, 'files': files, 'omissions': omissions,
                'parts': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in writer.parts],
                'file_count': len(files), 'logical_bytes': sum(f['bytes'] for f in files),
                'unique_objects': len(blobs), 'unique_bytes': sum(f['bytes'] for f in blobs.values())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('name','file_count','logical_bytes','unique_objects','unique_bytes')}, indent=2))


def restore(args):
    bundle = args.bundle.resolve()
    if args.dest is not None:
        args.dest = require_external_directory(args.dest)
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    targets = {}
    for row in manifest['files']:
        targets.setdefault(row['sha256'], []).append(row)
        if args.dest:
            safe(args.dest.resolve(), row['path'])
    parts = []
    for row in manifest['parts']:
        p = resolve_payload(bundle, row['path'], args.data_root)
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Corrupt compressed shard: ' + row['path'])
        parts.append(p)
    seen = set()
    with io.BufferedReader(ShardReader(parts)) as combined:
        with gzip.GzipFile(fileobj=combined, mode='rb') as gz:
            with tarfile.open(fileobj=gz, mode='r|') as tar:
                for member in tar:
                    digest = member.name.removeprefix('blobs/')
                    if member.name != 'blobs/' + digest or digest not in targets or not member.isfile() or digest in seen:
                        raise ValueError('Unexpected/duplicate archive member: ' + member.name)
                    seen.add(digest)
                    with tempfile.TemporaryFile() as temp:
                        source = tar.extractfile(member)
                        got = hashlib.sha256()
                        count = 0
                        while block := source.read(BLOCK):
                            temp.write(block)
                            got.update(block)
                            count += len(block)
                        if got.hexdigest() != digest or any(count != r['bytes'] for r in targets[digest]):
                            raise ValueError('Corrupt raw evidence object: ' + digest)
                        if args.verify_only:
                            continue
                        for row in targets[digest]:
                            dest = safe(args.dest.resolve(), row['path'])
                            if dest.exists():
                                if dest.is_file() and dest.stat().st_size == count and sha(dest) == digest:
                                    continue
                                raise ValueError('Refusing to overwrite differing file: ' + str(dest))
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            temp.seek(0)
                            with dest.open('xb') as out:
                                shutil.copyfileobj(temp, out, BLOCK)
            # Force the gzip reader to EOF so the CRC/trailer is also checked.
            while gz.read(BLOCK):
                pass
    if seen != set(targets):
        raise ValueError('Missing evidence objects')
    print(json.dumps({'status':'pass', 'bundle':manifest['name'], 'verified_unique_objects':len(seen),
                      'verified_paths':len(manifest['files']), 'restored':not args.verify_only}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--shard-mib', type=int, default=32)
    p.add_argument('paths', nargs='+')
    p = commands.add_parser('restore')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--data-root', type=Path, help='External payload root; otherwise use GPU_RA_DATA_ROOT or data.local.json')
    p.add_argument('--dest', type=Path)
    p.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'restore' and not args.verify_only and args.dest is None:
        parser.error('restore requires --dest or --verify-only')
    if args.command == 'pack' and not 1 <= args.shard_mib <= 48:
        parser.error('shards must be 1 through 48 MiB')
    (pack if args.command == 'pack' else restore)(args)
