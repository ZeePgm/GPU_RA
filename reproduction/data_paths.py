"""Resolve local experiment payloads independently of tracked metadata (stdlib)."""
import json
import os
from pathlib import Path, PurePosixPath


def repository_root():
    """Work in Git worktrees and exported source trees, including joint helpers."""
    for parent in Path(__file__).resolve().parents:
        if (parent / 'VERSION_MAP.md').is_file() or (parent / '.git').exists():
            return parent
    raise RuntimeError('Cannot locate GPU_RA repository root for data configuration')


def require_external_directory(path):
    """Resolve an output/data root and reject the current source checkout."""
    root = repository_root()
    directory = Path(path).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    if directory.is_relative_to(root):
        raise ValueError('Experiment data directory must be outside this checkout: '
                         + str(directory))
    return directory


def _relative_child(directory, relative):
    relative = str(relative).replace('\\', '/')
    name = PurePosixPath(relative)
    if (not name.parts or name.is_absolute() or '..' in name.parts
            or ':' in relative or '\x00' in relative):
        raise ValueError('Unsafe experiment payload path: ' + repr(relative))
    path = directory.joinpath(*name.parts).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Experiment payload leaves its data directory: ' + relative)
    return path


def _configured_root(data_root):
    root = repository_root()
    if data_root is None:
        data_root = os.environ.get('GPU_RA_DATA_ROOT')
    if data_root is None:
        config = root / 'data.local.json'
        if config.is_file():
            try:
                content = json.loads(config.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError) as error:
                raise ValueError('Cannot read local data configuration: ' + str(config)) from error
            if not isinstance(content, dict):
                raise ValueError('data.local.json must contain an object with data_root')
            data_root = content.get('data_root')
            if not isinstance(data_root, str) or not data_root.strip():
                raise ValueError('data.local.json requires a nonempty data_root string')
    if data_root is None:
        raise FileNotFoundError(
            'External experiment data is not configured. Pass --data-root, set '
            'GPU_RA_DATA_ROOT, or copy data.example.json to data.local.json. '
            'Experiment payloads are not read from this checkout.')
    if not isinstance(data_root, (str, os.PathLike)) or not str(data_root).strip():
        raise ValueError('External experiment data root must be a nonempty path')
    return require_external_directory(data_root)


def resolve_payload(metadata_dir, relative, data_root=None):
    """Map tracked bundle metadata to an external payload at the same relative path.

    An explicitly supplied bundle outside this checkout is self-contained and uses
    its own payload files. A tracked bundle never falls back to checkout payloads.
    Configuration precedence: argument, GPU_RA_DATA_ROOT, root data.local.json.
    Relative configured roots are interpreted against the repository root.
    """
    root = repository_root()
    metadata = Path(metadata_dir).resolve()
    if metadata.is_relative_to(root):
        external = _configured_root(data_root)
        directory = external if metadata == root else _relative_child(
            external, metadata.relative_to(root).as_posix())
        path = _relative_child(directory, relative)
        if path.is_relative_to(root):
            raise ValueError('External payload resolves into this checkout: ' + str(path))
    else:
        path = _relative_child(metadata, relative)
    if not path.is_file():
        raise FileNotFoundError('External experiment payload is missing: ' + str(path)
                                + '. Obtain the local data bundle matching its tracked manifest.')
    return path
