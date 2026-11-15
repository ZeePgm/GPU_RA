"""Restore the byte-exact V3 source snapshot using only Python's standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent / "source-bundle"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    root = args.destination.resolve()
    if os.name == "nt" and not str(root).startswith("\\\\?\\"):
        root = Path("\\\\?\\" + str(root))
    files = {row["path"]: row for row in manifest["files"]}
    if len(files) != len(manifest["files"]):
        raise ValueError("Duplicate source path")
    with tempfile.TemporaryFile() as archive:
        for part in manifest["parts"]:
            name = part["path"]
            if Path(name).name != name:
                raise ValueError("Invalid archive part path")
            data = (bundle / name).read_bytes()
            if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
                raise ValueError("Archive part mismatch: " + name)
            archive.write(data)
        archive.seek(0)
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            if len(names) != len(files) or set(names) != set(files):
                raise ValueError("Archive membership mismatch")
            for rel, row in files.items():
                path = PurePosixPath(rel)
                if path.is_absolute() or ".." in path.parts or ":" in rel:
                    raise ValueError("Unsafe source path: " + rel)
                dest = (root / rel).resolve()
                if not dest.is_relative_to(root):
                    raise ValueError("Source path escapes destination")
                with tempfile.TemporaryFile() as raw:
                    with zipped.open(rel) as source:
                        shutil.copyfileobj(source, raw, 1024 * 1024)
                    count = raw.tell()
                    raw.seek(0)
                    digest = hashlib.file_digest(raw, "sha256").hexdigest()
                    if count != row["bytes"] or digest != row["sha256"]:
                        raise ValueError("Source content mismatch: " + rel)
                    if not args.verify_only:
                        if dest.exists():
                            with dest.open("rb") as current:
                                if hashlib.file_digest(current, "sha256").hexdigest() != digest:
                                    raise ValueError("Refusing to overwrite different source: " + rel)
                        else:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            raw.seek(0)
                            with dest.open("xb") as output:
                                shutil.copyfileobj(raw, output, 1024 * 1024)
    print(json.dumps({"status": "pass", "verified_source_files": len(files),
                      "restored": not args.verify_only}))


if __name__ == "__main__":
    main()
