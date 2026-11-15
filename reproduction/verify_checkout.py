"""Verify a GPU_RA checkout against its byte-exact delivery manifest (stdlib)."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, manifest):
    root = root.resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    seen = set()
    total = 0
    for row in data["files"]:
        rel = row["path"]
        path = (root / rel).resolve()
        if rel in seen or not path.is_relative_to(root):
            raise ValueError(f"Invalid or duplicate manifest path: {rel}")
        seen.add(rel)
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Missing file or size mismatch: {rel}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {rel}")
        total += row["bytes"]
    print(json.dumps({"status": "pass", "files": len(seen), "bytes": total,
                      "version_family": data["version_family"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    verify(args.root, args.manifest or args.root / "provenance/delivery-manifest.json")
