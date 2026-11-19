"""Reuse the prior compiler launcher with this experiment's PTX rewriter."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "v3-q-auto-001"))
from compile_ptx import main

if __name__ == "__main__":
    raise SystemExit(main())
