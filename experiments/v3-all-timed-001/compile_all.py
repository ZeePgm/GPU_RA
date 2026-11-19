"""Use the four native ggml CUDA compilation units with the all-weight probe."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "v3-five-timed-001"))
from compile_five import main

if __name__ == "__main__":
    raise SystemExit(main())
