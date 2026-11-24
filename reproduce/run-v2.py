"""Compatibility entry point for the read-only FFN runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduction'))
from run_readonly_weights import MODEL_SHA256, PROMPT_SHA256, ROOT, clock_config, load, main, sha, union_ns


if __name__ == '__main__':
    main()
