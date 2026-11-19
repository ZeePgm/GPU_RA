"""Compile the four native ggml CUDA translation units with shared timing probes."""

from pathlib import Path
import subprocess
import sys

UNITS = {"convert.cu", "mmvf.cu", "binbcast.cu", "getrows.cu"}
sys.path.append(str(Path(__file__).resolve().parents[1] / "v3-q-auto-001"))

from compile_ptx import inspect_args, instrument


def main():
    driver, *args = sys.argv[1:]
    expanded = inspect_args(args)
    inputs = [Path(a) for a in expanded if Path(a).name.lower() in UNITS]
    if not inputs:
        return subprocess.run([driver, *args]).returncode
    if len(inputs) != 1:
        raise ValueError("expected one native CUDA source per compiler call")
    effective = args if "-c" in args and "-o" in args else expanded
    instrument(driver, effective, inputs[0].resolve(), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
