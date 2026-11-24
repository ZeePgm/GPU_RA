"""Run the all-weight experiment with its registered configuration and checks."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[3]
    runner = repository / "experiments" / "v3-all-timed-001" / "run.py"
    runpy.run_path(str(runner), run_name="__main__")
