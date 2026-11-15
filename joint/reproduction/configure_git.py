"""Keep the sealed source byte-exact when committing it from this checkout."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    top = Path(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], text=True).strip())
    if top.resolve() != ROOT.resolve():
        raise ValueError("Run this helper inside the independent GPU_RA repository")
    common = Path(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], text=True).strip())
    if not common.is_absolute():
        common = ROOT / common
    target = common.resolve() / "info/attributes"
    rule = "joint/source/** -text"
    old = target.read_text(encoding="utf-8") if target.exists() else ""
    if rule not in old.splitlines():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(("\n" if old and not old.endswith("\n") else "") +
                         "# GPU_RA: preserve sealed source bytes when staging changes.\n" + rule + "\n")
    print("Configured repository-local source byte preservation; global Git settings unchanged.")


if __name__ == "__main__":
    main()
