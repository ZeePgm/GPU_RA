"""Windows/Ninja compiler launcher for the two Q instrumentation translation units."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def win_argv(line: str) -> list[str]:
    split = ctypes.windll.shell32.CommandLineToArgvW
    split.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int))
    split.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    argv = split(line, ctypes.byref(count))
    if not argv:
        raise ctypes.WinError()
    try:
        return list(argv[:count.value])
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(argv, ctypes.c_void_p))


def inspect_args(args: list[str], ancestors: tuple[Path, ...] = ()) -> list[str]:
    """Inspect response files as well; normal pass-through retains the original argv."""
    result = []
    tokens = iter(args)
    for token in tokens:
        if token in {"--options-file", "-optf"}:
            paths = next(tokens).split(",")
        elif token.startswith(("--options-file=", "-optf=")):
            paths = token.split("=", 1)[1].split(",")
        elif token.startswith("@"):
            paths = [token[1:]]
        else:
            result.append(token)
            continue
        for value in paths:
            path = Path(value).resolve()
            if path in ancestors:
                raise ValueError(f"recursive NVCC response file {path}")
            raw = path.read_bytes()
            if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                data = raw.decode("utf-16")
            else:
                try:
                    data = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    data = raw.decode("mbcs")
            # CommandLineToArgvW treats the program name specially. The dummy
            # also handles leading whitespace and quoted paths consistently.
            response = win_argv("nvcc " + data.replace("\r", " ").replace("\n", " "))[1:]
            result.extend(inspect_args(response, (*ancestors, path)))
    return result


def nvcc_tail(plan: str) -> tuple[dict[str, str], list[list[str]]]:
    """Accept the CUDA 13 Windows whole-program compile suffix, or fail closed."""
    env = {name.upper(): value for name, value in os.environ.items()}
    commands = []
    for line in plan.splitlines():
        if not line.startswith("#$ "):
            continue
        action = line[3:].strip()
        var = re.fullmatch(r"([A-Za-z_]\w*)=(.*)", action)
        if var and not commands:
            # Values in this NVCC's prologue are generally already expanded.
            value = re.sub(r"%([^%]+)%", lambda m: env.get(m[1].upper(), m[0]), var[2])
            env[var[1].upper()] = value
            continue
        if not commands and not re.match(r"^ptxas(?:\.exe)?\s", action, re.I):
            continue
        commands.append(win_argv(action))
    names = [Path(args[0]).stem.lower() for args in commands]
    if names != ["ptxas", "fatbinary", "cl"]:
        raise ValueError(f"unexpected NVCC dry-run suffix {names}; expected one Windows device image")
    for args in commands:
        executable = shutil.which(args[0], path=env["PATH"])
        if not executable:
            raise FileNotFoundError(args[0])
        args[0] = executable
    return env, commands


def identity(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def publish(source: Path, destination: Path) -> None:
    # Only publish a completed product. Ninja can run both translation units at once.
    handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    os.close(handle)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def instrument(driver: str, args: list[str], source: Path, original_args: list[str]) -> None:
    from rewrite_ptx import rewrite

    if os.name != "nt" or "-c" not in args or args.count("-o") != 1:
        raise ValueError("the Q experiment requires an NVCC Windows -c/-o object build")
    unit = source.stem.lower()
    output_index = args.index("-o") + 1
    output = Path(args[output_index]).resolve()
    evidence = Path.cwd() / "q-auto-ptx" / unit
    evidence.mkdir(parents=True, exist_ok=True)
    header = source.with_name("hbflab_q_auto_device.cuh")
    if not header.is_file():
        raise FileNotFoundError(header)

    with tempfile.TemporaryDirectory(dir=evidence, prefix="w") as scratch:
        staging = Path(scratch)
        obj = staging / "q.obj"
        options = args.copy()
        options[output_index] = str(obj)
        command = [driver, "--keep", "--keep-dir", scratch, "--pre-include", str(header),
                   f"-DHBFLAB_Q_AUTO_TU={unit}", *options]
        # Let NVCC perform preprocessing, host stub generation and its private
        # dependency-filter step. In particular, preserve Ninja's original -MT/-MF.
        plan = subprocess.run([*command, "--dryrun"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=True)
        env, tail = nvcc_tail(os.fsdecode(plan.stdout))
        subprocess.run(command, check=True)
        ptx_inputs = [Path(a) for a in tail[0][1:] if a.lower().endswith(".ptx")]
        if len(ptx_inputs) != 1 or tail[0].count("-o") != 1:
            raise ValueError("unexpected ptxas input or output")
        ptx = ptx_inputs[0].resolve()
        cubin = Path(tail[0][tail[0].index("-o") + 1]).resolve()
        if ptx.parent != staging or cubin.parent != staging:
            raise ValueError("compilation phase references a different intermediate directory")
        native = staging / "native.ptx"
        shutil.copyfile(ptx, native)
        native_source = native.read_text(encoding="utf-8")
        try:
            rewritten, sites = rewrite(native_source, unit)
            if not sites or not rewritten or rewritten == native_source:
                raise ValueError(f"rewrite produced no instrumentation for {source}")
        except Exception:
            rejected = evidence / ("rejected-" + identity(native)["sha256"] + ".ptx")
            publish(native, rejected)
            print(f"rejected compiler PTX retained at {rejected}", file=sys.stderr)
            raise
        ptx.write_bytes(rewritten.encode("utf-8"))
        # The unchanged host stub includes the regenerated fatbin C header.
        # Remove the private first-pass products to require real regeneration.
        obj.unlink()
        cubin.unlink()
        fatbin_outputs = [Path(a.split("=", 1)[1]) for a in tail[1][1:]
                          if a.startswith(("--create=", "--embedded-fatbin="))]
        if len(fatbin_outputs) != 2 or any(p.parent.resolve() != staging for p in fatbin_outputs):
            raise ValueError("unexpected fatbin outputs")
        for product in fatbin_outputs:
            product.unlink()
        for part in tail:
            subprocess.run(part, env=env, check=True)
        fatbin = next(p for p in fatbin_outputs if p.suffix == ".fatbin")
        blob = fatbin.read_bytes()
        if not blob or not cubin.stat().st_size or blob not in obj.read_bytes():
            raise ValueError("instrumented fatbin missing from final host object")

        artifacts = {}
        for key, product, destination in [
            ("native_ptx", native, evidence / "native.ptx"),
            ("rewritten_ptx", ptx, evidence / "rewritten.ptx"),
            ("cubin", cubin, evidence / "instrumented.cubin"),
            ("object", obj, output),
        ]:
            publish(product, destination)
            artifacts[key] = identity(destination)
        manifest = {
            "schema": 1, "unit": unit, "source": identity(source), "header": identity(header),
            "compiler": identity(Path(shutil.which(driver) or driver)),
            "args": original_args, "sites": sites, "artifacts": artifacts,
            "embedded_fatbin_sha256": hashlib.sha256(blob).hexdigest(),
        }
        receipt = staging / "manifest.json"
        receipt.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        publish(receipt, evidence / "manifest.json")


def main() -> int:
    driver, *args = sys.argv[1:]
    expanded = inspect_args(args)
    inputs = [Path(a) for a in expanded if Path(a).name.lower() in {"convert.cu", "mmvf.cu"}]
    if not inputs:
        return subprocess.run([driver, *args]).returncode
    if len(inputs) != 1:
        raise ValueError("expected one selected CUDA translation unit per compile")
    # Keep NVCC's response-file parsing when the original compile/output
    # switches remain explicit (the normal CMake Ninja invocation).
    effective = args if "-c" in args and "-o" in args else expanded
    instrument(driver, effective, inputs[0].resolve(), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
