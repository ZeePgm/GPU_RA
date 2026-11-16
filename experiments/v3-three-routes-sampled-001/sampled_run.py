"""Direct C++ scientific process execution with the original full resource guard."""
import ctypes
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import stage1 as S
R = S.R


class FileTime(ctypes.Structure):
    _fields_ = [('low', W.DWORD), ('high', W.DWORD)]


class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', W.DWORD), ('faults', W.DWORD)] + [(name, ctypes.c_size_t) for name in (
        'peak_working_set', 'working_set', 'peak_paged_pool', 'paged_pool', 'peak_nonpaged_pool',
        'nonpaged_pool', 'pagefile', 'peak_pagefile', 'private_bytes')]


def process_metrics(process):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetProcessTimes.argtypes = [W.HANDLE] + [ctypes.POINTER(FileTime)] * 4
    psapi.GetProcessMemoryInfo.argtypes = [W.HANDLE, ctypes.POINTER(MemoryCounters), W.DWORD]
    created, exited, system, user = (FileTime() for _ in range(4))
    S.require(kernel.GetProcessTimes(W.HANDLE(int(process._handle)), ctypes.byref(created), ctypes.byref(exited),
                                   ctypes.byref(system), ctypes.byref(user)), 'OS CPU time unavailable')
    to_seconds = lambda t: ((int(t.high) << 32) + int(t.low)) / 10_000_000
    memory = MemoryCounters()
    memory.cb = ctypes.sizeof(memory)
    observed = bool(psapi.GetProcessMemoryInfo(W.HANDLE(int(process._handle)), ctypes.byref(memory), memory.cb))
    return {'cpu_seconds': to_seconds(system) + to_seconds(user), 'memory_observed': observed,
            'peak_working_set_bytes': int(memory.peak_working_set) if observed else None,
            'working_set_bytes': int(memory.working_set) if observed else None,
            'private_bytes': int(memory.private_bytes) if observed else None}


def unchanged(build, receipt):
    S.require(S.source_identity() == receipt['source'], 'Candidate source drift')
    S.require(S.audit_build(build) == receipt['build'], 'Candidate binaries or build inputs drifted')


def directory_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file()) if Path(directory).exists() else 0


def execute(args, command, name, child_output, *, extra_env=None, timeout=900, require_ggml=False, disk_cap=8*1024**3):
    command = list(map(str, command))
    before = R.probe(args.resource_root)
    S.require(not before['conflicts'], 'Resource conflict before launch: ' + repr(before))
    module = R.correctness_module()
    module.BIN, module.CUDA = args.build_dir / 'bin', args.cuda_root
    env = module.environment()
    env.update(extra_env or {})
    record = {'command': command, 'start_utc': R.stamp(), 'resource_probes': [before], 'loaded_modules': [], 'metrics': []}
    record['environment'] = {k: v for k, v in env.items() if k.startswith(('HBFLAB_', 'GGML_', 'CUDA_', 'LLAMA_'))}
    start = time.monotonic()
    print('START', name, record['start_utc'], flush=True)
    with (args.output / (name + '.log')).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            next_probe = 0
            while process.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed >= next_probe:
                    snapshot = R.probe(args.resource_root, process.pid)
                    record['resource_probes'].append(snapshot)
                    S.require(not snapshot['conflicts'], 'Resource conflict during scientific run: ' + repr(snapshot))
                    metrics = process_metrics(process)
                    metrics['wall_seconds'] = elapsed
                    record['metrics'].append(metrics)
                    S.require(not metrics['peak_working_set_bytes'] or metrics['peak_working_set_bytes'] <= 8*1024**3,
                              'Process Host working set exceeded 8 GiB pilot ceiling')
                    S.require(directory_bytes(child_output) <= disk_cap, 'Process output exceeded disk budget')
                    next_probe = elapsed + 5
                if not record['loaded_modules'] and elapsed >= .2:
                    observed = module.modules(process.pid)
                    if not require_ggml or any(Path(p).name.lower() == 'ggml-cuda.dll' for p in observed):
                        record['loaded_modules'] = observed
                S.require(elapsed < timeout, 'Scientific process timeout: ' + name)
                time.sleep(.1)
        except BaseException:
            record['error'] = traceback.format_exc()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            raise
        finally:
            record.update(returncode=process.poll(), end_utc=R.stamp(), wall_seconds=time.monotonic() - start,
                          output_bytes=directory_bytes(child_output))
            record['final_metrics'] = process_metrics(process)
            S.write(args.output / (name + '.process.json'), record)
    S.require(process.returncode == 0, 'Scientific process failed: ' + name)
    if require_ggml:
        loaded = {Path(p).name.lower(): Path(p).resolve() for p in record['loaded_modules']}
        S.require(all(loaded.get(p.name.lower()) == p.resolve() for p in (args.build_dir / 'bin').glob('*.dll')
                      if p.name.lower().startswith(('ggml', 'llama'))), 'Loaded GGML module path mismatch')
    print('PASS', name, round(record['wall_seconds'], 3), 'seconds', flush=True)
    return record


def gpu_activity(folder):
    summary = S.read(Path(folder) / 'gpu-trace-summary.json')
    S.require(summary['status'] == 'pass' and summary['dropped'] == summary['errors'] == 0, 'GPU trace incomplete')
    records = [json.loads(line) for line in (Path(folder) / 'gpu-activity.jsonl').read_text(encoding='utf-8').splitlines()]
    S.require(records and all(0 < r['start_ns'] <= r['end_ns'] for r in records), 'GPU activity timestamps unavailable')
    def union(rows):
        total, end = 0, 0
        for start, finish in sorted((r['start_ns'], r['end_ns']) for r in rows):
            total += max(0, finish - max(start, end))
            end = max(end, finish)
        return total / 1e9
    phases = {}
    for phase in sorted({r['phase'] for r in records}):
        rows = [r for r in records if r['phase'] == phase]
        phases[str(phase)] = {'activity_union_seconds': union(rows), 'kernel_records': sum(r['kind'] == 'kernel' for r in rows),
                             'copy_bytes': sum(r.get('bytes', 0) for r in rows if r['kind'] == 'copy'),
                             'memset_bytes': sum(r.get('bytes', 0) for r in rows if r['kind'] == 'memset')}
    return {'status': 'pass', 'activity_union_seconds': union(records), 'phases': phases,
            'scope': 'Union of traced kernel/copy/memset intervals in this process; no cross-epoch comparison'}
