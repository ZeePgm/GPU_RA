# A2 FIFO mechanism experiment

This uses the repository's actual `EventQueue` implementation with an experimental single-server FIFO adapter. It does not instantiate HostDriver, DeviceDriver, or H0. The Python oracle uses an independent scalar max recurrence.

On the measured Windows machine, build from a Visual Studio x64 developer PowerShell:

```powershell
./experiments/gpu_a2/build.ps1
python experiments/gpu_a2/run.py --study output/gpu-a2/<new-batch>
```

Do not use Python `-O`: acceptance checks intentionally use assertions. Output directories must be new. The runner freezes protocol and expected results before invoking the executable, runs twice, checks every request and event, checks real EventQueue probes, then runs output-corruption and invalid-input negative cases. It saves input, independent reference, actual traces, negative evidence, sources, executable and SHA256 manifest.

The CMake entry is portable source preparation, but its compiler probe failed on this machine due to the MSVC debug/PDB installation. The accepted binary was built directly with MSVC `/O2 /EHsc /MD` without debug information, using the two source files shown in build.ps1. Failed CMake logs are retained in the batch. The standalone runtime needs no CUDA or GPU.

The optional M/G/1 extension is not part of this completed deterministic experiment. See `docs/GPU_A2_RESULTS.md` for exact coverage and limitations.
