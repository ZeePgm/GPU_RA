# A3 fixed-H0 size response

Run from Visual Studio x64 developer PowerShell:

```powershell
./experiments/gpu_a3/build.ps1
python experiments/gpu_a3/run.py --study output/gpu-a3/<new-batch>
python experiments/gpu_a3/analyze.py --study output/gpu-a3/<new-batch>
```

Requires the preserved S8 H0 file and sealed EventLab dependency build. No GPU is required. Use a new output directory and do not use Python `-O`. The runner freezes plan, independent expected work, holdouts, candidate fit and 5% predictive acceptance before executing the backend. The count audit must pass before fitting. Three fresh sessions per point initialize the same 64KiB region with ordinary writes.

The inventory exception is explicit and limited to two unrelated archive tools; all old sealed source and artifact records must match, and linked libraries are hashed. Unexpected changes abort. The first batch aborted at this preflight; the second is the completed experiment.

See `docs/GPU_A3_RESULTS.md`. The fitted coefficients are empirical model response, not a physical-device calibration or a general service-time theorem. Raw statistics and trace files remain in the local batch.
