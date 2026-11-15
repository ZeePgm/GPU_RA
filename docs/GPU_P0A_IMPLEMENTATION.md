# P0-A measurement qualification

P0-A uses a dedicated runner (`experiments/gpu_s/p0a_run.py`). The legacy A1
`--formal` entry remains disabled to prevent bypassing qualification.

## Measurement contract

- Host boundaries use `steady_clock` nanoseconds: request entry, prefill end,
  token ID deliverable, and token JSON flushed to stdout. The last boundary is
  producer-side output return, not downstream consumer receipt or durable disk
  persistence. Model loading precedes request entry and has its own cost span.
- CUPTI kernel/transfer/API timestamps stay in a separate nanosecond domain.
  Simulation Tick stays in picoseconds. No cross-domain subtraction is allowed.
- The selected primary effect is a 100 ms injected delay at the first gate,
  evaluated at eight token-deliverable endpoints. The tolerance is fixed at 10%
  (10 ms). This qualifies a controlled perturbation, not a hardware speedup or
  smaller-dose experiments.
- Independent zero/zero pairs and standalone busy-wait calibration precede formal
  sampling. Complete verification data never enter the formal timing sample.
- Both process protocols use fixed warmup and balanced randomized crossover.
  Frequency/pstate/temperature/power snapshots are archived before and after
  runs (before/after the batch for reused-context runs); clocks are observed,
  not locked. Within-batch frequency transients are not resolved.
- Target and tolerance are frozen before calibration. The current runner freezes
  the formal pair budget after calibration using its predeclared formula,
  capped at 256 pairs. There is no optional stopping or tolerance relaxation.
  Inconclusive and failed intervals are retained.

## Reuse and mode equivalence

A reused context receives an unrelated token after the previous request, then
synchronizes and clears both KV metadata and KV data. The test asserts empty
sequence positions and a fresh backend Tick, recreates the sampler and backend,
and resets all gate/compute cursors and callback state. Model weights and
allocator resources intentionally remain resident. CUDA graphs are disabled by
runtime configuration. This is a fixed-workload functional reset qualification,
not a claim that every internal allocation/cache disappears.

Complete verification and lean verification compare every captured numerical
array byte-for-byte. Timing requests compare logits and token IDs. An ordered
FNV-1a identity covers node IDs, steps, operations, tensor/source names, source
shapes, strides and dtypes; pointers are excluded because process addresses
change. This is an accidental-corruption check, not a cryptographic proof.

Lean mode suppresses CUPTI activity collection, per-node trace serialization,
and diagnostic snapshots in timing runs. Dependency synchronization and token
boundaries are retained; no short windows are merged across those boundaries.
Full profiling therefore describes the instrumented path, not an additive
prediction of lean request latency.

## Cost accounting

`p0a-costs.json` and generated `p0a-timeline.json` preserve overlapping host
spans: load/initialization, host scheduling, backend CPU service,
synchronization, sampling, output return, export, backend drain and resource
close. GPU kernels/transfers remain on their own CUPTI track. Report category
unions and the host boundary chain; do not add nested categories. Unattributed
host gaps are explicitly reported. Final cost-file serialization and process
exit are outside the internal span envelope; the subprocess log records total
process wall time separately. No cross-clock critical path is claimed without
a calibrated clock mapping.

## Reproduction

From the repository root, with the verified CUDA build/runtime available:

```powershell
python experiments/gpu_s/p0a_run.py --study output/gpu-p0a/new-study --calibration-pairs 64
python experiments/gpu_s/p0a_validate.py output/gpu-p0a/new-study
python experiments/gpu_s/p0a_report.py output/gpu-p0a/new-study
python -m unittest discover -s experiments/gpu_s -p 'test_p0a*.py'
```

A study directory must be new. It contains the frozen target contract, binary
and source snapshots, independent calibration records, per-protocol contracts,
frequency records, raw boundaries, complete verification snapshots and final
interval decisions. The independent validator rechecks raw endpoints rather
than trusting exported error arrays. Student-t intervals assume independent,
approximately stable paired errors; order and temporal drift diagnostics are
reported separately and never used to overwrite primary decisions.
