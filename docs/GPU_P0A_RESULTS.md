# P0-A results — 2026-11-05

The selected **reused-context protocol** satisfies the P0-A exit criteria for the fixed Qwen3-0.6B workload and a 100 ms first-gate perturbation. The fresh-process protocol remains **inconclusive** and did not enter formal sampling.

## Frozen decisions

Target: 100 ms; tolerance: 10% (±10 ms), unchanged across all cohorts.
Independent calibration: 64 zero/zero pairs per protocol and 64 standalone injector samples.
Formal budget: 145 pairs, calculated from calibration with the predeclared rule and frozen before formal collection.
Crossover: 73 zero-first / 72 inject-first pairs, shuffled with seed 4102; four warmup requests.

Formal primary error is the observed paired host latency difference minus the frozen 100 ms target. It does not subtract observed injected waiting time and does not use a cumulative simulation-wait identity.

| Token | Simultaneous paired-error interval (ms) | Decision |
|---|---:|---|
| 0 | [-0.852, 1.202] | pass |
| 1 | [-1.159, 1.855] | pass |
| 2 | [-1.367, 2.411] | pass |
| 3 | [-1.762, 2.873] | pass |
| 4 | [-1.948, 2.787] | pass |
| 5 | [-2.169, 2.807] | pass |
| 6 | [-2.712, 2.625] | pass |
| 7 | [-3.682, 2.370] | pass |

Bonferroni Student-t intervals use the frozen 16-endpoint family. Inference assumes independent, approximately stable pair errors. Order-stratified means and first/second-half diagnostics are retained separately, without changing the primary decisions.

## Qualification and equivalence

- Reuse: all eight calibration uncertainty bounds below 10 ms (maximum 8.291 ms).
- Fresh process: final two bounds 10.001 and 11.062 ms; retained as inconclusive.
- 564 runs independently checked for logits, tokens, ordered access fingerprint, gate identity, clock boundaries and reset state.
- 3,608 full/lean verification snapshot arrays are byte-identical.
- Before each reused request, an unrelated token populates KV; reset clears KV data and metadata, recreates sampler/backend and checks initial positions/Tick. Model weights and allocator resources remain resident.
- Native target compiled successfully; nine Python tests passed.

## Cost and clock evidence

Each run retains host boundary records and overlapping cost spans. Full verification also retains CUPTI kernels, transfers and API records. Generated timelines report category unions, host coverage and unattributed gaps, rather than summing nested categories. Host, CUPTI and Tick clocks remain distinct. Output return means token JSON flushed to stdout, excluding downstream receipt. Final cost-file serialization/process exit are excluded from internal spans and covered only by external subprocess wall measurement.

Full verification is strongly perturbing and is excluded from the timing sample. Lean mode preserves dependency synchronizations and token boundaries; it removes diagnostic JSON/CUPTI work without merging across dependencies.

## Preserved attempts

- `20261105-001`: original high-overhead sampling; qualification inconclusive.
- `20261105-002`: aborted on invalid contamination-test position; no formal evidence.
- `20261105-003`: corrected reset test and lean path; 8-pair calibration inconclusive.
- `20261105-004`: independent 64-pair calibration; selected reuse protocol qualified and formally sampled.

No older cohort was pooled into the formal sample; no target or tolerance was enlarged. This is not qualification of shorter doses, arbitrary models/workloads, or an HBF hardware speedup. GPU frequency snapshots are before/after batches, so within-batch frequency transients remain unresolved.

## Evidence

- [Exit criteria](../output/gpu-p0a/20261105-004/exit-criteria.json)
- [Frozen formal contract](../output/gpu-p0a/20261105-004/reuse_context-formal-contract.json)
- [Results](../output/gpu-p0a/20261105-004/result.json)
- [Independent validation](../output/gpu-p0a/20261105-004/validation.json)
- [Order diagnostics](../output/gpu-p0a/20261105-004/order-diagnostics.json)
- [Implementation and reproduction](GPU_P0A_IMPLEMENTATION.md)
