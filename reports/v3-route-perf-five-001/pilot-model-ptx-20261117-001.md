# 五类代表只读权重三入口性能比较计划 — pilot-model-ptx-20261117-001

- Code commit A: `58be8458eed28f32a609a372e7df4361ce974c29`
- Base: `codex/v3-joint` / `9f6eaa71f79d89120b56602d84c99614ef22839e`
- Run: `v3-route-perf-five-001/pilot-model-ptx-20261117-001`
- Process status / exit: `completed` / `0`
- Author verdict: `passed`
- Data manifest SHA-256: `e70e257c07ca3d849afd8e361b0625609abe005ca0eb658c759c5c3ae423f0bd`

Fixed pilot passed after committed five-weight qualification. Complete native output/KV equality, runtime coverage, resource and identity/drain checks passed. Timing and pre/post GPU environment retained for budget estimation only; formal matrix not started.

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
