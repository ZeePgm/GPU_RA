# 五类代表只读权重三入口性能比较计划 — qual-w1-p-symbolic-20261117-002

- Code commit A: `0fb8776e8a5fb3d7783bd5c63fb5a14b6d5c2a89`
- Base: `codex/v3-joint` / `9f6eaa71f79d89120b56602d84c99614ef22839e`
- Run: `v3-route-perf-five-001/qual-w1-p-symbolic-20261117-002`
- Process status / exit: `completed` / `0`
- Author verdict: `passed`
- Data manifest SHA-256: `66d515c0f93f8f4e6dbbe0f441a5a0f7067760557f29e659c9a50134f8d279ec`

Single native GPU consumer qualification passed: captured activation/stride preserved; complete output bitwise equal; borrowed weight unchanged; all runtime lane/byte and drain checks passed; summary exports no raw lane/request/chunk traces. Excluded from formal statistics.

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
