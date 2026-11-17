# 五类代表只读权重三入口性能比较计划 — qual-w3-d-symbolic-20261117-001

- Code commit A: `0fb8776e8a5fb3d7783bd5c63fb5a14b6d5c2a89`
- Base: `codex/v3-joint` / `9f6eaa71f79d89120b56602d84c99614ef22839e`
- Run: `v3-route-perf-five-001/qual-w3-d-symbolic-20261117-001`
- Process status / exit: `completed` / `0`
- Author verdict: `passed`
- Data manifest SHA-256: `8e935e7ce1dc239de7002e032ddc55870ca7bba34d2bda811d0e1b7242066290`

Single native GPU consumer qualification passed: captured activation/stride preserved; complete output bitwise equal; borrowed weight unchanged; all runtime lane/byte and drain checks passed; summary exports no raw lane/request/chunk traces. Excluded from formal statistics.

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
