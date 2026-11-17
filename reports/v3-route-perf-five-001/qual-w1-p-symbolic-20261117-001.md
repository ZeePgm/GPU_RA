# 五类代表只读权重三入口性能比较计划 — qual-w1-p-symbolic-20261117-001

- Code commit A: `0f9479e7a1440cbd5b9acc6936d68a5842d62662`
- Base: `codex/v3-joint` / `9f6eaa71f79d89120b56602d84c99614ef22839e`
- Run: `v3-route-perf-five-001/qual-w1-p-symbolic-20261117-001`
- Process status / exit: `failed` / `1`
- Author verdict: `failed`
- Data manifest SHA-256: `ffa83943ffb67dff76e2abb03032c80fe4aec91ae2230430cc65df42ba32d95b`

Consumer output and drain passed inside C++, but overall run failed: the 0.25-second process exited before the OS observer captured loaded GGML modules. Sample excluded; retain record and require a new run after fixing post-timing identity observation.

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
