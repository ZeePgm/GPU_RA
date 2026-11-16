# 三种入口代表权重适配 — stage2-regression-20261116-001

- Code commit A: `a8d720f8e199293fe8434e02e7f02cff26ed92d9`
- Base: `codex/v3-joint` / `f4b754c8e228064364bc25877dbb39c698a2c167`
- Run: `v3-three-routes-sampled-001/stage2-regression-20261116-001`
- Process status / exit: `completed` / `0`
- Author verdict: `passed`
- Data manifest SHA-256: `d415edee6cb8f42eeb1838705ec5a4f13982c780f2ec17667bdfdb8602e9b531`

阶段2历史功能回归通过：F1后F2各8组共16模型进程全部通过原字节冻结验收器；完整logits/token、读取发布与逐lane依赖、KV全内容及前缀、预算和全部资源排空符合原合同。未修改历史清单/验收器/阈值；三入口代表权重实际读取及模型接入仍待阶段3/4。

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
