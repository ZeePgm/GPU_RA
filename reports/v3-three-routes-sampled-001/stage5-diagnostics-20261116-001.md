# 三种入口代表权重适配 — stage5-diagnostics-20261116-001

- Code commit A: `df715711c5a2a67aba1c3e80bb00ae3b495dbf0c`
- Base: `codex/v3-joint` / `f4b754c8e228064364bc25877dbb39c698a2c167`
- Run: `v3-three-routes-sampled-001/stage5-diagnostics-20261116-001`
- Process status / exit: `failed` / `2`
- Author verdict: `failed`
- Data manifest SHA-256: `4cf876fb97a560fc2d63d15b42cf2f14dfefc033655f037fcd72f9f50968cfa0`

汇总脚本准备阶段以Windows默认GBK读取UTF-8草稿失败，后续登记仍执行，实际登记代码A中缺少stage5.py；生成启动脚本执行后明确失败并排空。无新增模型样本，正式16模型及组件证据保持已封存状态。本次失败完整保留；后续显式UTF-8、提交完整脚本并使用新RUN。

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
