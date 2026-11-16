# 三种入口代表权重适配 — stage5-diagnostics-20261116-002

- Code commit A: `00d7548eee918dcc285b22ed0c5a9699c33b26df`
- Base: `codex/v3-joint` / `f4b754c8e228064364bc25877dbb39c698a2c167`
- Run: `v3-three-routes-sampled-001/stage5-diagnostics-20261116-002`
- Process status / exit: `completed` / `0`
- Author verdict: `inconclusive`
- Data manifest SHA-256: `2290bc78c3186a026fb5e67fccee71342e7af303ebb1ac9ee3e01b5201de8b41`

阶段5完成42格逐张量×入口功能支持矩阵与16进程成本汇总，功能验收全部通过。冻结资源预算审计通过：组件+正式模型墙钟14996.412155秒，GPU活动并集14042.776296秒，最终输出43115513396B，科学CPU保守上界27706.822855秒（组件协调器未单独计CPU，以849.521155秒×16逻辑CPU对整个组件区间保守封顶，加模型实际CPU）；各进程内存、磁盘及墙钟均在预算内。共同模拟时钟未合格，性能判定证据不足；墙钟/TTFT/TPOT/原始后端时钟分别报告，无统计性收益结论。001准备失败独立保留，无新增模型重放。

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
