# 五类代表权重三方式性能比较

当前仅有[实验计划](../../experiments/v3-route-perf-five-001/PLAN.md)及机器可读矩阵、配置、输入规格。尚未实施、构建、登记 RUN 或启动运行，没有性能结果或科学验收报告 B。

正式设计为逐权重 30 格 / 90 个进程，加完整模型 9 个进程；保留 symbolic、PTX、CUDA 各自实现及分块策略。计时、预算、前置资格验证与日志保留规则以计划为准。

未来每次运行须通过 `reproduction/experiment.py start/finish/verify` 登记并封存；本 README 不能替代 RUN 报告及完整核验。
