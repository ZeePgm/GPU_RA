# 第二版历史实验数据

本分支保留第一版的 `evidence/v1-ffn/`，并增加 `evidence/v2-readonly/`。第二版包保存 2026-11-08 新旧 F 对照的全部成功、失败及中断记录，以及 R1、R2 norm、R2 down、R3、真实上下文留出的原始证据；Python 字节码作为可再生缓存不归档，逐项原因记录在 manifest 的 omissions 中。原始文件没有有损缩减。

第二版包共 7,252 个路径、1,745,852,245 字节，按 SHA-256 去重后为 3,331 个对象，压缩分为 6 片，每片不超过 32 MiB。数据包含当时留证的源码/二进制快照；它们用于追溯历史身份，不作为新构建的链接输入。

从当前仓库恢复完整证据并离线重建：

```powershell
python reproduction/evidence_bundle.py restore --bundle evidence/v1-ffn --dest .
python reproduction/evidence_bundle.py restore --bundle evidence/v2-readonly --dest .
python reproduction/replay_evidence.py --root . --version v2 --report-dir output/offline-v2-001
```

恢复只依赖 Python 标准库；深度复核还需要 NumPy。恢复验证压缩分片及逐对象 SHA-256，保留原 `output/` 相对路径，拒绝覆盖不同内容。报告目录必须不存在。

离线复核包括：

- 第一版三档 F 原始 CUPTI 计算表重建、完整审计及 15 个损坏记录拒绝检查。
- 第二版同批 F1/F2 的输入 token IDs、完整 logits、原生 KV 日志、矩阵/请求/字节计数、逐 lane 依赖统计、CUPTI 与读取批次关联、GPU 时间并集、模拟时钟守恒，以及墙钟 TTFT/TPOT 的重新提取。
- `wall3` 资源窗口、第一次 F2 超时的精确五输出 logits 前缀、第二次中断记录均按历史分析器的条件重新检查。
- R1/R2 共 24 组完整 logits 字节对照；R3 112 个资格用例的独立逐字节访问多重集及依赖计数；160 对正式样本从逐进程原始观测重新聚合，复算 10 个条件的中位差、次序统计区间及正向配对数量。

历史分析器以原字节保存。运行时只去掉依赖原工作区源码/库存在及模型 GGUF 的身份条件，改由归档清单验证历史证据身份；被跳过的条件在复核报告中逐项列出。新源码构建和模型获取的身份检查由新运行入口承担。该工具没有重新推导全部 R1/R2/down/context 验收规则，也不重新执行 GPU 数值和返回字节检查；相应完整原记录已经归档。

`reproduction/evidence_validators/origin.json` 记录验证器来源哈希。F 同批比较仍以本批旧入口为分母，不能用第一版不同采集批次的历史数值替换。新的 GPU 实验需单独输出、独立冻结协议，见根 README。
