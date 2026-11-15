# 第一版历史实验数据

`v1-ffn/manifest.json` 对应 2026-11-05 第二次 F 的完整 F1/F2/F3 批次：129 个原始路径，158,571,957 字节，按 SHA-256 去重后为 77 个对象。所有输入配置、原始 GPU/消费者轨迹、完整 logits、逐 token 时钟、会话记录、报告、审计和执行日志均保留。压缩包约 11 MiB，每片不超过 32 MiB。

`artifacts/minexp/` 另行保存原 v0/minexp/S 数据，由其目录中的工具和清单管理。

先恢复再离线重建，所有路径相对于当前仓库，不需要历史工作区、模型 GGUF 或历史 DLL：

```powershell
python reproduction/evidence_bundle.py restore --bundle evidence/v1-ffn --dest .
python reproduction/replay_evidence.py --root . --version v1 --report-dir output/offline-v1-001
```

恢复工具仅用 Python 标准库，按分片哈希、解压后对象哈希和原始路径清单校验，并拒绝覆盖不同内容的文件。复核使用 NumPy 和历史原字节验证器；所需 Python 依赖按项目 requirements 安装。报告目录必须不存在。

离线复核重新从 CUPTI 原始 GPU 记录构造完整计算表，调用历史审计器检查完整 logits、模型消费者序列、矩阵覆盖、门控依赖顺序、fragment 计数、资源排空和逐 token 时钟守恒，并核对历史结果。每个 F 档还执行 5 类内存损坏副本的拒绝检查。历史源码清单见 `reproduction/evidence_validators/origin.json`。

仅验证归档、不恢复到磁盘：

```powershell
python reproduction/evidence_bundle.py restore --bundle evidence/v1-ffn --verify-only
```

新构建/新 GPU 采样与此历史离线复核分别记录；新采样入口见根 README。
