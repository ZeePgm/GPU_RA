# 第一版实验数据清单与外置恢复

仓库只保留数据清单、报告与工具；实验归档分片保存在仓库外的本地数据目录。数据总目录见根 [DATASETS.json](../DATASETS.json)。模型权重、构建缓存和外置实验载荷不进入 Git。

`v1-ffn/manifest.json` 对应 2026-11-05 第二次 F 的完整 F1/F2/F3 批次：129 个原始路径、158,571,957 字节、77 个去重对象。原始 GPU/消费者轨迹、完整 logits、逐 token 时钟、会话记录、报告、审计及日志均按原清单保留在本地归档，压缩分片为 11,550,087 字节。

原 v0/minexp/S 的清单在 `artifacts/minexp/`，恢复入口见[该目录说明](../artifacts/minexp/README.md)。

## 配置本地数据位置

复制根 `data.example.json` 为被忽略的 `data.local.json` 并填写 `data_root`，或设置环境变量：

```powershell
$env:GPU_RA_DATA_ROOT = '<仓库外归档根目录>'
```

外部根目录下保留 `evidence/v1-ffn/part-0000.gzpart`、`evidence/v1-migration-validation/part-0000.gzpart` 以及 `artifacts/minexp/payload.zip.part*` 的相对布局。每个恢复入口还支持 `--data-root`，优先级为显式参数 > 环境变量 > 本机配置；相对根目录按仓库根目录解释。清单由仓库读取，不从旧检出目录中的载荷回退。缺少配置或数据时会明确拒绝。

## 历史 F 全量恢复及离线重算

从仓库根目录运行，恢复目标与报告目录均选用仓库外的新目录，不需要模型 GGUF 或历史 DLL：

```powershell
python reproduction/evidence_bundle.py restore --bundle evidence/v1-ffn --dest '<仓库外恢复目录>'
python reproduction/replay_evidence.py --root '<仓库外恢复目录>' --version v1 --report-dir '<仓库外离线报告新目录>'
```

恢复工具只使用标准库，核对全部分片 SHA、解压对象 SHA 和原始路径清单，拒绝覆盖不同内容。复核使用 NumPy 与原字节历史验收器，重新从 CUPTI 原始记录构造完整计算表，检查完整 logits、消费者序列、矩阵覆盖、门控顺序、fragment 计数、排空及逐 token 时钟守恒。每个 F 档执行 5 类损坏副本的拒绝检查。历史验收器身份见 `reproduction/evidence_validators/origin.json`。

仅验证本地归档、不恢复文件：

```powershell
python reproduction/evidence_bundle.py restore --bundle evidence/v1-ffn --verify-only
```

显式提供仓库外、包含清单与载荷的完整 `--bundle` 目录时，工具允许直接读取该目录，不需要另外配置数据根目录。

## 本次迁移的新构建验证数据

`v1-migration-validation/manifest.json` 保存新源码构建后 F1 44 输入/8 输出的 43 个原始路径及构建日志，路径与历史批次分开，归档为 8,031,469 字节；不包含编译产物或模型权重。恢复后可核对原 F1：

```powershell
python reproduction/evidence_bundle.py restore --bundle evidence/v1-migration-validation --dest '<仓库外新验证恢复目录>'
python experiments/gpu_ffn/validate.py --output '<仓库外新验证恢复目录>/output/reproduction-f1-001/F1' --negative-checks
```

摘要见 `provenance/validation/v1-fresh-live-validation.json`。历史 F2/F3 由完整归档离线重算验证，本轮未新增完整 GPU 采样。新实验采用新的外部输出目录；不要覆盖本地历史载荷或把生成轨迹加入 Git。
