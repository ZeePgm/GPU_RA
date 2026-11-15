# GPU_RA 的版本与分支

本仓库为私有研究仓库。原 RealGPUAcess_v0 保留原状，此次整理内容只同步到 GPU_RA。

| 引用 | 内容 | 主要证据范围 |
|---|---|---|
| `main` | 第一版：v0/minexp 历史来源 + 2026-11-05 新后端 F 源码修订 | S 系列及第二次 F1/F2/F3 的报告、清单和外置数据复现入口 |
| `codex/v2-readonly` | 第二版：R1–R3 后完整 FFN 读取与发布，原生 F16 KV | R 系列、2026-11-08 F1/F2 的报告、清单和外置数据复现入口 |
| `codex/v3-joint` | 第三版：B2 只读与选定 K1/K4/K5 优化 KV；直接跟踪全部联合源码并保留冻结备份 | 16+142历史数据清单和外置复现入口、独立源码重建、608 CPU/28 GPU夹具、新F1/F2共16组验收及统计采集入口 |

第一、二版 F 实验均使用原生 F16 KV；不能把这些结果视作旧分页 KV 与优化分页 KV 的比较。第三版本轮未启用 K2 flush 合并和 K5 prefetch。

第二版的 R3 冻结点与后续 A/B1/B2 修订分别登记。第三版采用 B2；它的两个实验目录对应同一联合源码版本的不同批次。各版的具体复现步骤、历史数据与本次新验证范围见所在分支的 README 和 `provenance`。

第三版的源码、依赖、完整历史数据清单、参数化入口及新联合验收已单独整理验证。新GPU验收为F1/F2共16组；142进程统计批次全量离线重算，新的统计采集协议仅prepare/check，没有重新采样或生成新的性能统计结论。

根目录 `release-manifest.json` 是恢复的历史发布身份。此次整理后的仓库文件以 `provenance/delivery-manifest.json` 登记，使用 `python reproduction/verify_checkout.py` 验证；历史数据包另有原始逐文件与逐对象清单。

实验归档分片已移至仓库外的本地数据根目录。仓库保存 `DATASETS.json`、原始 manifest/SCOPE、报告与恢复入口；通过 `GPU_RA_DATA_ROOT`、`data.local.json` 或 `--data-root` 指定分片位置。源码归档 `joint/source-bundle` 保留。
