# GPU_RA — 第三版联合源码分支

本分支保存已封存的 **B2 只读路径 + 选定 K1/K4/K5 优化 KV** 联合工程。权威联合源码以无损分片保存在 `joint/source-bundle/`，恢复后为 `joint/source/`；根目录继承的第一版文件用于保留基线，联合程序应使用 `joint` 工程入口。

```powershell
python joint/restore_source.py
```

源码归档保留全部文件的原始字节，避免嵌套 Git 换行规则改变历史源码身份。恢复会核验 4,276 项 SHA-256，拒绝覆盖不同内容。`--verify-only` 可不落盘核验。

| 内容 | 位置与状态 |
|---|---|
| 封存源码与配置 | `joint/source-bundle`（恢复为 `joint/source`）、`joint/configs`；4,276 条源码记录重新核对大小及 SHA-256 通过 |
| 构建、正确性入口原件 | `joint/scripts`；原字节保留，尚有历史工具链/模型路径 |
| 来源和身份清单 | `joint/provenance`；保留当时完整源码/构建及证据清单 |
| 2026-11-12 正确性报告 | `joint/reports/RESULTS.md`；历史 16 个模型进程、608 项组件测试记录 |
| 2026-11-13 统计协议与结果 | `joint/statistical-batch`；历史 142 个模型进程，generation_wall_ms 口径 |

联合范围：Qwen3-0.6B F16、原始 44 输入/8 输出、单序列、F1/F2。读取 K=256 KiB、W=32 KiB；KV 启用 device 写来源、可取消追加、async 分页、coalesced 服务和异步副本；**未启用 K2 flush 合并或 K5 prefetch**。

本次优先完成第一、二版复现。第三版分支此次保存源码、协议、身份及报告，不分发旧构建二进制与完整原始批次数据；其进一步参数化、数据包及干净环境构建仍待单独整理，详见 `joint/MIGRATION_STATUS.json`。原始全量记录仍保存在工作区的 `integrated/20261112-dual-line-r1` 与 `integrated/20261113-dual-line-stat-r1`。

`joint/provenance/final-source-build-manifest.json` 的 build 条目及 evidence 清单描述历史完整封存目录，并非本分支当前分发清单；本分支交付身份由 `provenance/delivery-manifest.json` 管理。历史路径仅为溯源记录，不能把旧二进制存在或原报告通过当成本次重建通过。

三个版本的关系见 [VERSION_MAP.md](VERSION_MAP.md)。第一版为 `main`，第二版为 `codex/v2-readonly`，本分支为 `codex/v3-joint`。
