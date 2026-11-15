# v0/minexp 历史实验完整数据

本目录保留原 v0/minexp 的 S6–S9 及数值数组恢复清单和工具，载荷分片保存在仓库外的本地数据目录。它们作为第一版家族的历史证据；第二次 F（新后端）数据另列，不能混用后端身份或采样时间。

原始 22,379 个文件按 SHA-256 去重，分为 102 个不超过 2 MiB 的 ZIP 分片；恢复后约 4.44 GB。模型 GGUF 自行提供，模型、输入、代码及二进制身份保存在原始记录。

先配置根 `data.local.json` 的 `data_root`，或设置 `GPU_RA_DATA_ROOT`，使外部根目录下存在 `artifacts/minexp/payload.zip.part001` 至 `102`。`file-index.json.gz` 是原始文件的压缩哈希清单，和 `summary.json` 一起继续保留在仓库。

Python 标准库即可检查并恢复，目标必须在仓库外，冲突文件不会覆盖：

```powershell
python artifacts/minexp/restore.py --verify-only
python artifacts/minexp/restore.py --destination '<仓库外新数据目录>'
```

两个命令均可加 `--data-root '<仓库外归档根目录>'`，其优先级高于环境变量与本机配置。清单从本目录读取，分片从外部根目录的同一相对路径读取；显式 `--bundle '<仓库外完整包目录>'` 也可读取同时含清单和载荷的外部完整包。缺少数据不会回退到仓库内分片。统一使用此入口，`tools/restore_minexp_records.py` 保留为历史工具。

原始源码来源身份：`f600d9de3e7b6e4f5889f536a0c21ad05b57eebd`。分片和压缩索引均保持原字节；原记录中的本机路径保留为历史身份，不能当作新机器运行路径。新后端 F 的源码构建和实验入口见仓库主 README。
