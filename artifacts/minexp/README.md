# v0/minexp 历史实验完整数据

本包保留原 v0/minexp 的 S6–S9 及数值数组恢复记录，作为第一版家族的历史证据。第二次 F（新后端）的数据另列，不能与本包混用后端身份或采样时间。

原始 22,379 个文件按 SHA-256 去重，分为 102 个不超过 2 MiB 的 ZIP 分片；恢复后约 4.44 GB。分片保存于仓库外，Git 只保留 `file-index.json.gz` 数据索引、`summary.json`、报告与恢复入口。模型 GGUF 自行提供，模型、输入、代码及二进制身份保存在原始记录。

数据根目录依次由 `--data-root`、环境变量 `GPU_RA_DATA_ROOT` 或根目录被忽略的 `data.local.json` 的 `data_root` 选择；格式见 `data.example.json`。分片映射到 `<data_root>/artifacts/minexp/payload.zip.part*`，索引从本仓库读取。Python 标准库即可检查并恢复到仓库外，冲突文件不会覆盖：

```powershell
$env:GPU_RA_DATA_ROOT = 'E:/GPU_RA-local-data/bundles'
python artifacts/minexp/restore.py --verify-only
python artifacts/minexp/restore.py --destination E:/GPU_RA-local-data/restored/minexp
```

源码来源提交：`f600d9de3e7b6e4f5889f536a0c21ad05b57eebd`。分片和压缩索引均保持原字节；原记录中的本机路径保留为历史身份，不能当作新机器运行路径。新后端 F 的源码构建和实验入口见仓库主 README。
