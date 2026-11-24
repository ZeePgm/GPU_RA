# GPU_RA · GPU 权重访问与分页 KV 仿真

GPU_RA 将 llama.cpp 的真实 GPU 计算接入 HBF-EventLab 介质模型。第三版可以观察原生 CUDA 指令如何读取模型权重，也可以让选定权重的读取与分页 KV 的读写使用同一个介质服务。工程源码在 [joint/source](joint/source/)，模型和运行数据保存在仓库外。

## 已验证的能力

| 路径 | 工作负载与结果 | 证据 |
|---|---|---|
| 全权重在线计时 | Qwen3-0.6B F16、原生 F16 KV，44-token 预填充和一次单 token 解码。加载时登记 311 个 GPU 权重范围，两阶段各读到 311 个权重，分别服务 291,152 和 291,122 个 4 KiB 页。622 份算子输出摘要、两步完整 logits、KV 和输出 token 均与同构建原生组一致。 | [全权重结果](reports/v3-all-timed-001/README.md) |
| 权重与分页 KV 共享服务 | 同一模型，44-token 预填充和七次解码。选择一层或四层的 FFN gate、up、down 矩阵，4 个 GPU 页框和 4 个 pinned 槽承载 KV。两种规模分别产生 294,912 和 1,179,648 次权重请求，各完成 17,192 次 KV 请求，8 次事务后资源排空。 | [联合正确性](joint/reports/RESULTS.md) · [142 个模型进程的配对统计](joint/statistical-batch/reports/RESULTS.md) |

全权重路径按实际 GPU 读地址识别权重，每阶段首次触达页面时发起请求，消费者在回执就绪并达到模拟截止时间后读取原显存。联合路径用有界窗口传送所选权重，KV 的页读、写回、事务提交和回收共享同一服务。选择具体路径时，以表中对应的工作负载和计时口径为准。

## 核对已有结果

已验证的平台为 Windows x64、Python 3.12、VS 2022 C++ Build Tools、CMake/Ninja、CUDA 13.0 和 RTX 5070（SM120a）。在本版本工作树的根目录设置本机外部数据位置，`bundles` 与 `runs`、`seals` 为同级目录。

```powershell
$env:GPU_RA_DATA_ROOT = '<本机外部数据根>/bundles'
python reproduction/verify_checkout.py --manifest provenance/project-layout-manifest.json
python reproduction/experiment.py verify --id v3-all-timed-001 --run all-native-timed-20261119-001
```

`verify_checkout.py` 核对当前项目布局，`experiment.py verify` 核对全权重运行的代码、输入、报告与外部证据。该次通过运行的代码提交为 `bd8f77d0`，报告提交为 `b85d46c7`。[实验定义](experiments/v3-all-timed-001/README.md) · [归档索引](DATASETS.json)

## 构建与运行全权重计时

在独立实验工作树中创建 Python 环境，指定 CUDA、外部构建目录和构建身份文件。构建身份绑定已提交的源码、四个 CUDA 编译单元及生成的二进制。执行 `receipt` 时工作树须干净，输出文件须为新路径。

```powershell
python -m venv .venv
$python = (Resolve-Path .venv/Scripts/python.exe).Path
& $python -m pip install -r joint/reproduction/requirements-validated.txt

$cuda = '<CUDA 13.0 目录>'
$build = '<新的外部构建目录>'
$receipt = Join-Path $build 'build-receipt.json'
./joint/reproduction/full_weight_timing/build.ps1 -CudaRoot $cuda -BuildRoot $build -Python $python
& $python joint/reproduction/full_weight_timing/run.py receipt --build-dir $build --output $receipt
& $python joint/reproduction/full_weight_timing/run.py measure --help
```

全权重模型目标为 `weight_access_model`。同一 CMake 工程的 `weight_read_window_check` 可单独构建，用于检查窗口读取边界和消费者等待。正式测量通过 `reproduction/experiment.py start` 登记 `joint/reproduction/full_weight_timing/run.py measure`。输入包括指定 GGUF、冻结提示、构建身份、四个 PTX 清单，以及已有五类算子的参考清单。保留 `{resource_root}`、`{config}`、`{run_dir}` 三个整参数占位符，由登记命令生成 `launch.ps1`。运行结束用 `finish` 记录结论，提交报告后再 `verify`。[新实验工作流](reproduction/EXPERIMENTS.md) · [全权重输入与验收](experiments/v3-all-timed-001/config.json)

## 复现历史联合实验

`f4b754c` 保留 11 月 15 日交付的源码与构建协议。需要核对该批 4,276 个冻结源文件、重建联合正确性样本或复算统计时，在独立工作树中使用对应入口。

```powershell
git worktree add --detach ../GPU_RA-v3-frozen f4b754c
Push-Location ../GPU_RA-v3-frozen
python reproduction/verify_checkout.py
python joint/reproduction/verify_source.py
Pop-Location
```

在该冻结工作树中，`joint/reproduction/build.ps1` 重建联合程序并运行组件检查，`run_correctness.py` 运行一层和四层的正确性对照，`run_statistics.py` 管理固定配对设计。历史样本也可通过 `evidence_bundle.py` 恢复，再用 `replay_historical.py` 离线复算。[联合构建](joint/reproduction/BUILD.md) · [运行步骤](joint/reproduction/RUNNING.md) · [历史数据说明](joint/evidence/README.md)

各版本和冻结标签见 [版本索引](VERSION_MAP.md)。历史只读实验的三种入口及各自覆盖范围见 [只读路径验收](reports/v3-restore-r2-001/README.md)。
