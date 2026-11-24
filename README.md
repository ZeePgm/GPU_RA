# GPU_RA · GPU 推理中的 HBF 权重读取

GPU_RA 用真实 GPU 执行语言模型，研究 HBF 访存如何影响推理。主分支实现 FFN 权重的同步读取。llama.cpp 在 CUDA 上计算 Qwen3，每次使用选定的 gate、up 或 down 矩阵前，由 HBF-EventLab 模拟整块权重读取。所有子请求完成后，调用方放行对应的矩阵乘法。权重值和原生 F16 KV 均由 GPU 持有。

## 能研究什么

运行器先用 CUPTI 采集原生 GPU 的 kernel、memcpy 和 memset 活动，把每个计算区间内的活动时间合并为计算表。HBF 路径复用这张表，再加入各权重读取的等待时间。比较不同覆盖范围时，可以同时查看每步的读取量、模型首 token 延迟（TTFT）和后续 token 平均间隔（TPOT）。

| 权重覆盖 | Qwen3-0.6B 层号 | 每步读取 | 命令参数 |
|---|---|---:|---|
| 单层 | 10 | 3 个矩阵，18 MiB | `--coverage single-layer` |
| 四层 | 0、10、20、27 | 12 个矩阵，72 MiB | `--coverage four-layers` |
| 全部 28 层 | 0 至 27 | 84 个矩阵，504 MiB | `--coverage all-layers` |

四层和全层运行均先建立同批单层原生参考。省略覆盖参数会依次运行三种规模。若研究 CUDA 权重读取与发布，或权重读与分页 KV 读写的相互影响，分别参见只读路径和联合路径。[版本导航](VERSION_MAP.md)

## 依赖与构建

已验证环境为 Windows x64、Visual Studio 2022 C++ Build Tools、CMake/Ninja、Python 3.12、CUDA 13.0（含 CUPTI 和 cuBLAS）以及 RTX 5070。下列命令从仓库根目录执行。`$env:CUDA_PATH` 指向 CUDA 工具链，默认编译目标为 SM 120a。首次构建使用新的 `build-reproduction/` 目录。

```powershell
python -m venv .venv
$python = (Resolve-Path ./.venv/Scripts/python.exe).Path
& $python -m pip install -r requirements.txt
pwsh -NoProfile -File ./reproduction/build.ps1 -CudaRoot $env:CUDA_PATH -Python $python
```

模型使用 Qwen3-0.6B F16 GGUF，SHA-256 为 `74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2`。已有同哈希 GGUF 可直接使用；从原始权重重建时，按转换记录准备依赖并运行模型准备脚本。[模型转换记录](provenance/Qwen3-0.6B-conversion.json) · [模型准备脚本](reproduction/prepare_model.py)

## 最小示例

先检查一层、一个输出 token 的 GGUF 布局和读取计数。把 `$model` 改为本机 GGUF 路径。准备模式可以在构建前执行，每次在系统临时目录下生成新的输出目录。打开输出中的 `F1/plan.json` 可查看层号、输入身份和预期计数。

```powershell
$model = 'D:/models/Qwen3-0.6B-F16.gguf'
$preview = Join-Path ([IO.Path]::GetTempPath()) ('gpu-ra-ffn-' + [guid]::NewGuid())
& $python reproduction/run_ffn_weight_reads.py --model $model --output $preview --coverage single-layer --tokens 1 --prepare-only
Get-Content (Join-Path $preview 'F1/plan.json')
```

正式 GPU 采样使用 `reproduction/experiment.py new/start/finish/verify` 登记。从独立实验工作树构建并提交本次配置后，将 `$experiment` 设为该实验 ID，`$run` 设为新的运行 ID。以下命令登记单层的完整请求，随后用 PowerShell 7 执行生成的 `launch.ps1`。默认请求使用原始 44 个输入 token、8 个输出 token、greedy 解码、512 MiB HBF 配置和 256 个 64 B Host 驻留片段。[实验工作流](reproduction/EXPERIMENTS.md)

```powershell
& $python reproduction/experiment.py start --id $experiment --run $run --input $model -- $python reproduction/run_ffn_weight_reads.py --model $model --output "{run_dir}" --coverage single-layer --cuda-root $env:CUDA_PATH
```

## 验证与结果

```powershell
& $python reproduction/verify_checkout.py --manifest provenance/project-layout-manifest.json
& $python reproduction/verify_source.py
& $python -m unittest discover -s experiments/gpu_ffn -p test_run.py -v
```

当前交付清单检查项目文件身份，源码核验检查历史构建的 4,222 条源码哈希。正式采样另行核对完整 logits、token、选定矩阵的地址覆盖、消费顺序、时钟守恒及 Host/Device 排空。每种覆盖的输出目录含 `result.json` 和 `paired-tokens.csv`；前者汇总数值和计时，后者逐 token 列出计算与读取等待。模型计时使用皮秒，运行程序的墙钟开销单独记录。[FFN 实验与输出说明](experiments/gpu_ffn/README.md)

## 历史复现

历史批次以 F1、F2、F3 标记三种覆盖规模。旧入口 `reproduction/run_f.py`、`--stages F1 F2 F3` 和相应的输出目录仍可使用。历史三组均通过离线复核；迁移验证还重建并运行了单层 44 输入、8 输出样本，完整 logits 与原始参考逐位一致。[历史复核结果](provenance/validation/v1-offline-clean/acceptance.json) · [重建运行结果](provenance/validation/v1-fresh-live-validation.json)

使用 `GPU_RA_DATA_ROOT` 或本机 `data.local.json` 指定外部归档根。下面两步恢复历史运行并重算验收结果，恢复路径和报告路径应取仓库外的新目录。[数据清单与配置](evidence/README.md)

```powershell
& $python reproduction/evidence_bundle.py restore --bundle evidence/v1-ffn --dest '<恢复目录>'
& $python reproduction/replay_evidence.py --root '<恢复目录>' --version v1 --report-dir '<新报告目录>'
```

后端集成记录和数值误差研究保留了各自的实验条件及诊断过程。[后端基线](docs/history/baseline-2026-11-04.md) · [KV 舍入误差诊断](docs/numerics/kv-rounding-diagnosis.md)
