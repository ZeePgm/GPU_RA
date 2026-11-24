# GPU_RA · 完整 FFN 权重读取与原生 KV

本分支研究 Qwen3-0.6B 在真实 GPU 上推理时，指定层的 FFN 如何从 HBF 仿真后端读取权重，以及读取、等待与计算各占多少时间。每个选定层读取 gate、up、down 三份完整矩阵，共 18 MiB。KV 使用原生 F16。其他分支的实现范围见[版本导览](VERSION_MAP.md)。

## 读取路径与观测

GPU 每批读取 4 KiB，合并连续访问，每批完成后按偏移拼回权重。整份矩阵组装完成后写回原框架分配的显存，消费该矩阵的算子才继续执行。模型使用原始 249 字节提示词，单序列输入 44 个 token，完整运行生成 8 个 token。单层配置（`F1`）选择第 10 层，四层配置（`F2`）选择第 0、10、20、27 层。后端配置为 H0 512 MiB，主机最多容纳 256 个 64 B 片段。[读取与发布实现](experiments/readonly_f_compare/candidate.cpp) · [介质配置](reproduce/inputs/H0-512MiB.ini)

| 运行模式 | 主要输出 |
| --- | --- |
| `native` | 用原生 GPU 路径生成对照 token 和完整 logits。 |
| `capture` | 旁路采集 GPU 活动和读取批次。`--clock` 将本次采集交给共同模拟时钟，重放每个 token。 |
| `live` | 在读取过程中校验返回字节、每个 lane 的等待、全矩阵覆盖和结束时排空。 |

## 依赖与构建

使用 Windows x64、PowerShell 7、VS 2022 C++ Build Tools、CMake、Ninja、CUDA 13.0（含 CUPTI 和 cuBLAS）及 Python 3.12。使用较短的本地工作树路径，从仓库根目录运行以下命令。将 `CUDA_PATH` 指向已安装的 Toolkit。

```powershell
python -m venv .venv
$python = (Resolve-Path ./.venv/Scripts/python.exe).Path
& $python -m pip install -r requirements.txt
& $python reproduce/verify-v2-source.py
./reproduction/build_readonly.ps1 -CudaRoot $env:CUDA_PATH -BuildDir ./build-v2
```

准备 Qwen3-0.6B F16 GGUF，入口核对 SHA-256 `74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2`。模型版本、转换参数及各输入文件身份见[模型记录](provenance/Qwen3-0.6B-conversion.json)。

## 最短运行

在同一 PowerShell 会话中替换模型路径，指定一个尚不存在的仓库外输出目录。以下运行覆盖单层、一个输出 token，依次启动原生对照、旁路采集和实时校验。

```powershell
$model = 'E:/models/Qwen3-0.6B-F16.gguf'
$output = 'E:/GPU_RA-local-data/checks/readonly-smoke-001'
& $python reproduction/run_readonly_weights.py --model $model --cuda-root $env:CUDA_PATH --build-dir ./build-v2 --output $output --scale F1 --mode all --outputs 1 --clock
```

`summary.json` 给出各模式的完成状态、进程墙钟时间、logits 哈希、请求数及覆盖字节。每个模式目录保存完整 logits 和 token；`F1-clock.json` 保存本次采集重放的模拟时间。完整验收沿用上述命令，换用新目录并设 `--outputs 8`。四层选 `--scale F2`。正式新增实验按[登记与封存流程](reproduction/EXPERIMENTS.md)运行。

## 验证

项目交付清单按路径、字节数和 SHA-256 核查当前文件。构建后，在 CTest 可用且已配置 CUDA DLL 搜索路径的环境运行读取适配器的边界、排空和 PTX 策略测试。

```powershell
& $python reproduction/verify_checkout.py --manifest provenance/project-layout-manifest.json
ctest --test-dir ./build-v2 -R '^v2_' --output-on-failure
```

2026-11-15 的源码重建记录覆盖 4,423 条历史来源，11 项资格测试通过。该次单层 44 输入、8 输出的三个模式均通过，1,215,488 个 logits 与历史原生结果逐字节一致。实时读取覆盖 144 MiB，完成 294,912 个请求及 9,437,184 次逐 lane 校验。四层保留历史运行及 442,384 个时钟动作的转换核对。[源码重建与单层验收](provenance/clean-validation/validation.json) · [时钟核对](provenance/v2-clock-transform-validation.json)

## 历史复现

本地数据根目录指向外部 `bundles`。历史全 FFN 对照使用第一版与第二版两个证据包，恢复到同一外部目录后即可离线复算 logits、请求覆盖和模拟时钟。使用新的报告目录。

```powershell
$env:GPU_RA_DATA_ROOT = 'E:/GPU_RA-local-data/bundles'
$restored = 'E:/GPU_RA-local-data/restored/readonly'
& $python reproduction/evidence_bundle.py restore --bundle evidence/v1-ffn --dest $restored
& $python reproduction/evidence_bundle.py restore --bundle evidence/v2-readonly --dest $restored
& $python reproduction/replay_evidence.py --root $restored --version v2 --report-dir E:/GPU_RA-local-data/reports/readonly-offline-001
& $python reproduction/restore_v2_validation.py --verify-only
```

原有 `reproduce/build-v2.ps1` 和 `reproduce/run-v2.py` 继续接受原参数。历史样本、清单和报告沿用各自采集时的名称。进一步阅读可从[历史基线](docs/history/baseline-2026-11-04.md)、[KV 舍入差异诊断](docs/numerics/kv-rounding-diagnosis.md)、[完整 FFN 对照](docs/READONLY_F_COMPARE_20261108.md)和[实验数据目录](DATASETS.json)进入。
