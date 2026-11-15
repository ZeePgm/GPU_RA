# GPU_RA — 第一版：新后端 F 基线

本分支 main 保存实验总账第一版的新后端 F 基线。历史 v0 和 minexp 提交保留在 Git 历史；第二版 R3 只读位于 `codex/v2-readonly`，第三版联合优化位于 `codex/v3-joint`，具体状态见 [VERSION_MAP.md](VERSION_MAP.md)。项目用于真实 GPU 接入的 HBF 仿真。

本源码由 2026-11-05 的冻结快照恢复，按第二次 F 的源码身份补齐完整 FFN 入口。后端固定为 HBF-EventLab `8ccb1f353630f893f761b017d621812b7b9d5bf2`（无 K5 overlay），llama.cpp 固定为 `b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9` 加当时的前端改动。历史源码与恢复记录位于 provenance/；原冻结 README 位于 [docs/README_FROZEN_20261105.md](docs/README_FROZEN_20261105.md)。

F1/F2/F3 使用 Qwen3-0.6B F16、原始 CRLF prompt、44 输入/8 输出、greedy、原生 F16 KV、H0 512 MiB、Host 256×64 B 驻留预算。F1 覆盖第10层三个 FFN 矩阵；F2 覆盖0/10/20/27层；F3覆盖28层全部84矩阵。此实验的消费者前门控保留完整矩阵父读取和原调度，未启用分页 KV 或后续 KV 优化。

## 构建与运行

要求 Windows x64、Visual Studio 2022 C++ Build Tools、CMake/Ninja、Python 3.12、CUDA 13.0 含 CUPTI/cuBLAS，以及兼容 CUDA GPU。本次平台目标为 RTX 5070 / SM120a。源码和外部工具链、模型分别管理；仓库不附带 GGUF 权重。

从仓库根目录启动。CudaRoot、Python 和模型路径均为调用方参数，不依赖原工作区历史库或构建目录。

Windows 建议选择短克隆目录（例如 E:/GPU_RA）；源码包含 llama.cpp UI 长文件名，克隆时启用 core.longpaths。新源码校验入口支持 Windows 扩展路径。

```powershell
git -c core.longpaths=true clone https://github.com/ZeePgm/GPU_RA.git
cd GPU_RA
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./reproduction/build.ps1 -CudaRoot '<CUDA目录>' -Python ./.venv/Scripts/python.exe -Jobs 4
./.venv/Scripts/python.exe reproduction/run_f.py --model '<Qwen3-0.6B-F16.gguf>' --output '<仓库外新批次目录>' --cuda-root '<CUDA目录>'
```

模型需满足 SHA-256 `74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2`（1,509,347,712字节）。输入以哈希确认，可提供已有同哈希模型；另行转换得到的 GGUF 必须先核对哈希，转换工具与权重版本相同也不自动证明字节身份相同。

精确转换记录见 [provenance/Qwen3-0.6B-conversion.json](provenance/Qwen3-0.6B-conversion.json)，含 HuggingFace revision 和各文件哈希。需要从原始权重重建时，先安装该源码中 `third_party/llama.cpp-integrated/requirements/requirements-convert_hf_to_gguf.txt` 的转换依赖，再使用以下可选入口（首次下载约1.5GB）：

```powershell
./.venv/Scripts/python.exe -m pip install -r third_party/llama.cpp-integrated/requirements/requirements-convert_hf_to_gguf.txt
./.venv/Scripts/python.exe reproduction/prepare_model.py --input-dir external/Qwen3-0.6B --download --output models/Qwen3-0.6B-F16.gguf
```

脚本固定原始 revision，逐项检查输入文件大小和 SHA-256，调用仓库内转换器，并拒绝与历史 GGUF 哈希不同的转换结果。已有原始权重可省略 `--download`；省略 `--output` 仅检查输入。

构建入口从仓库内源码新建 model 和 FFN 两个目录，封存新的构建身份，执行在线审计及 Python 检查。所有运行必须使用新输出目录。F2/F3引用本次新F1的原生轨迹，拒绝混用历史二进制身份。默认完整F3可能耗时较长；--stages F1 --tokens 1 可做有限范围的入口资格检查，不能替代正式44输入/8输出实验。

若仅需继续核对同一新构建，可显式传入 `-ReuseVerifiedModel -BuildRoot <已构建目录>`；脚本会先核验该目录的源码和构建身份，并要求 CUDA 工具链路径、CUDA目标架构与PTX架构匹配。模型编译跳过，FFN按同样目标完成构建和检查。首次重建仍使用新的 BuildRoot。

仅核对输入、布局和预期计数而不执行 GPU：

```powershell
python reproduction/run_f.py --model '<Qwen3-0.6B-F16.gguf>' --output '<仓库外准备目录>' --prepare-only
```

## 身份和证据

provenance/historical-f-build-identity.json 保存历史二进制、DLL与4,222条源码哈希；provenance/source-recovery.json 记录源码恢复来源。build-reproduction/model/release-identity.json 在新构建后产生，专门标识新源码构建，和历史二进制身份分开。

根目录 `release-manifest.json` 保留恢复的历史发布身份。当前GPU_RA源码、复现入口和数据清单以 `provenance/delivery-manifest.json` 登记，运行 `python reproduction/verify_checkout.py` 复核；该交付清单与历史源码清单、历史实验清单各自保留。

`python reproduction/verify_source.py` 可重新核对全部4,222条历史源码记录。FFN入口本身仅编译C++，CUDA目标架构由新构建model依赖库的 `-CudaArchitectures` 决定。

F系列历史数据与离线重建入口由 evidence/ 中的数据清单及复核说明导航。历史计时保留当时的轨迹和测量条件；新采样另存新批次，不替换历史结果。minexp/S 的清单和恢复工具在 artifacts/minexp/。已有原始F代码的详细运行与输出合同见 [experiments/gpu_ffn/README.md](experiments/gpu_ffn/README.md)。

## 实验数据外置

仓库保存源码、配置、报告、数据清单和复现入口；原始轨迹、logits、日志归档等实验载荷保留在仓库外的本地数据目录。克隆不会取得这些载荷，先取得与 [DATASETS.json](DATASETS.json) 及各批 manifest 哈希一致的本地数据。

可复制 [data.example.json](data.example.json) 为 `data.local.json`，填写本机的 `data_root`；本机配置已被 Git 忽略。也可设置 `$env:GPU_RA_DATA_ROOT='<外部归档根目录>'`，或在恢复命令中传 `--data-root`。优先级为显式参数、环境变量、本机配置；相对数据根目录按仓库根目录解释。三个版本可以共用一个外部根目录。

外部根目录保持仓库相对布局：`evidence/v1-ffn/part-0000.gzpart`、`evidence/v1-migration-validation/part-0000.gzpart` 和 `artifacts/minexp/payload.zip.part*`。清单继续从仓库读取，载荷只从外部位置读取。恢复目标必须在仓库外；新实验的 `--output` 也建议指向仓库外的新目录。新归档只能写到仓库外，提交时只纳入对应清单、配置和报告。详细命令见[数据说明](evidence/README.md)。

## 此次迁移的实际验证（2026-11-15）

- 恢复树中完整源码构建通过：model 317 步、FFN 4 步；在线审计 CTest 1/1、Python 10/10；最终参数化构建入口复核通过。
- 新构建 F1 完成原始 44 输入/8 输出，数值、覆盖、依赖、时钟及排空审计通过，5 类损坏记录拒绝检查通过。新 native/F1 完整 logits 与历史结果逐字节一致。
- 新后端 F1/F2/F3 历史数据在独立干净克隆中恢复并通过深度离线复核；v0/minexp 的 22,379 个原始文件另行恢复且全部哈希匹配。
- 本轮没有重新采集完整 F2/F3，也没有重新下载或转换模型。新采集不是历史样本的替换，验证结论不扩展为新的性能统计。

新验证摘要见 [v1-fresh-live-validation.json](provenance/validation/v1-fresh-live-validation.json)，原始新轨迹和构建日志的清单见 `evidence/v1-migration-validation`，数据载荷在配置的外部根目录。恢复及复核命令见 [数据说明](evidence/README.md)。
