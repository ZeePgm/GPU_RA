# GPU_RA — 第一版：新后端 F 基线

本分支 main 保存实验总账第一版的新后端 F 基线。历史 v0 和 minexp 提交保留在 Git 历史；第二版 R3 只读和第三版联合优化使用各自分支。项目用于真实 GPU 接入的 HBF 仿真。

本源码由 2026-11-05 的冻结快照恢复，按第二次 F 的源码身份补齐完整 FFN 入口。后端固定为 HBF-EventLab `8ccb1f353630f893f761b017d621812b7b9d5bf2`（无 K5 overlay），llama.cpp 固定为 `b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9` 加当时的前端改动。历史源码与恢复记录位于 provenance/；原冻结 README 位于 [docs/README_FROZEN_20261105.md](docs/README_FROZEN_20261105.md)。

F1/F2/F3 使用 Qwen3-0.6B F16、原始 CRLF prompt、44 输入/8 输出、greedy、原生 F16 KV、H0 512 MiB、Host 256×64 B 驻留预算。F1 覆盖第10层三个 FFN 矩阵；F2 覆盖0/10/20/27层；F3覆盖28层全部84矩阵。此实验的消费者前门控保留完整矩阵父读取和原调度，未启用分页 KV 或后续 KV 优化。

## 构建与运行

要求 Windows x64、Visual Studio 2022 C++ Build Tools、CMake/Ninja、Python 3.12、CUDA 13.0 含 CUPTI/cuBLAS，以及兼容 CUDA GPU。本次平台目标为 RTX 5070 / SM120a。源码和外部工具链、模型分别管理；仓库不附带 GGUF 权重。

从仓库根目录启动。CudaRoot、Python 和模型路径均为调用方参数，不依赖原工作区历史库或构建目录。

```powershell
./reproduction/build.ps1 -CudaRoot '<CUDA目录>' -Python '<Python可执行文件>' -Jobs 4
python reproduction/run_f.py --model '<Qwen3-0.6B-F16.gguf>' --output output/fresh-f-001 --cuda-root '<CUDA目录>'
```

模型需满足 SHA-256 `74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2`（1,509,347,712字节）。输入以哈希确认，可提供已有同哈希模型；另行转换得到的 GGUF 必须先核对哈希，转换工具与权重版本相同也不自动证明字节身份相同。

精确转换记录见 [provenance/Qwen3-0.6B-conversion.json](provenance/Qwen3-0.6B-conversion.json)，含 HuggingFace revision 和各文件哈希。需要从原始权重重建时，先安装该源码中 `third_party/llama.cpp-integrated/requirements/requirements-convert_hf_to_gguf.txt` 的转换依赖，再使用以下可选入口（首次下载约1.5GB）：

```powershell
python reproduction/prepare_model.py --input-dir external/Qwen3-0.6B --download --output models/Qwen3-0.6B-F16.gguf
```

脚本固定原始 revision，逐项检查输入文件大小和 SHA-256，调用仓库内转换器，并拒绝与历史 GGUF 哈希不同的转换结果。已有原始权重可省略 `--download`；省略 `--output` 仅检查输入。

构建入口从仓库内源码新建 model 和 FFN 两个目录，封存新的构建身份，执行在线审计及 Python 检查。所有运行必须使用新输出目录。F2/F3引用本次新F1的原生轨迹，拒绝混用历史二进制身份。默认完整F3可能耗时较长；--stages F1 --tokens 1 可做有限范围的入口资格检查，不能替代正式44输入/8输出实验。

仅核对输入、布局和预期计数而不执行 GPU：

```powershell
python reproduction/run_f.py --model '<Qwen3-0.6B-F16.gguf>' --output output/prepare-001 --prepare-only
```

## 身份和证据

provenance/historical-f-build-identity.json 保存历史二进制、DLL与4,222条源码哈希；provenance/source-recovery.json 记录源码恢复来源。build-reproduction/model/release-identity.json 在新构建后产生，专门标识新源码构建，和历史二进制身份分开。

`python reproduction/verify_source.py` 可重新核对全部4,222条历史源码记录。FFN入口本身仅编译C++，CUDA目标架构由新构建model依赖库的 `-CudaArchitectures` 决定。

F系列历史数据与离线重建入口由 evidence/ 中的数据清单及复核说明导航。历史计时保留当时的轨迹和测量条件；新采样另存新批次，不替换历史结果。minexp/S 原始数据由 artifacts/minexp/ 的分片与恢复工具管理。已有原始F代码的详细运行与输出合同见 [experiments/gpu_ffn/README.md](experiments/gpu_ffn/README.md)。
