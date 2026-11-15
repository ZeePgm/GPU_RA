# GPU_RA — 第二版：R3 完整 FFN 读取与原生 KV

本分支固定 2026-11-08 的第二版边界：R1/R2/R3 之后的完整 FFN 读取、逐 4 KiB 批次等待、plan/scatter、返回字节发布与消费者门控。KV 使用原生 F16，不启用原分页 KV 或后续 K 系列优化。A/B1/B2 不在本版冻结点内。

支持 Qwen3-0.6B F16、原始 249 字节 CRLF prompt、44 输入/8 输出、单序列、F1/F2。F1 第 10 层；F2 第 0/10/20/27 层；每层 gate/up/down 三个完整矩阵。保留 H0 512 MiB 及 Host 256 个 64 B fragment 的预算。

新机器使用私有仓库访问权限克隆本分支，推荐较短路径：

```powershell
git -c core.longpaths=true clone --branch codex/v2-readonly https://github.com/ZeePgm/GPU_RA.git GPU_RA-v2
cd GPU_RA-v2
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe reproduction/verify_checkout.py
```

## 源码与依赖

- 根目录框架/运行时/后端来源由 `build-r2-final-20261107` 的历史身份清单恢复，4,225 项源码 SHA-256 全部匹配。身份及逐文件来源记录在 [provenance](provenance/v2-r2-source-origin.json)。
- 对照运行时、原候选和 R3 读取核保持历史字节；R3 core 与 CUDA helper 取自 R3 最终快照。分别在 `experiments/readonly_f_compare/runtime*`、`r3_eventlab` 和 `experiments/readonly_r3`。
- 新 CMake 将旧 `build-r2-final`/`build-r3-current` 的 lib/DLL 依赖替换为源码构建目标；helper PTX 重新编译。历史 CMake 原件保留在 `provenance/historical`。
- 外部依赖为 Windows x64、VS2022 C++ Build Tools、CMake/Ninja、CUDA 13.0（CUPTI/cuBLAS）、Python 3.12，以及模型文件。模型 SHA-256 必须为 `74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2`。
- 模型获取与转换请沿用第一版模型记录；模型本体不包含在源码内。新运行入口会拒绝不同哈希、不同 prompt 和既有输出目录。

## 构建与来源检查

在本分支仓库根目录运行；路径可以是任意本机目录。

Windows 构建建议使用较短的克隆根目录，例如 `E:/GPU_RA`，为 CUDA/MSVC 生成路径留出空间。源码核验器使用 Windows extended-length 路径，能够核查深目录中的 llama UI 历史文件。

```powershell
python reproduce/verify-v2-source.py
./reproduce/build-v2.ps1 -CudaRoot <CUDA目录> -BuildDir ./build-v2 -CudaArchitectures 120a-real -Jobs 4
```

构建入口显式构建 llama、完整 R2 GGML CUDA 运行时、R3 core、候选、共同模拟时钟以及 R3 资格程序。既有实验二进制和生成头不作为构建输入。

## 完整 F1/F2 复现入口

```powershell
python reproduce/run-v2.py --model <Qwen3-0.6B-F16.gguf> --cuda-root <CUDA目录> --build-dir ./build-v2 --output ./output/new-f1 --scale F1 --mode all --clock
python reproduce/run-v2.py --model <Qwen3-0.6B-F16.gguf> --cuda-root <CUDA目录> --build-dir ./build-v2 --output ./output/new-f2 --scale F2 --mode all --clock
```

每个规模重新启动 native、旁路 GPU 轨迹采集和启用逐 lane 审计的 live 候选三个独立进程。保留原始输入/输出 token IDs、完整 logits、实际读取覆盖与发布、逐 lane 等待、正常关闭排空检查；完整 8 输出时还校验历史 native 的 1,215,488 个 logits 哈希。环境显式关闭分页 KV 与后续优化 hooks，固定 F32 矩阵计算与关闭 fusion。

`--clock` 只重放本次新采集的 GPU 活动/读取批次，输出逐 token 模拟时钟及守恒检查。读取核与其他 GPU 工作交叠、覆盖缺失或 trace 丢失会拒绝通过。当前生成的时钟结果与新采集轨迹一起保存在新输出目录；它们不覆盖历史样本。

`--outputs 1` 可用于短入口验证，其结果明确标为 smoke，不计为新增 44/8 实验。运行正式性能采样前应协调独占资源窗口。以上新入口是可重复的验收批次，不自动提供墙钟统计置信区间；历史墙钟对照仍按原报告解释。

## R3 资格与历史阶段

在配置 CUDA DLL 的同一环境执行：

```powershell
ctest --test-dir ./build-v2 -R '^v2_' --output-on-failure
```

注册的 10 个 GPU 夹具覆盖 1/2/4/8/16 B、尾 warp、稀疏/布局回退、越界失败及 drain；另有统一 PTX 策略的正例与非法模板拒绝。`r3_run` 支持原 R3 JSON 配置，供重跑完整适配器资格使用。

历史 R2/R3 脚本与报告保留原记录，包含当时机器路径；新构建与模型入口使用 `reproduce/`。历史结果依次见 [R1](docs/READONLY_R1_RESULTS_20261107.md)、[R2](docs/READONLY_R2_RESULTS_20261107.md)、[R2 down](docs/READONLY_R2_DOWN_RESULTS_20261107.md)、[R3](docs/READONLY_R3_RESULTS_20261108.md)、[F 对照](docs/READONLY_F_COMPARE_20261108.md)。新代码的时钟动作转换已用历史 F1/F2 轨迹逐项核对，分别匹配 110,608/442,384 个动作；记录在 [转换核对](provenance/v2-clock-transform-validation.json)。

历史实验数据采用内容寻址分片包，按原 `output/` 路径恢复。在本分支根目录执行：

```powershell
python reproduction/evidence_bundle.py restore --bundle evidence/v1-ffn --dest .
python reproduction/evidence_bundle.py restore --bundle evidence/v2-readonly --dest .
python reproduction/replay_evidence.py --root . --version v2 --report-dir output/offline-v2-001
```

离线复核不需要模型或旧构建库，重新检查历史记录的完整性、CUPTI/批次关联、时钟、logits、请求覆盖及 R 系列验收。新 GPU 运行与历史结果离线复核分开保存。旧 F 基线从第一版 `main` 的对应源码及数据复现；第二版只读候选与它保持独立身份。

## 2026-11-15 交付入口核验

独立恢复目录中的 4,423 条历史来源记录全部匹配；新构建的 345 条编译来源均位于交付源码树内，没有引用原 `build-r*` 库。重入构建无额外工作，11 个新注册资格测试全部通过。

新源码构建的 F1 完整 44/8 native、capture、live 三个进程通过，三组完整 logits 与历史 native 的 1,215,488 个值逐字节一致。live 完成 294,912 个请求、144 MiB 完整读取和 9,437,184 次逐 lane 字节/等待检查，正常排空。本轮 capture 的共同模拟时钟独立重放通过；F2 未新采集，保留完整历史证据复核与 442,384 个时钟动作的转换核对。

首次入口在加载模型前发现 CUPTI DLL 搜索目录不完整，失败记录保留；补全可配置 CUDA 根目录下的 runtime 路径后使用新目录完成验收，未改变历史二进制实现或验收阈值。结构化结果、构建/测试日志及不含二进制/模型的原始验收数据包见 [clean-validation](provenance/clean-validation/validation.json)。

第三方 GGML 自动探测的 Git commit 可能是 GPU_RA 的外层提交；上游 llama 身份以冻结来源清单中的源码和 overlay 为准。根目录旧 `release-manifest.json` 为历史记录，本次分支交付身份使用 `provenance/delivery-manifest.json`。
