# R2 同算子只读接入对照

本目录比较同一 norm 权重乘法的 PTX 改写与显式 CUDA 标量接入，采用固定的 2×128 启动形状，每个有效元素一次 4 B 权重请求。两种入口共用 adapters/llama_cpp/integrated_weight_mul.cu 中的算术循环和 hbflab::ReadonlyAccess。常规算子对照使用 compute_75 PTX，CUDA Driver 在同一 GPU 上加载执行。

## 构建

在仓库根目录加载 configs/enter-environment.ps1 -BuildTools，使用 VS2022、CUDA 13 与 Ninja 配置此 CMake 项目。

```powershell
cmake -S experiments/readonly_r2 -B build-r2-20261107 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release `
  -DCMAKE_CUDA_ARCHITECTURES=75-virtual
cmake --build build-r2-20261107 --parallel 4
```

R1_BUILD 默认为 build-r1-20261107，读取其中已构建的 realgpu_runtime.lib 和 eventlab/hbf_eventlab.lib。通过 -DR1_BUILD=... 指定其他保留构建时，应核对 ABI、源码版本与库哈希。实验协议会冻结实际使用的库、程序、PTX、CMakeCache 和相关源文件哈希。

## 单个算子用例

```powershell
build-r2-20261107/readonly_r2.exe ptx modeled 1024 3 1 `
  build-r2-20261107/norm.ptx build-r2-20261107/down.ptx output/check.json
```

参数依次为 strategy、mode、width、测量启动次数、设备计时开关、norm PTX、down PTX、输出 JSON。strategy 支持 ptx/cuda，mode 支持 native/bypass/modeled。每次进程先执行一次预热，再执行指定数量的测量启动。

- native 使用相同原始 norm PTX 作为两种策略的算术参考。
- bypass 分别运行原始 PTX 或显式 CUDA 核的无后端分支。
- modeled 运行真实 mailbox、共享服务与 HBF 后端。
- 设备计时仅用于 modeled 正确性审计，检查每个元素在其建模延迟后才消费数据。性能测量关闭该插桩。

主指标 wall_ns 从启动 API 前到同步完成。submit_ns 为调用提交成本。stream_interval_ns 是 CUDA event 所围区间，包含其中的主机提交间隙。process_cpu_ns 是整个进程 CPU 时间增量，需考虑 Windows CPU 时钟量化。模型审计另用 CUPTI 保存实际 kernel 起止时间。

## 冻结批次

```powershell
.venv/Scripts/python.exe experiments/readonly_r2/run.py `
  --build build-r2-20261107 --output output/r2-new-batch
```

输出目录必须为新目录。协议包含 10 个宽度的三模式双策略正确性检查、两条路径的 A/A 先导、六条件各 12 个独立进程配对。每进程测量三个启动，统计单位为进程配对。计时阶段应独占 GPU 实验时段，避免同时编译或运行其他负载。

本批次采用固定探索预算，未设置应用 δ_min，报告差值和名义区间。选择默认策略需要另外冻结决策成本和测量资格。首次与模型构建并发的批次应保留为先导，另开目录在空闲时段重跑。

## 模型入口

模型构建使用 adapters/llama_cpp/integrated_norm_explicit.cu，纳入既有构建身份的 adapters 源清单。设置环境变量选择 norm 路径。

```powershell
$env:HBFLAB_R2_NORM_STRATEGY='cuda_scalar'
```

ptx 为默认策略，其他值明确拒绝。tools/run_readonly_r2_model.py 对四组模型输入分别运行两策略三模式，并检查数值、token、归一化父访问、图覆盖、启动形状和故障处理。兼容日志中的 ptx-calls/ptx-summary 名称继续表示 norm 调用组，实际策略由 module/readonly-summary 记录。

## down 能力探测

down_probe.cu 派生自当前 integrated_weights.cu 的选定行算术，保留 half2 读取、F32 累加、warp 归约、shared 汇总和 CTA 同步。辅助 MAD 与 warp reduction 来自固定 llama.cpp 的 common.cuh，适用其 MIT 许可。原始 PTX 先由 CUDA Driver 加载确认，再送入现有改写器。

初轮子集拒绝该 PTX。诊断保留于各输出的 down_rejection，原始生成物为 down.ptx。此探测只确认接入能力边界，不提供 down 双路径性能结果。后续适配需验证 shared、barrier、half conversion、FMA、shuffle 等实际所需语义，并处理失败线程与同步的关系。


## down 补全（2026-11-07）

当前异步改写器已支持本例需要的 half2 解包、FMA、shared、bar.sync 0 与 shfl.sync.bfly。完整结果见 docs/READONLY_R2_DOWN_RESULTS_20261107.md。

在相同环境下构建本目录到新的 build-r2-down-20261107；新增 readonly_r2_down.exe 用于等粒度对照，r2_down_contract 和 CTest 用于解析器回归。运行：

```powershell
.venv/Scripts/python.exe experiments/readonly_r2/run_down.py --build build-r2-down-20261107 --output output/r2-down-new
```

单次参数为 strategy mode ffn repeats timed down.ptx tokens output.json [pattern]。tokens 为 1/2，pattern 为 0/1。正常模式 native/bypass/modeled；故障测试用 fault/invalid_range，成功捕获故障输出 expected_failure。声明范围为偶数 FFN 2..16384；模型默认 down warp 路径不由本脚本切换。

独立 CUPTI 审计项目位于 experiments/readonly_r2_audit，需在主性能批次结束后单独构建/执行，避免采集和编译扰动。该目录 compile_cost.py 保留了失败的 Python/NVCC 调用诊断；最终编译观测使用 PowerShell 直接调用 NVCC，证据在 output/r2-down-20261107/compile-cost-powershell。
