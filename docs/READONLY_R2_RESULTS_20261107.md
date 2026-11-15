# R2 同算子双路径实验结果

> 更新：down 已补全，R2 对照验收 PASS；见 [补全报告](READONLY_R2_DOWN_RESULTS_20261107.md)。以下保留初轮结果及当时的限制。

日期为 2026-11-07。**初轮 R2 为 PARTIAL，norm 功能验收 PASS，性能判定 INCONCLUSIVE，down 双路径为 UNSUPPORTED。** down 的原始核需要超出当前改写器的指令与同步支持，尚未完成其双路径对照。

## 1. 已实现的入口

- 新增 [显式 norm 核](../adapters/llama_cpp/integrated_norm_explicit.cu)，与 PTX 原始入口共用 [相同算术循环](../adapters/llama_cpp/integrated_weight_mul.cu)。
- 两条路径均为 F32 norm 权重乘法，固定 2×128，活动元素各发出一次 4 B 标量权重请求。模型为 F16 时，所选 norm tensor 仍按既有 F32 合同处理。
- [共同只读模块](../include/hbflab/readonly_access.hpp) 增加 CudaScalar 策略，统一校验和审计。模型通过 HBFLAB_R2_NORM_STRATEGY=ptx 或 cuda_scalar 选择，默认 ptx，其他值明确拒绝。
- 适配器、验证器与统一计数识别实际策略。兼容的 ptx-calls/ptx-summary 文件名继续表示 norm 调用组，实际路径由 module/readonly-summary 记录。
- 正式模型构建 build-r2-final-20261107 已封存，显式核位于 adapters 目录，包含在既有源码身份清单中。

## 2. 功能验收

| 检查 | 结果与证据 |
|---|---|
| 模型真实入口 | 四组输入 × 两种策略 × 三种模式，共 24 次正常运行，全部通过现有模型验收器 |
| 数值与 token | 每组 native/bypass/modeled 的双策略 logits 文件逐字节相同，token 完全相同；每路径的 bypass/modeled 也满足既有合同 |
| 请求与启动 | 归一化后父访问分组、多重集和图 expected/handled 一致，CUPTI 确认 norm 为 2×128 |
| 默认路径回归 | 最终构建 ptx 路径的 12 组模式/输入 logits 与 R1 保留文件逐字节一致 |
| 独立算子 | 10 个宽度 × 两策略 × 三模式，共 60 次进程会话，所有结果与独立 F32 乘法参考一致 |
| 逐访问语义 | 每宽度精确覆盖 [0,width×4)，每元素一个 4 B、一个 sector 的父访问，无遗漏或重复 |
| GPU 依赖 | 正确性专用计时版本按每个元素检查实际消费间隔不短于对应建模等待，time_scale=100；性能版本关闭该插桩 |
| 固定到达回放 | 对两路径实际父访问签名重放，共 13,698 个请求/路径，状态、起止时间和 sector 一致，整数时间残差为零 |
| 共同模块 | 三策略生命周期/负例与 18 次固定后端读取通过，非法 PTX 加载后服务可排空 |
| 模型失败路径 | ptx/cuda_scalar 均在 fault 模式报告 shared read terminal failure，返回 1，无成功 summary；非法策略被明确拒绝 |
| 新构建回归 | 独立算子/PTX/输入拒绝 18 项通过，Python 验收契约 5 项通过 |

宽度集合为 32、33、255、256、257、576、1024、1025、2048、8192，覆盖尾线程和 256 线程步长边界。每个算子进程一次预热、三次观测。模型每次生成 2 token，覆盖 prefill 与一次 decode，长输入跨 ubatch。

| 模型输入 | 每策略 norm 请求 | 每策略全部选定权重请求 |
|---|---:|---:|
| qwen-short | 46080 | 48240 |
| qwen-long | 142336 | 149008 |
| smol-short | 23616 | 24600 |
| qwen-1.7b | 92160 | 96480 |

每策略模型 norm 请求合计 304,192，连同 down 为 318,328。独立 GPU oracle、模型验收及父访问对照分别提供数值、发布/资源合同与访问语义证据。

## 3. 性能配对结果

正式批次为 final-study，220 个独立进程会话，其中 60 个正确性、16 个 A/A 先导、144 个 A/B 测量进程。A/B 为六条件各 12 对，启动顺序平衡并随机化，统计单位为独立进程配对，进程内三个启动先取均值。

算子编译目标统一为 compute_75 PTX，运行于 RTX 5070，驱动 616.56。正式批次在模型编译和 GPU 验收全部结束后串行执行。原 study 批次与构建并发，保留为先导，未并入下表。完整协议、随机顺序和输入哈希均保留。

主指标为 API 启动前至同步结束的稳态算子墙钟。区间为按进程配对重采样 20,000 次、对六条件作 Bonferroni 调整的名义 95% 探索性区间。小样本区间没有被当作有限样本保证。未设置应用最小效应 δ_min，因此这些结果用于估计差异，不用于默认策略切换。

Δ 定义为 PTX−CUDA，正值表示该条件下 CUDA 的点估计耗时较小。

| 宽度 | 模式 | PTX 均值 ms | CUDA 均值 ms | Δ ms | 调整后探索区间 ms |
|---|---|---:|---:|---:|---|
| 576 | bypass | 0.0110 | 0.0221 | -0.0111 | [-0.0453, 0.0058] |
| 576 | modeled | 25.6496 | 25.1094 | 0.5402 | [-0.6218, 1.6700] |
| 1024 | bypass | 0.0101 | 0.0118 | -0.0017 | [-0.0067, 0.0022] |
| 1024 | modeled | 40.8379 | 41.6710 | -0.8331 | [-1.9101, 0.2049] |
| 2048 | bypass | 0.0286 | 0.0152 | 0.0134 | [-0.0127, 0.0450] |
| 2048 | modeled | 77.9993 | 78.0738 | -0.0746 | [-1.7554, 1.7036] |

**六个区间均跨零，未建立性能胜负或等效结论。** 模拟服务、主机调度与等待共同构成当前墙钟，不以点估计选默认路径。原始文件还保留每对顺序和 A/A 差值，供判断波动与顺序效应。

### 分项计时的含义

- submit_ns 记录启动调用成本。process_cpu_ns 为全进程 CPU 时间，包含服务与驱动线程，Windows 量化可能使短区间返回零。
- stream_interval_ns 为 CUDA event 围住的区间，包含其间主机发射间隙。实际 kernel 起止由模型审计的 CUPTI 文件另外保存。
- module_setup_ns 包含模块改写、链接与加载等工作，显式 CUDA 的延迟初始化可能落在 first_launch_ns 中。比较首次启动时需联合查看这两项。新进程不等于清空驱动 JIT cache。
- native 作为统一的原始 PTX 算术参考；bypass 分别运行原始 PTX、显式 CUDA 无后端分支；modeled 才进入共同 mailbox/后端。

### 编译观测

在正式 GPU 测量结束后，对相同 compute_75 目标分别调用 NVCC --ptx，各进行三次编译进程观测。以下仅为描述性结果。

| 源入口 | 编译进程耗时中位数 ms |
|---|---:|
| raw | 385.599 |
| explicit | 861.798 |

PTX 路径的运行时模块/JIT 成本保存在每个会话的 module_setup_ns 与 first_launch_ns，编译命令、输出与日志另存 compile-cost。

## 4. down 能力边界

[down_probe.cu](../experiments/readonly_r2/down_probe.cu) 从当前选定行内核派生，保留 half2 读取、F32 累加及两级归约，共享内存和 CTA 同步。生成的 down.ptx 已由 CUDA Driver 成功加载。现有改写器处理它时明确拒绝。

```text
PTX line 73: unsupported character
```

该位置是编译器输出的 .pragma 字符串。生成物还包含当前子集未覆盖的 shared 声明/读写、bar.sync、内联 half 转换、FMA 和 shuffle 等结构。仅去掉这一条 pragma 不能建立完整支持。

因此，本轮没有取得 down 的等价 PTX 入口，也没有 down 双路径性能结果。完成 R2 的该部分，需要扩展改写器并验证这些结构的操作数、资源与控制流，特别是失败线程遇到 CTA/warp 同步时的安全终止。这是 R2 尚未整体 PASS 的原因。

## 5. 适用范围与后续决策

norm 的连续 F32 逐元素读路径，两种方式已可切换并满足功能合同。PTX 接入复用原始核并集中检查访存，显式 CUDA 复用相同算术源码并直接调用客户端。性能选择还需要适合具体决策的测量精度与预算。当前 down 保持已验证的显式 warp 路径，扩大 PTX 支持需要单独的语义扩展验收。

## 6. 证据与复现

- [结构化摘要](validation/readonly-r2-20261107.json)
- [正式算子协议](../output/r2-20261107/final-study/protocol.json)
- [正式性能与能力结果](../output/r2-20261107/final-study/result.json)
- [模型对照](../output/r2-20261107/final-models/result.json)
- [R1 默认路径回归](../output/r2-20261107/r1-default-regression.json)
- [编译观测](../output/r2-20261107/compile-cost/result.json)
- [独立算子检查](../output/r2-20261107/final-operators/validation.json)
- [实验入口与命令](../experiments/readonly_r2/README.md)
- [模型双策略脚本](../tools/run_readonly_r2_model.py)

模型运行前加载 configs/enter-environment.ps1，使用 build-r2-final-20261107。HBFLAB_R2_NORM_STRATEGY=ptx/cuda_scalar 选择 norm 接入。模型对照固定 HBFLAB_KV_WRITE_MODE=host，避免将 KV 路径变化带入比较。全部保留批次及原始日志位于 output/r2-20261107。
