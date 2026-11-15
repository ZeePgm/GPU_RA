# R2 down 补全与最终验收

2026-11-07。**R2 同算子双路径对照验收 PASS；性能优势仍为 inconclusive。** 本报告补齐此前缺失的 down 对照，与 [初轮 norm 结果](READONLY_R2_RESULTS_20261107.md) 共同构成 R2 交付。

## 根因与修复

初轮失败是 PTX 解析能力不足，不是 down 原始 CUDA 算法错误。最先触发的字符串词法错误来自 `.pragma "nounroll"`；继续解析还需要 scoped half2 解包、`cvt.f32.f16`、`fma.rn.f32`、共享内存及同步归约支持。

[src/ptx.cpp](../src/ptx.cpp) 现在在异步模式中支持这些实际所需结构：只接受 nounroll pragma、受校验的局部 half2 解包结构、4 字节对齐且总量不超过 48 KiB 的共享声明、32 位共享地址上的指定 load/store、无显式线程数的 `bar.sync 0`、寄存器操作数的 `shfl.sync.bfly.b32`。其他未知结构继续拒绝，同步版改写器保持原支持范围。

原先将线程提前退出直接等同于屏障死锁风险的描述过于笼统。SM75+ 上这里使用的屏障/shuffle 等待未退出线程；失败线程退出后归约数值可能无效，因此仍必须由服务报告失败、丢弃整次运行结果，并验证排空。本轮实际测试了这条失败链路。规范依据：[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

两条 down 路径共用 [算术源码](../adapters/llama_cpp/integrated_down_scalar.cu)，保持选定第 0 行、half2 读取、F32 FMA 顺序、256 线程和两级 xor 归约。双方每个 half2 均为一次 4 B 标量请求，避免与旧 128 B warp 合并路径混比。原模型默认 down 仍使用已验证的 warp 接入；本次新增入口用于算子级 R2 对照，不宣称已完成全模型 PTX 接入或 R3 合并优化。

## 验收结果及理由

| 项目 | 结果 |
|---|---|
| 正确性 | 12 个 FFN 宽度 × 1/2 CTA × 两种输入，共 48 条件；双策略三模式共 288 次独立运行，逐位匹配独立 CPU FMA/归约 oracle |
| 选择范围 | 仅写每 token 的第 0 行，其他输出位置保持零；双 CTA 使用不同激活数据 |
| 请求覆盖与身份 | 每 CTA 精确覆盖 `ffn/2` 个 4 B 请求，无重复/遗漏/越界；权重字节哈希一致 |
| 依赖 | 正确性计时版逐次检查消费间隔不短于对应建模延迟，time_scale=100；性能版关闭该插桩 |
| 固定回放 | 每路径共 92,748 次读取，状态、起止整数时间与 sector 完全一致 |
| 失败处理 | 两策略 × 单/双 CTA × 越界/后端失败，8 项均被拒绝；析构排空后 CUDA 同步成功，无成功结果发布 |
| 解析器 | 18 项 CTest 通过，包含新增合法结构、11 种非法变体及原有拒绝合同 |
| norm 回归 | 60 次运行通过；输出、父访问集合、原始及改写 PTX 哈希均与初轮正式批次一致 |
| GPU 审计 | 18 个 CUPTI 会话，无丢记录/采集错误，确认 2×256，保存真实 kernel 起止及资源用量 |

宽度为 2、62、64、66、510、512、514、1536、2050、3072、6144、16384。1536/3072/6144 来自三个现有模型的 FFN；其余覆盖 half2 最小输入、warp 边界、CTA 步长边界、不整齐循环尾部及较大循环。数据包含精确二进制分数与非整齐小数，独立 oracle 按冻结运算顺序计算。离散访问语义和同算术结果要求精确相等，沿用原 R2 合同，未放宽阈值。

## 性能结果

正式批次 448 次正常进程会话，含 288 次正确性、16 次 A/A、144 次 A/B。六条件各 12 对进程，顺序随机化且平衡，每进程预热一次再测三次。主指标为启动 API 至同步结束的墙钟。RTX 5070、compute_75；运行时及后端使用封存 R1 库，实验程序直接编译新版解析器。协议保存全部源码和二进制哈希，完成后已复核未变化。

单位 ms，Δ=PTX−CUDA。区间为 20,000 次进程配对 bootstrap、六条件 Bonferroni 调整后的名义 95% 探索区间。

| FFN | 模式 | PTX | CUDA | Δ | 区间 |
|---|---|---:|---:|---:|---|
| 1536 | bypass | 0.0695 | 0.0115 | 0.0579 | [0.0001, 0.2076] |
| 1536 | modeled | 33.1522 | 33.1655 | -0.0132 | [-1.5910, 1.3767] |
| 3072 | bypass | 0.0172 | 0.0200 | -0.0028 | [-0.0141, 0.0086] |
| 3072 | modeled | 60.5801 | 60.4451 | 0.1351 | [-2.2637, 2.2920] |
| 6144 | bypass | 0.0133 | 0.0146 | -0.0013 | [-0.0073, 0.0059] |
| 6144 | modeled | 117.9014 | 118.2951 | -0.3938 | [-3.2349, 2.5118] |

三个完整建模条件的区间均跨零。1536 bypass 的探索区间略高于零，但样本波动大、未预设应用 δ_min，不能据此确立默认策略选择或稳定收益。R2 允许 inconclusive，其目标是取得等价工作下可归因的比较，未预定哪条路径必须胜出。

分项记录包含启动调用、进程 CPU 时间、CUDA event 区间、首次启动、模块设置/JIT。event 区间含主机提交间隙；真实 kernel 时间由独立 CUPTI 批次提供，该批次为双 CTA 且带采集，不能混入单 CTA 主性能样本。实际访问到达时间另存各会话 arrivals，固定回放单独隔离后端行为。

NVCC 编译进程各三次，raw 中位数 766.354 ms、explicit 806.653 ms，仅为描述性观测。Python subprocess 调用 NVCC 时出现主机编译器进程 ACCESS_VIOLATION；相同源码经 PowerShell 直接调用成功，保留失败日志与完整成功批次，未把失败调用计入耗时比较。

## 证据与复现

- [正式协议](../output/r2-down-20261107/final-study/protocol.json)、[正式结果](../output/r2-down-20261107/final-study/result.json)、[故障结果](../output/r2-down-20261107/final-study/failures.json)
- [norm 回归与 GPU 审计](../output/r2-down-20261107/supplement/result.json)、[CUPTI 分项](../output/r2-down-20261107/supplement/cupti-summary.json)
- [编译耗时](../output/r2-down-20261107/compile-cost-powershell/result.json)、[测试日志](../output/r2-down-20261107/ctest.log)
- [批次脚本](../experiments/readonly_r2/run_down.py)、[审计脚本](../experiments/readonly_r2_audit/run.py)

加载 configs/enter-environment.ps1 -BuildTools 后，按 experiments/readonly_r2/README.md 构建，使用 `run_down.py --build build-r2-down-20261107 --output <新目录>` 复现。保留 source-snapshot 与输入哈希；性能测量期间不要并发编译或运行其他 GPU 实验。
