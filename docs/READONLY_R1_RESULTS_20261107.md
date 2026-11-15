# R1 统一只读接入模块验收结果

日期为 2026-11-07。结论为 **PASS**，范围为原选定 norm PTX 标量路径与 down CUDA warp 路径的共同管理重构。新构建位于 `build-r1-20261107`，后端版本为 `8ccb1f353630f893f761b017d621812b7b9d5bf2`。

## 实现

新增 [ReadonlyAccess](../include/hbflab/readonly_access.hpp)，集中管理 tensor 名称、借用范围绑定、重叠检测、策略校验、步骤序号、图覆盖计数、启动/请求审计及结束/失败生命周期。两个前端的真实启动都通过共同 dispatch，范围与策略检查发生在回调启动前。内存字节、CUDA 模块和 LiveTimingService 沿用各自既有所有者。

[权重适配](../adapters/llama_cpp/integrated_weights.cu) 使用注册表完成匹配、服务注册与覆盖检查，[PTX 适配](../adapters/llama_cpp/integrated_ptx.cpp) 接收同一个注册表。新增 readonly-summary.json 提供两种策略的统一计数。原调用日志继续用于独立逐访问核对。

| 策略 | 当前支持集合 | 原请求语义 |
|---|---|---|
| ptx_scalar | profile 选定的 F32 norm 权重乘法，现有合法连续布局与形状 | 每活动线程标量读取 |
| cuda_warp | profile 选定的 F16 down 第 0 行，现有合法连续布局与形状 | 128 B warp 跨度 |

## 新旧模型对照

每个输入分别使用旧、新程序运行 native、bypass、modeled，共 24 次正常运行，均生成 2 token，包含 prefill 和一次 decode。长输入跨越 ubatch 边界。

| 输入 | 提示 token | 处理 token 数 | norm 请求 | down 请求 | 每版本选定请求合计 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| qwen-short | 44 | 45 | 46080 | 2160 | 48240 | PASS |
| qwen-long | 138 | 139 | 142336 | 6672 | 149008 | PASS |
| smol-short | 40 | 41 | 23616 | 984 | 24600 | PASS |
| qwen-1.7b | 44 | 45 | 92160 | 4320 | 96480 | PASS |

所有 12 组模式配对的 logits.f32 逐字节一致，提示和生成 token 完全一致。四组 modeled 对照按策略、step、row/row_base、rows 标识逻辑启动，去除实际指针基址与运行时 ID 后，对比每个父访问的 offset、bytes、sector、span_count 多重集，全部一致。每个启动下保留父请求分组。图 expected/handled 记录一致，统一策略请求合计与原 summary 一致。现有模型验收器同时检查独立 native 容差、KV 发布、预算、借用权重字节、依赖证据和最终关闭排空。

## 按原合同逐项判定

| R1 标准 | 证据与结果 |
|---|---|
| 逻辑访问与覆盖保持一致 | 四组旧/新父访问分组及多重集、图覆盖检查通过；每版本累计 318,328 个选定请求 |
| 固定到达后端结果与时间一致 | 两种策略分别执行 6 次固定地址/间隔读取，共 12 次状态、begin/end ps、sector 和最终统计完全一致 |
| 数值与完成合同 | 24 次模型运行通过现有验收器；12 组旧/新 logits 文件字节一致 |
| 两种策略共用生命周期 | 两种策略均检查重复/迟注册、绑定、范围重叠、越界、错误策略、无效步骤、覆盖缺失、owner、关闭及回调失败 |
| 加载失败与资源结束 | 新构建真实 PTX 非法模块加载被拒绝，共享服务可 finish 并 drained，GPU allocation/context 释放成功 |
| 最终失败保持 fail-closed | 旧/新故障模式均返回 1，错误为 shared read terminal failure，均无成功 summary |
| 构建与回归 | 新构建独立检查 18 项通过，Python 验收契约 5 项通过；既有二进制相关回归 50/50 通过 |

严重事件/Session 终态失败按既有合同停止，不将该故障用例声称为成功排空。正常模型运行和可恢复加载失败分别验证排空。

## 时间差异与覆盖边界

实时 GPU 到达驱动的模拟结束时间在旧/新运行之间存在差异，详见结构化结果中的 old_model_time_ps/new_model_time_ps；本次没有执行统计性能实验。固定输入时间计划的精确对照已通过。验收范围保持在当前选定算子、布局及单 host owner 的模型生命周期，双路径同算子比较和扩大权重覆盖属于 R2/R3。

两 token 运行用于验证 prefill/decode 接交、跨 ubatch 和结束合同，长时运行与性能收益需要相应后续实验。缓存容量、并发和新指令的能力声明以原支持表为准。

## 构建与证据

旧程序及 DLL 在修改前复制到 output/r1-20261107/baseline/bin，6 个文件均与原 sealed build 的 artifact 哈希一致，原 release identity 已保存。新构建独立配置并封存，未使用旧 build 重新链接。首次配置遇到本机 MSVC C1902 调试数据库问题，采用 CMAKE_TRY_COMPILE_CONFIGURATION=Release 后构建通过，失败与恢复日志均保留。

- [结构化验收摘要](validation/readonly-r1-20261107.json)
- [原设计合同](PARALLEL_OPTIMIZATION_EXPERIMENTS_20261107.md)
- [完整模型对照及原始证据](../output/r1-20261107/comparison/result.json)
- [只读共同契约测试日志](../output/r1-20261107/readonly-contract.log)
- [新构建算子验收](../output/r1-20261107/operators/validation.json)
- [已有二进制相关回归](../output/r1-20261107/existing-regression.log)
- [旧程序哈希核对](../output/r1-20261107/baseline/binary-check.json)

复现模型对照时，先加载 configs/enter-environment.ps1，再使用 [run_readonly_r1.py](../tools/run_readonly_r1.py) 的 --baseline、--candidate、--output 参数，输出目录必须为新目录。共同模块测试运行 build-r1-20261107/bin/realgpu_readonly_test.exe。新构建的 seal/verify 入口沿用 tools/integrated_build_identity.py。
