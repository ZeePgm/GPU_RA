# 第二次 F 与完整只读接入的对照

结果入口：[报告](E:/v1_update/docs/READONLY_F_COMPARE_20261108.md)。本目录是隔离实验，不替换共享 KV、运行时或默认入口。

## 固定条件

- 基线：2026-11-05 第二次 F，封存 `build-sf-f-entry-20261105/bin/gpu_ffn.exe` 及 DLL。
- 候选：R3 合并读取核、完整矩阵字节组装/发布、图段前门控；显式 CUDA 路径。
- Qwen3-0.6B F16，第二次 F 的原始 CRLF prompt，44 个输入 token、8 个输出 token；原生 F16 KV。
- F1 第 10 层，F2 第 0/10/20/27 层；每层 gate/up/down 共 18 MiB。保留全模型 FFN 布局中的逻辑地址。
- 不统一调度、请求粒度或同步批次。统一 token 端点和模拟时钟纳入范围。

## 文件

| 文件 | 作用 |
|---|---|
| `candidate.cpp` | 真实 llama 执行、完整权重读取和发布、原生 KV、输入及 logits 导出、累计 token 墙钟 |
| `capture.hpp` | 旁路 GPU 活动与读取批次记录；拒绝 CUPTI 丢失 |
| `clock.cpp`、`sim_audit.hpp` | 同一后端的方案 A 重建、fragment/完成/排空校验 |
| `local/` | 隔离的 TensorAccessBridge chunk 配置入口，不改变类布局 |
| `runtime/`、`runtime_include/` | 隔离运行时副本，仅增加稀疏逻辑基址登记，以保留第二次 F 地址布局 |
| `run_old.py` | 验证封存身份、采集旧原生 GPU 表并重跑 F1/F2 |
| `run_candidate.py` | 候选旁路采集、带字节/时间审计的 live 验收、双系统共同模拟时钟 |
| `run_wall.py` | 全部构建/时钟回放完成后串行执行稳定墙钟批次 |
| `process_probe.py` | 使用 Windows 进程枚举检查资源竞争，计时中不查询 CUDA/NVML 驱动 |
| `analyze.py` | 独立复核输入、完整 logits、覆盖、批次关联、时钟守恒并提取指标 |
| `write_report.py` | 从验收 JSON 生成中文报告和文档目录中的验收副本 |

## 顺序

在工作区根目录依次运行 `run_old.py`、`run_candidate.py`、`run_wall.py`、`analyze.py`、`write_report.py`。`run_old.py` 要求新输出目录，防止覆盖已封存记录。`run_candidate.py` 仅在原配置完全相同时复用成功记录。现有输出保存在 `output/readonly-f-compare-a-20261108`；重做整批时应统一修改脚本的输出目录并重新封存。

正式墙钟样本为 `wall3`。`run_wall.py` 需要主机进程可见性，若运行在仅能看到自身少数进程的沙箱中，会拒绝启动；本次由获准的主机执行环境运行。先等工作区实验与编译器退出，再每 5 秒原生枚举进程，记录任何竞争并拒绝接纳受干扰批次。复测前已确认 KV 聊天完成并空闲。

构建使用 `configs/enter-environment.ps1 -BuildTools` 设置的 MSVC/CUDA，目标为 `build-readonly-f-compare-20261108` 中的 `f_candidate` 与 `f_clock`。依赖 llama 使用 `build-r2-final-20261107`，EventLab core 使用 `build-r3-current-20261107` 的封存静态库。预检核对当前 EventLab 头文件与封存身份的 57 个哈希。

## 计时规则

模拟时间按各自 GPU 活跃区间的并集和 HBF 服务推进，排除 CPU 仿真求解用时。旧入口保留单节点消费者和整矩阵读取，候选保留图段前门控与逐 4 KiB 同步。候选一个批次的 8 个 512 B 请求在共同就绪点提交；有限读取核耗时在服务后计入，不假定批内重叠。完整旧轨迹必须在共同后端逐 token 逐皮秒复现。

旁路性能采集关闭额外字节/设备计时审计；另一个 live 运行开启这些审计并要求逐位 logits 一致。正式墙钟运行仍保留请求身份、覆盖、完成和排空检查。旧入口保留其原 CUPTI/在线审计成本，所以墙钟衡量各自可运行系统的总成本。

TTFT = 首个 token 完成 − 请求开始；TPOT =（第 8 个 token 完成 − 首个 token 完成）/7。模型加载、地址发现、初始化不计入这两个稳态指标；整个进程墙钟另外报告。单次稳定运行给出描述性差异，不提供统计置信区间。

## 排错保留

`new-F1-capture` 是未修正 Windows 文本读取的失败记录，不参与结论。该入口把 CRLF 转成 LF；输入 token 数仍为 44，但 token ID 不同。修正为二进制读取后，新旧原生输出的 1,215,488 个 logits 逐位一致。`candidate-F1-capture` 为带额外验收拷贝的采集，仅作验证；模拟性能使用 `candidate-*-capture-perf`。

首个 `wall` 批次的候选 F2 在第 5 个输出后发生等待超时，时间与 K4 并行编译重叠；`wall2` 又观察到 K4 测试，因而停止。两个批次均排除。未修改 GPU 二进制或放宽请求/服务超时，在 KV 线完成后的 `wall3` 四个完整运行均通过，253 次进程采样没有观察到竞争。具体超时触发点未唯一定位，保留失败记录与这个限制。
