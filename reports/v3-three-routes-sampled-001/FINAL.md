# 三入口代表权重适配结项

> 保留策略更新（2026-11-16）：已按用户批准清理全量轨迹，保留报告和必要参考结果，详见[清理记录](RETENTION.md)。本页40.15GiB等数据为清理前的实际生成量；当前使用显式 `verify --retained` 核验，不能再逐条复算已删除的原始时间轨迹。

阶段1–5已完成。**功能验收通过；性能判定为证据不足。** 结论范围为Qwen3-0.6B F16的14个代表权重、三种入口、两种输入与两种KV模式；其他297张量维持原生路径。

## 阶段证据

| 阶段 | 结果 | 运行/证据 |
|---|---|---|
| 1 清单与需求 | 311张量中14个选定，CUDA0连续存储，无别名；44/15输入各8输出，112消费者/输入 | [清单与需求](STAGE1.md)，stage1-20261116-004；B `157a2483bde452e0103caffceaa7474ca394403a` |
| 2 公共注册及历史回归 | 三入口各25负例、共享KV预留/排空；608 CPU、28 B2及历史F1/F2共16模型通过 | [阶段2](STAGE2.md)，registration-002 B `a8d720f8e199293fe8434e02e7f02cff26ed92d9`，regression-001 B `53465f9bc758b3260811bd58e542f7cdecffde63` |
| 3 完整独立组件 | 三入口均完整覆盖14张量及embedding151936行，尾部/偏移/重复行/边界/缓冲复用及8负例通过 | [阶段3](STAGE3.md)，stage3-full-20261116-001；B `8da8fa0f0d0f1c1ba8212faa3aa7625508c442b4` |
| 4 完整模型与联合KV | 正式16进程通过；12候选同KV完整logits和KV状态逐字节一致 | [阶段4](STAGE4.md)，stage4-full-20261116-001；B `df715711c5a2a67aba1c3e80bb00ae3b495dbf0c` |
| 5 成本与结项 | 42格支持矩阵、16进程成本及最终预算核对通过；共同模拟时钟未合格，性能证据不足 | stage5-diagnostics-20261116-002；A `00d7548eee918dcc285b22ed0c5a9699c33b26df`，报告B以verify输出为准 |

历史F1/F2回归对应阶段2当时的候选身份；最终候选另有608项CPU测试、阶段3完整组件及阶段4正式16进程模型证据。历史manifest、冻结验收器和原数值阈值保持。

## 逐张量支持

下表每格均包含实际注册、完整独立组件及4个模型条件（2输入×2KV），每张量每入口32个模型消费者。F16/F32均按实际连续布局处理。

| 张量 | 类型 | 物理字节 | symbolic | PTX | CUDA |
|---|---|---:|---|---|---|
| blk.10.attn_k.weight | F16 | 2097152 | 通过 | 通过 | 通过 |
| blk.10.attn_k_norm.weight | F32 | 512 | 通过 | 通过 | 通过 |
| blk.10.attn_norm.weight | F32 | 4096 | 通过 | 通过 | 通过 |
| blk.10.attn_output.weight | F16 | 4194304 | 通过 | 通过 | 通过 |
| blk.10.attn_q.weight | F16 | 4194304 | 通过 | 通过 | 通过 |
| blk.10.attn_q_norm.weight | F32 | 512 | 通过 | 通过 | 通过 |
| blk.10.attn_v.weight | F16 | 2097152 | 通过 | 通过 | 通过 |
| blk.10.ffn_down.weight | F16 | 6291456 | 通过 | 通过 | 通过 |
| blk.10.ffn_gate.weight | F16 | 6291456 | 通过 | 通过 | 通过 |
| blk.10.ffn_norm.weight | F32 | 4096 | 通过 | 通过 | 通过 |
| blk.10.ffn_up.weight | F16 | 6291456 | 通过 | 通过 | 通过 |
| output.weight | F16 | 311164928 | 通过 | 通过 | 通过 |
| output_norm.weight | F32 | 4096 | 通过 | 通过 | 通过 |
| token_embd.weight | F16 | 311164928 | 通过 | 通过 | 通过 |

完整映射及每种模型条件见[support-matrix.json](support-matrix.json)。embedding模型需求保留实际行及重复次数：原输入51次/36个不同行，短输入22次/16个不同行；全表读取仅用于独立夹具。

symbolic仅发起范围请求和门控，不执行模型阶段的载荷返回/发布；PTX使用实际AsyncModuleInterceptor改写的straight-line/predicated读取模板；CUDA使用显式读取内核。PTX/CUDA逐块返回和发布哈希一致。所有选定原生图消费者均在对应门控完成后放行，PTX/CUDA另核对发布早于放行；实际GPU消费者及与参考一致的终端内核已审计；未支持的偏移/控制流拒绝，不做入口回退。

正式12候选共64,245,912请求、2,055,869,184次lane依赖检查；每进程112消费者。完整8×151936 logits及同KV状态逐字节一致；分页对native最大max_abs=0.0168949068、RMS=0.00280749986，原0.2/0.025及精确token合同保持。KV完整/前缀字节、版本1..8、依赖及allocation/pin/dirty/lease/pinned释放全部通过。

## 资源与成本

使用2GiB逻辑介质，完整权重逻辑布局加KV占1,565,556,736 B，剩余581,926,912 B，另有32KiB物理备用量；该容量与历史F的1GiB条件不同。Host resident fragments=256，KV GPU页框=4、pinned槽=4保持。symbolic mirror为0，PTX/CUDA mirror各256KiB，三入口各256KiB时间缓冲；输出矩阵均有界分块。

| 正式组件+模型批次 | 最终记录/上界 | 冻结上限 |
|---|---:|---:|
| 墙钟 | 14996.412155秒（4.17小时） | 8小时 |
| GPU活动区间并集 | 14042.776296秒（3.90小时） | 8小时 |
| 科学CPU保守上界 | 27706.822855秒（7.70小时） | 24小时 |
| 最终输出 | 43,115,513,396 B（约40.15GiB） | 64GiB |
| 单模型最长墙钟 | 1330.484秒 | 3600秒 |
| C++进程Host最大工作集 | 3,326,812,160 B | 8GiB |

CPU上界包含模型C++/协调器实际CPU，并将未单独计CPU的整个组件区间按849.521155秒×16逻辑CPU保守封顶；不把该上界当作实际消耗。显存为阶段采样及有界reader分配身份，没有连续显存高水位。详见[预算核对](budget-audit.json)、[16进程成本表](DIAGNOSTICS.md)和[诊断数据](diagnostics.json)。

初始化、TTFT/TPOT墙钟、KV审计、drain、最终权重哈希、全进程墙钟、请求/数据工作、GPU活动、原始后端时钟、内存与日志开销分别保留。**联合共同模拟时钟尚未合格，性能结论为证据不足。** 本批单次功能样本不构成统计性收益确认，不使用CPU墙钟替代模拟时间；没有追加统计重复。

## 实现与复现

V3实现位于joint/source：[公共注册](../../joint/source/src/sampled_readonly.cpp)、[三入口reader](../../joint/source/src/sampled_reader.cu)、[模型接入](../../joint/source/adapters/llama_cpp/sampled_model.cu)。使用[构建入口](../../experiments/v3-three-routes-sampled-001/build_candidate.ps1)生成独立候选构建及receipt；[stage3.py](../../experiments/v3-three-routes-sampled-001/stage3.py)执行组件，[stage4.py](../../experiments/v3-three-routes-sampled-001/stage4.py)执行参考诊断或正式模型矩阵，[stage5.py](../../experiments/v3-three-routes-sampled-001/stage5.py)只汇总既有证据。

复现遵循[实验入口规范](../../reproduction/EXPERIMENTS.md)：新实验/唯一RUN、代码A、start冻结输入、直接执行生成launch.ps1、finish、报告B、verify。模型/原始轨迹/构建产物保存在外部GPU_RA-local-data，Git仅保存代码、配置、派生报告和manifest。

## 失败及结项边界

所有失败/中断保留：stage1-001长路径、002编译器ABI沙箱中断、003实际CPU embedding及双gather假设拒绝；stage2-registration-001路径JSON序列化；stage3-pilot-001 PTX局部寄存器命名；stage5-diagnostics-001 Windows默认GBK导致脚本准备失败但后续登记执行。对应修复均使用新提交和新RUN，未覆盖失败目录或修改历史清单。先导和smoke不计正式16样本。

本次关闭代表权重适配与验收工作。扩大其他层、其他布局/模型，或建立合格共同模拟时钟，属于后续独立实验。
