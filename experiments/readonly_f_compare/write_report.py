import json
from pathlib import Path
R=Path.cwd();O=R/'output/readonly-f-compare-a-20261108';d=json.loads((O/'comparison.json').read_text(encoding='utf-8'));cases=d['cases']
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
sim=[];wall=[];mechanism=[];gpu=[];process=[]
for scale,x in cases.items():
 for metric in ['ttft','tpot']:
  sim.append([scale,metric.upper(),f"{x['old_sim_ms'][metric]:.6f}",f"{x['new_sim_ms'][metric]:.6f}",f"{x['sim_slowdown'][metric]:.2f}×"])
  wall.append([scale,metric.upper(),f"{x['old_wall_s'][metric]:.3f}",f"{x['new_wall_s'][metric]:.3f}",f"{x['wall_speedup'][metric]:.2f}×",f"{100*(1-x['new_wall_s'][metric]/x['old_wall_s'][metric]):.2f}%"])
 mechanism.append([scale,x['selected_matrices'],int(x['read_MiB']),x['old_matrix_parents'],x['new_warp_parents'],x['new_batch_count'],x['read_fragments_each']])
 b=x['gpu_activity_breakdown'];gpu.append([scale,f"{x['old_compute_total_ms']:.3f}",f"{b['model_other_kernel']['active_ms']:.3f}",f"{b['read_kernel']['active_ms']:.3f}",f"{b['scatter']['active_ms']:.3f}",f"{b['copy']['active_ms']:.3f}",f"{x['new_compute_total_ms']:.3f}"])
 process.append([scale,f"{x['old_process_s']:.3f}",f"{x['new_process_s']:.3f}",f"{x['old_read_calls_s']:.3f}",f"{x['new_gate_calls_s']:.3f}",f"{x['old_backend_init_s']:.3f}",f"{x['new_adapter_setup_s']:.3f}"])
assert all(x['sim_slowdown'][k]>1 for x in cases.values() for k in ['ttft','tpot'])
s='''# 第二次 F 与 R1–R3 后完整只读接入：F1/F2 统一对照

状态：运行条件、数值、覆盖、依赖、共同模拟时钟及排空验收全部 PASS。实测墙钟成本下降，但模拟 TTFT/TPOT 上升。本次没有观察到相对第二次 F 的模拟推理延迟收益。

本次基线明确为 **2026-11-05 的第二次 F**，不是第一次 F，也不是上轮每 lane 16 B 的标量入口。使用该次封存的 `build-sf-f-entry-20261105/bin/gpu_ffn.exe` 与 DLL，二进制 SHA256 为 `78257def4fd8ccb92ef9d29c2bb0b1ca42030d9e96b1ac6045f3d687ac3ebfe3`。本轮从头运行并重新采集旧入口 GPU 计算表，历史表里的 6.883839/8.359689 ms 不直接作为本轮分母。

## 范围与运行条件

- Qwen3-0.6B F16、第二次 F 的原始 249 字节 CRLF prompt、44 个输入 token、8 个输出 token；greedy、F32 矩阵精度、原生 F16 KV，n_ctx=512、batch/ubatch=256、线程=8、关闭 flash attention。
- F1：第 10 层 gate/up/down，18 MiB；F2：第 0/10/20/27 层，72 MiB。保留全模型 FFN 布局中的原逻辑地址；后端共同采用 H0 512 MiB 配置。
- 旧系统：单节点派发、完整矩阵父读取、每个后端 Global 4096 B。候选：显式 CUDA 合并路径、图段前门控、每 warp 512 B、每 4 KiB 批次同步、scatter 与实际返回字节发布。各自调度保持原样。
- 旧 F 是模型消费者前的符号时序门控；候选还执行 GPU 数据读取、组装和发布。这是完整系统能力对照，新增工作会计入成本；本轮不能被解释成只测 PTX/CUDA API 开销。
- GPU：RTX 5070，驱动 616.56。核对旧二进制/DLL、模型身份；57 个 EventLab 头文件与封存 core 身份一致，已有 34 项实验输入未漂移。
- 候选在隔离目录中编译：封存 R2 llama DLL、R3 core 静态库、只读运行时副本。仅在副本中补充稀疏逻辑地址登记和模拟 chunk 配置；未修改共享 KV 实现，未启用 K1/K2、paged KV 或 capacity/KV hook。

## 统一计时口径

**模拟时钟（方案 A）**：各自真实 GPU 活跃区间的并集，加各自请求粒度、提交顺序和同步边界产生的 HBF 服务/依赖等待。排除 CPU 仿真求解、日志及主机执行空隙；不把忙等 CPU 服务造成的实际 GPU 核持续时间直接注入模拟值。候选从自己的旁路 GPU 轨迹采集有限计算、读取、scatter、拷贝成本，不能套用旧 F 的计算表。

候选每个 4 KiB 批次建模为同一就绪点的 8 个 512 B 请求，全部完成后才能继续下一批；有限读取核时间在服务后加入，采用保守的批内不重叠处理。这个批次模型不声称还原每个 warp 的细粒度发射时差。读取核与其他 GPU 活动如有交叠，当前重建会拒绝，而不是悄悄串行化。额外验收拷贝和设备计时插桩在性能采集中关闭。

共同后端分别回放两种策略；对旧 F 的 F1/F2，全部 16 个 token 完成时刻与旧入口自身结果逐皮秒一致。所有 token 满足 `relative_ps = compute_ps + wait_ps`，守恒残差为零。

**实际墙钟**：t0 为测量请求开始，t1/t8 为第 1/8 个 token 完成；TTFT=t1−t0，TPOT=(t8−t1)/7。加载、地址发现与初始化在稳态指标外，另报全进程时间。旧入口保留原 CUPTI/在线审计；候选性能运行保留请求身份、覆盖、完成与排空检查，逐 lane 字节/时间审计放在独立验收运行。因此墙钟比较的是两套可运行系统及其必要观测成本。

最初旧入口资格运行遇到短暂构建并行，不用它作为主墙钟结果。第一次墙钟批次的候选 F2 在第 5 个输出后超时；事后发现 K4 从 19:10 开始并行构建，覆盖 19:12 的故障时段，首批不满足稳定条件而整批排除。并行构建与周期性驱动查询均是可能干扰，现有日志不能唯一确定超时根因。没有修改 GPU 二进制或放宽 1 秒请求/60 秒服务超时，改用全系统原生进程观察，第二次批次又观察到 `build-k4/hbflab_k4.exe`，因此停止并保留记录；在协调独占计时窗口后重新执行完整 `wall3` 批次。

主结果取 `wall3`，顺序为旧 F1→候选 F1→候选 F2→旧 F2。每系统每规模一次完整稳定墙钟运行、一次 GPU 轨迹；以下是描述性结果，不是中位数或统计置信界。失败记录保留，因此本次完整成功复测也不构成长时间或并发负载下的稳定性证明。

## 模拟 TTFT/TPOT

'''+table(['规模','指标','第二次 F 重跑（ms）','候选（ms）','候选/旧，延迟增加'],sim)+'''
两种规模的模拟延迟均退化。不能以实际程序运行更快来宣称模拟推理更快。

## 实际墙钟

'''+table(['规模','指标','旧（s）','候选（s）','旧/候选','耗时下降'],wall)+'''
'''+table(['规模','旧全进程 s','候选全进程 s','旧读取调用 s','候选门控调用 s','旧后端初始化 s','候选适配器设置 s'],process)+'''
初始化列用于展示各自设置阶段，范围不同，不能单独作初始化算法的因果比较。全进程列包含加载、设置、推理及退出，提供完整运行成本。

## 为什么墙钟降低而模拟延迟上升

'''+table(['规模','矩阵数','累计 MiB','旧矩阵父请求','候选 warp 父请求','候选 4 KiB 批次','双方 64 B fragments'],mechanism)+'''
两边工作覆盖相同，但请求结构不同。旧 F 的一个矩阵父请求可持续补充多个 4096 B Global；候选在一个 4 KiB 批次的 8 个 512 B 请求完成后，才提交下一批。这压低跨批次在途量，增加建模等待。R3 相对自身标量版本的“32 倍请求减少”不能搬到旧 F 对照：此处旧 F 本来就是整矩阵提交，候选父请求反而更多。

8 个 token 的 GPU 活跃时间分解如下；旧表包含自身模型工作及原生拷贝，候选其他核、读取核、scatter 与拷贝分列。

'''+table(['规模','旧 GPU 总 ms','候选模型/其他核 ms','候选读取核 ms','候选 scatter ms','候选拷贝 ms','候选 GPU 总 ms'],gpu)+'''
'''+table(['规模','旧 HBF 累计等待 ms','候选 HBF 累计等待 ms','共同 core 旧轨迹回放 s','共同 core 候选轨迹回放 s'],[[k,f"{x['old_wait_total_ms']:.6f}",f"{x['new_wait_total_ms']:.6f}",f"{x['common_core_old_replay_s']:.3f}",f"{x['common_core_new_replay_s']:.3f}"] for k,x in cases.items()])+'''
候选需要大量短读取核、scatter、plan 拷贝与同步，增加 GPU 活跃工作；整矩阵持续供给改为小批次屏障也增加 HBF 依赖等待。这两项直接解释模拟延迟上升。

软件墙钟还受父请求管理、在线审计、事件驱动开销及 GPU/CPU 交互影响，不能由模拟 picosecond 推断。共同 core 的 F1 轨迹回放也呈现软件开销差异，说明墙钟与建模完成时间可以朝相反方向变化。回放包含初始化和审计，不能从 live 墙钟直接相减当作“纯接入开销”。旧 F2 离线回放期间曾与 K3 实验并行，其回放墙钟仅作运行日志，不纳入性能归因；确定性的模拟结果仍须逐 ps 验收。最终 live 墙钟批次每 5 秒原生枚举工作区进程与编译器，有冲突则拒绝验收；计时期间不轮询 GPU 驱动。旧新 live 入口的后端版本/封装、观测机制与策略不同，整项改善不归因于 R3 单独一项。

## 验收与排错

| 验收项 | 标准与理由 | 结果 |
|---|---|---|
| 输入和数值 | 输入 token ID 相同、8 个输出相同、全部 1,215,488 个 logits 逐位相同；请求读取不能改变计算结果 | PASS |
| 工作覆盖 | 每步每选定层三个完整矩阵；偏移、字节与 64 B fragment 多重集完整；避免省略工作制造收益 | PASS |
| 依赖 | 旧消费者 GPU API 在门控完成后启动；候选逐 lane 时间/字节检查且先发布再放行图段；防止提前消费 | PASS |
| 时钟 | 旧完整轨迹在共同后端逐 token 逐 ps 复现，GPU/批次记录一一对应，所有 token 守恒为零 | PASS |
| 生命周期/KV | 所有会话成功、关闭、排空，无 KV 优化 hook；避免跨运行状态污染 | PASS |
| 性能目标 | 同口径报告真实方向，不预设任意加速百分比作为通过线 | 墙钟下降；模拟延迟未改善 |

F1/F2 候选验证分别完成 9,437,184 / 37,748,736 次 lane 数据/依赖检查。输入读取曾暴露 Windows 文本模式差异：CRLF 被转换成 LF，虽然仍为 44 个 token，却不是同一输入。改为二进制读取后，新旧原生完整 logits 逐位一致，再完成所有接入验证。失败记录 `new-F1-capture` 保留但不计入结果；带额外验收拷贝的 `candidate-F1-capture` 也不作性能轨迹。

## 三阶段实际得到什么

| 阶段 | 已有成果 | 本次能支持的结论 |
|---|---|---|
| R1 | 统一只读注册、策略、覆盖及生命周期管理；原 norm/down 回归通过 | 工程与可审计性收益；本次未对 R1 单独做性能消融 |
| R2 | 相同算术/范围的 PTX 与显式 CUDA 双入口，补齐 down 对照 | 适配与等价验证能力；本次主候选为显式 CUDA，未判定 PTX/CUDA 普遍优劣 |
| R3 | 连续 lane 合并、非连续回退、完整张量/多层覆盖及模型内发布 | 完整 F1/F2 接入保持数值、覆盖与依赖合同，实际运行成本下降；逐 4 KiB 的当前组织方式未带来相对旧 F 的模拟延迟收益 |

R1/R2 的阶段成果沿用各自封存报告；本次完整 FFN 实验通过独立图回调驱动 R3 读取/发布路径，不等于重新验证所有 R1 注册策略与所有 PTX 核。后续若目标是降低模拟 TTFT/TPOT，应优先扩大可在途窗口、减少每 4 KiB 的核启动/scatter/同步、复用 plan 与发布缓冲区，并保持先读取完成再消费的依赖合同。这些尚未计入本轮收益。

本次覆盖完整 llama 推理中的声明 FFN 范围，不包含全部模型权重迁移、长生成或优化后的 KV 事务路径。

## 证据与复现

- [机器可读结果与验收](E:/v1_update/output/readonly-f-compare-a-20261108/comparison.json)
- [全部原始输出](E:/v1_update/output/readonly-f-compare-a-20261108)
- [运行条件](E:/v1_update/output/readonly-f-compare-a-20261108/preflight.json)、[墙钟批次条件](E:/v1_update/output/readonly-f-compare-a-20261108/wall3-preflight.json)、[进程观察](E:/v1_update/output/readonly-f-compare-a-20261108/wall3-process-observations.json)
- [候选构建/源码身份](E:/v1_update/output/readonly-f-compare-a-20261108/candidate-identity.json)
- [实验入口和复现顺序](E:/v1_update/experiments/readonly_f_compare/README.md)
- [R1](E:/v1_update/docs/READONLY_R1_RESULTS_20261107.md)、[R2 norm](E:/v1_update/docs/READONLY_R2_RESULTS_20261107.md)、[R2 down](E:/v1_update/docs/READONLY_R2_DOWN_RESULTS_20261107.md)、[R3](E:/v1_update/docs/READONLY_R3_RESULTS_20261108.md)
'''
(R/'docs/READONLY_F_COMPARE_20261108.md').write_text(s,encoding='utf-8')
(R/'docs/validation/readonly-f-compare-20261108.json').write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
print('Wrote report and validation JSON')
