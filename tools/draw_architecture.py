"""Generate the numbered implementation diagram and Chinese reference together."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'
A = 'adapters/llama_cpp/'
H = 'third_party/HBF-EventLab/'

# Title, diagram summary, core paths, implementation and ownership explanation.
MODULES = [
('配置与模型入口', 'GGUF / 能力校验 / 三模式',
 [A+'integrated_profile.cpp', A+'integrated_runner.cpp', 'tools/integrated_profile.py'],
 '从真实 GGUF 元数据和转换记录生成 profile，核对模型身份、形状、上下文及容量预算，装配 llama.cpp 并保存 token、logits 和摘要。为 02、04、07 提供配置。native 为原生参考；bypass 保留分页但绕过介质计时；modeled 启用后端计时。'),
('PTX 改写与模块管理', '选定访存 / ABI / 链接缓存',
 ['src/ptx.cpp', 'src/async_module.cpp', A+'integrated_ptx.cpp'],
 '校验支持的 PTX 子集，对选定只读访问插入辅助调用；管理隐藏参数 ABI、辅助 PTX 链接、函数查找与缓存。同步模块入口另见 src/module.cpp。由模型集成或 03 使用，改写后经 06 请求计时；不支持任意 PTX 指令的无条件改写。'),
('Windows 注入〔可选〕', 'Detours / CUDA Driver 拦截',
 ['src/inject/launcher.cpp', 'src/inject/dll.cpp', 'include/hbflab/injection.h'],
 'launcher 启动注入目标，DLL 通过 Detours 接入 CUDA Driver 模块加载与启动路径，管理错误、统计和线程递归保护。复用 02 与 11；不是 integrated_runner 的必经入口。'),
('真实 GPU 与权重接入', 'F16 / norm、down 选定访问',
 [A+'integrated_weights.cu', A+'integrated_weight_mul.cu', A+'integrated_kv.cu'],
 '在固定 llama.cpp CUDA 集成中执行真实数值运算。选定 norm 权重标量访问与 down 权重 warp 跨度进入计时；norm 的 PTX 接入由 02 实现。分页注意力经 07/08 取得真实 K/V。未选中张量走 05，选中访存经 06→11 申请许可。后端 DataToken 不生成真实计算结果。'),
('原生 CUDA 存储', '未选中张量 / 原生参考 KV',
 [A+'integrated_weights.cu', A+'integrated_runner.cpp'],
 '未被选择的模型张量继续使用原生 CUDA 分配；native 参考模式使用原生 GPU KV，分页模式将 KV 接入容量层。这是 04 的并行原生分支，不表示已实现全权重卸载或全部 CUDA 访问的 HBF 建模。'),
('设备侧异步协议', 'mailbox / grant / 依赖等待',
 ['include/hbflab/async_device.cuh', 'include/hbflab/async_protocol.hpp', 'src/async_ptx_helper.cu'],
 'GPU 通过有界映射 mailbox 发布稳定身份的请求，等待 grant、结果及依赖延迟后才继续使用真实数据；支持标量和连续 warp 读取。连接 04 与 11/12，错误阻止依赖计算；请求被接受不等于最终完成。'),
('分页 KV 与发布事务', '分层 K/V / append / commit',
 ['src/kv.cpp', 'include/hbflab/kv.hpp', A+'integrated_kv.cu'],
 '按 token block、layer、K/V 分配逻辑空间，begin_append/write_layer/commit 管理追加；各层 K/V 完成后才发布序列长度。未完成写入的 abort 会使缓存失效，不声称回滚。经 08 访问 GPU 页，09 管理 backing。通用 API 有多序列接口，当前模型验收仍限定单序列 F16 KV。'),
('GPU 页镜像与租约', '核边界分页 / H2D、D2H',
 ['src/capacity_gpu.cpp', 'include/hbflab/capacity_gpu.hpp'],
 'acquire 获取 GPU 页指针及租约；finish 等待指定 stream、回收写入字节并解除 pin，统计真实上传下载。连接 07 与 09。这是显式核边界换页，不是设备侧缺页异常或统一虚拟内存。'),
('容量与真实字节', '4 KiB / pin / 淘汰 / 写回',
 ['src/capacity.cpp', 'include/hbflab/capacity.hpp'],
 'CapacityStore 管理逻辑分配、有限页框、generation/revision、租约、淘汰和 flush。脏驻留页持有最新字节；成功完成建模写回后 backing 才取得相应权威数据。真实字节交给 10，建模读写经 CapacityMedia 交给 11；不可恢复错误后 poison，防止继续访问。'),
('Backing 与有界预取', '内存默认 / 文件、预取可选',
 ['src/page_file.cpp', 'src/capacity.cpp', 'src/capacity_gpu.cpp'],
 '默认内存 backing；PageFile 提供版本化文件、显式 checkpoint 和重开，只接受完整检查点，不是崩溃恢复日志。预算内同步预取不抢占 pinned/dirty 页；淘汰普通 clean 页需显式启用。向 09/08 提供真实字节，介质工作仍走共享计时链。'),
('共享计时所有者', 'LiveTimingService / 单线程',
 ['src/live_runtime.cpp', 'include/hbflab/live_runtime.hpp'],
 '一个绑定 CUDA 上下文的工作线程独占 AsyncRuntime 和后端，串行处理 GPU 请求、容量介质工作与采样；框架线程保有服务句柄及借用数据生命周期。06 与 09 的工作在此汇合，经 12 复用同一设备状态和时钟。'),
('异步请求调度', 'frontier / grant / 时间映射',
 ['src/async_runtime.cu', 'include/hbflab/async_runtime.hpp', 'include/hbflab/frontier.hpp'],
 '轮询 mailbox，维护有序到达、生产者 frontier、grant 和完成回填，将 GPU readiness 映射到模拟时间，保存 service/media audit。按合法水位向 13 提交，完成后通知 06；结束时检查排空。同步兼容路径另见 src/runtime.cu，不另建共享异步会话。'),
('TimingBackend 适配', '64 B 并集 / Global / 块快照',
 ['src/backend.cpp', 'include/hbflab/backend.hpp'],
 '将字节范围转成准确的 64 B sector 并集，仅合并同 DLU 内连续 sector，不覆盖空洞。管理父请求、Global 子请求、块内容身份和 RAW/WAR/WAW；页写回形成整逻辑块快照。仅经 14 提交 Global，所有子 Global 最终逻辑结果齐备后，以最晚完成时间汇总父请求；命令拆分和物理 replay 属于 16/17。'),
('在线 SimulationSession', '单时钟 / 因果反馈 / 关闭',
 [H+'src/runtime/simulation_session.cpp', H+'include/hbf_eventlab/runtime/simulation_session.hpp'],
 '拥有 EventQueue、原 HostDriver 和 DeviceDriver。advance_before(T) 仅执行严格早于 T 的事件，支持同 Tick 反馈、yield/resume、关闭与 Completed/Stalled/限额诊断。13 使用该在线入口，不逐请求调用独立同步 TensorAccessBridge。前端 end_stream 是暂时排空，finish 才永久关闭外部输入。'),
('事件队列与定向唤醒', '同 Tick 因果序 / 资源通知',
 [H+'src/infrastructure/event_queue.cpp', H+'src/runtime/admission_wait_registry.cpp', H+'include/hbf_eventlab/common/wait_reason.hpp'],
 'EventQueue 管理事件时间、同 Tick 顺序、取消及限额；AdmissionWaitRegistry 按原因与对象维护等待订阅。资源变化后异步唤醒相关等待者，避免无条件全量扫描。14、16、18～20 共用这些机制，通知不另加物理时延、不允许同步重入接收路径。'),
('Host 输入与发送', '驻留预算 / 索引 / 最终结果',
 [H+'src/host/host_driver.cpp', H+'src/host/host_input.cpp', H+'src/host/host_indices.cpp'],
 '同一 Host 支持离线/增量输入，GlobalRequestSplitter 拆分逻辑工作，发送策略选择合法候选。host_storage.cpp 管理活动片段预算和回收；host_indices.cpp 维护 Pending 重试、驻留引用及读顺序索引。经 IHostPort 与 18 交互，channel_send_blockers 预检 channel 统一阻塞，try_submit 最终判定；汇总最终 Global 后反馈至 14/13。'),
('Host 重试与恢复', '页重试 / token 保留 / replay',
 [H+'src/host/write_retry.cpp', H+'src/host/recovery_manager.cpp', H+'src/host/retention_store.cpp'],
 '根据设备响应管理失败尝试、页级重试、恢复范围和 replay；WriteRetentionStore 在有限预算内保留版本化符号 token，不保存真实 KV backing。它是 16 的恢复分支，正常请求不必经过。前端 replayed_pages 是普通快照页数，不等于 Host recovery replay 命令数。'),
('Device 传输与协调', 'UCIe / RX、TX / MOCS / Bank',
 [H+'src/runtime/device_driver.cpp', H+'src/runtime/command_admission.cpp', H+'include/hbf_eventlab/runtime/link_buffers.hpp'],
 '协调链路、有限 RX/TX、正式接受、共用 command credit、合法 NAND 候选与响应交付。区分 NotTaken、Taken、Accepted、issue、result、delivered；分派并列的 19/20，独立 Bank 可重叠。credit 在响应交付时释放；Host 活动预算、设备 MOCS 与 GPU frame 数互相独立。'),
('读取与 Bank Cache', 'Pending / 窗口 / 结果租约',
 [H+'src/runtime/read_path.cpp', H+'src/runtime/bank_cache.cpp', H+'src/policies/read_dispatch.cpp'],
 '处理 Pending 观察、逻辑依赖、读窗口资格、物理绑定、cache hit/sense 和结果生命周期。每 Bank 两个 4 KiB cache 槽，保护结果直到有限 Egress 接手；保留同组返回与同 Bank sense 顺序。18 调度合法读；资源变化定向唤醒重查。这里的符号 cache 不同于 08 的真实 GPU 镜像。'),
('写入、介质与映射', '组装 / erase、program / 退役',
 [H+'src/runtime/write_path.cpp', H+'src/runtime/nand_media_state.cpp', H+'src/runtime/capacity_retirement_coordinator.cpp'],
 'WriteAssembly 收齐完整 4 KiB 才形成 program 候选，维护接收/编程顺序及 erase 生命周期。NandMediaState 保存符号介质状态；ZoneMapping、BlockReplacement 与退役协调器管理故障映射和容量。与 19 并列，由 18 取得 Bank 后发射；重试/replay 策略属于 17，不搬运真实 CUDA 数据。'),
('环境与暴露〔可选〕', '字节能耗 / RC 温度 / exposure',
 ['src/environment.cpp', 'include/hbflab/environment.hpp', 'src/capacity.cpp'],
 '依据后端介质计数增量、模拟时间和调用者提供的 GPU 功率计算能耗、集总 RC 温度与温度加速暴露量，为 22 提供判断依据。采样使用共享时钟。参数未经物理校准，GPU watts 是输入，不是 kernel 墙钟推导的遥测。'),
('保留期刷新〔可选〕', 'maintain / READ + 块快照',
 ['src/capacity.cpp', 'src/capacity_gpu.cpp', A+'integrated_kv.cu'],
 'maintain 在预算内处理过期和待刷新页；被租用页延期，不强行夺取。显式 READ 加逻辑块快照消耗同一介质时间，并统计过期、阻塞和刷新字节。形成 21→22→09/11 维护反馈；需要显式服务调用，模型适配器可在层边界调用，环境默认关闭时不自动启用。'),
('证据、验收与排空', 'logits / 等待 / 内容 / failure',
 ['tools/run_llama_integrated.py', 'tools/check_llama_integrated.py', 'tools/validate_baseline.py'],
 '比较 native、paged bypass、modeled 的数值与 token，检查访问覆盖、依赖等待、KV 发布、Host 最终结果、预算及最终关闭排空。tests/backend_contract.cpp、tests/shared_backend.cpp、tests/kv_transaction.cpp 等覆盖正常/失败契约。观察 01～22，不参与调度；本次文档更新不等于重跑全部 GPU 验收。'),
('A1 / S 系列〔实验〕', '真实 GPU 证据 / 延迟传播',
 ['experiments/gpu_s/native_a1.cpp', 'experiments/gpu_s/a1_measure.py', 'experiments/gpu_s/a1_analyze.py'],
 '记录真实 GPU 执行与选定延迟注入，关联时钟契约、依赖传播、时间线和数值证据。S 系列提供阶段性原型，A1 提供机制实验与统计分析。复用相关接口，历史记录保留各自身份；不是产品默认调度模块，也不是物理 HBF 吞吐校准。'),
('A2 / A3〔独立实验〕', 'FIFO oracle / 固定 H0 响应',
 ['experiments/gpu_a2/fifo.cpp', 'experiments/gpu_a3/scan.cpp', 'experiments/gpu_a3/run.py'],
 'A2 使用真实 EventQueue 与实验性单服务台 FIFO，通过独立标量 oracle 核对结果，不实例化 Host/Device/H0。A3 固定 H0，启动新会话检查工作量、地址、内容、排空并分析尺寸响应。A2 复用 15，A3 复用后端；两者都不是 GPU 模型运行的必经路径。'),
]


def validate():
    assert len(MODULES) == 25
    for _, _, files, _ in MODULES:
        for file in files:
            assert (ROOT/file).is_file(), file


def render():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(27, 19))
    ax.set(xlim=(0,27), ylim=(0,19)); ax.axis('off')
    ink = '#172b3b'
    colors = ['#3168a0','#258276','#bf7c25','#9c5970','#72618c']
    fills = ['#edf4fc','#eaf6f2','#fff5e6','#fbf0f3','#f3f0f9']
    ax.text(.65,18.55,'前后端实现架构与核心代码',fontsize=25,weight='bold',color=ink)
    ax.text(.65,18.06,'RealGPUAcess_v0  |  当前工作区实现 · 2026-11-04  |  25 个编号与 ARCHITECTURE.md 一一对应',fontsize=12,color='#526878')
    lanes = ['A  模型与接入：真实 CUDA 执行，选择性访存建模',
             'B  协议与容量：真实字节、显式分页、KV 发布',
             'C  共享时序：GPU 请求与容量介质工作使用同一所有者',
             'D  HBF 后端：Host 命令、有限资源、符号介质',
             'E  维护与证据：可选维护及独立实验，各列不代表串行流水线']
    positions = {}
    references = {
        1: '配置 → 02 / 04 / 07', 2: '改写接入 → 04',
        3: '复用 02 / 11', 4: '访存 → 06；分页 KV → 07',
        15: '供 14 / 16 / 18～20 共用',
        20: '介质计数经共享采样 → 21',
        21: '输入：09 / 11 的时钟与介质计数',
        22: '维护反馈 → 09 / 11（显式调用）',
        23: '观察 01～22；不参与调度',
        24: '复用 GPU / 时序接口；独立实验入口',
        25: 'A2 → 15；A3 → 14 / 16 / 18',
    }
    for row in range(5):
        top = 17.25-row*3.15
        ax.text(.65,top+.08,lanes[row],fontsize=13,color=colors[row],weight='bold',
                bbox=dict(facecolor='white',edgecolor='none',pad=2),zorder=5)
        ax.axhline(top-.13,xmin=.024,xmax=.975,color=colors[row],alpha=.22)
        for col in range(5):
            idx = row*5+col+1
            x,y,w,h = .65+col*5.2,top-2.65,4.82,2.22
            positions[idx]=(x,y,w,h)
            title,sub,files,_=MODULES[idx-1]
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=.1',edgecolor=colors[row],facecolor=fills[row],lw=1.2,zorder=3))
            ax.text(x+.17,y+h-.21,f'{idx:02d}  {title}',fontsize=12.2,weight='bold',color=ink,va='top',zorder=4)
            ax.text(x+.17,y+h-.70,sub,fontsize=10.5,color=colors[row],va='top',zorder=4)
            for i,file in enumerate(files):
                short=file.replace(A,'A/').replace(H+'src/','H/').replace(H+'include/hbf_eventlab/','HI/')
                ax.text(x+.17,y+h-1.18-i*.29,short,fontsize=8.6,color=ink,va='top',zorder=4)
            if idx in references:
                ax.text(x+.17,y+.17,references[idx],fontsize=8.5,color=colors[row],va='bottom',zorder=4)
    def port(n,side):
        x,y,w,h=positions[n]
        return {'l':(x,y+h/2),'r':(x+w,y+h/2),'t':(x+w/2,y+h),'b':(x+w/2,y)}[side]
    def edge(points,color,dashed=False,both=False):
        for p,q in zip(points[:-2],points[1:-1]):
            ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=1.35,ls='--' if dashed else '-',zorder=2)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='<->' if both else '-|>',mutation_scale=12,color=color,lw=1.35,linestyle='--' if dashed else '-',zorder=2))
    def horizontal(a,b,row,both=False,dashed=False):
        edge([port(a,'r'),port(b,'l')],colors[row],dashed,both)
    horizontal(1,2,0,dashed=True)
    edge([port(3,'l'),port(2,'r')],colors[0],True)
    horizontal(4,5,0)
    for a,b in [(7,8),(8,9),(9,10)]: horizontal(a,b,1,True)
    for a,b in [(11,12),(12,13),(13,14)]: horizontal(a,b,2,True)
    horizontal(14,15,2)
    horizontal(16,17,3,True)
    horizontal(18,19,3)
    p,q=port(18,'b'),port(20,'b')
    edge([p,(p[0],p[1]-.20),(q[0],p[1]-.20),q],colors[3])
    p,q=port(16,'t'),port(18,'t')
    edge([p,(p[0],p[1]+.18),(q[0],p[1]+.18),q],colors[3])
    horizontal(21,22,4)
    edge([port(6,'b'),port(11,'t')],colors[2])
    p,q=port(9,'b'),port(11,'t');mid=p[1]-.25
    edge([p,(p[0],mid),(q[0],mid),q],colors[1])
    ax.text(12.8,mid+.04,'容量介质请求 → 11',fontsize=9,color=colors[1],ha='center')
    p,q=port(14,'b'),port(16,'t');mid=p[1]-.27
    edge([p,(p[0],mid),(q[0],mid),q],colors[2])
    ax.text(9.8,mid+.04,'Global 输入 / 最终完成反馈',fontsize=9,color=colors[2],ha='center')
    p,q=port(4,'t'),port(6,'l')
    edge([p,(p[0],17.04),(.26,17.04),(.26,q[1]),q],colors[0],True)
    ax.text(.65,1.05,'图例  虚线：配置 / 改写 / 接入关联    实线：主要调用、请求或数据关系    双向：请求与结果 / 页数据交互',fontsize=11,color=ink)
    ax.text(.65,.69,'文件前缀  A/ = adapters/llama_cpp/    H/ = third_party/HBF-EventLab/src/    HI/ = third_party/HBF-EventLab/include/hbf_eventlab/',fontsize=10,color=ink)
    ax.text(.65,.33,'数据边界  CUDA、CapacityStore 与 backing 持有真实字节；HBF 后端模拟行为与时序。跨层完整关系及可选开关见说明文档。',fontsize=11,color='#526878')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    for ext in ('png','svg','pdf'):
        fig.savefig(OUT/f'architecture.{ext}',dpi=200,facecolor='white')
    plt.close(fig)


def write_reference():
    lines = ['# 前后端架构与编号说明','',
        '更新时间：2026-11-04。依据当前工作区源代码，包含已有未提交的实现。“前端”指 CUDA/llama.cpp 执行与接入层，“后端”指 HBF-EventLab 离散事件模拟层，不是 Web 前后端。','',
        '![当前架构图](architecture.png)','',
        '[原尺寸 PNG](architecture.png) · [可缩放 SVG](architecture.svg) · [矢量 PDF](architecture.pdf)。25 个功能方框均有唯一编号；彩色分区标题只用于分层。图中只画主要关系，完整交互见下文。','',
        '## 1. 主链路与数据边界','',
        '```text','配置 01 → 权重/PTX 接入 02、04 → 协议 06 → 共享服务 11',
        'KV 07 ↔ GPU 页 08 ↔ CapacityStore 09 → 共享服务 11',
        '                         ↕                  ↓',
        '                     backing 10       AsyncRuntime 12',
        '                                            ↓',
        'TimingBackend 13 → Session 14 → Host 16 → Device 18 → 读 19 / 写 20',
        '       ↑                        ↓',
        '       └──── 最终 Global 结果 ────┘ → 父请求完成 → 06 解除依赖','```','',
        '- **真实数据**由 CUDA 分配、GPU 页镜像、CapacityStore 脏页与 backing 按租约/写回状态持有，H2D/D2H 是真实传输。',
        '- **模拟行为与时序**由 13～20 负责。DataToken 是符号内容证据，不是 payload 或页哈希，后端不向 GPU 返回模型计算字节。',
        '- **独立预算**包括 GPU frame、前端父请求数、Host resident_fragments、Device MOCS；活动预算不限制全部历史记录内存。',
        '- **时间解释**保留 GPU readiness/timestamp 驱动策略；程序墙钟不能直接作为物理 HBF 的 TTFT/TPOT 或吞吐预测。','',
        '## 2. 编号索引','', '| 编号 | 方框 | 分层 |','|---|---|---|']
    layers=['模型与接入','协议与容量','共享时序','HBF 后端','维护与证据']
    for i,(title,*_) in enumerate(MODULES,1):
        lines.append(f'| {i:02d} | [{title}](#module-{i:02d}) | {layers[(i-1)//5]} |')
    lines += ['', '图中 A/、H/、HI/ 为节省空间的前缀；以下核心代码提供完整仓库相对路径及链接。','', '## 3. 每个方框的实现说明','']
    for i,(title,sub,files,detail) in enumerate(MODULES,1):
        lines += [f'<a id="module-{i:02d}"></a>',f'### {i:02d}　{title}','',detail,'','**核心代码**：','']
        lines += [f'- [{file}](../{file})' for file in files]
        lines += ['']
    lines += ['## 4. 跨层关系与完成语义','',
        '| 关系 | 传递内容与规则 |','|---|---|',
        '| 01 → 02/04/07 | 验证后的模型形状、tensor 选择、KV 几何和预算。 |',
        '| 03 → 02/11 | 可选 Driver 接入，复用现有模块和服务。 |',
        '| 04 → 06 → 11 → 12 → 13 | 选定访存、到达时间、grant 与依赖状态。 |',
        '| 07 ↔ 08 ↔ 09 ↔ 10 | 真实 KV 页、租约与 backing；事务 commit 后发布长度。 |',
        '| 09 → 11 → 12/13 | 缺页读、写回和预取的建模介质工作，与权重共享会话。 |',
        '| 13 → 14 → 16 | Global、块快照及水位；不绕过 Host 投 Device 命令。 |',
        '| 14 使用 15 | 同一事件时钟与定向资源通知。 |',
        '| 16 ↔ 18 | 命令移交、接受与最终响应；容量不足由原所有者保留工作。 |',
        '| 16 ↔ 17 | 失败时的重试与恢复，正常命令可直接走 18。 |',
        '| 18 → 19 / 20 | 并列读写分支，共享链路、credit 与 Bank。 |',
        '| 16 → 14 → 13 → 12 → 06 | 子 Global 全部得到最终逻辑结果后父请求才完成，依赖计算才继续。 |',
        '| 11/09 → 21 → 22 → 09/11 | 可选计数采样、暴露量与显式刷新反馈。 |',
        '| 23 / 24 / 25 | 验证与独立实验，不串入产品默认运行路径。 |','',
        '## 5. 默认、可选能力与限制','',
        '| 项目 | 当前含义 |','|---|---|',
        '| 模型验收 | Qwen3-0.6B、Qwen3-1.7B、SmolLM2-135M；dense Qwen3/llama，单序列，F16 选定 norm/down 权重及全层分页 KV。 |',
        '| 在线主路径 | 共享服务 → TimingBackend → SimulationSession → 原 Host/Device；同步 TensorAccessBridge 保留作独立不可变权重参考。 |',
        '| 默认策略 | Host/NAND FCFS、内存 backing、不附加完整历史统计 observer；其他策略接口不表示默认开启。 |',
        '| 可选能力 | 注入、文件、预取、环境与显式刷新已有实现和针对性测试，不表示全选项在同一次模型验收中同时启用。 |',
        '| 两种保留机制 | 21/22 的温度暴露与保留期刷新，不同于 17 中的 Host 恢复 token 保留区。 |',
        '| 实验边界 | A2 不实例化完整后端；A3 是固定 H0 的模型响应；A1/S 历史计时保留各自来源身份。 |',
        '| 未覆盖 | 物理 HBF、全权重卸载、量化/MoE、多序列服务、vLLM、完整多 stream 重叠建模、新的 measured-compute TTFT/TPOT 执行器。 |','',
        '## 6. 相比旧图的变化与核对依据','',
        '保留“真实 CUDA + 容量 backing + 共享在线模拟”的边界，拆开设备协议/主机调度、KV/GPU 镜像、Host/恢复、Device/读写路径。新增可选注入、定向唤醒、Host 索引与发送预检、有限 RX/TX/credit、映射退役和独立实验层；每个方框加入编号与核心文件。','',
        '依据上述代码、[适配契约](INCREMENTAL_BACKEND.md)、[当前基线](BASELINE_20261104.md)、[后端架构](../third_party/HBF-EventLab/docs/architecture.md) 和 [README](../README.md)。基线文档记录后端上游版本 `8ccb1f353630f893f761b017d621812b7b9d5bf2`；本次没有重跑源码来源哈希或完整 GPU 验收。','',
        '本次文档检查：25 个编号唯一、图文使用同一模块清单、核心文件与相对链接存在、PNG/SVG/PDF 可生成并检查排版。仅更新架构文档和绘图脚本，不更改运行实现。','',
        '## 7. 重新生成','',
        '依赖见 requirements-dev.txt。在仓库根目录执行：','',
        '```powershell',r'.\.venv\Scripts\python.exe tools/draw_architecture.py','```','',
        '[绘图源文件](../tools/draw_architecture.py) 中的 MODULES 是编号、文件与解释的共同来源，运行后生成 PNG/SVG/PDF 和本文。Windows 使用微软雅黑；其他平台需配置可显示中文的 Matplotlib 字体。','']
    content='\n'.join(lines)
    for target in re.findall(r'\]\((\.\./[^)]+)\)',content):
        assert (OUT/target).is_file(),target
    (OUT/'ARCHITECTURE.md').write_text(content,encoding='utf-8',newline='\n')


if __name__ == '__main__':
    validate()
    render()
    write_reference()
    print('Generated PNG/SVG/PDF and Chinese reference: 25 modules, core paths verified.')
