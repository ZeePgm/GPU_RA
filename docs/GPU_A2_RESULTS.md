# A2：受控FIFO机制与独立解析对拍

日期：2026-11-03；批次 `output/gpu-a2/20261103-fifo-001/`。**A2确定性FIFO机制验证通过：938个案例、15,058个请求逐项一致，最大时间残差0ps。** 验证对象是仓库真实EventQueue加实验层单服务台FIFO适配器；没有实例化HostDriver、DeviceDriver或H0，不构成H0等待模型验证。依据：[结果摘要](validation/gpu-a2.json)、[冻结协议](validation/gpu-a2/protocol.json)、[逐案例结果](validation/gpu-a2/cases.csv)。

## 1. 设计与被测实现

按[A组实验设计第3节](GPU_A1_A3_EXPERIMENT_DESIGN.md)，本轮采用可精确对拍的单服务台、FIFO、非抢占、整数ps机制。服务时间由测试输入给定，不来自GPU测量或H0参数；无重试、无其他竞争资源。FIFO队列、busy状态和完成后继续服务由实验适配器实现，事件调度与模拟时钟使用仓库已有[EventQueue头文件](../third_party/HBF-EventLab/include/hbf_eventlab/infrastructure/event_queue.hpp)及[生产实现](../third_party/HBF-EventLab/src/infrastructure/event_queue.cpp)。这两份原文件未作实验修改，编译时直接纳入原cpp翻译单元；没有另写一个替代EventQueue。

每个请求的到达事件入队，服务器空闲时启动队首并登记未来完成事件；完成后释放服务器、继续下一请求。观测所有arrival/start/complete及事件round。源代码：[FIFO适配器](../experiments/gpu_a2/fifo.cpp)。本轮不是将H0配置简化为FIFO，而是明确建立独立机制案例，仅覆盖实际使用到的EventQueue功能。

同刻到达按输入位置稳定排序；已有同tick round内Completion先于Arrival，Deadline在普通事件轮次排空后处理；Arrival回调中新登记的同tick Completion进入后续round。零服务时长合法，允许同tick完成但不凭空添加1ps。输入与计算不允许uint64溢出。规则依据上述EventQueue实现，并由本轮专项probe实际核对。

## 2. 独立参考和验收

Python参考仅读取原始输入，按到达时刻和输入位置排序，以标量递推计算：

- b_i=max(a_i,c_(i−1))；初始服务器空闲，c_0=0。
- c_i=b_i+s_i；w_i=b_i−a_i。

参考没有事件队列，不调用被测调度代码，也不读取实际输出生成期望。协议、输入与完整reference.json在执行C++程序之前保存。[独立参考及审计脚本](../experiments/gpu_a2/run.py)。

逐请求核对ID、到达、服务、开始、完成、等待及完成顺序；再逐事件核对时刻单调、状态迁移、单服务台无重叠、每请求恰好三个记录及最终排空。全部字段精确一致，不能用均值抵消误差。事件队列每请求执行一个到达和一个完成事件，start为回调内记录，故执行事件数为2N、审计记录数为3N。两次运行的完整结果文件SHA256必须相同。[冻结协议](validation/gpu-a2/protocol.json)。

## 3. 正向结果

| 测试集合 | 案例 | 请求 | 覆盖 |
|---|---:|---:|---|
| 命名边界案例 | 9 | 71 | 空请求、单请求、空闲间隔、同刻到达、积压、恰在完成时刻到达、无序输入、零服务链、uint64边界 |
| 三请求小空间穷举 | 729 | 2,187 | 每个到达取0/1/2ps，每个服务取0/1/3ps，覆盖全部组合 |
| 固定种子随机轨迹 | 200 | 12,800 | 每轨迹64请求，到达0..299ps，服务取0/1/2/10/100ps，种子2026110300..2026110499 |
| 合计 | 938 | 15,058 | 单次执行审计45,174条记录 |

所有请求字段与独立解析结果一致，最大绝对时间残差0ps，所有案例排空；两次完整执行输出逐字节相同。穷举范围仅为上述有限空间，随机例也不是统计证明所有输入正确。证据：[逐案例CSV](validation/gpu-a2/cases.csv)、[九个边界案例原始轨迹副本](validation/gpu-a2/boundary-traces.json)、[结果与输出哈希](validation/gpu-a2.json)。

## 4. EventQueue专项与负例

专项probe实际调用生产API，取得以下结果：[详细结构化记录](validation/gpu-a2.json)。

| 检查 | 观察结果 |
|---|---|
| 先登记Arrival、后登记Completion，同tick | 执行为Completion→Arrival→Deadline |
| Arrival中新增同tick Completion | 两个原Arrival先执行，再执行新增Completion；最终tick仍为0 |
| 取消 | 首次取消成功、再次取消失败；回调未执行，无待决项 |
| 空闲推进 | 推进至事件端点不执行该事件；越过活事件、回退时钟及登记过去事件均拒绝 |
| 事件预算与恢复 | 执行1个后限额停止，保留1个待决项且tick=1；显式提高预算后完成第二个，tick=2 |

独立审计器对7种实际结果副本篡改全部拒绝：完成顺序错误、重复请求结果、漏完成事件、完成时间错误、等待错误、提前start及未知事件。这些为构造负控，不声称生产引擎真实发生这些错误。

真实C++入口拒绝6类输入：负到达、负服务、小数服务、重复ID、完成时间溢出、非整数到达。均非零退出、未生成成功结果，stderr日志保留。结果来源于批次negative-audit、negative-input目录及[摘要](validation/gpu-a2.json)。

## 5. 可以得出的结论

在本轮受控机制、合法整数输入及已测案例内，生产EventQueue驱动的FIFO适配器产生的逐请求时刻和顺序与独立解析递推一致；同刻阶段/新轮次、空闲边界、取消与预算恢复的已测行为符合预期。结果支持这些机制和时钟行为的实现正确性，不证明所有EventQueue功能或所有输入均无错误。

FIFO服务模型属于实验适配器；Host发送、Device资源、NAND并行、重试和H0服务时延均未进入本实验。因此不能从0ps残差推导H0等待正确或真实硬件参数准确。可选M/G/1统计扩展本轮未执行，不报告排队论稳态均值或置信区间。该边界与[A2设计](GPU_A1_A3_EXPERIMENT_DESIGN.md)一致。

## 6. 构建、复现和证据保留

本轮运行不需要GPU/CUDA。独立CMake配置在本机MSVC编译器探测时遇到调试/PDB组件不匹配，失败日志保留；最终使用MSVC直接以`/std:c++17 /O2 /EHsc /MD /utf-8`编译实验适配器和原event_queue.cpp。失败构建未产生实验验收结果，最终二进制在协议中绑定SHA256。

在Visual Studio x64开发者PowerShell运行：

```powershell
./experiments/gpu_a2/build.ps1
python experiments/gpu_a2/run.py --study output/gpu-a2/<new-batch>
```

必须使用新目录且不要使用Python `-O`，因为验收断言需启用。详见[执行说明](../experiments/gpu_a2/README.md)。批次保存input.json、reference.json、两次actual、专项和负控、原始源码/二进制快照、构建日志及artifact-index.json；仓库保存报告、脚本、结构化摘要、逐案例CSV和边界轨迹副本。该实验没有生成GPU数值快照，未上传GitHub。
