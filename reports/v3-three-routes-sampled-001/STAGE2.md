# 阶段2：公共注册与历史回归

状态：公共注册、资源及原F1/F2回归通过；三入口实际读取和代表权重模型接入继续由阶段3/4验收。

公共注册运行：`stage2-registration-20261116-002`，代码A=`7eac3c17a096d878ed68139558cb9e367151d681`。

- 三种入口显式选择，无效值拒绝；每种入口独立注册14个实际尺寸范围，共653,800,448 B。
- 按32KiB预留后初始化权重653,950,976 B；KV基址1,506,836,480，KV预留58,720,256 B，逻辑介质2GiB。
- 每种入口25项负例通过；覆盖容量、物理/逻辑重叠、边界、错误内存、错误owner、启动顺序、注册上限、无效对齐和失败后的终止状态。
- 共享读写读及排空通过，Host驻留峰值256；未扩大Host驻留预算。
- 原608 CPU回归、28个B2 GPU场景、6,244请求/199,808字节依赖检查、B2越界负例和共享介质夹具均通过。

历史功能回归运行：`stage2-regression-20261116-001`，代码A=`a8d720f8e199293fe8434e02e7f02cff26ed92d9`。

F1后F2，各八组独立进程：native、readonly bypass/modeled、KV bypass/modeled、joint bypass/modeled、joint host写来源，共16进程通过。使用未修改的`joint/scripts/run_correctness.py`，SHA-256为`3e93f076b1add9426309de741c062e7fd44b1721961663304ae89726d04ba318`。

同算术路径的完整logits逐字节一致，token一致；分页attention对native保留max_abs≤0.2、RMS≤0.025。原权重读取/发布、逐lane依赖、KV内容和前缀、事务、Host预算以及allocation/pin/dirty/lease/pinned正常排空合同全部通过。默认4KiB注册路径保持原行为，候选32KiB路径单列，不修改历史清单或阈值。

注册首批001因runner收据序列化错误失败；原始夹具与异常已独立封存，不计通过样本。未采集新的统计性能样本。
