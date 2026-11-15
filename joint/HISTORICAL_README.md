# 双线联合正确性固化版本

范围：Qwen3-0.6B F16，固定原始 CRLF prompt、44 输入、8 输出、单序列。先 F1（层 10 的 gate/up/down），再 F2（层 0/10/20/27）。仅做正确性验收，不做统计性能确认。

来源：K5 R3 封存源码及 K5 R2 后端 overlay；B2 最终封存读取核与验收逻辑。来源逐文件哈希见 `provenance/source-origin.json`，任务起点见 `provenance/task.json`。原工作区源码与历史输出保持原样。

共同配置：K256 KiB、W32 KiB；2048 mailbox、1 个活动读取 launch、513 条审计缓存；Host 256 个 64B fragment；4 页/块、每 Bank 32768 块、总介质容量 1 GiB；完整 FFN 地址布局 504 MiB，KV 地址区紧随其后。未选权重仅保留逻辑间隙，不初始化。mirror 6 MiB，验证缓冲 256 KiB。

请求/服务超时固定为 1/60 秒。页框与 Host 驻留预算保持独立。初始化、真实模型计算及额外正确性审计均完整执行；进程耗时作为执行成本记录。

KV 使用 K1 device 写来源、K4 可取消追加及 K5 async 分页、coalesced 服务、异步副本；4 个 GPU 页框、4 个 pinned 槽。host 写来源用于同算术与真实 KV 字节对照。K2 flush 合并和 K5 prefetch 本轮不启用，不据本次结果扩展其通过范围。

所有组使用 F32 MUL_MAT，关闭 graphs/fusion/flash。完整 FFN 使用原矩阵乘算子，旧 norm/down 小范围 hook 关闭。权重和 KV 使用同一服务，分别验证 GPU 读取与介质请求计数。

## 运行

构建：`scripts/build.ps1`。组件验证：`scripts/build_tests.ps1`。模型矩阵：使用含 numpy 的 Python 执行 `scripts/run_correctness.py --scale F1 --batch <新批次>`，通过后执行相同批次的 `--scale F2`。

每规模包含 native、只读 bypass/modeled、纯 KV bypass/modeled、联合 bypass/modeled，以及联合 host 写来源，共 8 个独立进程。所有输出目录拒绝覆盖。原始失败及中断单独保留。

模型路径和 SHA-256 位于 `configs/qwen3-0.6b.json`；工具链位于构建脚本。外部依赖是 Windows x64、VS2022、CUDA 13.0 和该 GGUF。源码树包含 llama.cpp、HBF-EventLab 和所有链接代码；不使用工作区旧 exe/lib/obj 作为构建输入。

## 验收

完整输入/输出 token 一致。只读对 native 及同算术组间 logits 逐字节一致；分页 attention 对 native 使用原 profile 的 max_abs≤0.2、RMS≤0.025 合同。每个选定矩阵每步完整读取并发布，逐 lane 消费等待通过。分页组最终 KV 全内容和 44-token 前缀哈希一致，正常结束 allocation/pin/dirty/lease/pinned 全部为零。

状态及实际耗时以最终 `reports/RESULTS.md` 和机器验收文件为准。

封存后运行 `scripts/verify.py` 可只读核对源码、二进制及证据清单。需要修改时建立新修订目录，保留本目录及已产生的输出。
