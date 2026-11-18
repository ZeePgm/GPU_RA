# V3 只读入口恢复

| 入口 | 配置与实际覆盖 |
|---|---|
| 第二次 F 完整入口 | `symbolic_f model.gguf prompt.txt output count timing config.json`。沿用原单节点调度、CUPTI 计算段采集与表回放、S3Gate、TensorAccessBridge 和在线审计。原 F1/F2 区域与 H0 配置可直接使用。 |
| 选定权重 symbolic | 主机对每个矩阵或每次实际 embedding 行消费提交一个范围父请求，原生 GPU 消费者使用原权重。图按单节点派发。有效范围按 64 B 对齐，Global 上限 4096 B。扩展布局只增大 H0 的块数。此入口测任务墙钟；带计算段及 token 模拟时钟的原 F 使用上行入口。 |
| PTX | 完整模型及组件的单个 F32 norm 逐行调用 R2 的 `m6_weight_mul` 算术核，2×128、每活动元素 4 B，原生输入与输出角色沿用 R2。模型需要连续的 hidden 宽度和 MUL 消费者。独立 `r2_down ptx modeled ffn repeats 0 down.ptx tokens result.json` 使用原 half2/F32 FMA、256 线程两级归约，覆盖第 0 行。完整 down、attention、embedding 和输出矩阵均超出 R2 PTX 原范围。 |
| CUDA | V3 B2 的 `gate` 和 `b2_kernel(true,false)`，K=256 KiB、W=32 KiB，64 个 warp 信用，每个请求 512 B。读完单个消费范围后一次发布。对 embedding 每个实际行单独调用。当前要求范围是完整 512 B warp 跨度，可处理非整 K 的末块。选择输出矩阵时，镜像峰值约 296.75 MiB。 |

选定模型入口为 `five_model model.gguf prompt.txt output selection.json outputs`。在清单 `all_tensors` 中用 `selected` 选择权重，`HBFLAB_SAMPLED_ROUTE` 取 `symbolic`、`ptx`、`cuda` 或原生参考 `native`。环境保留 `HBFLAB_JOINT_KV=native`、`HBFLAB_SAMPLED_TOKEN_ONLY=1`、`HBFLAB_SAMPLED_AUDIT=summary|full`、`HBFLAB_SAMPLED_PROFILE=<模型配置>`。symbolic 另设 `HBFLAB_SYMBOLIC_CONFIG=joint/source/experiments/gpu_ffn/H0-512MiB.ini`。GPU 图、融合、算术精度及资源守卫设置参见本目录 smoke.py 所用的原 V3 启动器。

独立权重使用 `five_weight model.gguf selection.json fixture.json output route repeats summary`，profile 和 symbolic 配置与模型入口相同。原 V3 B2 联合 KV 模式仍由 `realgpu_llama` 驱动。选定权重入口此次验收使用 native KV。历史 sampled 候选与对应 runner 在其封存源码 A 上复现。

本目录 build.ps1 从当前工作树构建。smoke.py 的四个 case 通过仓库的 new/start/finish/verify 流程管理，逐个启动。测试只验证实现、数据、依赖和排空；正式性能矩阵另行决定。
