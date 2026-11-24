# GPU_RA 的版本与分支

GPU_RA 将真实 GPU 上的权重和 KV 访问接入 HBF 介质仿真。三条版本分支分别保有自己的构建入口、适用范围与验收记录。

| 分支 | 实现与已验证能力 |
|---|---|
| `main` | FFN 权重访问计时。使用原生 F16 KV，支持单层、选定四层和全部 28 层的复现配置。 |
| `codex/v2-readonly` | 完整 FFN 权重读取与发布。使用原生 F16 KV，提供原生对照、GPU 轨迹采集和在线读取。 |
| `codex/v3-joint` | 两条已验证路径。按实际 GPU 地址对 311 个权重在线计时，使用原生 KV。选定 FFN 权重和分页 KV 也可共用一个介质服务。 |

从所在分支的 [README](README.md) 进入构建和最小运行。V3 实现位于 `joint/source/`。窗口读取、KV 舍入误差溯源等名称描述各自的用途。原实验 ID 和阶段参数保留在 [实验工作流](reproduction/EXPERIMENTS.md) 对应的历史配置中。

## 历史复现

| 版本 | 历史交付快照 |
|---|---|
| V1 | `70d0787fa749a292dd196bb587994f5a9eb0bf83` |
| V2 | `e890debd0d9d2c064bd4da9c715547f8d269e03c` |
| V3 | `f4b754c8e228064364bc25877dbb39c698a2c167` |

在独立工作树检出相应提交，可用该提交里的核验器检查原交付清单。各次实验另有固定的执行提交和报告提交。在当前分支运行 `python reproduction/verify_checkout.py` 可核对当前交付清单。

模型与归档载荷放在外置数据目录。使用 [数据索引](DATASETS.json) 和 [实验工作流](reproduction/EXPERIMENTS.md) 中的数据根配置恢复或核验。早期工程记录见 [2026-11-04 基线](docs/history/baseline-2026-11-04.md) 和 [KV 舍入误差溯源](docs/numerics/kv-rounding-diagnosis.md)。
