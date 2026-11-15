# 工作区回答准则

本项目的实验对象是真实 GPU 接入的 HBF 仿真。当前没有 HBF 硬件，这是工作区的默认背景。

所有面向用户的回答（含进度更新和最终回复）都直接使用这一背景陈述结果。省略「不代表硬件收益」「需要硬件校准」及同类防御性说明，也无需在每次回答中重申默认背景。仅在用户明确询问相关硬件问题时展开。报告实验时，照常说明影响当前结论的具体条件、测量不确定性与判定结果。

## 实验工作规范

- 按 [实验入口说明](reproduction/EXPERIMENTS.md) 使用 `python reproduction/experiment.py new/start/finish/verify`。实验 ID 唯一，运行 ID 不复用；实验分支为 `codex/exp-ID`，默认独立工作树为 `GPU_RA/experiments/ID`。
- 指定版本基线：v1=`main`，v2=`codex/v2-readonly`，v3=`codex/v3-joint`。V3 的真实实现是 `joint/source`；不要修改继承的第一版根目录 `src` 来替代第三版实现。
- 先把源码、脚本、配置及 `experiments/ID/{experiment.json,config.json}` 提交为 A，保持工作树干净后再 `start`。外部运行根为 `GPU_RA-local-data/runs/ID/RUN`，保存 `run.json`、配置快照、生成脚本和日志；数据根目录是其同级 `bundles`，本机配置保持 Git 忽略。
- `start --id ID --run RUN [--input FILE ...] -- 实际命令` 只冻结代码提交、输入和参数，生成运行根下的 `launch.ps1` 后退出。随后直接用 PowerShell 7 执行 `pwsh -NoProfile -File <launch.ps1路径>`；启动脚本统一生成，不手工另写。模型前进行预检、模型后进行身份收尾，模型计时期间不驻留额外 Python 控制包装器，保持原科学资源守卫不变。
- `start` 支持整参数占位符 `{run_dir}`（尚不存在的外部 `data/`）、`{resource_root}`（完整扫描根）、`{config}`（已提交配置），记录模板和实际 argv。直接登记原科学 runner，例如 `-- python path/to/actual_runner.py --output "{run_dir}" --resource-root "{resource_root}"`，按其实际参数填写；不要再用一个长期驻留的 Python 脚本调用原 Python runner。
- 启动脚本提供 `GPU_RA_RUN_DIR` 和 `GPU_RA_RESOURCE_ROOT`。自定义科学程序可在运行时读取环境并创建输出目录；不要提前建 `data/`，也不要由父 PowerShell 提前展开变量。失败与中断记录保留，旧运行不覆盖；源码或条件变化后使用新提交、新运行，不混合样本。
- 完整资源根优先为外层原工作区 Git 根，否则为共同工作树管理根。直接登记原 V3 科学 runner 时必须显式传 `--resource-root "{resource_root}"`；不能沿用实验子工作树导致缩小的默认范围，也不能修改原守卫或利用目录迁移漏检其他进程。
- `--input` 的输入快照存于外部共享 `inputs/<sha256>`，运行 `run.json` 记录哈希和大小。`finish` 更新 `DATASETS.json` 的 `experiment_runs` 索引，并写入一次性的外部 seals 封存锚点；入口不自动 commit/push。
- `finish --id ID --run RUN --verdict passed|failed|inconclusive --conclusion TEXT` 生成 `reports/ID/RUN.{json,md}` 和 `evidence/ID/RUN/manifest.json`。报告提交 B 引用运行时的 A，不伪造自指提交。Git 只保存元数据、报告和清单；raw、模型、构建产物、缓存及数据归档载荷留在仓库外。
- `verify --id ID --run RUN` 核查可追溯性；不能把哈希检查当成科学通过，也不自动批准结论、调整验收阈值。验证后通过 merge/PR 合回所属版本；第二、三版不合入 `main`。
- 历史 manifest、冻结验收器及原脚本的身份校验保持不变；不能为了新源码通过而改写旧清单或放宽规则。修改实现的实验使用自己的候选源码/构建身份及验收入口。

- 最终交付前，`verify` 必须显示 `report_committed=true`，并给出报告提交 B；未提交报告时的身份核验只作预检。
