# 后续实验与仓库协作

每个实验固定所属版本、代码身份、配置和原始数据；统一通过 `python reproduction/experiment.py new/start/finish/verify` 登记。原始数据留在本地外部目录，Git 保存可审查的源码、配置、报告与数据清单。

## 版本与目录

| 版本 | 派生基线 | 实现位置 |
|---|---|---|
| v1 | `main` | 第一版根目录源码 |
| v2 | `codex/v2-readonly` | 第二版源码及 `reproduce/` 入口 |
| v3 | `codex/v3-joint` | `joint/source`；构建/运行入口在 `joint/reproduction/` |

第三版继承的第一版根目录 `src` 不代表第三版实现。实验 ID 不能重复，运行 ID 不复用；每个实验在 `codex/exp-ID` 分支、独立工作树中进行，当前三工作树布局下默认位置为 `GPU_RA/experiments/ID`。

| 内容 | 位置 |
|---|---|
| 实验定义与配置，进入 Git | `experiments/ID/experiment.json`、`config.json` |
| 每次运行报告，进入 Git | `reports/ID/RUN.json`、`RUN.md` |
| 原始文件身份清单，进入 Git | `evidence/ID/RUN/manifest.json` |
| 运行索引，进入 Git | `DATASETS.json` 的 `experiment_runs` |
| `run.json`、配置快照、`launch.ps1` 和日志，留本地 | `GPU_RA-local-data/runs/ID/RUN/` |
| 模型/程序原始输出，留本地 | `GPU_RA-local-data/runs/ID/RUN/data/` |
| 输入快照，外部共享 CAS | `inputs/<sha256>` |
| 归档载荷，留本地 | `GPU_RA-local-data/bundles/` 下对应目录 |

配置的 `data_root` 指向外部 `bundles`，其同级 `runs` 保存新运行。可通过 `GPU_RA_DATA_ROOT` 或被 Git 忽略的 `data.local.json` 设置本机路径；格式见根目录 `data.example.json`。不上传 raw、模型权重、构建产物、缓存和数据归档载荷。

Windows 新克隆建议将长路径设置持久化到克隆配置：

```powershell
git clone --config core.longpaths=true https://github.com/ZeePgm/GPU_RA.git GPU_RA
```

## 1. 建立实验

先更新所需版本基线，再从任意 GPU_RA 工作树创建实验，例如：

```powershell
$env:GPU_RA_DATA_ROOT = 'E:/v1_update/repository-prep/GPU_RA-local-data/bundles'
python reproduction/experiment.py new --id joint-ab-001 --version v3 --title '联合路径对照' --purpose '固定条件下比较指定路径'
```

`new` 创建实验分支、工作树和定义/配置。`--worktree-root` 可覆盖默认工作树父目录，`--data-root` 可显式指定外部数据根目录。进入该独立工作树后，修改本轮源码、脚本与配置；写清输入、对照、测量端点、验收条件和统计规则。入口不自动 commit/push。

修改实现时，建立本实验候选源码/构建身份和相应验收入口。历史 manifest、冻结验收器及原运行脚本的身份校验继续保持；不能为新源码过关而重写历史清单、跳过原身份条件或放宽验收规则。

## 2. 先提交 A，再登记运行

按实际路径暂存本轮源码、运行脚本、配置和实验定义，提交为 **A**。`git status --short` 应为空；构建产物和本机数据配置保持忽略。

```powershell
git commit -m 'Prepare joint-ab-001 experiment'
git status --short
```

直接登记已经提交的科学运行命令。下面的 `path/to/actual_runner.py` 代表实际 runner；按其支持的参数填写，并补齐模型、CUDA 等必要输入，不再额外编写一个 Python 程序调用原 Python runner：

```powershell
python reproduction/experiment.py start --id joint-ab-001 --run run-20261115-001 --input experiments/joint-ab-001/config.json -- python path/to/actual_runner.py --output "{run_dir}" --resource-root "{resource_root}"
```

`--input FILE` 可以重复，登记模型、数据文件等实际输入的身份，并把输入快照保存到外部共享 CAS `inputs/<sha256>`；运行 `run.json` 记录每项输入的哈希、大小和快照身份。`start` 只冻结 A、输入及参数，在外部运行根中保存 `run.json`、配置快照和统一生成的 `launch.ps1`，然后退出，尚未执行科学运行命令。

`start` 支持三个**整参数占位符**：`"{run_dir}"` 为外部运行根下尚不存在的 `data/`，`"{resource_root}"` 为完整资源扫描根，`"{config}"` 为已提交的实验配置路径。占位符单独占一个参数，不与选项或其他字符串拼接；登记时解析为实际路径，同时保存模板 argv 和实际 argv。

## 3. 直接执行生成的启动脚本

下一步由 PowerShell 7 直接执行 `start` 输出的脚本路径。例如上述数据根目录下：

```powershell
pwsh -NoProfile -File 'E:/v1_update/repository-prep/GPU_RA-local-data/runs/joint-ab-001/run-20261115-001/launch.ps1'
```

`launch.ps1` 是统一入口生成的启动脚本，不需要人工编写。它设置 `GPU_RA_RUN_DIR=<外部运行根>/data`；运行根已经存在，但 `data/` 在程序启动前保持不存在，以便原有 runner 自行创建并继续执行“拒绝已有输出目录”的规则。

调用既有 runner 时直接通过登记命令中的 `--output "{run_dir}"` 传入输出位置。自定义科学程序可在运行时通过 `os.environ['GPU_RA_RUN_DIR']` 等环境接口取路径，并自行创建目录；若它再调用原 Python runner，会增加一层驻留父进程，应改为直接登记原 runner 和占位符。不要预先创建 `data/`，也不要在父 PowerShell 中提前展开该环境变量。

启动脚本只在模型前调用内部预检、在模型后调用身份收尾；模型计时期间不驻留额外 Python 控制包装器。这样保持原有 V3 科学资源守卫，无需修改它来接受工作流辅助进程。直接登记本轮科学运行入口，不再额外套一层持续运行的 Python 控制程序。

启动脚本还提供 `GPU_RA_RESOURCE_ROOT`，与 `"{resource_root}"` 一致，优先选择外层原工作区 Git 根，否则选择共同工作树管理根。独立实验工作树会改变旧 V3 runner 的默认扫描范围；直接登记原科学 runner 时，必须显式传入 `--resource-root "{resource_root}"`。扫描应覆盖整个工作区或共同 GPU_RA 根，不能缩小范围、改动原资源守卫或靠目录迁移漏检其他进程。

运行凭据记录 A 的 commit、分支、所属版本基线、配置和输入哈希、实际 argv 与环境。运行失败或中断也保留原目录和凭据；不能覆盖失败数据或替换样本。源码、配置或运行条件改变后，先形成新代码提交，再使用新的运行 ID。

## 4. 完成报告，再提交 B

检查原始输出及本轮科学验收结果后，用明确判定和结论完成登记：

```powershell
python reproduction/experiment.py finish --id joint-ab-001 --run run-20261115-001 --verdict inconclusive --conclusion '已完成运行，当前证据尚不足以通过本轮科学判据'
```

`--verdict` 支持 `passed`、`failed`、`inconclusive`；应按实际证据选择。`finish` 生成原始数据清单和 JSON/Markdown 报告，更新 `DATASETS.json` 的 `experiment_runs` 索引，并写入一次性的外部 seals 封存锚点，供后续核验且不覆盖已封存运行。报告描述执行条件、具体结果、不确定性、失败及排除情况，原始数据继续保存在外部运行目录。

审查生成的报告和清单，提交为 **B**：

```powershell
git add DATASETS.json reports/joint-ab-001 evidence/joint-ab-001
git commit -m 'Record joint-ab-001 run-20261115-001'
python reproduction/experiment.py verify --id joint-ab-001 --run run-20261115-001
```

B 引用真实执行代码的 A；不要把报告中尚不存在的自身提交写成运行身份。A 与 B 分开登记，后续代码修改不能混入既有样本。

## 5. 核验与合回

`verify` 检查代码、配置、输入、运行凭据、报告与原始清单是否可追溯且一致。哈希相同只证明文件身份，不能代替数值、资源、计时或统计判据。入口不自动批准科学验收，也不修改原定阈值；判据变化须形成新的实验配置和运行。

验证通过后，用 Git merge 或 PR 合回实验所属版本：v1 回 `main`，v2 回 `codex/v2-readonly`，v3 回 `codex/v3-joint`。第二、三版实验不合入 `main`。合并报告与清单时，原始数据仍留在本地外部目录。

当前三工作树布局的默认实验父目录为 `GPU_RA/experiments`；普通独立 clone 的默认父目录为其同级 `<clone-name>-experiments`，以 `new` 输出的实际路径为准，也可用 `--worktree-root` 覆盖。取消正在执行的生成启动脚本时，用 PowerShell 在该外部运行根创建 `cancel.request`，无需额外 Python 进程；Job 整树结束后登记为 `interrupted`。硬杀控制器或无法证明排空的运行保持 `incomplete`，不能 `finish --verdict passed`，也不能复用 RUN。

最终交付前，`verify` 应显示 `report_committed=true` 并返回报告提交 B；报告尚未提交时的身份校验仅用于预检。

## 工作流自身的回归检查

`python reproduction/tests/test_experiment_workflow.py --output-root <仓库外验证目录>` 在隔离临时Git库中核验完整流程和拒绝条件，不启动GPU或修改真实版本引用，保留全部检查日志。
