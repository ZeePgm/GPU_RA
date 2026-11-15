# 第三版参数化运行入口

以下命令在 `codex/v3-joint` 仓库根目录执行。源码随 Git 直接检出；若 `joint/source` 缺失，先运行 `python joint/restore_source.py` 从原字节备份恢复。模型、CUDA 与构建目录均显式传入，不使用原工作区的 `integrated` 二进制或绝对数据路径。

以下示例先设置 `$env:GPU_RA_DATA_ROOT = '<外部数据根目录>'`，运行结果、恢复数据和复核报告均写到该根目录同级的本地目录。分片取数也支持 `data.local.json` 或恢复命令的 `--data-root`。

运行前检查新构建的 `build-acceptance.json`、4276 项历史源码、全部新 exe/DLL、固定模型和原始 prompt 身份。配置由冻结 JSON 生成，只改 `model_path` 和 `conversion` 路径；数值容差、预算和模型形状不可由命令行放宽。模型获取与转换入口沿用仓库 `reproduction/prepare_model.py` 及固定转换记录。

## 正确性：F1 后 F2，共 16 个模型进程

```powershell
python joint/reproduction/run_correctness.py --model <Qwen3-0.6B-F16.gguf> --cuda-root <CUDA目录> --build-dir joint/build-reproduction/model --output "$env:GPU_RA_DATA_ROOT/../runs/v3-correctness-new-001"
```

默认 F1/F2 各运行 native、readonly bypass/modeled、KV bypass/modeled、joint bypass/modeled 和 joint host 写来源八组。F2 必须在同一新批次 F1 八组完整验收后启动；`--scales F1` 只执行 F1。`--prepare-only` 仅核验输入、构建及身份并生成配置，不启动模型。

`--gpu-device` 映射物理 CUDA 设备，模型内部逻辑设备固定 0。`--resource-root` 默认覆盖同级版本工作树；可按实际项目目录指定。OS 进程可见性必须包含 System，遇到其他编译器或模型进程即拒绝/停止。计时中不查询 CUDA/NVML 驱动；每 5 秒记录 OS 进程快照。

每组保留命令、环境、进程退出和实际加载模块路径/哈希。使用原字节冻结验收器检查逐 lane 返回字节/等待、权重覆盖/发布、KV 全内容/地址/事务及排空。同算术路径要求 logits 逐字节一致；分页 attention 对 native 沿用 max_abs≤0.2、RMS≤0.025。失败目录保留且不覆盖，也不自动替换失败样本。

结果位于 `<output>/cases`，分别有 `F1-acceptance.json`、`F2-acceptance.json`。批次状态及新依赖身份位于 `<output>/status.json`、`dependencies.json`。这些运行用于完整入口与正确性验证，不形成新的统计性能结论。

## 固定 142 进程统计协议：先 prepare，再显式 run

```powershell
python joint/reproduction/run_statistics.py prepare --model <Qwen3-0.6B-F16.gguf> --cuda-root <CUDA目录> --build-dir joint/build-reproduction/model --correctness-dir "$env:GPU_RA_DATA_ROOT/../runs/v3-correctness-new-001" --output "$env:GPU_RA_DATA_ROOT/../runs/v3-statistics-new-001"
python joint/reproduction/run_statistics.py check --output "$env:GPU_RA_DATA_ROOT/../runs/v3-statistics-new-001"
```

prepare 启动 **零个模型进程**，先验收 F1/F2 新正确性参考，将 14 组参考的 `summary.json`、`logits.f32` 及已有的 `kv-content.json`（KV 内容校验摘要和哈希）复制到新批次输入包，冻结新模型、脚本、二进制、配置及参考身份。准备后的协议不借用历史批准时间或历史构建身份。固定种子 131113 的完整 142 行顺序与历史设计逐项核对。

只有下面的显式 run 动作会执行约 142 个独立模型进程的长批次：

```powershell
python joint/reproduction/run_statistics.py run --output "$env:GPU_RA_DATA_ROOT/../runs/v3-statistics-new-001"
```

每规模 7 组，分别 1 次预热、8 个配对区块及 4 对 joint-modeled A/A；F1 完整验收后进入 F2。主指标为包含 prefill、7 次 decode、最终审计/排空而排除模型加载/初始化的 `generation_wall_ms`。重用原统计函数，按 8 个配对差值 min/max 区间、共同 A/A 最大绝对差及原判定门槛分析，交互效应仅作描述性记录。

样本量、顺序和停止规则不随显著性变化；任何超时、数值或资源冲突终止批次并保留记录。已有 `run-start.json` 或样本/尝试目录拒绝重启，重做必须新建整个输出目录。GPU 状态查询仅在 F1/F2 边界和结束时执行，实际模型计时期间只观察 OS 进程。

原始报告及历史 142 进程数据的恢复/离线重算是另一条入口，见数据说明。重算历史数据不会被记为新采样。默认构建目录为 `joint/build-reproduction/model`；本轮验证使用其他新 BuildRoot 时，需把上述 `--build-dir` 替换成该 BuildRoot 下的 `model`。

## 本次新批次归档的独立离线复核

先按 `joint/evidence/README.md` 配置 `GPU_RA_DATA_ROOT`（或 `data.local.json`、恢复命令的 `--data-root`），分别从外部归档恢复历史与本次迁移数据，然后运行：

```powershell
python joint/reproduction/evidence_bundle.py restore --bundle joint/evidence/v3-migration-validation --dest "$env:GPU_RA_DATA_ROOT/../restored/v3-fresh"
python joint/reproduction/replay_fresh.py --root "$env:GPU_RA_DATA_ROOT/../restored/v3-fresh" --historical-root "$env:GPU_RA_DATA_ROOT/../restored/v3-historical" --report-dir "$env:GPU_RA_DATA_ROOT/../reports/v3-fresh-001"
```

此入口核验本次固定命名的 `correctness-new-001/002` 和 `statistics-prepare-001`，不启动模型或加载 DLL。独立核验归档的 8 个新 exe/DLL、16 组原始正确性记录、逐请求读取和 KV 介质流、80 条实际加载模块身份、进程退出/资源/不重叠顺序，以及 14 组统计 gold、142 行计划与零样本状态。新旧各模式/规模的 logits 逐字节比较单独留证；若存在差异会输出明确的哈希、最大差和 RMS，并拒绝把它记为完整逐字节复现。

失败的首次正确性尝试独立记录，不混入完整 16 组。原始数据未导出每个 lane 的全部时间数组和 KV 字节本体，离线检查仅重算实际存在的 warp 最大消费时刻、逐 lane 内部校验计数及 KV 哈希；新 GPU 字节校验由新正确性运行执行。报告目录必须不存在，所有输入保持只读。

实验打包工具要求明确的仓库外 `--bundle` 输出目录，拒绝把新实验分片写回源码工作树。新的数据清单、配置和报告经核验后单独纳入版本控制，实际 payload 继续保留本地。
