# 第三版联合实验完整历史证据

`v3-historical` 无损保存 2026-11-12 正确性批次的 16 个模型进程，以及 2026-11-13 固定统计批次的 142 个模型进程。全部成功、首次失败、沙箱预检失败、构建/组件日志、输入配置、完整 logits、KV 内容哈希、逐请求地址/时序、进程/已加载模块/资源观察记录、统计原始样本和协议均保留。

包内为原始 `integrated/...` 相对路径；2,685 个文件合计 11,680,662,120 B，按 SHA-256 去重为 919 个对象后压缩，分为 44 个不超过 32 MiB 的分片。精确字节数、逐路径/对象/分片哈希、Python 字节码遗漏清单见 `v3-historical/manifest.json`。分类和排除原因见 `SCOPE.json`。

源码已单独保存在 `joint/source-bundle`，不在此重复；obj/lib/CMake 缓存等可再生构建产物未纳入。统计协议实际依赖的 8 个历史 exe/DLL 约 36.5 MB 原始字节作为审计对象保留，**只计算历史身份哈希，不执行或参与新源码构建链接**。模型 GGUF 权重不包含在包内。

## 恢复与离线重算

在仓库根目录运行，选择新恢复/报告目录。恢复只用 Python 标准库，离线重算需要本分支指定的 NumPy：

```powershell
python joint/reproduction/evidence_bundle.py restore --bundle joint/evidence/v3-historical --dest joint/output/historical-restored
python joint/reproduction/replay_historical.py --root joint/output/historical-restored --report-dir joint/output/historical-replay-001
```

默认恢复保留全部原始字节，并拒绝覆盖不同内容；`--verify-only` 可只校验归档。恢复后约需 11.7 GB 空间。报告目录必须不存在。较大临时恢复目录应放在 `joint/output` 或外部验证目录，不能提交到 Git。

复核首先验证原始两个封存 evidence-manifest 的全部数据哈希，再使用原字节验收函数核查 16+142 个模型输出、数值合同、跨路径 KV 哈希、模块与实际归档二进制 SHA-256、进程退出及资源窗口记录。逐行重算全部权重请求的精确地址/512 B 覆盖、票据、生产者、launch 关联及最大消费时刻 deadline；检查全部 KV 介质地址、延迟、服务/恢复时序和最终只读审计步骤。统计从原进程输出重新构造 142 个样本，复算全部组分布、8 对差值、4 对 A/A、六项中位数区间/判定及描述性交互项，核查 AB/BA 平衡与预声明覆盖率计算。

原验收器保存在 `joint/reproduction/evidence_validators/`，来源 SHA-256 在 `origin.json`。工具只选取不启动模型的原函数，注入恢复后的数据根目录，所有新报告写入独立报告目录；不会执行旧进程枚举、模型获取或二进制启动。依赖与模块身份由归档的实际二进制重算哈希，不只比较两个身份 JSON。

历史原始记录只导出了每 warp 的最大消费时刻、GPU 内逐 lane 校验计数及 KV 全内容/前缀 SHA-256，没有逐 lane 的全部设备时间数组或 KV 字节本体。离线工具重算实际可导出的全部记录；新源码构建与新 GPU 字节检查另行验证。原 608 项 CPU 测试和 28 个 GPU fixture 在本工具中核验留证结果，不把读取历史日志当成本轮重新执行。

## 本次迁移的新构建与新批次

`v3-migration-validation` 保存本次新源码构建与验收的全部原始留证，共 406 个路径、1,352,562,979 B；去重为 192 个对象，压缩为 182,055,231 B、6 个不超过 32 MiB 的分片：

- `correctness-new-001`：资源守卫停止的首次尝试；前两个成功进程和第三个中断进程完整保留，不计入第二批验收。
- `correctness-new-002`：完整 F1/F2 各 8 组，16 个新模型进程。
- `statistics-prepare-001`：统计入口零模型准备/检查、14 个数值参考组和冻结协议；未启动新的 142 进程性能采样。
- 新构建、组件测试、运行及失败日志；新 runtime 的 3 个 exe 和 5 个 DLL 只供离线 SHA-256 核查，不作为重建输入。模型权重、obj/lib 和编译缓存不入包。

两次正确性各 19 项冻结依赖和新统计准备的 61 项依赖均已逐项核对归档中的路径、大小和哈希，无遗漏。恢复到独立目录的命令：

```powershell
python joint/reproduction/evidence_bundle.py restore --bundle joint/evidence/v3-migration-validation --dest joint/output/fresh-restored
```

新数据离线验收入口为 `joint/reproduction/replay_fresh.py`，以恢复后的 fresh 数据根和上节恢复的 historical 数据根为输入。它与历史完整统计的重算分别生成报告。本次归档本身的独立恢复记录见 `joint/provenance/migration-validation/fresh-data-validation.json`。

```powershell
python joint/reproduction/replay_fresh.py --root joint/output/fresh-restored --historical-root joint/output/historical-restored --report-dir joint/output/fresh-replay-001
```
