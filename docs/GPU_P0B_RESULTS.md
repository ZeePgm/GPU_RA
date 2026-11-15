# P0-B 模型功能复验

2026-11-05 批次验收通过。12 个主格全部为 pass，对应 36 次模式运行。其中 27 次来自九个已校验的封存格，9 次来自本批补跑的三个 Qwen3-0.6B 格。Qwen3-0.6B 的两个长输入格实际将 138 个输入 token 拆成 64、64、10 三个 ubatch。各通过格的输出、字节范围、父子完成和释放均有逐格记录。

[逐格核查表](../output/gpu-p0b/20261105-001/checklist.csv) · [版本与验收结果](../output/gpu-p0b/20261105-001/acceptance.json) · [终止边界原始结果](../output/gpu-p0b/20261105-001/boundaries/results.json) · [组件测试记录](../output/gpu-p0b/20261105-001/components.xml)

本批次将三个模型的短／长输入和 32／64 输出组合成 12 个主格。每格使用 native、bypass、modeled 三种模式。短输入配置 ubatch 256，长输入配置 ubatch 64。两个 Qwen 模型的 KV 行为 2,048 B，当前布局在 4,096 B 页内对齐。SmolLM2 的 384 B 行实际触发跨页访问。每格的跨页判定单独列在核查表。

## 版本与参照

目标构建为 `build-matrix-20261104`，后端为 `8ccb1f353630f893f761b017d621812b7b9d5bf2`。九个已有格逐一核对封存构建、原始参照 manifest、允许的模型及 conversion 路径映射、输入与完整产物哈希后复用。三个新增格在同一构建上执行。A-short-64 的前 32 行 logits 和 token 还与旧 A-short-32 的同模式结果逐字节核对。两个新增长输入格采用已映射的 Qwen 长提示词，并在各自格内建立完整 native 参照。

沿用原数值合同。每格要求输出 token 数和完整 logits 行数正确，三模式 token 与文本一致。分页路径按原容差比较 native，bypass 和 modeled 的 logits 逐位一致。modeled 另核验访问字节范围、KV 事务、父子完成、命令排空和资源释放。

## 终止边界

| 场景 | 证据来源 | 判定 |
| --- | --- | --- |
| 0／1 输出 | 三模式实际调用，合成请求参数 | unsupported。入口按 2..448 合同拒绝 |
| 自然 EOG | Qwen3-0.6B 真实生成，三模式各交付两个 token 后触发 | unsupported。固定长度 runner 抛出 Early EOG，失败退出无进程内释放审计产物 |
| frame、host 和 GPU 预算不足 | 合成预算配置通过真实 runner 的入口校验 | pass。按预期诊断拒绝 |
| 请求级协作取消、超时 | 通用模型 runner 合同检查 | unsupported |
| 依赖取消、超时重试耗尽、失败排空、KV 回滚、容量及跨页释放 | 同一封存源码构建的组件测试 | 10 项 pass |

## 复现

从工作区根目录运行。模型、旧参照与封存构建须通过身份校验。输出路径应选新目录。

```powershell
python experiments/p0b/run.py --output output/gpu-p0b/new-batch
python experiments/p0b/boundaries.py output/gpu-p0b/new-batch
```

组件测试使用封存源码 `output/matrix-20261104/source`，启用 `BUILD_TESTING` 和 `HBFLAB_BACKEND_TESTS`，关闭 `HBFLAB_ENABLE_CUDA`。构建 `hbflab_capacity_contract`、`hbflab_kv_transaction`、`hbflab_incremental_bridge`、`hbf_eventlab_host_completion_tests` 四个目标。用 CTest 将下列用例输出为批次目录内的 `components.xml`。

```text
^(lab_capacity_(failure|kv_failure|kv_capacity|kv_cross_page)|lab_kv_transaction_(failure|abort)|lab_incremental_(failure_drain|partial_initialization)|host_completion_(dependency_cancel|timeout_retry))$
```

```powershell
python experiments/p0b/summarize.py output/gpu-p0b/new-batch --component-build build-p0b-components-r2
```
