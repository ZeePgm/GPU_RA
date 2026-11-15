# 双线联合统计性能确认

范围：仅 Qwen3-0.6B F16，原始 44 输入 / 8 输出、单序列，先 F1 后 F2。

封存依赖：`E:/v1_update/integrated/20261112-dual-line-r1`，使用其原二进制、配置、prompt 和正确性验收函数，不修改其文件。全部新的运行记录写入本目录。

批准方案：每规模 7 组、每组 1 次预热、8 轮完整配对对照、4 对联合 modeled A/A，共 142 个独立模型进程。组间 AB/BA 顺序各 4 次，顺序与统计门槛在 `protocol.json` 固化。

使用含 numpy 的工作区 Python 运行 `scripts/run.py prepare` 固化，再运行 `scripts/run.py run`。脚本拒绝覆盖既有样本，不自动替换失败样本，不根据显著性扩充预算。

主计时保留所有已有审计，排除模型加载和初始化，包含 prefill、7 次 decode、最终审计与排空。二进制和依赖在运行前后核验；每个进程验证已加载模块、完整 logits、token、KV 内容与请求/资源统计。

原始记录见 `raw/`；全量样本见 `samples.json`；进度见 `status.json`；最终统计报告见 `reports/RESULTS.md`。源目录与模型为外部只读依赖，哈希和路径完整记录于协议。
