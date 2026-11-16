# 验收后日志清理记录

2026-11-16，按用户本次明确批准，对本实验5个已通过运行清理不再需要的全量轨迹。失败、中断、未通过运行未处理。

共删除116个文件，44,064,479,254 B（41.0382 GiB）；5个运行保留351,308,911 B（335.03 MiB）。这包含正式组件/模型两批以及已通过的回归、先导、参考诊断。

| 运行 | 删除文件 | 删除GiB | 保留MiB |
|---|---:|---:|---:|
| stage2-regression-20261116-001 | 26 | 1.0431 | 80.17 |
| stage3-full-20261116-001 | 12 | 2.2457 | 1.57 |
| stage3-pilot-20261116-002 | 12 | 0.0092 | 1.43 |
| stage4-full-20261116-001 | 60 | 37.7111 | 200.86 |
| stage4-smoke-20261116-001 | 6 | 0.0292 | 51.01 |

删除范围：lane-times.bin、requests.bin、weight-requests.jsonl、gpu-activity.jsonl、chunks.jsonl、kv-media.jsonl。每个文件均在删除前匹配原始封存SHA-256，并逐文件确认绝对路径位于对应外部运行根、没有符号链接或目录重解析点；没有递归删除。

保留代码、配置、科学报告、原manifest和seals、输入/CAS、模型、构建身份，以及完整logits、native KV序列化状态、KV内容/前缀哈希、版本、生命周期、需求和消费者摘要、分块权重哈希、PTX源码/改写身份与必要失败复现资料。原始生成量与运行成本记录保持不变。

精确清单与用户批准位于每个evidence/实验ID/运行ID/retention-plan.json，删除结果位于retention-result.json。计划提交为642a9c05163f624b970470500a8b66266793c4b1；外部GPU_RA-local-data/retentions/实验ID/运行ID保存计划、逐文件意图/完成日志和结果锚点。清理结果分别提交，均通过显式保留证据核验。

当前核验范围为已保留证据的身份和完整性：report_committed=true、retention_committed=true、full_payload_present=false、full_raw_trace_recheck_available=false。原报告记录的是清理前实际完成的科学验收；现在不能重新逐条复算已删除的时序轨迹。普通完整verify保持严格，不会将缺失文件视为原始数据仍完整。

```powershell
python reproduction/experiment.py verify --id v3-three-routes-sampled-001 --run stage4-full-20261116-001 --retained
```

完整核验输出见[retention-verification.json](retention-verification.json)。清理工具的65项CPU隔离检查（22项清理保护/恢复、43项原工作流）通过，无GPU模型重跑。

工作区AGENTS.md及GPU_RA的main/v2/v3均已写入规则：每次实验结束完成验收、报告和完整核验后，先向用户列出清理范围、大小和保留项并询问；仅明确批准后执行，未回复不清理，本次批准不适用于下次实验。原始轨迹不默认长期保留几十GiB。
