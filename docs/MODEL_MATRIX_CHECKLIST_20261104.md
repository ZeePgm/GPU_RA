# 当前版本完整模型矩阵核查表（2026-11-04）

结论：九个主案例、27 次三模式模型运行全部通过；全部 27 组完整 logits 与对应历史结果逐字节一致，输入和生成 token 亦一致。独立算子 18 项、Python 合同 5 项通过。

## 身份与参照

- 后端：8ccb1f353630f893f761b017d621812b7b9d5bf2，上游源码逐文件核验。
- 当前前端源码冻结于本批次 source/，逐文件与冻结时 E:/v1_update 相同；新构建为 E:/v1_update/build-matrix-20261104。
- 原默认构建因绘图脚本变化未通过全源清单校验；本次未重写它的身份，使用独立源码快照重新编译、封存和验收。
- 历史参照：D:/HBF-Sim/HBFLab/output/llama-m6/final。原始 profile、prompt、summary 和完整 logits 均核对原 manifest 哈希。独立 reference-view 只映射模型和 conversion 路径；其余参数及原数值容差不变，原数据未修改。
- 同模式历史比较要求完整 logits 字节一致；native 与分页路径按既有数值合同比较，bypass 与 modeled 要求逐位一致。

## 九格核查表

模型代号：A＝Qwen3-0.6B，B＝Qwen3-1.7B，C＝SmolLM2-135M。

| 案例 | 输入/输出 token | ubatch | native | bypass | modeled | 数值 | 访问/KV/计费 | 完成/释放 | 三模式历史对照 |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| A-short-32 | 44/32 | 256 | pass | pass | pass | pass | pass | pass | logits/token 一致 |
| B-short-32 | 44/32 | 256 | pass | pass | pass | pass | pass | pass | logits/token 一致 |
| B-short-64 | 44/64 | 256 | pass | pass | pass | pass | pass | pass | logits/token 一致 |
| B-long-32 | 138/32 | 64 | pass | pass | pass | pass | pass | pass | logits/token 一致 |
| B-long-64 | 138/64 | 64 | pass | pass | pass | pass | pass | pass | logits/token 一致 |
| C-short-32 | 40/32 | 256 | pass | pass | pass | pass | pass | pass | logits/token 一致 |
| C-short-64 | 40/64 | 256 | pass | pass | pass | pass | pass | pass | logits/token 一致 |
| C-long-32 | 163/32 | 64 | pass | pass | pass | pass | pass | pass | logits/token 一致 |
| C-long-64 | 163/64 | 64 | pass | pass | pass | pass | pass | pass | logits/token 一致 |

## KV 与最终完成实数

| 案例 | KV 长度 | 批事务 | 跨页行消费 | frame 字节 | 分配页 | faults | 淘汰 | 写回 | Global 完成/登记 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-short-32 | 75 | 32 | 0 | 16384 | 2128 | 56728 | 56724 | 2968 | 163339/163339 |
| B-short-32 | 75 | 32 | 0 | 16384 | 2128 | 56728 | 56724 | 2968 | 243741/243741 |
| B-short-64 | 107 | 64 | 0 | 16384 | 3024 | 140952 | 140948 | 4760 | 403741/403741 |
| B-long-32 | 169 | 34 | 0 | 16384 | 4760 | 148960 | 148956 | 5600 | 548037/548037 |
| B-long-64 | 201 | 66 | 0 | 16384 | 5656 | 317408 | 317404 | 7392 | 792261/792261 |
| C-short-32 | 71 | 32 | 3180 | 16384 | 600 | 16200 | 16196 | 2280 | 71762/71762 |
| C-short-64 | 103 | 64 | 8280 | 16384 | 840 | 39840 | 39836 | 4320 | 122762/122762 |
| C-long-32 | 194 | 34 | 10920 | 16384 | 1560 | 48120 | 48116 | 3240 | 181322/181322 |
| C-long-64 | 226 | 66 | 23400 | 16384 | 1800 | 101280 | 101276 | 5280 | 261842/261842 |

所有 modeled 案例检查逻辑片段全部成功、accepted=delivered、输入关闭、active lease/pin/dirty/释放后 allocation 全部为零。地址、PTX/helper 范围与页事务使用现有独立公式验收；非空计数不是单独的通过依据。

## 边界与不支持项

- 18 项独立检查覆盖三个模型、MHA/GQA/宽行/三页行、PTX 宽度、错误位置/形状、frame 不足、错误层数/GQA/审计预算/dtype/架构拒绝。
- 5 项 Python 合同覆盖页规划和跨 ubatch 事务。
- 0、1、449 个输出均实际调用并明确拒绝，且未创建成功输出；当前 runner 合同为 2..448。0/1 输出支持标记为 unsupported，不计入九格正常推理通过数。
- 当前固定长度 runner 将提前 EOG 视为错误。本轮没有建立自然 EOS、模型级取消或所有故障路径的完整支持矩阵，也没有重跑 A1 的180对墙钟统计实验；这些仍是后续能力扩展。
- 本轮是既有支持域的功能/数值回归，不是硬件性能或真实 HBF 参数校准。

## 产物

原始批次：[output/matrix-20261104](../output/matrix-20261104/)。以下文件均位于该批次目录。

本目录 checklist.csv 为逐格完整核查表，mode-checklist.csv 为27次执行明细，matrix-result.json 为机器可读汇总。evidence/ 保存每格输入、原始记录、validation 和 manifest；operators/、token-count-boundaries.json、python-contract.log 保存补充检查。source-snapshot.json 与 reference-provenance.json 记录身份和路径映射。

冻结后活跃工作区源码变化：无。
