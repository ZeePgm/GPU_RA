# 阶段1：清单与需求计划验收

状态：通过。运行 `stage1-20261116-004`；运行代码 A 为 `773639a1b114b23b4e7546fca2e99bf4ba8c901e`。

实际GGUF共311个张量，选定14个、其余297个保持原生路径。模型SHA-256：`74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2`。文件存储与加载后均无选定张量别名，全部选定权重为连续CUDA0存储。

| 张量 | 类型 | 形状 | 字节 | 消费者 |
|---|---|---|---:|---|
| blk.10.attn_k.weight | F16 | 1024 × 1024 | 2,097,152 | MUL_MAT / src[0] |
| blk.10.attn_k_norm.weight | F32 | 128 | 512 | MUL / src[1] |
| blk.10.attn_norm.weight | F32 | 1024 | 4,096 | MUL / src[1] |
| blk.10.attn_output.weight | F16 | 2048 × 1024 | 4,194,304 | MUL_MAT / src[0] |
| blk.10.attn_q.weight | F16 | 1024 × 2048 | 4,194,304 | MUL_MAT / src[0] |
| blk.10.attn_q_norm.weight | F32 | 128 | 512 | MUL / src[1] |
| blk.10.attn_v.weight | F16 | 1024 × 1024 | 2,097,152 | MUL_MAT / src[0] |
| blk.10.ffn_down.weight | F16 | 3072 × 1024 | 6,291,456 | MUL_MAT / src[0] |
| blk.10.ffn_gate.weight | F16 | 1024 × 3072 | 6,291,456 | MUL_MAT / src[0] |
| blk.10.ffn_norm.weight | F32 | 1024 | 4,096 | MUL / src[1] |
| blk.10.ffn_up.weight | F16 | 1024 × 3072 | 6,291,456 | MUL_MAT / src[0] |
| output.weight | F16 | 1024 × 151936 | 311,164,928 | MUL_MAT / src[0] |
| output_norm.weight | F32 | 1024 | 4,096 | MUL / src[1] |
| token_embd.weight | F16 | 1024 × 151936 | 311,164,928 | GET_ROWS / src[0] |

原输入249字节CRLF、44 token、8输出保持；短输入为15 token、8输出。每份均有112条消费者执行前记录及对应执行后记录，所有范围、算子、源槽位、输出形状与算术复用次数符合独立需求。embedding分别读取51次/36个不同行、22次/16个不同行；重复访问不去重。未实际使用的embedding行留待独立全行夹具。

原加载配置实际将embedding留在CPU_Mapped，并构建普通/mixed两条gather。候选显式覆盖embedding为CUDA0，启用纯token图；与初次native诊断对照，原输入8×151936完整logits逐字节一致，输入和输出token一致。所有后续参考与候选均固定该配置，历史默认图行为保留。

选定原始权重653,800,448 B。完整GGUF布局按32KiB对齐，连同58,720,256 B KV预留为1,565,556,736 B；2GiB逻辑介质剩余581,926,912 B，另有每bank一个备用block，共32KiB物理备用量。此介质容量与旧F配置的1GiB不同。

Host resident fragments保持256×64B；KV保持4个GPU页框及4个pinned槽。模型驻留后，symbolic的256KiB验证缓冲以及PTX/CUDA各256KiB mirror+256KiB验证缓冲均分配成功。注册数14≤16，当前范围无需扩大上限。初始化、历史摘要、模型与Host总内存以及运行成本还须在先导中实测，不能把活动fragment预算当作进程内存上限。

六个篡改负例全部被拒绝：漏消费者、缺执行后记录、错误gather偏移、错误设备、错误算子、错误算术复用。

阶段2仍需安装GET_ROWS/MUL/MUL_MAT门控，实现32KiB注册预留对齐与统一入口选择。阶段3/4的返回字节、发布、消费者依赖和联合KV合同尚未通过；本阶段不产生三入口科学通过或性能判定。

001长路径失败、002沙箱中断、003设备/图假设拒绝均独立保留并提交报告。原始观察、完整logits和构建身份留在外部运行/构建目录；Git中的inventory.json和demand-plan.json为审查用清单与预期计划。
