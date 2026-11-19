# All Qwen3-0.6B weights with address-based live HBF timing — all-native-timed-20261119-001

- Code commit A: `bd8f77d0f215f11cf7b8c5191c9094b93faa3ba4`
- Base: `codex/v3-joint` / `f6c97691e8b360dc27b3bf5e26b96a938c52b134`
- Run: `v3-all-timed-001/all-native-timed-20261119-001`
- Process status / exit: `completed` / `0`
- Author verdict: `passed`
- Data manifest SHA-256: `2bcdc3b4f1a41dad73ce7375375897389822f3c94e82ee81a66fa3faedd97357`

Qwen3-0.6B F16 的 311 个权重在加载后一次登记。44-token 预填充、新增一个 token 的解码分别验收 291,152 和 291,122 个实读页，两个阶段所有权重都有动态读取。覆盖位图与逐权重读取字节符合预注册预期，582,274 页均通过先完成服务再放行检查，提前放行、缺失时间戳均为零，阶段结束队列排空。两步完整 logits、原生 F16 KV 内容与位置、序列释放记录均与原生组逐字节一致。两个阶段 622 份完整算子输出的 SHA-256 一致，五类冻结参考通过，两组输出 token 均为 [32, 6366]。

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
