# 三种入口代表权重适配 — stage2-registration-20261116-001

- Code commit A: `1523d84c11095186a12835fef4d02cdd4f1b12a9`
- Base: `codex/v3-joint` / `f4b754c8e228064364bc25877dbb39c698a2c167`
- Run: `v3-three-routes-sampled-001/stage2-registration-20261116-001`
- Process status / exit: `failed` / `1`
- Author verdict: `failed`
- Data manifest SHA-256: `efe20aef10915e7595375f4b3680b76586583a39343c862e610e3a60631a6773`

symbolic公共注册夹具完成并输出通过，但runner在保存进程命令时将Path对象直接写入JSON导致失败，后续入口与回归未执行。保留原始夹具及异常，修复命令序列化后新建运行。

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
