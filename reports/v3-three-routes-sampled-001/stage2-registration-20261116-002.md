# 三种入口代表权重适配 — stage2-registration-20261116-002

- Code commit A: `7eac3c17a096d878ed68139558cb9e367151d681`
- Base: `codex/v3-joint` / `f4b754c8e228064364bc25877dbb39c698a2c167`
- Run: `v3-three-routes-sampled-001/stage2-registration-20261116-002`
- Process status / exit: `completed` / `0`
- Author verdict: `passed`
- Data manifest SHA-256: `164e6a6c5ce5dd2a727fe7f8ba6407a0257ef3a85fe1b1e80072f78c7c89fcd4`

阶段2公共注册子验收通过：三入口配置各自完成14实际范围/653800448B权重注册、固定KV预留、共享读写读及排空，25负例/入口被拒绝；608 CPU、28 B2场景/6244请求/199808检查、B2越界负例与共享介质夹具通过。历史F1/F2及各入口读取/模型接入另行验收。

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
