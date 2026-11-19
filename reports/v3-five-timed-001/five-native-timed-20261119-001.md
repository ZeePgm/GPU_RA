# Five native GPU weight paths with live HBF timing — five-native-timed-20261119-001

- Code commit A: `6ef8ed35b08ac3ab896d435366c276f8168edcc2`
- Base: `codex/v3-joint` / `a3ee58745940bbef0dfe4d0c65fcd7e8820d02f4`
- Run: `v3-five-timed-001/five-native-timed-20261119-001`
- Process status / exit: `failed` / `1`
- Author verdict: `failed`
- Data manifest SHA-256: `e37356028d84f22bf3a6ec5423ec454fd053017ff183cc8890c4f0bf88d09649`

原生对照完成并核对封存十个输出后进入计时组。计时组预填充词嵌入已完成，进入第 10 层 Q 投影后 CUDA stream 同步报 unspecified launch failure。流程确认退出、无残留子进程。未取得完整阶段回执或联合数值验收，原始日志和已有输出封存。

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
