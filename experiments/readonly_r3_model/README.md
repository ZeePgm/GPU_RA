# R3 真实上下文留出

该补充使用封存的 R2 llama DLL 执行原生模型，采集 SmolLM2 第 29 层 norm/down 的真实输入，再用 R3 完整张量读取适配器回放。不会把回放误认为完整模型内 HBF 集成。

固定 `n_ubatch=32`，使用 31、32、33 个真实 tokenizer token；每个上下文执行 prefill 和一次 greedy decode。回调保存所有微批输入，验证末次 prefill 微批及首步 decode，覆盖完整微批和 U+1 尾批。三种上下文的 decode 激活必须具有不同哈希。

两条合并读取路径各读取完整权重并执行同样的原生 CUDA 消费者。验收要求完整镜像、全部 GPU 消费输出逐位一致，沿用 CPU 算术检查和逐 lane 依赖检查；不引入新的数值容差。共 24 次运行，不加入 R3 性能统计。

`replay.cpp` 是已冻结 R3 runner 的独立副本，仅增加从 `activation_file` 加载激活和记录哈希，保护既有性能实验的源文件/二进制身份。`kernels.cu` 和运行时仍使用 R3 的原实现。

构建：加载 `configs/enter-environment.ps1 -BuildTools` 后执行：

```powershell
cmake -S experiments/readonly_r3_model -B build-r3-context-20261108 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75-virtual
cmake --build build-r3-context-20261108 --parallel 4
.venv/Scripts/python.exe experiments/readonly_r3_model/run.py
```

脚本拒绝覆盖已有结果；再次执行应先改用新的输出目录。运行协议保存模型、源码、DLL 与程序哈希，结束时复核。证据位于 `output/r3-context-v2-20261108/`。此前两次采集准备失败的日志保留：首次缺失 CUDA DLL 搜索路径，第二次未识别 norm 权重位于 MUL 第二输入；均未进入最终验收。
