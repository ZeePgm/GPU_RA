# 第 10 层 Q 权重的单次登记实验

使用 Qwen3-0.6B F16、原 44-token 输入、两个输出 token 和原生 F16 KV。一次登记 `blk.10.attn_q.weight` 的 4 MiB GPU 地址范围。在同一候选构建中依次运行探针关闭组与探针开启组，保留框架原有的算子选择、参数与启动。

实验构建对 ggml 的 `convert.cu` 和 `mmvf.cu` 两个编译单元应用同一套 PTX 处理规则。在编译器生成的受支持 global load 前插入地址探针。构建选择以源文件为单位，PTX 处理无核名、参数位置或选定权重的知识。探针只统计落入已登记范围的读取，保存原 PTX 位置、指令宽度、各位置的动态次数及逐 2 B 元素计数。该实验的端点为读取覆盖和数值等价。

预填充与解码分别检查全部 2,097,152 个 F16 元素、按指令宽度重算的访问字节、Q 投影、完整 logits、token、KV 内容与位置及序列释放。覆盖计数允许原生核重复读取，并据实报告次数。

`build.ps1` 使用 Windows CUDA 13、VS2022、Ninja 构建。原始 PTX、插桩 PTX、cubin、对象文件和编译清单保存在外部构建目录。提交实验代码后，用 `run.py receipt` 封存候选构建身份，再用仓库统一工作流直接登记 `run.py measure`。显式传入 `--resource-root "{resource_root}"`、`--config "{config}"`、`--output "{run_dir}"`；模型、构建收据及编译清单作为输入登记。
