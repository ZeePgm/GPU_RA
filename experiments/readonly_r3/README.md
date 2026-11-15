# R3 只读访问合并与覆盖扩展实验

运行环境：在仓库根目录加载 configs/enter-environment.ps1 -BuildTools。独立构建当前源码运行时，避免复用旧 ABI 库。

```powershell
cmake -S experiments/readonly_r3 -B build-r3-current-20261107 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DCMAKE_CUDA_ARCHITECTURES=75-virtual
cmake --build build-r3-current-20261107 --parallel 4
ctest --test-dir build-r3-current-20261107 --output-on-failure
.venv/Scripts/python.exe experiments/readonly_r3/run.py --stage qualify --output output/r3-new/qualification
.venv/Scripts/python.exe experiments/readonly_r3/run.py --stage coverage --output output/r3-new/coverage
.venv/Scripts/python.exe experiments/readonly_r3/run.py --stage hybrid --output output/r3-new/hybrid
.venv/Scripts/python.exe experiments/readonly_r3/run.py --stage pilot --output output/r3-new/pilot
.venv/Scripts/python.exe experiments/readonly_r3/run.py --stage performance --pilot output/r3-new/pilot --output output/r3-new/performance
```

各输出必须使用新目录，GPU 批次串行执行，性能采样期间不要并发构建。每阶段保存源码/程序哈希，结束时再次验证。

## 固定策略

活跃 lane 按 lane 顺序组成连续字节跨度，且整个跨度属于同一个注册区间时合并；重复、空洞、逆序、跨注册区间均保持标量访问。物理 sector/DLU 边界允许跨越，由后端正常拆分。

PTX uniform_warp_reads 只接受无分支、无提前返回、所有全局读先于写的直线入口，线程块必须由完整 warp 构成。helper 在每个读站点收集 predicate，任何成员失败则整体退出，防止下一次 collective 缺少参与者。原标量策略保持可用。

hybrid 对本实验已知等价的读取模板先检查 PTX 合并能力；检查不通过则选择显式 CUDA。控制流回退夹具只添加不改变语义的跳转，不将未知任意 PTX 静默替换成 CUDA。

## 覆盖层级和计时

完整张量路径是有界读取适配器：每批 256 lane，将不可变权重读入 GPU 镜像，在每个实际消费者许可后检查字节和时间，再将完整镜像交给原算术顺序的 CUDA 矩阵/权重乘法消费者。对应原始 GGUF 权重的完整 GPU 输出逐位比较，另有 CPU 算术抽查。它验证完整张量与声明层集合的读取和计算，不是完整 llama 推理入口，也不包含全模型所有权重。

coverage 使用真实 Qwen3-0.6B 第 10 层、SmolLM2 第 0/23 层、Qwen3-1.7B 第 27 层的 norm/down；hybrid 使用额外第 29 层。token 批大小为 1/3/7，额外策略留出为 5，不把这些批大小当作文本上下文长度。

主性能终点是单次读取适配器 API 启动至同步结束的墙钟；测量时关闭 GPU 时间插桩，每进程预热一次再测三次。性能结果不等同于完整模型 TTFT/TPOT。主条件每条接入路径 32 对独立进程，使用配对差值中位数及二项分布次序统计区间；非连续、单 lane、稀疏和跨 DLU 为描述性条件。原始顺序、先导及资格判定均保留。

完整张量的事件预算按声明标量访问量配置；它是软件资源上限，不是正确性容差。超时、越界与后端失败仍须明确终止并排空。
