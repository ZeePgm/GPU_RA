# 第三版独立源码构建与组件资格

本入口从 `joint/source-bundle` 恢复的 4,276 项封存源码构建第三版，不使用原 `integrated`、`build-r*` 或历史 lib/DLL/PTX。源文件保持原始字节；新构建编译的程序及库单独生成身份记录。

环境：Windows x64、Visual Studio 2022 C++ Build Tools、CMake/Ninja、Python 3.12+、CUDA 13.0（含 CUPTI/cuBLAS）及匹配的 GPU。Python 使用调用方指定的解释器，并显式传给 CMake 的 Python 实验测试。

在 GPU_RA 第三版分支根目录运行，推荐较短的本机目录：

```powershell
python joint/restore_source.py
python joint/reproduction/verify_source.py
./joint/reproduction/build.ps1 -CudaRoot <CUDA目录> -Python <Python解释器> -BuildRoot ./joint/build-reproduction -CudaArchitectures 120a-real -Jobs 4
```

默认需要新的 BuildRoot。`-VsRoot`、`-CMake`、`-Ninja`、`-PtxArchitecture` 亦可指定；默认 PTX 架构为 `compute_75`。CUDA 编译架构与嵌入 PTX 架构分别记录。工具链、模型和源码分别管理；此构建步骤不需要 GGUF 模型，也不执行完整模型推理。

构建顺序：CPU core 及完整 608 项回归；联合模型、B2 读取夹具及共享介质 GPU 夹具；28 个 B2 窗口/边界/慢生产者场景；越界负例；共享介质读写读与排空。验证工具独立核对 B2 的场景集合、请求/字节依赖检查数、驻留预算、持续补充证据以及负例退出和日志。

日志写入新的 `joint/reproduction-validation/<时间戳>-<标识>/`，不覆盖先前尝试。最终摘要同时写入该报告目录和 `<BuildRoot>/build-acceptance.json`，包括：

- 历史源码逐文件校验；新构建源根和输出根。
- 选择的 Python、CUDA/架构、VS/CMake/Ninja 配置。
- CMake 声明的编译输入；生成的 Unity 单元逐项展开为封存源码，其他外部源码或历史构建链接被拒绝。
- CPU 与 GPU 夹具的具体通过范围。
- 新的三个执行程序及 `model/bin/*.dll` 全部文件大小和 SHA-256。

模型运行器应先校验该摘要的 `status=pass`、`tests_executed=true`、`final_parameterized_entry=true`，再核对当前源码/归档清单和全部新执行程序、DLL 的哈希。移动已构建目录后应重新构建，不能用旧路径身份代替新目录身份。

同一源码树中继续核验或重入构建时可显式传 `-Reuse`；入口检查现有 CMake 源根归属，重新核对源码和组件。`-SkipTests` 仅用于开发构建，结果标为 `tests_executed=false`，不能用作完整复现资格。

2026-11-15 本轮结果：core 147 步、model 291 步首次源码构建完成；最终参数化入口重入时两项目均 `ninja: no work to do`；显式解释器下 608/608 CPU 回归、28 个 B2 场景（6,244 请求、199,808 次字节/等待检查）、范围负例及共享介质夹具全部通过。首轮自动解释器选择及 Unity 审计误判日志保留，最终入口已纠正，封存源码未改动。完整模型验证由联合运行入口另行执行并记录。
