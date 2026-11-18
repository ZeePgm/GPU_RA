# GPU_RA — 第三版：只读与优化 KV 联合版本

后续新实验统一使用 [实验工作流](reproduction/EXPERIMENTS.md) 和 `python reproduction/experiment.py new/start/finish/verify`，按代码提交 A、外置运行数据、报告提交 B 建立关联；规则见 [AGENTS.md](AGENTS.md)。

`codex/v3-joint` 保存 B2 只读路径与选定 K1/K4/K5 优化 KV 的联合源码、依赖、历史数据清单和复现入口。实际工程为 `joint/`，源码可直接在 `joint/source/` 查看；根目录第一版代码保留为历史基线。

固定范围：Qwen3-0.6B F16、原始249字节CRLF prompt、44输入/8输出、单序列、F1/F2。读取K=256KiB、W=32KiB；KV启用device写来源、可取消追加、async分页、coalesced服务和异步副本，4个GPU页框、4个pinned槽；未启用K2 flush合并或K5 prefetch。权重/KV共用服务，Host驻留256×64B，逻辑介质容量1GiB。

V3 选定只读权重可以使用第二次 F 的主机范围门控、R2 的 PTX 算术入口或 B2 CUDA。可用权重、调用方式及小规模验证见[入口说明](experiments/v3-restore-r2-001/README.md)和[恢复验收](reports/v3-restore-r2-001/README.md)。

## 克隆与依赖

使用私有仓库访问权限，在较短的Windows路径克隆：

```powershell
git clone --config core.longpaths=true --branch codex/v3-joint https://github.com/ZeePgm/GPU_RA.git GPU_RA-v3
cd GPU_RA-v3
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r joint/reproduction/requirements-validated.txt
./.venv/Scripts/python.exe reproduction/verify_checkout.py
./.venv/Scripts/python.exe joint/reproduction/verify_source.py
```

验证平台为Windows x64、Python3.12.10、VS2022 C++ Build Tools、CMake/Ninja、CUDA13.0含CUPTI/cuBLAS、RTX5070/SM120a。NumPy/PyYAML实际验证版本已固定。工具链、Python、模型和构建路径均由调用方配置；不依赖原工作区旧build目录。

4276项历史源码保持原字节，冻结备份仍在 `joint/source-bundle`；`python joint/restore_source.py` 可从备份恢复，拒绝覆盖不同内容。修改并提交源码前运行 `python joint/reproduction/configure_git.py`，只配置当前独立仓库的字节保留规则，不改全局Git设置；新的实验修订应重新登记身份。

## 本地外置数据配置

实验归档、完整 logits、逐请求轨迹及原始采样留在仓库外；Git 只保存源码、配置、报告、数据清单及复现入口。数据包和逐片身份见 `DATASETS.json`、各包 `manifest.json`。源码备份 `joint/source-bundle` 继续随源码保存。

运行数据恢复前设置外部数据根目录，或复制 `data.example.json` 为被 Git 忽略的 `data.local.json` 后配置 `data_root`：

```powershell
$env:GPU_RA_DATA_ROOT = '<外部数据根目录>'
```

分片按仓库内清单的相对目录存放，例如 `<外部数据根目录>/joint/evidence/v3-historical/part-0000.gzpart`。可用 `--data-root` 为单次恢复显式覆盖设置；没有数据根或缺少分片时给出错误，不从仓库内寻找实验分片。取得数据后仍用仓库中的清单逐片校验。

## 从源码构建与新联合验收

```powershell
./joint/reproduction/build.ps1 -CudaRoot '<CUDA目录>' -Python ./.venv/Scripts/python.exe -Jobs 4
./.venv/Scripts/python.exe joint/reproduction/run_correctness.py --model '<Qwen3-0.6B-F16.gguf>' --cuda-root '<CUDA目录>' --build-dir joint/build-reproduction/model --resource-root '<实验工作区父目录>' --output "$env:GPU_RA_DATA_ROOT/../runs/v3-correctness-new-001"
```

构建默认写入 `joint/build-reproduction`，重建core/model及B2/KV夹具，运行608项CPU回归、28个B2场景、范围负例及共享介质读写读/排空；生成绑定源码和全部新exe/DLL的 `build-acceptance.json`。已有构建必须显式 `-Reuse`。参数见 [BUILD.md](joint/reproduction/BUILD.md)。

模型权重不分发，所需GGUF SHA256为 `74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2`。固定HF revision及9项原始输入见 `provenance/Qwen3-0.6B-conversion.json`，获取/转换入口为根目录 `reproduction/prepare_model.py`。

正确性默认F1/F2各八组，共16进程：native、只读bypass/modeled、KV bypass/modeled、联合bypass/modeled、联合host写来源。F1完整验收后进入F2。核对实际DLL路径/哈希、完整logits/token、逐lane字节/等待、覆盖/发布、KV内容/事务和排空。同算术路径逐字节一致；分页attention对native沿用max_abs≤0.2、RMS≤0.025。生成profile只改路径，不能放宽预算或数值合同。已有输出拒绝覆盖，资源竞争停止并留证。

## 外置数据恢复和离线重算

外置历史包包含2685原始路径、11,680,662,120B，去重后919对象，压缩1,451,241,878B、44个≤32MiB分片。包括原16正确性及142统计进程、成功/失败/中断与资源记录、8个仅供离线哈希审计的历史exe/DLL。缓存和其他构建中间产物的未纳入理由在SCOPE.json中。

```powershell
./.venv/Scripts/python.exe joint/reproduction/evidence_bundle.py restore --bundle joint/evidence/v3-historical --dest "$env:GPU_RA_DATA_ROOT/../restored/v3-historical"
./.venv/Scripts/python.exe joint/reproduction/replay_historical.py --root "$env:GPU_RA_DATA_ROOT/../restored/v3-historical" --report-dir "$env:GPU_RA_DATA_ROOT/../reports/v3-historical-001"
```

恢复工具仅用标准库，检查分片、对象和原始路径哈希，拒绝覆盖不同内容。离线工具复用原字节验收器和统计函数，逐行重验请求覆盖/票据/消费deadline、KV介质时序、完整logits/KV哈希、模块身份、进程与资源顺序，以及A/A、区间和六项判定。详见 [数据说明](joint/evidence/README.md)。历史导出只有warp最大消费时刻和KV哈希，没有全部lane独立时间或完整KV字节本体；新GPU验收另行检查完整字节与依赖合同。

`joint/evidence/v3-migration-validation` 的清单登记外置归档，保存本轮失败001、成功002、零模型统计准备、构建/测试日志与新bin身份，共406路径、压缩182,055,231B/6片。它与历史样本分开，恢复命令同上换bundle路径；新的只读语义复核入口为 `joint/reproduction/replay_fresh.py`。

## 固定142进程统计入口

```powershell
./.venv/Scripts/python.exe joint/reproduction/run_statistics.py prepare --model '<模型>' --cuda-root '<CUDA目录>' --build-dir joint/build-reproduction/model --correctness-dir "$env:GPU_RA_DATA_ROOT/../runs/v3-correctness-new-001" --output "$env:GPU_RA_DATA_ROOT/../runs/v3-statistics-new-001"
./.venv/Scripts/python.exe joint/reproduction/run_statistics.py check --output "$env:GPU_RA_DATA_ROOT/../runs/v3-statistics-new-001"
```

prepare启动零个模型，重新比较并封存14组gold、身份和协议SHA。顺序、7组、每规模8个配对区块和4对A/A与原142进程设计逐项相同，篡改会在启动前拒绝。只有显式 `python joint/reproduction/run_statistics.py run --output "$env:GPU_RA_DATA_ROOT/../runs/v3-statistics-new-001"` 才启动长统计批次，保留原端点、完整审计、停止/区间/判定规则，不替换失败样本或追加至显著。详见 [RUNNING.md](joint/reproduction/RUNNING.md)。

## 本轮实际验证（2026-11-15）

- 独立源码重建及最终参数化入口通过；4276源匹配，重入无须重编。
- 608/608 CPU、28个B2场景、6244请求及199808字节/等待检查、范围负例、共享介质读写读/排空全部通过。
- 新 `correctness-new-002` F1/F2共16组全部通过，完整logits、KV内容、依赖、预算和排空符合原合同。
- 新142进程入口prepare/check通过，14gold及142行顺序核验，启动零个统计模型；三类协议篡改负例被拒绝。
- 历史16+142进程全量离线重验通过：2659原证据哈希、790模块哈希、42,762,240权重请求及997,136 KV记录，所有原统计重算一致。
- 初次模型批次被辅助Python资源冲突守卫停止，失败批次和最初构建解释器/Unity接线日志全部保留；新目录完整重跑，未调整验收阈值、资源规则或超时。

本轮未重新执行142进程统计采样，不形成新的统计性能结论。结构化验证见 `joint/provenance/migration-validation`；分发身份由 `provenance/delivery-manifest.json` 管理，旧release-manifest和原封存build条目仅作历史记录。第一版为main，第二版为codex/v2-readonly，本版的数据外置接线与三版统一规则一致。
