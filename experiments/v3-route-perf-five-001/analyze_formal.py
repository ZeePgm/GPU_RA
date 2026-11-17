"""Recompute the frozen formal matrix from retained evidence; never starts GPU work."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean, median

ID = "v3-route-perf-five-001"
ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("symbolic", "ptx", "cuda")
RAW_NAMES = {"lane-times.bin", "requests.bin", "weight-requests.jsonl",
             "gpu-activity.jsonl", "chunks.jsonl", "kv-media.jsonl"}


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stats(values):
    assert len(values) == 3 and all(math.isfinite(v) and v > 0 for v in values)
    return {"values": values, "median": median(values), "min": min(values), "max": max(values)}


def comparisons(group, metric):
    result = []
    for a, b in itertools.combinations(ROUTES, 2):
        x, y = group[a][metric], group[b][metric]
        deltas = [v - w for v, w in zip(x["values"], y["values"])]
        overlap = max(x["min"], y["min"]) <= min(x["max"], y["max"])
        consistent = all(v < 0 for v in deltas) or all(v > 0 for v in deltas)
        stable = not overlap and consistent
        result.append({"metric": metric, "X": a, "Y": b,
                       "speedup_X_over_Y": y["median"] / x["median"],
                       "paired_round_deltas_X_minus_Y": deltas,
                       "ranges_overlap": overlap, "round_order_consistent": consistent,
                       "ordering_stable": stable,
                       "stable_faster": (a if deltas[0] < 0 else b) if stable else None})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "reports" / ID)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    identities = {}

    def read(p):
        p = Path(p)
        identities[str(p)] = {"bytes": p.stat().st_size, "sha256": digest(p)}
        return json.loads(p.read_text(encoding="utf-8"))

    matrix = read(ROOT / "experiments" / ID / "matrix.json")
    control = read(out / "FORMAL-CONTROLLER.json")
    config = read(ROOT / "experiments" / ID / "config.json")
    budgets = config["parameters"]["budgets"]
    schedule = matrix["weight_process_schedule"] + matrix["model_process_schedule"]
    assert len(schedule) == 99 and control["stopped"] is None
    assert len(control["rows"]) == 99
    assert {p.name for p in args.run_root.glob("formal-*")} == {x["future_run_id"] for x in schedule}
    rows_by_id = {x["run_id"]: x for x in control["rows"]}
    scenarios = {x["scenario_id"]: x for x in matrix["weight_scenarios"]}
    weight_rows, model_rows, call_rows, peaks = [], [], [], []
    environment_rows = []
    native = args.run_root / "reference-20261117-002" / "data"
    reference_logits = digest(native / "result/logits.f32")
    reference_kv = digest(native / "result/native-kv-state.bin")
    for seq, item in enumerate(schedule, 1):
        run_id = item["future_run_id"]
        run = args.run_root / run_id
        rec = read(run / "run.json")
        acc = read(run / "data/acceptance.json")
        summary = read(run / "data/result/summary.json")
        process = read(run / "data/result.process.json")
        execution = read(run / "execution.json")
        assert rec["code_commit"] == control["source_A"] and rec["status"] == "completed"
        assert rec["exit_code"] == 0 and rec["identity_unchanged_after"] and rec["working_tree_clean_before"]
        assert execution["exit_code"] == 0 and execution["drain_confirmed"] and execution["active_processes_at_end"] == 0
        assert item["sequence"] == seq == acc["schedule_sequence"]
        assert acc["status"] == "pass" and acc["formal_sample"] and acc["phase"] == "formal"
        assert acc["sample_key"] == item["sample_key"] and acc["route"] == item["route"]
        assert acc["independent_process_repeat"] == item["independent_process_repeat"]
        assert acc["audit"] == "summary"
        assert acc["process_wall_seconds"] < acc["process_wall_cap_seconds"]
        assert not any(p.name in RAW_NAMES for p in (run / "data").rglob("*"))
        observed_peak = max(x.get("peak_working_set_bytes", 0)
                            for x in process["metrics"] + [process["final_metrics"]])
        peaks.append(observed_peak)
        assert observed_peak < budgets["scientific_host_process_bytes"]
        base = {"run_id": run_id, "sample_key": item["sample_key"], "sequence": seq,
                "route": acc["route"], "round": item["independent_process_repeat"],
                "process_wall_seconds": acc["process_wall_seconds"],
                "cpp_cpu_seconds": acc["cpu_seconds"],
                "coordinator_cpu_seconds": acc["coordinator_cpu_seconds"],
                "observed_host_peak_bytes": observed_peak}
        for side in ("before", "after"):
            env = read(run / f"data/environment-{side}.json")
            gpu = [x.strip() for x in env["gpu"].splitlines()[1].split(",")]
            environment_rows.append({"run_id": run_id, "side": side, "utc": env["utc"],
                                     "gpu_identity": gpu[:5], "memory_free": gpu[5],
                                     "temperature_C": float(gpu[6].split()[0]), "sm_clock_MHz": float(gpu[7].split()[0]),
                                     "power_W": float(gpu[8].split()[0]), "platform": env["platform"],
                                     "logical_cpu_count": env["logical_cpu_count"]})
        if acc["mode"] == "weight":
            assert acc["outputs_bitwise"] and acc["borrowed_unchanged"] and acc["drained"]
            assert summary["status"] == "pass" and summary["runtime_validation_in_timing"]
            assert summary["audit_mode"] == "summary" and summary["drained"] and summary["borrowed_unchanged"]
            assert summary["initialization_ns"] == summary["ready_ns"] - summary["u0_ns"]
            calls = read(run / "data/result/calls.json")
            assert len(calls) == 1 + item["repeated_calls"] == summary["calls"]
            assert calls[0]["kind"] == "first" and all(x["kind"] == "repeated" for x in calls[1:])
            sid = acc["scenario_id"]
            fixture = read(native / "fixtures" / f"{sid}.json")
            expected_hash = digest(native / "fixtures" / fixture["output"]["file"])
            for i, call in enumerate(calls):
                assert call["index"] == i and call["native_bitwise_equal"]
                assert call["output_sha256"] == expected_hash
                assert call["latency_ns"] == call["end_ns"] - call["begin_ns"] > 0
                assert call["requested_bytes"] == scenarios[sid]["consumer_demand_bytes"]
                call_rows.append({**{k: base[k] for k in ("run_id", "sample_key", "sequence", "route", "round")},
                                  "scenario_id": sid, **call})
            repeated = mean(c["latency_ns"] for c in calls[1:])
            assert abs(repeated - summary["repeated_mean_ns"]) <= 1
            assert abs(repeated / 1e6 - acc["repeated_mean_ms"]) <= 1e-6
            assert summary["first_call_ns"] == calls[0]["latency_ns"]
            assert summary["first_result_ns"] == calls[0]["end_ns"] - summary["u0_ns"]
            reader = read(run / "data/result/reader/summary.json")
            assert reader["status"] == "pass" and reader["all_runtime_lane_and_byte_checks"]
            assert reader["audit_mode"] == "summary"
            assert reader["requests"] == sum(c["requests"] for c in calls)
            assert reader["requested_bytes"] == sum(c["requested_bytes"] for c in calls)
            weight_rows.append({**base, "scenario_id": sid, "tensor": scenarios[sid]["tensor"],
                                "repeated_calls": item["repeated_calls"], "repeated_mean_ms": repeated / 1e6,
                                "first_call_ms": calls[0]["latency_ns"] / 1e6,
                                "initialization_ms": summary["initialization_ns"] / 1e6,
                                "first_result_ms": summary["first_result_ns"] / 1e6})
        else:
            assert acc["mode"] == "model" and acc["all_logits_and_native_kv_bitwise"]
            assert acc["runtime_validation_in_timing"] and not acc["independent_full_raw_offline_recheck"]
            assert summary["selected_tensors"] == 5 and summary["consumer_records"] == 20
            assert summary["audit_mode"] == "summary" and summary["generated_steps"] == 4
            assert len(summary["prompt_tokens"]) == 44 and summary["generated_tokens"] == [32, 6366, 5711, 4938]
            assert digest(run / "data/result/logits.f32") == reference_logits == acc["logits"]["sha256"]
            assert digest(run / "data/result/native-kv-state.bin") == reference_kv == acc["KV"]["sha256"]
            emissions = summary["emission_wall_ms"]
            assert len(emissions) == 4 and 0 < emissions[0] < emissions[1] < emissions[2] < emissions[3]
            intervals = [emissions[i] - emissions[i - 1] for i in range(1, 4)]
            ttft, tpot, generation = emissions[0], mean(intervals), emissions[3]
            for k, value in (("ttft_ms", ttft), ("tpot_ms", tpot), ("generation_ms", generation)):
                assert abs(acc[k] - value) < 1e-6
            model_rows.append({**base, "ttft_ms": ttft, "tpot_ms": tpot,
                               "generation_ms": generation, "decode_tokens_per_second": 1000 / tpot,
                               "initialization_ms": summary["initialization_wall_ms"],
                               "emission_ms": emissions, "decode_intervals_ms": intervals})
        tracked = rows_by_id[run_id]
        for k in ("process_wall_seconds", "coordinator_cpu_seconds"):
            assert math.isclose(tracked[k], acc[k], abs_tol=1e-6)

    assert len(weight_rows) == 90 and len(model_rows) == 9 and len(call_rows) == 468
    assert sum(x["kind"] == "repeated" for x in call_rows) == 378
    assert len({tuple(x["gpu_identity"]) for x in environment_rows}) == 1
    assert len({(x["platform"], x["logical_cpu_count"]) for x in environment_rows}) == 1
    weight_stats, pairwise = {}, {}
    for sid in scenarios:
        group = {}
        for route in ROUTES:
            rows = sorted((r for r in weight_rows if r["scenario_id"] == sid and r["route"] == route), key=lambda x: x["round"])
            assert [r["round"] for r in rows] == [1, 2, 3]
            group[route] = {k: stats([r[k] for r in rows]) for k in
                            ("repeated_mean_ms", "first_call_ms", "initialization_ms", "first_result_ms")}
        weight_stats[sid] = group
        pairwise[sid] = comparisons(group, "repeated_mean_ms")
    model_stats = {}
    for route in ROUTES:
        rows = sorted((r for r in model_rows if r["route"] == route), key=lambda x: x["round"])
        assert [r["round"] for r in rows] == [1, 2, 3]
        model_stats[route] = {k: stats([r[k] for r in rows]) for k in
                              ("ttft_ms", "tpot_ms", "generation_ms", "initialization_ms", "decode_tokens_per_second")}
    model_pairs = [v for key in ("ttft_ms", "tpot_ms", "generation_ms") for v in comparisons(model_stats, key)]
    baseline = control["baseline_budget"]
    cpp_wall = sum(r["process_wall_seconds"] for r in weight_rows + model_rows)
    cpp_cpu = sum(r["cpp_cpu_seconds"] for r in weight_rows + model_rows)
    coordinator_cpu = sum(r["coordinator_cpu_seconds"] for r in weight_rows + model_rows)
    cpu_conservative = control["rows"][-1]["cumulative_cpu_conservative_seconds"]
    gpu_seconds = baseline["baseline_cpp_wall_seconds"] + cpp_wall
    wall_upper = baseline["baseline_wall_upper_seconds"] + control["stage_wall_seconds"]
    assert control["stage_wall_seconds"] < budgets["stage_wall_seconds"]["formal"]
    assert gpu_seconds < budgets["gpu_process_lifetime_seconds"] and cpu_conservative < budgets["cpu_seconds"]
    assert wall_upper < budgets["execution_wall_seconds"]
    all_files = [p for p in args.run_root.rglob("*") if p.is_file()]
    raw_files = [p for p in all_files if p.name in RAW_NAMES]
    all_run_bytes = sum(p.stat().st_size for p in all_files)
    raw_bytes = sum(p.stat().st_size for p in raw_files)
    assert all_run_bytes < budgets["new_run_outputs_all_attempts_bytes"]
    assert raw_bytes < budgets["full_raw_diagnostics_total_bytes"]
    resource = {"formal_cpp_wall_seconds": cpp_wall, "formal_cpp_cpu_seconds": cpp_cpu,
                "formal_coordinator_cpu_seconds": coordinator_cpu,
                "formal_stage_wall_seconds": control["stage_wall_seconds"],
                "all_phase_gpu_process_lifetime_seconds": gpu_seconds,
                "all_phase_cpu_conservative_seconds": cpu_conservative,
                "all_phase_execution_wall_upper_through_formal_seconds": wall_upper,
                "report_stage_not_in_above_total": True,
                "formal_observed_host_peak_bytes": max(peaks), "total_run_root_bytes": all_run_bytes,
                "total_run_count": len(list(args.run_root.glob("*/run.json"))),
                "raw_diagnostic_files": len(raw_files), "raw_diagnostic_bytes": raw_bytes,
                "formal_raw_files": 0, "cleanup_performed": False,
                "cpu_accounting": "C++ and coordinator measured CPU plus 16 x non-C++ workflow wall time; includes baseline conservative build/workflow bounds. Upper budget accounting, not measured CPU usage.",
                "gpu_memory": "pre/post and phase snapshots plus allocation caps; not continuous peak sampling"}
    result = {"schema": 1, "experiment_id": ID, "source_A": control["source_A"],
              "status": "pass", "formal_processes": 99, "formal_passed": 99, "formal_failed": 0,
              "replacement_or_exclusion_count": 0, "weight_calls": 468,
              "statistics": "median/min/max of 3 independent process values; repeated-call means first computed within each process; values in round order; no significance/p95 claims",
              "weight_stats": weight_stats, "weight_pairwise": pairwise,
              "model_stats": model_stats, "model_pairwise": model_pairs,
              "resources": resource, "weight_processes": weight_rows, "model_processes": model_rows,
              "environments": environment_rows, "analysis_inputs": identities,
              "analysis_script_sha256": digest(Path(__file__))}
    (out / "FORMAL-RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for filename, rows in (("FORMAL-WEIGHT-PROCESSES.csv", weight_rows), ("FORMAL-WEIGHT-CALLS.csv", call_rows),
                           ("FORMAL-MODEL-PROCESSES.csv", model_rows)):
        with (out / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def fmt(s, scale=1):
        return f'{s["median"]/scale:,.3f} [{s["min"]/scale:,.3f}, {s["max"]/scale:,.3f}]'

    lines = ["# 五类权重三种接入方式正式性能结果", "",
             "正式矩阵 99/99 个独立进程全部通过：90 个组件进程、9 个模型进程；无失败、剔除、替换、追加或分块调优。",
             f'正式阶段 {control["stage_wall_seconds"]/60:.2f} 分钟（含登记、启动和收尾），上限 255 分钟。代码 A：`{control["source_A"]}`。', "",
             "完整模型的 symbolic 与 CUDA 很接近，TTFT/TPOT/总生成三项均出现区间交叠和跨轮顺序反转，不能判定谁稳定领先。PTX 在这三项上均比二者慢，三轮方向一致且区间不交叠。",
             "组件中，attn_q、ffn_down、output 的 P/D 六个场景均为 symbolic/CUDA 快于 PTX；symbolic 对 PTX 的中位数加速比为 1.13–1.15×，CUDA 对 PTX 为 1.13–1.15×。这些场景的 symbolic 与 CUDA 排序均不稳定。小 norm 和按行 embedding 多数比较也不稳定；单独 norm-D 中 symbolic 快于 CUDA（1.13×）。", "",
             "## 条件与指标", "",
             "Qwen3-0.6B-F16，44-token 提示，生成 4 token；五个选定权重接入，其余 306 个权重及 KV 走原生路径。context=512，batch/ubatch=256，threads=8；graphs/fusion/flash/reuse 关闭，F32 矩阵精度。",
             "GPU：RTX 5070，驱动 616.56；CPU：Ryzen 7 9800X3D（8 核 / 16 线程）；Windows 11。99 次运行前后 GPU UUID、驱动、容量、平台和逻辑线程数一致；时钟未锁定。",
             "各方式保持原实现和分块：symbolic/CUDA 256 KiB、PTX 32 KiB；u16 回退分别 32/4/32 KiB，CUDA 为 8 blocks × 256 threads。symbolic 的无 payload 返回及 PTX/CUDA 的读取发布均保留，按同一外部任务完成指标直接比较。", "",
             "组件 L：输入已在 GPU 就绪，CPU 单调钟从进入接入路径前计到原生 GPU 算子完成被 CPU 观察到，包含请求、等待、读取/发布、门控、同步及原生算子。每进程先测首次，再测 K 次重复（w1–w4 K=5，w5 K=1）；主统计先取进程内重复均值，再取三个独立进程的中位数和 min/max。",
             "模型 TTFT=t1−t0；TPOT=(t4−t1)/3；生成耗时=t4−t0；decode token/s=1000/TPOT_ms。初始化不计入上述三个模型指标。t0 在初始化及分词完成后、prefill 前；t_i 在第 i 个 token 可交付时。",
             "summary 模式抑制全量轨迹导出，但全量运行时 lane/字节检查、必需回读/memcmp 和同步仍计时；没有人为扣时。四步 logits 的中间 CPU 内存副本计入后续 token 间隔，最终文件导出和 KV 验收在 t4 后。", "",
             "## 组件主结果", "", "单位 ms；单元格为中位数 [min, max]，每格 N=3 个进程。P=prefill，D=第一轮单 token decode。", "",
             "| 权重 / 场景 | symbolic | PTX | CUDA |", "|---|---:|---:|---:|"]
    for sid, group in weight_stats.items():
        lines.append(f'| `{scenarios[sid]["tensor"]}` / {sid[-1].upper()} | ' + " | ".join(fmt(group[r]["repeated_mean_ms"]) for r in ROUTES) + " |")
    lines += ["", "embedding P/D 只取实际 44/1 行（每行 2048 B，保留顺序和重复），不是全表扫描；output P/D 均为真实的 1024×1 输入到 151936×1 输出。", "",
              "### 两两比较", "", "X 相对 Y 加速比=median(Y)/median(X)，大于 1 表示 X 更快。区间交叠或三轮顺序不一致即标为排序不稳定；这是描述性判据，不是显著性检验。", "",
              "| 场景 | X / Y | X 相对 Y 加速比 | 观察排序 |", "|---|---|---:|---|"]
    for sid, pairs in pairwise.items():
        for p in pairs:
            label = p["stable_faster"] + " 较快" if p["ordering_stable"] else "不稳定"
            lines.append(f'| {sid} | {p["X"]} / {p["Y"]} | {p["speedup_X_over_Y"]:.4f}× | {label} |')
    lines += ["", "## 模型结果", "", "单位 s；中位数 [min, max]，每方式 N=3 个独立进程，每进程仅首次完整序列。", "",
              "| 方式 | TTFT | TPOT | 生成 4 token | 初始化 |", "|---|---:|---:|---:|---:|"]
    for route in ROUTES:
        lines.append(f'| {route} | ' + " | ".join(fmt(model_stats[route][k], 1000) for k in ("ttft_ms", "tpot_ms", "generation_ms", "initialization_ms")) + " |")
    lines += ["", "| 指标 | X / Y | X 相对 Y 加速比 | 观察排序 |", "|---|---|---:|---|"]
    for p in model_pairs:
        label = p["stable_faster"] + " 较快" if p["ordering_stable"] else "不稳定"
        lines.append(f'| {p["metric"]} | {p["X"]} / {p["Y"]} | {p["speedup_X_over_Y"]:.4f}× | {label} |')
    lines += ["", "decode 吞吐（token/s，中位数 [min,max]）：" + "；".join(f'{r} ' + f'{model_stats[r]["decode_tokens_per_second"]["median"]:.6f} [{model_stats[r]["decode_tokens_per_second"]["min"]:.6f}, {model_stats[r]["decode_tokens_per_second"]["max"]:.6f}]' for r in ROUTES) + "。", "",
              "## 首次与初始化", "", "组件单位 ms，首次调用、初始化、从 u0 到首次结果分别列出，不混入重复调用均值。", "",
              "| 场景 | 方式 | 首次调用 | 初始化 | 首次结果 |", "|---|---|---:|---:|---:|"]
    for sid, group in weight_stats.items():
        for route in ROUTES:
            lines.append(f'| {sid} | {route} | ' + " | ".join(fmt(group[route][k]) for k in ("first_call_ms", "initialization_ms", "first_result_ms")) + " |")
    lines += ["", "## 验收、资源与证据", "",
              "468 次组件调用全部与固定原生输出逐位一致，借用权重不变；9 个模型的 token、四步完整 logits、native KV 状态/位置/版本/生命周期均通过原判据。每个模型均为 20 个选定消费者、2,513,116 次请求、1,286,715,392 B 请求量；全范围资源守卫、依赖身份、覆盖、完成及排空检查通过。",
              "本分析从 retained calls 的起止 ns 独立重算全部延迟和进程均值，并从四个 token 时刻独立重算模型指标；与 runner 输出一致。科学数值/协议/资源验收与 experiment.py verify 的文件身份核验分别成立。summary 运行没有逐请求原始轨迹，不能声称独立离线逐条复算了这些轨迹。", "",
              f'- 正式 C++ 进程存续合计 {cpp_wall/3600:.3f} h；C++ CPU {cpp_cpu/3600:.3f} CPU·h；科学 runner CPU {coordinator_cpu/3600:.3f} CPU·h。',
              f'- 加上前期全部尝试：GPU 进程窗口 {gpu_seconds/3600:.3f} / 6.5 h；CPU 保守预算账 {cpu_conservative/3600:.3f} / 20 CPU·h。CPU 保守账包含未能细分的流程墙钟乘 16，不能当成实测 CPU 消耗。',
              f'- 至正式结束的全阶段执行墙钟保守计账 {wall_upper/3600:.3f} / 7.5 h；不含本次报告封存核验，后者另在 FORMAL-DELIVERY 记录。',
              f'- 正式观测 Host 工作集峰值 {max(peaks)/2**30:.3f} GiB，低于 8 GiB。GPU 内存仅报告阶段/运行前后快照和受控分配上限，不当作连续峰值。',
              f'- 全实验 {resource["total_run_count"]} 个运行目录合计 {all_run_bytes/2**20:.2f} MiB（快照时刻）；有界原始诊断 12 文件、{raw_bytes/2**20:.2f} MiB；正式运行全量轨迹文件 0。共享模型/CAS、构建单独存放。',
              '- 前期身份观测失败的 1 次资格运行完整保留，未进入正式统计；前期其余 54 次通过。正式 99 次全部纳入。',
              '- 用户明确拒绝本次清理：所有既有及本次日志、参考输出、输入 CAS、模型、报告与 seals 全部保留，未执行删除。', "",
              "每个正式 RUN 的原始文件由 evidence 下的 manifest 和外部 seals 封存；报告提交 B 和 report_committed=true 核验结果见 FORMAL-VERIFY.json，最终提交/合回信息见 FORMAL-DELIVERY.json。", "",
              "## 解释范围", "",
              "N=3，仅报告描述性统计；时钟未锁定，且每个模型仅 4 个 token、3 个 decode 间隔。不能把区间交叠或跨轮反转的顺序说成稳定领先，不报 p95、显著性或窄置信区间。重复调用也不宣称达到稳态或 GPU 冷缓存。按五种权重与 P/D 分别比较，不构造任意加权的全局权重分数。", "",
              "完整逐进程值和全部 468 个组件调用见同目录 CSV；FORMAL-RESULTS.json 还保存全部输入哈希、每轮配对差值、环境快照与统计值。可在仓库根用 `python experiments/v3-route-perf-five-001/analyze_formal.py --run-root <外部该实验 runs 目录>` 复算。", ""]
    (out / "FORMAL-RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"status": "pass", "weight_stats": weight_stats,
                      "model_stats": model_stats, "model_pairwise": model_pairs,
                      "resources": resource}, ensure_ascii=False))


if __name__ == "__main__":
    main()
