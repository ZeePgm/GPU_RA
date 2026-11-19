"""Run native and page-timed Q inference and audit the original loads independently."""
import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
prior_spec = importlib.util.spec_from_file_location("prior_q_observation", HERE.parent / "v3-q-auto-001/run.py")
prior = importlib.util.module_from_spec(prior_spec)
prior_spec.loader.exec_module(prior)
read, write, require, sha = prior.read, prior.write, prior.require, prior.sha


def media_evidence(folder, step, row, cfg, signature, coverage, gpu_base):
    phase = row["phase"]
    weight_bytes, page_bytes = cfg["bytes"], cfg["page_bytes"]
    pages = weight_bytes // page_bytes
    require(page_bytes == 4096 and pages == 1024 and pages * page_bytes == weight_bytes, "Q media geometry")
    hits = coverage["executed_sites"]
    require(len(hits) == 1 and all(hits[0][key] == value for key, value in signature.items()),
            f"{phase}: changed native input instruction or multiplicity")
    require(coverage["minimum_word_reads"] == coverage["maximum_word_reads"] == 1 and
            coverage["matched_bytes"] == weight_bytes, f"{phase}: duplicate or missing Q half")

    files = {}
    for key, suffix in (("media", "json"), ("observed", "u64"), ("resumed", "u64")):
        name = f"{key}-{step}.{suffix}"
        require(row[key + "_file"] == name, f"{phase}: wrong {key} evidence file")
        files[key] = folder / name
    receipts = read(files["media"])
    require(isinstance(receipts, list) and len(receipts) == pages, f"{phase}: incomplete page receipts")
    columns = ("logical", "request_ns", "begin_ps", "end_ps", "delay_ns", "sectors", "completion_order", "status",
               "released_base")
    for page, receipt in enumerate(receipts):
        require(all(type(receipt[key]) is int and 0 <= receipt[key] < 2**64 for key in columns),
                f"{phase} page {page}: invalid receipt field")
        require(receipt["logical"] == cfg["logical_base"] + page * page_bytes and
                receipt["released_base"] == gpu_base + page * page_bytes and receipt["status"] == 2 and
                receipt["sectors"] == page_bytes // 64,
                f"{phase} page {page}: wrong address, completion, or sector count")
        duration = receipt["end_ps"] - receipt["begin_ps"]
        require(duration > 0 and receipt["delay_ns"] == (duration + 999) // 1000 and
                0 < receipt["request_ns"] < 2**64 - receipt["delay_ns"],
                f"{phase} page {page}: invalid service time or GPU deadline")
    by_completion = sorted(receipts, key=lambda receipt: receipt["completion_order"])
    first = by_completion[0]["completion_order"]
    require(first == 1, f"{phase}: missing first page service")
    for index, receipt in enumerate(by_completion):
        require(receipt["completion_order"] == first + index and
                (index == 0 or receipt["begin_ps"] >= by_completion[index-1]["end_ps"]),
                f"{phase}: incomplete ordering or overlapping serialized page service")

    words_per_page = page_bytes // 2
    stride = signature["width"] // 2
    arrays = {}
    for key in ("observed", "resumed"):
        file = files[key]
        require(file.stat().st_size == pages * words_per_page * 8, f"{phase}: incomplete {key} times")
        by_page = np.fromfile(file, dtype="<u8").reshape(pages, words_per_page)
        samples = by_page[:, ::stride]
        require(samples.size == signature["dynamic_reads"] and np.all(samples > 0) and
                np.count_nonzero(by_page) == samples.size, f"{phase}: missing or misplaced {key} samples")
        arrays[key] = samples
    request = np.array([r["request_ns"] for r in receipts], dtype="<u8")[:, None]
    deadline = np.array([r["request_ns"] + r["delay_ns"] for r in receipts], dtype="<u8")[:, None]
    observed, resumed = arrays["observed"], arrays["resumed"]
    require(np.all(observed >= request), f"{phase}: GPU observation earlier than its page request")
    require(np.all(resumed >= observed), f"{phase}: resume before observation")
    require(np.all(resumed >= deadline), f"{phase}: a native Q load resumed before its page deadline")
    return {
        "pages": pages, "sectors": sum(r["sectors"] for r in receipts), "matched_loads": int(resumed.size),
        "earliest_begin_ps": by_completion[0]["begin_ps"], "latest_end_ps": by_completion[-1]["end_ps"],
        "first_completion_order": first, "last_completion_order": by_completion[-1]["completion_order"],
        "modeled_service_ps": sum(r["end_ps"] - r["begin_ps"] for r in receipts),
        "minimum_gpu_margin_after_deadline_ns": int(np.min(resumed - deadline)),
        "early_resumes": 0, "missing_timestamps": 0,
        "file_sha256": {key: sha(path) for key, path in files.items()},
    }


def inspect_process(folder, mode, cfg, signatures, sites, prompt_tokens):
    model, q = read(folder/"summary.json"), read(folder/"q-auto.json")
    require(model["status"] == "pass" and model["generated_tokens"] == cfg["output_tokens"] and
            model["prompt_tokens"] == prompt_tokens and model["generated_steps"] == model["consumer_records"] == 2 and
            model["selected_tensors"] == 1 and model["kv"] == "native F16", f"{mode}: model scope")
    selected = [t for t in read(folder/"loaded-tensors.json") if t["selected"]]
    require(len(selected) == 1 and selected[0]["name"] == cfg["weight"] and
            selected[0]["address"] == selected[0]["storage_address"] == q["device_address"] and
            selected[0]["cuda_memory_type"] == 2 and selected[0]["cuda_device"] == q["cuda_device"] == 0,
            f"{mode}: registered Q allocation identity")
    route = read(folder/"route-summary.json")
    require(route["drained"] and route["steps"] == route["consumers"] == 2 and
            route["requested_bytes"] == route["range_or_scalar_calls"] == 0,
            f"{mode}: another weight access route was active")
    timed = mode == "timed"
    require(q["finished"] and q["mode"] == mode and q["tensor"] == cfg["weight"] and q["bytes"] == cfg["bytes"] and
            q["logical_base"] == cfg["logical_base"] and q["registrations"] == 1 and
            q["device_registrations"] == int(timed) and len(q["steps"]) == 2, f"{mode}: registration scope")
    if timed:
        require(q["media_finished"] is True, "timed: EventLab did not finish and drain")
    logits = (folder/"logits.f32").read_bytes()
    vector_bytes = cfg["vocabulary"] * 4
    require(len(logits) == 2 * vector_bytes, f"{mode}: expected two complete logit vectors")
    phases = []
    for step, row in enumerate(q["steps"]):
        phase, columns = (("prefill", cfg["prompt_length"]), ("decode", 1))[step]
        require(row["step"] == step and row["phase"] == phase and row["weight_slot"] == 0 and
                row["op"] == "MUL_MAT" and row["consumer"] and row["activation_columns"] == columns and
                row["projection_values"] == columns * cfg["shape"][1] and
                row["projection_file"] == f"q-output-{step}.f32", f"{mode}/{phase}: wrong graph consumer")
        output = folder / row["projection_file"]
        require(output.stat().st_size == 4 * cfg["shape"][1] * columns and sha(output) == cfg["projection_sha256"][step],
                f"{mode}/{phase}: native Q projection")
        vector_hash = hashlib.sha256(logits[step*vector_bytes:(step+1)*vector_bytes]).hexdigest()
        require(vector_hash == cfg["logit_vector_sha256"][step], f"{mode}/{phase}: native logit reference")
        item = {"phase": phase, "graph_consumer": row["consumer"], "projection_sha256": sha(output),
                "logit_vector_sha256": vector_hash}
        if timed:
            item["coverage"] = prior.trace_coverage(folder, step, row, sites, cfg["bytes"])
            item["media"] = media_evidence(folder, step, row, cfg, signatures[step], item["coverage"],
                                           q["device_address"])
        phases.append(item)
    if timed:
        require(phases[0]["media"]["latest_end_ps"] <= phases[1]["media"]["earliest_begin_ps"],
                "Media clock ran backwards between native Q phases")
    positions = read(folder/"native-kv-positions.json")
    require(positions == [{"step": i, "min": 0, "max": cfg["prompt_length"]+i-1} for i in range(2)] and
            read(folder/"framework-release.json") == {"status": "pass", "sequence": 0, "empty": True} and
            sha(folder/"native-kv-state.bin") == cfg["native_kv_state_sha256"], f"{mode}: native KV state, positions or release")
    return {"phases": phases, "output_tokens": model["generated_tokens"], "kv_positions": positions,
            "kv_sha256": sha(folder/"native-kv-state.bin"), "device_registrations": q["device_registrations"],
            "registrations": q["registrations"], "media_finished": q.get("media_finished", False)}


def measure(args):
    definition = read(args.config)
    cfg = definition["parameters"]
    signatures = definition["acceptance"]["native_access_signatures"]
    out = args.output
    require(args.resource_root.resolve() == Path("E:/v1_update").resolve(), "Expected full outer resource root")
    require(args.config.resolve() == (HERE/"config.json").resolve(), "Use the registered configuration")
    require(cfg["modes"] == ["native", "timed"], "Native control and timed process required")
    out.mkdir(parents=True, exist_ok=False)
    receipt = read(args.receipt)
    require(receipt == prior.build_identity(args.build_dir), "Candidate build differs from frozen receipt")
    sites = prior.site_inventory(receipt)
    require(sha(args.model) == prior.S.MODEL_SHA and sha(prior.PROMPT) == prior.S.PROMPT_SHA, "Model or prompt identity")
    selection = out / "selection.json"
    prior.select_weight(cfg, selection)
    profile = read(prior.FIVE/"profile.json")
    profile.pop("sampled_scope")
    profile["model_path"] = str(args.model)
    write(out/"profile.json", profile)
    env = {"HBFLAB_JOINT_KV": "native", "HBFLAB_M4_PAGED": "0", "LLAMA_GRAPH_REUSE_DISABLE": "1",
           "HBFLAB_SAMPLED_TOKEN_ONLY": "1", "HBFLAB_SAMPLED_AUDIT": "summary", "HBFLAB_SAMPLED_ROUTE": "native",
           "HBFLAB_SAMPLED_PROFILE": str(out/"profile.json"),
           "HBFLAB_SYMBOLIC_CONFIG": str(ROOT/"joint/source/experiments/gpu_ffn/H0-512MiB.ini")}
    prompt_tokens = read(prior.FIVE/"input-spec.json")["prompt_tokens"]
    require(len(prompt_tokens) == cfg["prompt_length"], "Historical prompt token count")
    evidence = {}
    for mode in cfg["modes"]:
        child = out / mode
        command = [args.build_dir/"bin/five_model.exe", args.model, prior.PROMPT, child, selection,
                   str(len(cfg["output_tokens"]))]
        prior.X.execute(args, command, mode, child, extra_env={**env, "HBFLAB_AUTO_Q_MODE": mode}, require_ggml=True,
                        timeout=cfg["process_timeout_seconds"], disk_cap=cfg["process_disk_limit_bytes"])
        evidence[mode] = inspect_process(child, mode, cfg, signatures, sites, prompt_tokens)
    for name in ("logits.f32", "native-kv-state.bin", "native-kv-positions.json", "framework-release.json",
                 "q-output-0.f32", "q-output-1.f32"):
        require((out/"native"/name).read_bytes() == (out/"timed"/name).read_bytes(), "Native/timed bytes: " + name)
    require(receipt == prior.build_identity(args.build_dir), "Source/build changed during run")
    write(out/"acceptance.json", {"status": "passed", "code_commit": receipt["commit"], "source_tree": receipt["source_tree"],
          "build_receipt_sha256": sha(args.receipt), "configuration_sha256": sha(args.config), "modes": evidence,
          "timing_service_used": True, "media_page_bytes": cfg["page_bytes"]})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    receipt = sub.add_parser("receipt")
    run = sub.add_parser("measure")
    for p in (receipt, run):
        p.add_argument("--build-dir", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    for flag in ("cuda-root", "model", "receipt", "config", "resource-root"):
        run.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.action == "receipt":
        require(not args.output.exists() and not prior.git("status","--porcelain"), "Receipt requires new file and clean commit")
        write(args.output, prior.build_identity(args.build_dir))
    else:
        measure(args)


if __name__ == "__main__":
    main()
