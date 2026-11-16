#include "llama.h"
#include "ggml-backend.h"
#include "instrumented_bridge.hpp"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
static void need(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
static void ck(cudaError_t value) { if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value)); }
static void save(const fs::path &path, const json &value) {
    std::ofstream f(path); f << value.dump(2) << '\n'; f.close(); need(bool(f), "inventory write failed");
}
static json memory() {
    size_t available = 0, total = 0; ck(cudaMemGetInfo(&available, &total));
    return {{"free_bytes", available}, {"total_bytes", total}};
}
static json info(const ggml_tensor *t) {
    const ggml_tensor *storage = t;
    while (storage->view_src) storage = storage->view_src;
    return {{"name", t->name}, {"dtype", ggml_type_name(t->type)},
        {"shape", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}},
        {"strides", {t->nb[0], t->nb[1], t->nb[2], t->nb[3]}},
        {"bytes", ggml_nbytes(t)}, {"address", uintptr_t(t->data)},
        {"contiguous", ggml_is_contiguous(t)}, {"view_offset", t->view_offs},
        {"storage_address", uintptr_t(storage->data)},
        {"buffer", storage->buffer ? ggml_backend_buffer_name(storage->buffer) : "none"}};
}
struct Observation {
    int step = -1;
    uint64_t sequence = 0;
    std::map<const ggml_tensor *, std::string> selected;
    std::ofstream consumers;
    std::ofstream completed;
    std::string failure;
};
static bool observe(ggml_tensor *node, bool ask, void *opaque) {
    auto &s = *static_cast<Observation *>(opaque);
    if (s.step < 0) return false;
    if (!ask) {
        for (const auto *src : node->src) if (src && s.selected.count(src))
            s.completed << json({{"step", s.step}, {"weight", s.selected.at(src)},
                {"node_name", node->name}, {"op", ggml_op_name(node->op)}}).dump() << '\n';
        return true;
    }
    // Preserve the V3 F32 arithmetic choice for every matrix multiplication.
    hbflab_m2_graph_node(node);
    bool selected = false;
    try {
        for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
            const auto *src = node->src[slot];
            if (!src) continue;
            const auto found = s.selected.find(src);
            if (found == s.selected.end()) continue;
            selected = true;
            json row = {{"sequence", s.sequence++}, {"step", s.step},
                {"phase", s.step == 0 ? "prefill" : "decode"},
                {"weight", found->second}, {"source_slot", slot},
                {"op", ggml_op_name(node->op)}, {"node", info(node)}, {"source", info(src)},
                {"gate", "not_installed_stage1"}, {"ranges", json::array()}};
            if (node->op == GGML_OP_GET_ROWS) {
                const auto *ids = node->src[1];
                need(slot == 0 && ids && ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids), "unsupported gather indices");
                std::vector<int32_t> tokens(ggml_nelements(ids));
                ggml_backend_tensor_get(ids, tokens.data(), 0, tokens.size() * sizeof(int32_t));
                for (size_t i = 0; i < tokens.size(); ++i) {
                    need(tokens[i] >= 0 && tokens[i] < src->ne[1], "gather index outside weight");
                    row["ranges"].push_back({{"row", tokens[i]}, {"occurrence", i},
                        {"offset", uint64_t(tokens[i]) * src->nb[1]}, {"bytes", src->nb[1]}});
                }
                row["indices"] = tokens;
            } else {
                row["ranges"].push_back({{"offset", 0}, {"bytes", ggml_nbytes(src)}});
            }
            s.consumers << row.dump() << '\n'; need(bool(s.consumers), "consumer log failed");
        }
    } catch (const std::exception &e) { s.failure = e.what(); }
    return selected;
}
static int run(int argc, char **argv) {
    need(argc == 6, "usage: sampled_inventory model prompt output selection.json outputs");
    const char *token_only = std::getenv("HBFLAB_SAMPLED_TOKEN_ONLY");
    need(token_only && std::string(token_only) == "1", "explicit token-only graph setting required");
    const fs::path out = argv[3]; need(!fs::exists(out), "output already exists"); fs::create_directories(out);
    const int outputs = std::stoi(argv[5]); need(outputs == 8, "inventory freezes eight outputs");
    std::ifstream spec_file(argv[4]); json spec; spec_file >> spec;
    std::ifstream prompt_file(argv[2], std::ios::binary); need(bool(prompt_file), "prompt missing");
    const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
    json budgets = {{"before_model", memory()}};
    llama_backend_init(); auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999; mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.main_gpu = 0;
    auto *gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU); need(gpu, "GPU backend missing");
    llama_model_tensor_buft_override overrides[] = {{"^token_embd\\.weight$", ggml_backend_dev_buffer_type(gpu)}, {nullptr, nullptr}};
    mp.tensor_buft_overrides = overrides;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    need(bool(model), "model load failed"); budgets["after_model"] = memory();
    Observation observations; json tensors = json::array();
    for (const auto &item : spec.at("all_tensors")) {
        const auto name = item.at("name").get<std::string>();
        const auto *t = hbflab_m2_find_tensor(model.get(), name.c_str()); need(t, "tensor missing after load");
        need(ggml_nbytes(t) == item.at("bytes").get<size_t>(), "loaded tensor size mismatch");
        auto row = info(t); row["selected"] = item.at("selected");
        cudaPointerAttributes attributes{}; ck(cudaPointerGetAttributes(&attributes, t->data));
        row["cuda_memory_type"] = int(attributes.type); row["cuda_device"] = attributes.device;
        tensors.push_back(row);
        if (item.at("selected").get<bool>()) observations.selected.emplace(t, name);
    }
    need(observations.selected.size() == 14, "selected tensor identity mismatch");
    save(out / "loaded-tensors.json", tensors);
    observations.consumers.open(out / "consumers.jsonl"); need(bool(observations.consumers), "consumer log open failed");
    observations.completed.open(out / "consumers-completed.jsonl"); need(bool(observations.completed), "completion log open failed");
    const auto *vocab = llama_model_get_vocab(model.get());
    int n = -llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()), nullptr, 0, true, true);
    need(n > 0 && n + outputs <= 512, "prompt outside context");
    std::vector<llama_token> tokens(n);
    need(llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()), tokens.data(), n, true, true) == n, "tokenization failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = 256; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F16; cp.offload_kqv = cp.op_offload = true;
    cp.cb_eval = observe; cp.cb_eval_user_data = &observations;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    need(bool(ctx), "context init failed"); budgets["after_context"] = memory();
    // Probe candidate buffer sizes with the actual loaded model, outside generation.
    json allocations = json::array();
    for (const char *route : {"symbolic", "ptx", "cuda"}) {
        const size_t mirror = std::string(route) == "symbolic" ? 0 : 262144;
        const size_t validation = 262144;
        void *a = nullptr, *b = nullptr;
        if (mirror) ck(cudaMalloc(&a, mirror)); ck(cudaMalloc(&b, validation));
        allocations.push_back({{"route", route}, {"mirror_bytes", mirror}, {"validation_bytes", validation},
            {"allocation_check", "pass"}, {"free_after_allocation", memory()}});
        if (a) ck(cudaFree(a)); ck(cudaFree(b));
    }
    budgets["route_buffers"] = allocations;
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
    need(bool(batch) && bool(sampler), "batch/sampler initialization failed");
    auto input = tokens; llama_pos position = 0; json generated = json::array(), steps = json::array();
    std::ofstream logits_file(out / "logits.f32", std::ios::binary);
    for (int step = 0; step < outputs; ++step) {
        observations.step = step; llama_batch_ext_clear(batch.get());
        for (auto token : input) {
            const int index = llama_batch_ext_add_token(batch.get(), 0, token); const auto pos = position++;
            need(index >= 0 && llama_batch_ext_set_pos(batch.get(), index, &pos), "batch token failed");
        }
        need(llama_batch_ext_set_output_logits(batch.get(), int32_t(input.size() - 1), true), "logits selection failed");
        need(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "model process failed");
        llama_synchronize(ctx.get()); need(observations.failure.empty(), observations.failure.c_str());
        float *logits = llama_get_logits_ith(ctx.get(), -1); need(logits, "missing logits");
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) need(std::isfinite(logits[i]), "nonfinite logits");
        logits_file.write(reinterpret_cast<char *>(logits), llama_vocab_n_tokens(vocab) * sizeof(float));
        const auto token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        need(!llama_vocab_is_eog(vocab, token), "early EOG");
        steps.push_back({{"step", step}, {"input_tokens", input}, {"output_token", token}});
        generated.push_back(token); input.assign(1, token);
    }
    observations.step = -1; observations.consumers.close(); observations.completed.close(); logits_file.close();
    need(bool(observations.consumers) && bool(observations.completed) && bool(logits_file), "output close failed");
    budgets["after_generation"] = memory(); save(out / "resources.json", budgets);
    save(out / "summary.json", {{"status", "pass"}, {"purpose", "stage1_inventory_only"},
        {"prompt_tokens", tokens}, {"generated_tokens", generated}, {"steps", steps},
        {"selected_tensors", 14}, {"consumer_records", observations.sequence},
        {"arithmetic", "F32 MUL_MAT"}, {"kv", "native F16"}, {"graphs", false}, {"fusion", false}, {"flash", false},
        {"context", 512}, {"batch", 256}, {"ubatch", 256}, {"threads", 8},
        {"embedding_buffer_override", "CUDA0"}, {"token_only_graph", true}});
    return 0;
}
int main(int argc, char **argv) {
    try { return run(argc, argv); }
    catch (const std::exception &e) { std::cerr << "sampled inventory: " << e.what() << '\n'; return 1; }
}
