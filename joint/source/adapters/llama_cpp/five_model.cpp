#include "llama.h"
#include "ggml-backend.h"
#include "instrumented_bridge.hpp"
#include "restored_routes.hpp"
#include "hbflab/platform.hpp"
#include "five_capture.hpp"
#include "q_auto_model.hpp"
#ifdef HBFLAB_FIVE_MEDIA_BUILD
#include "five_media_api.hpp"
#include "five_media_projection.hpp"
#endif
#include <algorithm>
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
using Clock = std::chrono::steady_clock;
static void need(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
static void ck(cudaError_t value) { if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value)); }
static void save(const fs::path &path, const json &value) {
    std::ofstream f(path); f << value.dump(2) << '\n'; f.close(); need(bool(f), "inventory write failed");
}
#ifdef HBFLAB_FIVE_MEDIA_BUILD
struct FiveMediaCase {
    bool enabled = false, active_device = false;
    std::uint64_t tiles_at_begin = 0;
    fs::path directory;
    json report;
    ~FiveMediaCase() { if (active_device) hbflab_five_media_finish(); }

    void start(const char *mode, const std::vector<const ggml_tensor *> &weights,
               const std::vector<std::uint64_t> &bases, const fs::path &out) {
        if (!mode) return;
        const std::string arm(mode);
        need(arm == "native" || arm == "timed", "five media mode must be native or timed");
        need(weights.size() == 5 && bases.size() == weights.size(), "five media selection count");
        directory = out;
        enabled = true;
        std::vector<five_media::RangeSpec> ranges;
        json registered = json::array();
        for (std::size_t i = 0; i < weights.size(); ++i) {
            const auto *w = weights[i];
            cudaPointerAttributes attributes{};
            ck(cudaPointerGetAttributes(&attributes, w->data));
            need(attributes.type == cudaMemoryTypeDevice && attributes.device == 0 && ggml_is_contiguous(w),
                 "selected range must be contiguous CUDA0 device memory");
            const std::string name(w->name);
            const bool retain = name == "blk.10.ffn_norm.weight" || name == "token_embd.weight";
            const auto bytes = ggml_nbytes(w);
            ranges.push_back({w->data, bytes, bases[i], std::uint32_t(retain)});
            registered.push_back({{"name", name}, {"base", reinterpret_cast<std::uintptr_t>(w->data)},
                {"bytes", bytes}, {"logical", bases[i]}, {"retain_words", retain}});
        }
        if (arm == "timed") {
            ck(static_cast<cudaError_t>(hbflab_five_media_begin(ranges.data(), ranges.size())));
            active_device = true;
        }
        report = {{"mode", arm}, {"registrations", active_device ? ranges.size() : 0}, {"media_finished", false},
            {"ranges", std::move(registered)}, {"output_tile_rows", five_media_output_tile_rows},
            {"output_tile_launches", json::array()}};
    }
    void step_begin() {
        if (!enabled) return;
        tiles_at_begin = hbflab_five_projection_tiles();
        if (active_device) ck(static_cast<cudaError_t>(hbflab_five_media_step_begin()));
    }
    void step_end(unsigned step) {
        if (!enabled) return;
        if (active_device) ck(static_cast<cudaError_t>(hbflab_five_media_step_end(directory.c_str(), step)));
        const auto launches = hbflab_five_projection_tiles() - tiles_at_begin;
        constexpr auto output_rows = 151936u;
        need(launches == (output_rows + five_media_output_tile_rows - 1) / five_media_output_tile_rows,
             "full native output projection tile count mismatch");
        report["output_tile_launches"].push_back(launches);
    }
    void finish() {
        if (!enabled) return;
        if (active_device) {
            const auto result = hbflab_five_media_finish();
            active_device = false;
            ck(static_cast<cudaError_t>(result));
            report["media_finished"] = true;
        }
        save(directory / "five-media.json", report);
    }
};
#endif
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
    bool single_node = false;
    FiveCapture capture;
    QAutoModel q_auto;
};
static bool observe(ggml_tensor *node, bool ask, void *opaque) {
    auto &s = *static_cast<Observation *>(opaque);
    if (s.step < 0) return false;
    if (!ask) {
        s.q_auto.observe_after(node);
        s.capture.after(node);
        return true;
    }
    s.capture.before(node,s.step,s.selected);
    restored_before(node);
    for(const auto* src:node->src)if(src && s.selected.count(src))++s.sequence;
    const bool capture_q = s.q_auto.observe_before(node);
    return capture_q || s.single_node || !s.capture.root.empty();
}
static int run(int argc, char **argv) {
    const auto program_start = Clock::now();
    need(argc == 6, "usage: five_model model prompt output selection.json outputs");
    const char *token_only = std::getenv("HBFLAB_SAMPLED_TOKEN_ONLY");
    need(token_only && std::string(token_only) == "1", "explicit token-only graph setting required");
    const fs::path out = argv[3]; need(!fs::exists(out), "output already exists"); fs::create_directories(out);
    const char *route = std::getenv("HBFLAB_SAMPLED_ROUTE");
    const bool sampled = route != nullptr;
    need(sampled,"five model requires explicit native/symbolic/ptx/cuda route");

    const char *audit=std::getenv("HBFLAB_SAMPLED_AUDIT");need(audit && (std::string(audit)=="full" || std::string(audit)=="summary"),"explicit five audit mode required");
    const bool summary=std::string(audit)=="summary";
    const char *kv = std::getenv("HBFLAB_JOINT_KV");
    const bool paged = sampled && kv && std::string(kv) == "paged";
    if (sampled) {
        need(kv && (std::string(kv) == "native" || std::string(kv) == "paged"), "explicit sampled KV setting required");
        hbflab::platform::set_environment("HBFLAB_M4_PAGED", paged ? "1" : "0");
    }
    need(!paged,"restored selected routes require native KV");
    const int outputs = std::stoi(argv[5]); need(outputs>=1 && outputs<=4, "selected model supports 1 to 4 outputs");
    std::ifstream spec_file(argv[4]); json spec; spec_file >> spec;
    std::ifstream prompt_file(argv[2], std::ios::binary); need(bool(prompt_file), "prompt missing");
    const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
    json budgets = {{"before_model", memory()}};
    if (sampled) {
        const char *profile_path = std::getenv("HBFLAB_SAMPLED_PROFILE"); need(profile_path, "sampled profile required");
        std::ifstream file(profile_path); json profile; file >> profile;
        const auto gpu_required=profile.at("budgets").at("gpu_required_bytes").get<uint64_t>();
        const auto host_required=profile.at("budgets").at("host_required_bytes").get<uint64_t>();
        need(gpu_required>=2308702208ull && host_required>=1006632960ull,"sampled profile weakens original minimum budgets");
        const auto host_available=hbflab::platform::available_host_memory();
        need(host_available>=host_required && budgets.at("before_model").at("free_bytes").get<uint64_t>()>=gpu_required,"sampled model memory budget unavailable");
        budgets["host_available_before_model"]=host_available;
        need(profile.at("numerical").at("max_abs")==0.2 && profile.at("numerical").at("rms")==0.025 &&
             profile.at("numerical").at("tokens_exact")==true,"sampled numerical policy changed");
    }
    llama_backend_init(); auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999; mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.main_gpu = 0;
    auto *gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU); need(gpu, "GPU backend missing");
    llama_model_tensor_buft_override overrides[] = {{"^token_embd\\.weight$", ggml_backend_dev_buffer_type(gpu)}, {nullptr, nullptr}};
    mp.tensor_buft_overrides = overrides;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    need(bool(model), "model load failed"); budgets["after_model"] = memory();
    Observation observations; observations.single_node = std::string(route)=="symbolic";
    const char *five_media_mode = std::getenv("HBFLAB_FIVE_MEDIA");
    if (const char *capture = std::getenv("HBFLAB_FIVE_CAPTURE")) observations.capture.root = capture;
    else if (five_media_mode) observations.capture.root = out / "fixtures";
    if (!observations.capture.root.empty()) {
        need(std::string(route) == "native", "capture requires native graph");
        need(!fs::exists(observations.capture.root), "capture root exists");
        fs::create_directories(observations.capture.root);
    }
    json tensors = json::array();
    for (const auto &item : spec.at("all_tensors")) {
        const auto name = item.at("name").get<std::string>();
        const auto *t = hbflab_m2_find_tensor(model.get(), name.c_str()); need(t, "tensor missing after load");
        need(ggml_nbytes(t) == item.at("bytes").get<size_t>(), "loaded tensor size mismatch");
        if(item.at("selected").get<bool>())need(item.at("dtype")==(t->type==GGML_TYPE_F32?"F32":t->type==GGML_TYPE_F16?"F16":"unsupported"),"selected weight type mismatch");
        auto row = info(t); row["selected"] = item.at("selected");
        cudaPointerAttributes attributes{}; ck(cudaPointerGetAttributes(&attributes, t->data));
        row["cuda_memory_type"] = int(attributes.type); row["cuda_device"] = attributes.device;
        tensors.push_back(row);
        if (item.at("selected").get<bool>()) observations.selected.emplace(t, name);
    }
    need(!observations.selected.empty(), "selected tensor identity mismatch");
    save(out / "loaded-tensors.json", tensors);
    const auto *vocab = llama_model_get_vocab(model.get());
    int n = -llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()), nullptr, 0, true, true);
    need(n > 0 && n + outputs <= 512, "prompt outside context");
    std::vector<llama_token> tokens(n);
    need(llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()), tokens.data(), n, true, true) == n, "tokenization failed");
    struct Cleanup {~Cleanup(){restored_abort();}} cleanup;
    std::vector<const ggml_tensor*> selected;
    std::vector<uint64_t> bases;
    for(const auto& entry:spec.at("all_tensors"))if(entry.at("selected").get<bool>()){
        auto* w=hbflab_m2_find_tensor(model.get(),entry.at("name").get<std::string>().c_str());
        selected.push_back(w);bases.push_back(entry.at("logical_base"));
    }
    // Register sparse ranges in logical order independently of JSON list order.
    std::vector<std::pair<uint64_t,const ggml_tensor*>> ordered;
    for(size_t i=0;i<bases.size();++i)ordered.emplace_back(bases[i],selected[i]);
    std::sort(ordered.begin(),ordered.end());
    for(size_t i=0;i<ordered.size();++i){bases[i]=ordered[i].first;selected[i]=ordered[i].second;}
    if (const char *mode = std::getenv("HBFLAB_AUTO_Q_MODE")) {
        need(std::string(route) == "native" && selected.size() == 1 && observations.capture.root.empty(),
             "Q auto trace requires one selected tensor on the native route");
        observations.q_auto.bind(mode, selected.front(), bases.front(), out, n, outputs);
    }
    restored_bind(route,selected.data(),bases.data(),unsigned(selected.size()),
        spec.at("logical_weight_layout_bytes"),std::getenv("HBFLAB_SAMPLED_PROFILE"),
        std::getenv("HBFLAB_SYMBOLIC_CONFIG"),argv[3],!summary);
#ifdef HBFLAB_FIVE_MEDIA_BUILD
    FiveMediaCase five_media_case;
    if (five_media_mode) {
        need(std::string(route) == "native" && !std::getenv("HBFLAB_AUTO_Q_MODE") && !paged &&
             n == 44 && outputs == 2, "five media requires native graph, 44 input tokens and 2 output tokens");
        five_media_case.start(five_media_mode, selected, bases, out);
    }
#else
    need(!five_media_mode, "five media mode requires an instrumented build");
#endif
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = 256; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F16; cp.offload_kqv = cp.op_offload = true;
    cp.cb_eval = observe; cp.cb_eval_user_data = &observations;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    need(bool(ctx), "context init failed"); budgets["after_context"] = memory();
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
    need(bool(batch) && bool(sampler), "batch/sampler initialization failed");
    auto input = tokens; llama_pos position = 0; json generated = json::array(), steps = json::array();
    const auto vocabulary=llama_vocab_n_tokens(vocab);
    std::vector<float> all_logits(size_t(outputs)*vocabulary);
    std::ofstream logits_file(out / "logits.f32", std::ios::binary);
    ck(cudaDeviceSynchronize());
    const auto generation_start = Clock::now();
    const double initialization_ms = std::chrono::duration<double,std::milli>(generation_start - program_start).count();
    std::vector<double> emissions;
    json native_positions = json::array();
    for (int step = 0; step < outputs; ++step) {
        if (sampled) restored_step_begin(step);
        observations.q_auto.begin_step(step);
#ifdef HBFLAB_FIVE_MEDIA_BUILD
        five_media_case.step_begin();
#endif
        observations.step = step; llama_batch_ext_clear(batch.get());
        for (auto token : input) {
            const int index = llama_batch_ext_add_token(batch.get(), 0, token); const auto pos = position++;
            need(index >= 0 && llama_batch_ext_set_pos(batch.get(), index, &pos), "batch token failed");
        }
        need(llama_batch_ext_set_output_logits(batch.get(), int32_t(input.size() - 1), true), "logits selection failed");
        need(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "model process failed");
        llama_synchronize(ctx.get());
#ifdef HBFLAB_FIVE_MEDIA_BUILD
        five_media_case.step_end(unsigned(step));
#endif
        observations.q_auto.end_step();
        if (sampled) restored_step_end();
        if (sampled && !paged) {
            auto memory = llama_get_memory(ctx.get());
            const auto maximum = llama_memory_seq_pos_max(memory,0);
            need(llama_memory_seq_pos_min(memory,0) == 0 && maximum == n + step - 1, "native KV sequence position mismatch");
            native_positions.push_back({{"step",step},{"min",0},{"max",maximum}});
        }
        float *logits = llama_get_logits_ith(ctx.get(), -1); need(logits, "missing logits");
        
        const auto token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        need(!llama_vocab_is_eog(vocab, token), "early EOG");
        emissions.push_back(std::chrono::duration<double,std::milli>(Clock::now() - generation_start).count());
        std::copy(logits,logits+vocabulary,all_logits.data()+size_t(step)*vocabulary);
        steps.push_back({{"step", step}, {"input_tokens", input}, {"output_token", token}});
        generated.push_back(token); input.assign(1, token);
    }
    const auto generation_end = Clock::now();
    for(float x:all_logits)need(std::isfinite(x),"nonfinite logits");
    logits_file.write(reinterpret_cast<const char *>(all_logits.data()),all_logits.size()*sizeof(float));
    if(!observations.capture.root.empty())need(observations.capture.completed==10 && observations.capture.pending.empty(),"ten native fixtures required");
    if (sampled && !paged) {
        const auto bytes = llama_state_seq_get_size(ctx.get(),0);
        need(bytes > 0 && bytes <= 128ull*1024*1024, "native KV serialization budget");
        std::vector<uint8_t> snapshot(bytes);
        need(llama_state_seq_get_data(ctx.get(),snapshot.data(),snapshot.size(),0) == bytes, "native KV snapshot incomplete");
        std::ofstream snapshot_file(out / "native-kv-state.bin",std::ios::binary);
        snapshot_file.write(reinterpret_cast<const char *>(snapshot.data()),snapshot.size());snapshot_file.close();
        need(bool(snapshot_file), "native KV snapshot write failed");
        save(out / "native-kv-positions.json",native_positions);
    }
    if (sampled) {
        restored_finish();
        observations.q_auto.finish();
#ifdef HBFLAB_FIVE_MEDIA_BUILD
        five_media_case.finish();
#endif
        auto memory = llama_get_memory(ctx.get());
        need(llama_memory_seq_rm(memory,0,0,-1), "framework sequence release failed");
        need(llama_memory_seq_pos_min(memory,0) < 0 && llama_memory_seq_pos_max(memory,0) < 0, "framework sequence not empty after release");
        save(out / "framework-release.json",{{"status","pass"},{"sequence",0},{"empty",true}});
    }
    const double final_audit_ms = std::chrono::duration<double,std::milli>(Clock::now() - generation_end).count();
    observations.step=-1;logits_file.close();need(bool(logits_file),"logits write failed");
    budgets["after_generation"] = memory(); save(out / "resources.json", budgets);
    save(out / "summary.json", {{"status", "pass"}, {"purpose", sampled ? "sampled_model" : "stage1_inventory_only"},
        {"prompt_tokens", tokens}, {"generated_tokens", generated}, {"steps", steps},
        {"selected_tensors", observations.selected.size()},{"audit_mode",summary?"summary":"full"},{"generated_steps",outputs},{"capture_fixtures",observations.capture.completed}, {"consumer_records", observations.sequence},
        {"arithmetic", "F32 MUL_MAT"}, {"kv", paged ? "modeled F16" : "native F16"}, {"graphs", false}, {"fusion", false}, {"flash", false},
        {"context", 512}, {"batch", 256}, {"ubatch", 256}, {"threads", 8},
        {"embedding_buffer_override", "CUDA0"}, {"token_only_graph", true},
        {"initialization_wall_ms",initialization_ms},{"emission_wall_ms",emissions},{"ttft_wall_ms",emissions.front()},
        {"tpot_wall_ms",outputs>1?(emissions.back()-emissions.front())/(outputs-1):0.0},{"final_audit_and_drain_wall_ms",final_audit_ms},
        {"generation_until_last_token_wall_ms",emissions.back()},{"common_clock_qualified",false}});
    return 0;
}
int main(int argc, char **argv) {
    try { return run(argc, argv); }
    catch (const std::exception &e) { std::cerr << "five model: " << e.what() << '\n'; return 1; }
}
