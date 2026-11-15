#include "integrated_profile.hpp"
#include "hbflab/platform.hpp"
#include "llama.h"
#include "instrumented_bridge.hpp"
#include "paged_bridge.hpp"
#include "ggml-backend.h"
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static bool record_node_metadata=false;
struct M2Cleanup { ~M2Cleanup(){hbflab_m2_abort();} };
static std::mutex trace_mutex;
static std::ofstream kernel_log, node_log;
static std::atomic<int> current_step{-1};
static std::atomic<uint64_t> kernel_count{0}, dropped{0}, trace_errors{0};

static void check(CUptiResult result) {
    if (result != CUPTI_SUCCESS) {
        const char * text = nullptr;
        cuptiGetResultString(result, &text);
        throw std::runtime_error(std::string("CUPTI: ") + (text ? text : "unknown error"));
    }
}

static void CUPTIAPI request_buffer(uint8_t ** buffer, size_t * size, size_t * max_records) {
    *size = 8 * 1024 * 1024;
    *buffer = static_cast<uint8_t *>(hbflab::platform::aligned_allocate(*size, 8));
    *max_records = 0;
    if (!*buffer) { *size = 0; ++trace_errors; }
}

static void CUPTIAPI complete_buffer(CUcontext ctx, uint32_t stream, uint8_t * buffer, size_t, size_t valid) {
    try {
        CUpti_Activity * activity = nullptr;
        CUptiResult result;
        std::lock_guard<std::mutex> guard(trace_mutex);
        while ((result = cuptiActivityGetNextRecord(buffer, valid, &activity)) == CUPTI_SUCCESS) {
            if(activity->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
                const auto* c=reinterpret_cast<const CUpti_ActivityMemcpy6*>(activity);
                kernel_log<<json({{"kind","copy"},{"step",current_step.load()},{"start_ns",c->start},{"end_ns",c->end},
                                 {"bytes",c->bytes},{"direction",c->copyKind},{"stream",c->streamId},{"correlation",c->correlationId}}).dump()<<'\n';
                continue;
            }
            if (activity->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
            const auto * k = reinterpret_cast<const CUpti_ActivityKernel10 *>(activity);
            const int step = current_step.load();
            json row = {{"step", step}, {"phase", step == 0 ? "prefill" : "decode"},
                {"name", k->name}, {"start_ns", k->start}, {"end_ns", k->end},
                {"device", k->deviceId}, {"context", k->contextId}, {"stream", k->streamId},
                {"correlation", k->correlationId}, {"grid", {k->gridX,k->gridY,k->gridZ}},
                {"block", {k->blockX,k->blockY,k->blockZ}}, {"registers", k->registersPerThread},
                {"static_shared", k->staticSharedMemory}, {"dynamic_shared", k->dynamicSharedMemory}};
            kernel_log << row.dump() << '\n';
            ++kernel_count;
        }
        if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) ++trace_errors;
        size_t lost = 0;
        if (cuptiActivityGetNumDroppedRecords(ctx, stream, &lost) != CUPTI_SUCCESS) ++trace_errors;
        dropped += lost;
        if (!kernel_log) ++trace_errors;
    } catch (...) { ++trace_errors; }
    hbflab::platform::aligned_release(buffer);
}

static json tensor_info(const ggml_tensor * t) {
    json item = {{"name", t->name}, {"type", ggml_type_name(t->type)},
        {"shape", {t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},
        {"strides", {t->nb[0],t->nb[1],t->nb[2],t->nb[3]}}, {"bytes", ggml_nbytes(t)}};
    const auto * storage = t;
    while (!storage->buffer && storage->view_src) storage = storage->view_src;
    item["buffer"] = storage->buffer ? ggml_backend_buffer_name(storage->buffer) : "none";
    item["address"] = reinterpret_cast<uintptr_t>(t->data);
    return item;
}

static bool observe_node(ggml_tensor * t, bool ask, void *) {
    if (!ask) return true;
    hbflab_m2_graph_node(t);
    if (!record_node_metadata) return false;
    try {
        const int step = current_step.load();
        if (step < 0) return false;
        json row = {{"step", step}, {"phase", step == 0 ? "prefill" : "decode"},
            {"op", ggml_op_name(t->op)}, {"tensor", tensor_info(t)}, {"sources", json::array()}};
        for (auto * src : t->src) if (src) row["sources"].push_back(tensor_info(src));
        node_log << row.dump() << '\n';
        if (!node_log) ++trace_errors;
    } catch (...) { ++trace_errors; }
    // Metadata only. The scheduler may still synchronize at split boundaries.
    return false;
}

static std::ofstream output(const fs::path & p, bool binary = false) {
    std::ofstream f(p, std::ios::out | (binary ? std::ios::binary : std::ios::openmode(0)));
    if (!f) throw std::runtime_error("Cannot write " + p.string());
    return f;
}

static int run(int argc, char ** argv) {
    if (argc != 8) throw std::runtime_error("Usage: realgpu_llama model.gguf prompt.txt output_dir token_count observe_0_or_1 native_or_bypass_or_modeled profile.json");
    const fs::path out = argv[3];
    const int count = std::stoi(argv[4]);
    const bool observe = std::stoi(argv[5]) != 0;
    const char* kv_setting=std::getenv("HBFLAB_JOINT_KV");
    if(!kv_setting || (std::string(kv_setting)!="native" && std::string(kv_setting)!="paged"))throw std::runtime_error("explicit joint KV setting required");
    hbflab::platform::set_environment("HBFLAB_M4_PAGED",std::string(kv_setting)=="paged"?"1":"0");
    if (count < 2 || count > 448) throw std::runtime_error("token_count must be 2..448");
    if (fs::exists(out)) throw std::runtime_error("Output directory already exists; refusing to overwrite evidence");
    fs::create_directories(out);
    m6_load_profile(argv[7],argv[3]);const auto& profile=integrated_profile();
    if(hbflab::platform::available_host_memory()<profile.host_required)throw std::runtime_error("M6 host memory budget unavailable");
    std::size_t free_gpu=0,total_gpu=0;if(cudaMemGetInfo(&free_gpu,&total_gpu)!=cudaSuccess || free_gpu<profile.gpu_required)throw std::runtime_error("M6 GPU weight/scratch budget unavailable");
    std::ifstream prompt_file(argv[2], std::ios::binary);
    if (!prompt_file) throw std::runtime_error("Cannot open prompt");
    const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
    if (prompt.empty()) throw std::runtime_error("Empty prompt");
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices != 1) throw std::runtime_error("M0 requires exactly one visible CUDA GPU");
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess) throw std::runtime_error("CUDA device query failed");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) throw std::runtime_error("Model load failed");
    char architecture[64]{};llama_model_meta_val_str(model.get(),"general.architecture",architecture,sizeof(architecture));
    if(profile.architecture!=architecture || llama_model_n_layer(model.get())!=profile.layers || llama_model_n_embd(model.get())!=profile.hidden)throw std::runtime_error("M6 actual model metadata mismatch");
    std::ifstream pf(argv[7]);json metadata;pf>>metadata;
    for(const auto& t:metadata.at("tensors")){const std::string name=t.at("name");m6_check_tensor(hbflab_m2_find_tensor(model.get(),name.c_str()),name.c_str());}
    const auto * vocab = llama_model_get_vocab(model.get());
    int n = -llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (n <= 0) throw std::runtime_error("Empty tokenized prompt");
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), n, true, true) != n)
        throw std::runtime_error("Tokenization failed");
    if(const char* setting=std::getenv("HBFLAB_K5_PROMPT_TOKENS")){
        const std::string value=setting;std::size_t end=0;const auto limit=std::stoul(value,&end);
        if(end!=value.size() || !limit || limit>tokens.size())throw std::runtime_error("invalid prompt token prefix");
        tokens.resize(limit);n=static_cast<int>(limit);
    }
    if(n+count>static_cast<int>(profile.context))throw std::runtime_error("Prompt outside context limits");
    std::vector<std::vector<llama_token>> sequence_inputs{tokens};
    if(const char* second=std::getenv("HBFLAB_K3_SECOND_PROMPT")){
        std::ifstream f(second,std::ios::binary);if(!f)throw std::runtime_error("second prompt open failed");
        std::string text((std::istreambuf_iterator<char>(f)),{});
        int size=-llama_tokenize(vocab,text.data(),static_cast<int32_t>(text.size()),nullptr,0,true,true);
        if(size<=0 || size+count>profile.context)throw std::runtime_error("second prompt limits");
        std::vector<llama_token> t(size);if(llama_tokenize(vocab,text.data(),static_cast<int32_t>(text.size()),t.data(),size,true,true)!=size)throw std::runtime_error("second tokenization");sequence_inputs.push_back(t);
    }
    const auto sequence_prompts=sequence_inputs;const int sequence_count=static_cast<int>(sequence_inputs.size());
    auto cp = llama_context_default_params();
    cp.n_ctx = profile.context*sequence_count; cp.n_batch = profile.batch; cp.n_ubatch = profile.ubatch; cp.n_seq_max = sequence_count;
    if(std::getenv("HBFLAB_K3_SCENARIO"))cp.kv_unified=true; // Preserve mixed ubatches, including descending sequence order.
    cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = true; cp.op_offload = true; cp.no_perf = false;
    record_node_metadata=observe;
    cp.cb_eval = observe_node; cp.cb_eval_user_data = nullptr;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("Context init failed");
    hbflab_m2_configure(argv[6],argv[3]);
    M2Cleanup m2_cleanup;
    const char* scale_setting=std::getenv("HBFLAB_JOINT_SCALE");
    const std::string scale=scale_setting?scale_setting:"";
    if(scale!="F1" && scale!="F2")throw std::runtime_error("joint scale must be F1 or F2");
    std::vector<const ggml_tensor*> selected;
    const std::vector<int> selected_layers=scale=="F1"?std::vector<int>{10}:std::vector<int>{0,10,20,27};
    for(int layer:selected_layers)for(const char* role:{"gate","up","down"}){
        const auto name="blk."+std::to_string(layer)+".ffn_"+role+".weight";
        selected.push_back(hbflab_m2_find_tensor(model.get(),name.c_str()));
    }
    hbflab_joint_bind(selected.data(),static_cast<unsigned>(selected.size()));
    if(n!=44 || count!=8 || sequence_count!=1)throw std::runtime_error("joint scope requires 44 input and 8 output tokens, one sequence");
    if (observe) {
        kernel_log = output(out / "kernels.jsonl");
        node_log = output(out / "nodes.jsonl");
        check(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer));
        check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    }
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
    if (!batch || !sampler) throw std::runtime_error("Batch/sampler init failed");
    auto logits_file = output(out / "logits.f32", true);
    auto generated_file = output(out / "generated.txt", true);
    auto steps_file = output(out / "steps.jsonl");
    const int nvocab = llama_vocab_n_tokens(vocab);
    json generated = json::array(), times = json::array();
    std::vector<llama_pos> positions(sequence_count,0);
    const auto generation_start=Clock::now();

    const char* scenario_env=std::getenv("HBFLAB_K3_SCENARIO");
    const std::string scenario=scenario_env?scenario_env:"";
    if(!scenario.empty()) {
        if(sequence_count!=2 || count<4 || (scenario!="mixed" && scenario!="lifecycle"))throw std::runtime_error("K3 scenario requires two sequences and >=4 rounds");
        const bool paged=std::string(argv[6])!="native";const std::size_t keep=17;
        if(sequence_prompts[0].size()<=keep)throw std::runtime_error("K3 prompt must cross truncate boundary");
        json replay,peer_replay,inputs_trace={{"steps",json::array()}};
        if(const char* path=std::getenv("HBFLAB_K3_REPLAY")){std::ifstream f(path);f>>replay;}
        if(const char* path=std::getenv("HBFLAB_K3_PEER_REPLAY")){std::ifstream f(path);f>>peer_replay;}
        auto lifecycle_file=output(out/"lifecycle.jsonl");
        auto memory=llama_get_memory(ctx.get());
        auto snapshot=[&](const std::string& label){
            json record={{"label",label},{"framework",json::array()}};
            for(int id=0;id<sequence_count;++id)record["framework"].push_back({{"sequence",id},{"min",llama_memory_seq_pos_min(memory,id)},{"max",llama_memory_seq_pos_max(memory,id)}});
            if(paged){const auto path=out/("audit-"+label+".json");m4_sequence_audit(path.string().c_str(),keep);std::ifstream f(path);f>>record["kv"];}
            lifecycle_file<<record.dump()<<'\n';return record;
        };
        for(int round=0;round<count;++round){
            if(scenario=="lifecycle" && round==1){
                snapshot("before-truncate");
                if(!llama_memory_seq_rm(memory,0,static_cast<llama_pos>(keep),-1))throw std::runtime_error("framework truncate refused");
                if(paged)m4_sequence_truncate(0,keep);
                positions[0]=keep;sequence_inputs[0]={sequence_prompts[0][keep]};snapshot("after-truncate");
            }
            if(scenario=="lifecycle" && round==2){
                snapshot("before-release");
                if(!llama_memory_seq_rm(memory,0,0,-1))throw std::runtime_error("framework release refused");
                if(paged)m4_sequence_release(0);
                positions[0]=0;sequence_inputs[0]=sequence_prompts[1];snapshot("after-release");
            }
            if(!peer_replay.is_null())sequence_inputs[1]=peer_replay.at("steps").at(round).at("inputs").at(1).get<std::vector<llama_token>>();
            if(!replay.is_null()){
                const auto& expected=replay.at("steps").at(round);
                if(expected.at("positions_before").get<std::vector<llama_pos>>()!=positions)throw std::runtime_error("fixed replay position mismatch");
                sequence_inputs=expected.at("inputs").get<std::vector<std::vector<llama_token>>>();
            }
            std::vector<int> order=round%2?std::vector<int>{1,0}:std::vector<int>{0,1};
            inputs_trace["steps"].push_back({{"round",round},{"inputs",sequence_inputs},{"positions_before",positions},{"order",order}});
            std::size_t total=0;for(const auto& input:sequence_inputs)total+=input.size();
            if(total>profile.batch)throw std::runtime_error("mixed batch exceeds profile budget");
            current_step=round;hbflab_m2_step_begin(round);llama_batch_ext_clear(batch.get());std::vector<int> last(sequence_count,-1);
            for(int id:order){
                for(auto token:sequence_inputs[id]){int index=llama_batch_ext_add_token(batch.get(),id,token);llama_pos pos=positions[id]++;
                    if(index<0 || !llama_batch_ext_set_pos(batch.get(),index,&pos))throw std::runtime_error("mixed token insertion failed");last[id]=index;}
                if(last[id]<0 || !llama_batch_ext_set_output_logits(batch.get(),last[id],true))throw std::runtime_error("mixed output selection failed");
            }
            const auto start=Clock::now();
            if(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())!=0)throw std::runtime_error("mixed llama_process failed");
            llama_synchronize(ctx.get());hbflab_m2_step_end();
            const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            if(observe)check(cuptiActivityFlushAll(0));
            for(int id=0;id<sequence_count;++id){
                float* logits=llama_get_logits_ith(ctx.get(),last[id]);if(!logits)throw std::runtime_error("mixed logits missing");
                for(int i=0;i<nvocab;++i)if(!std::isfinite(logits[i]))throw std::runtime_error("nonfinite mixed logits");
                logits_file.write(reinterpret_cast<char*>(logits),static_cast<std::streamsize>(nvocab)*sizeof(float));
                auto token=llama_sampler_sample(sampler.get(),ctx.get(),last[id]);
                if(llama_vocab_is_eog(vocab,token))throw std::runtime_error("early EOG in K3 scenario");
                generated.push_back(token);times.push_back(ms);
                steps_file<<json({{"sequence",id},{"step",round},{"input_tokens",sequence_inputs[id].size()},{"token",token},{"process_wall_ms",ms}}).dump()<<'\n';
                sequence_inputs[id].assign(1,token);
                if(llama_memory_seq_pos_max(memory,id)!=positions[id]-1)throw std::runtime_error("framework sequence position mismatch");
            }
            snapshot("round-"+std::to_string(round));
        }
        for(int id=0;id<sequence_count;++id){if(!llama_memory_seq_rm(memory,id,0,-1))throw std::runtime_error("final framework release refused");if(paged)m4_sequence_release(id);}
        snapshot("final-release");lifecycle_file.close();if(!lifecycle_file)throw std::runtime_error("lifecycle audit write");
        auto trace_file=output(out/"inputs.json");trace_file<<inputs_trace.dump(2)<<'\n';trace_file.close();if(!trace_file)throw std::runtime_error("input trace write");
    } else {
    int dispatch_step=0;
    for (int step = 0; step < count*sequence_count; ++step) {
        const int sid=step%sequence_count;auto& input=sequence_inputs[sid];auto& position=positions[sid];
        current_step = step;
        const auto start = Clock::now();
        std::size_t chunks=0;
        for(std::size_t first=0;first<input.size();first+=profile.batch){
        const auto count_here=std::min(std::size_t(profile.batch),input.size()-first);
        hbflab_m2_step_begin(dispatch_step++);
        llama_batch_ext_clear(batch.get());
        for (std::size_t i=0;i<count_here;++i) {const auto token=input[first+i];
            const int index = llama_batch_ext_add_token(batch.get(), sid, token);
            const llama_pos pos = position++;
            if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &pos)) throw std::runtime_error("Batch token failed");
        }
        if (!llama_batch_ext_set_output_logits(batch.get(), static_cast<int32_t>(count_here - 1), true)) throw std::runtime_error("Batch logits failed");
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) throw std::runtime_error("llama_process failed");
        llama_synchronize(ctx.get());
        hbflab_m2_step_end();
        ++chunks;
        }
        const double ms = std::chrono::duration<double,std::milli>(Clock::now() - start).count();
        // All work is synchronized and flushed before changing step attribution.
        if (observe) check(cuptiActivityFlushAll(0));
        float * logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits) throw std::runtime_error("Missing logits");
        for (int i = 0; i < nvocab; ++i) if (!std::isfinite(logits[i])) throw std::runtime_error("Non-finite logits");
        logits_file.write(reinterpret_cast<char *>(logits), static_cast<std::streamsize>(nvocab) * sizeof(float));
        const auto token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, token)) throw std::runtime_error("Early EOG: baseline requires the requested token count");
        char piece[1024];
        const int length = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (length < 0) throw std::runtime_error("Token piece too large");
        generated_file.write(piece, length);
        std::cout.write(piece, length);
        std::cout.flush();
        generated.push_back(token); times.push_back(ms);
        json row = {{"sequence",sid},{"step",step}, {"phase",step < sequence_count ? "prefill" : "decode"}, {"input_tokens",input.size()},
                    {"token",token}, {"logit",logits[token]}, {"process_wall_ms",ms},{"chunks",chunks}};
        steps_file << row.dump() << '\n';
        input.assign(1, token);
    }
    }
    hbflab_m2_finish();
    const double generation_wall_ms=std::chrono::duration<double,std::milli>(Clock::now()-generation_start).count();
    if (observe) {
        check(cuptiActivityFlushAll(0));
        check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
        kernel_log.flush(); node_log.flush();
        if (!kernel_log || !node_log || dropped || trace_errors || !kernel_count) throw std::runtime_error("Incomplete CUPTI/node trace");
    }
    logits_file.flush(); generated_file.flush(); steps_file.flush();
    if (!logits_file || !generated_file || !steps_file) throw std::runtime_error("Output write failed");
    json summary = {{"status","success"}, {"llama_commit",M0_LLAMA_COMMIT}, {"model",argv[1]},
        {"gpu",device.name}, {"compute_capability",{device.major,device.minor}}, {"observed",observe},
        {"cuda_graphs",false}, {"flash_attention",false}, {"hbf_modeled",std::string(argv[6])=="modeled"}, {"profile_sha256",profile.hash},{"n_ctx",profile.context},
        {"n_batch",profile.batch}, {"n_ubatch",profile.ubatch}, {"n_threads",8}, {"n_gpu_layers_requested",999},
        {"model_layers",llama_model_n_layer(model.get())}, {"prompt",prompt}, {"prompt_tokens",tokens},
        {"generated_tokens",generated}, {"vocab_size",nvocab}, {"scenario",scenario},{"sequence_prompts",sequence_prompts},{"sequence_count",sequence_count},{"logits_shape",{count*sequence_count,nvocab}},
        {"logits_format","little-endian float32; one next-token row per step"},
        {"sampler","greedy"}, {"process_wall_ms",times}, {"generation_wall_ms",generation_wall_ms},{"kernel_records",kernel_count.load()},
        {"cupti_dropped",dropped.load()}, {"trace_errors",trace_errors.load()}};
    auto summary_file = output(out / "summary.json");
    summary_file << summary.dump(2) << '\n';
    summary_file.close();
    if (!summary_file) throw std::runtime_error("Summary write failed");
    llama_perf_context_print(ctx.get());
    std::cout << '\n';
    return 0;
}

int main(int argc, char ** argv) {
    try { return run(argc, argv); }
    catch (const std::exception & e) { std::cerr << "M6 failure: " << e.what() << '\n'; return 1; }
}
