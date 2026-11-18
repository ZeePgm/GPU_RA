#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "llama.h"
#include "restored_routes.hpp"
#include "hbflab/ptx.hpp"
#include "hbflab/platform.hpp"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
using J=nlohmann::ordered_json;namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static uint64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
static void need(bool b,const char *s){if(!b)throw std::runtime_error(s);}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static J read(const fs::path &p){std::ifstream f(p);J j;f>>j;need(bool(f),"JSON input failed");return j;}
static std::string binary(const fs::path &p){std::ifstream f(p,std::ios::binary);need(bool(f),"binary input missing");return std::string(std::istreambuf_iterator<char>(f),{});}
static void save(const fs::path &p,const J &j){std::ofstream f(p);f<<j.dump(2)<<'\n';f.close();need(bool(f),"report write failed");}
static void binary_save(const fs::path &p,const std::string &v){std::ofstream f(p,std::ios::binary);f.write(v.data(),v.size());f.close();need(bool(f),"output write failed");}
struct Memory {
    ggml_context *ctx=nullptr;ggml_backend_t backend=nullptr;ggml_backend_buffer_t buffer=nullptr;
    ~Memory(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);if(backend)ggml_backend_free(backend);}
};
static ggml_tensor *tensor(ggml_context *ctx,const J &d){
    const auto shape=d.at("shape").get<std::array<int64_t,4>>();
    auto *t=ggml_new_tensor(ctx,ggml_type(d.at("type").get<int>()),4,shape.data());
    for(unsigned k=0;k<4;++k)t->nb[k]=d.at("strides").at(k).get<size_t>();
    need(ggml_nbytes(t)==d.at("bytes").get<size_t>(),"fixture stride/byte mismatch");return t;
}
static J hashes(ggml_tensor *t){
    J result=J::array();std::string block(262144,'\0');
    for(size_t offset=0;offset<ggml_nbytes(t);){const auto count=std::min(block.size(),ggml_nbytes(t)-offset);
        ggml_backend_tensor_get(t,block.data(),offset,count);result.push_back(hbflab::sha256(std::string_view(block.data(),count)));offset+=count;}
    return result;
}
int main(int argc,char **argv)try{
    need(argc==8,"usage: five_weight model inventory fixture.json output route repeats audit");
    const auto inventory=read(argv[2]),fixture=read(argv[3]);const fs::path out=argv[4],fixture_root=fs::path(argv[3]).parent_path();
    need(!fs::exists(out),"weight output exists");fs::create_directories(out);
    const unsigned repeats=std::stoul(argv[6]);need(repeats<=5,"repeat limit");
    const std::string audit=argv[7];need(audit=="full" || audit=="summary","audit mode");
    const unsigned count=1+repeats;const auto expected=binary(fixture_root/fixture.at("output").at("file").get<std::string>());
    J entry;for(const auto &t:inventory.at("all_tensors"))if(t.at("name")==fixture.at("tensor"))entry=t;
    need(!entry.is_null() && entry.at("selected").get<bool>(),"unselected fixture weight");
    need(fs::file_size(argv[1])==inventory.at("model").at("bytes").get<uint64_t>(),"model size mismatch");
    const auto u0=now();llama_backend_init();ggml_backend_load_all();Memory memory;memory.backend=ggml_backend_cuda_init(0);need(memory.backend,"CUDA backend missing");
    size_t free_gpu=0,total_gpu=0;ck(cudaMemGetInfo(&free_gpu,&total_gpu));
    need(free_gpu>=2308702208ull && hbflab::platform::available_host_memory()>=1006632960ull,"original memory floors unavailable");
    memory.ctx=ggml_init({32*1024*1024,nullptr,true});need(memory.ctx,"ggml context");
    auto *weight=tensor(memory.ctx,fixture.at("weight"));ggml_set_name(weight,entry.at("name").get<std::string>().c_str());
    std::array<ggml_tensor *,GGML_MAX_SRC> sources{};sources[fixture.at("weight_slot").get<unsigned>()]=weight;
    for(const auto &input:fixture.at("inputs"))sources[input.at("slot").get<unsigned>()]=tensor(memory.ctx,input);
    std::vector<ggml_tensor *> outputs;std::vector<ggml_cgraph *> graphs;
    size_t output_bytes=0;
    for(unsigned k=0;k<count;++k){ggml_tensor *output=nullptr;const auto op=fixture.at("op").get<std::string>();
        if(op=="MUL")output=ggml_mul(memory.ctx,sources[0],sources[1]);
        else if(op=="MUL_MAT"){output=ggml_mul_mat(memory.ctx,sources[0],sources[1]);ggml_mul_mat_set_prec(output,GGML_PREC_F32);}
        else if(op=="GET_ROWS")output=ggml_get_rows(memory.ctx,sources[0],sources[1]);
        else throw std::runtime_error("unsupported native consumer");
        need(ggml_nbytes(output)==expected.size() && ggml_is_contiguous(output),"consumer output layout mismatch");
        for(unsigned d=0;d<4;++d)need(output->ne[d]==fixture.at("output").at("shape").at(d).get<int64_t>(),"consumer output shape");
        auto *graph=ggml_new_graph_custom(memory.ctx,64,false);ggml_build_forward_expand(graph,output);
        outputs.push_back(output);graphs.push_back(graph);output_bytes+=ggml_nbytes(output);
    }
    need(output_bytes<=4*1024*1024,"isolated output buffer budget");
    memory.buffer=ggml_backend_alloc_ctx_tensors(memory.ctx,memory.backend);need(memory.buffer,"GPU tensor allocation");
    std::ifstream model(argv[1],std::ios::binary);model.seekg(entry.at("file_offset").get<uint64_t>());std::vector<char> block(262144);
    for(size_t offset=0;offset<ggml_nbytes(weight);){const auto size=std::min(block.size(),ggml_nbytes(weight)-offset);
        model.read(block.data(),size);need(model.gcount()==std::streamsize(size),"weight truncated");ggml_backend_tensor_set(weight,block.data(),offset,size);offset+=size;}
    for(const auto &input:fixture.at("inputs")){const auto data=binary(fixture_root/input.at("file").get<std::string>());
        auto *t=sources[input.at("slot").get<unsigned>()];need(data.size()==ggml_nbytes(t),"activation bytes");ggml_backend_tensor_set(t,data.data(),0,data.size());}
    const auto original=hashes(weight);
    const ggml_tensor* selected[]={weight};
    const uint64_t logical_base=entry.at("logical_base");
    struct Cleanup {~Cleanup(){restored_abort();}} cleanup;
    restored_bind(argv[5],selected,&logical_base,1,inventory.at("logical_weight_layout_bytes"),
        std::getenv("HBFLAB_SAMPLED_PROFILE"),std::getenv("HBFLAB_SYMBOLIC_CONFIG"),argv[4],audit=="full");
    ck(cudaDeviceSynchronize());const auto ready=now();
    struct Sample{uint64_t begin,end;};std::vector<Sample> samples;samples.reserve(count);
    for(unsigned k=0;k<count;++k){const auto begin=now();
        restored_step_begin(k);restored_before(outputs[k]);
        need(ggml_backend_graph_compute(memory.backend,graphs[k])==GGML_STATUS_SUCCESS,"native consumer compute failed");
        ggml_backend_synchronize(memory.backend);restored_step_end();
        samples.push_back({begin,now()});
    }
    J calls=J::array();std::string result(expected.size(),'\0');double repeated_ns=0;
    for(unsigned k=0;k<count;++k){ggml_backend_tensor_get(outputs[k],result.data(),0,result.size());
        binary_save(out/("call-"+std::to_string(k)+".bin"),result);const bool equal=result==expected;
        calls.push_back({{"index",k},{"kind",k?"repeated":"first"},{"begin_ns",samples[k].begin},{"end_ns",samples[k].end},
            {"latency_ns",samples[k].end-samples[k].begin},{"native_bitwise_equal",equal},{"output_sha256",hbflab::sha256(result)}});
        save(out/"calls.json",calls);need(equal,"native consumer output differs");if(k)repeated_ns+=samples[k].end-samples[k].begin;
    }
    const auto final_hashes=hashes(weight);need(original==final_hashes,"borrowed weight changed");save(out/"weight-hashes.json",{{"before",original},{"after",final_hashes}});
    restored_finish();
    save(out/"summary.json",{{"status","pass"},{"scenario_id",fixture.at("scenario_id")},{"tensor",entry.at("name")},{"route",argv[5]},
        {"audit_mode",audit},{"u0_ns",u0},{"ready_ns",ready},{"initialization_ns",ready-u0},{"first_result_ns",samples.front().end-u0},
        {"first_call_ns",samples.front().end-samples.front().begin},{"repeated_calls",repeats},{"repeated_mean_ns",repeats?repeated_ns/repeats:0.0},
        {"calls",count},{"output_buffer_bytes",output_bytes},{"drained",true},{"borrowed_unchanged",true},
        {"free_gpu_before_allocation",free_gpu},
        {"post_timing_identity_observation_hold_ms",1000}});
    // The unchanged host observer samples modules after 200ms. Keep short
    // successful processes alive outside every measured boundary for that proof.
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout<<"PASS "<<fixture.at("scenario_id")<<" "<<argv[5]<<"\n";return 0;
}catch(const std::exception &e){std::cerr<<"five weight: "<<e.what()<<'\n';return 1;}
