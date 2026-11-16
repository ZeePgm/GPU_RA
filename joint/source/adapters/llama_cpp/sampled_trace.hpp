#pragma once
#include "hbflab/platform.hpp"
#include <cupti.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>

class SampledGpuTrace {
    inline static SampledGpuTrace *active_=nullptr;
    std::ofstream file_;
    std::filesystem::path output_;
    std::mutex mutex_;
    std::atomic<int> phase_{-1};
    std::atomic<uint64_t> kernels_{0},copies_{0},memsets_{0},dropped_{0},errors_{0};
    bool finished_=false;
    static void check(CUptiResult result){
        if(result==CUPTI_SUCCESS)return;const char *text=nullptr;cuptiGetResultString(result,&text);
        throw std::runtime_error(std::string("sampled CUPTI: ")+(text?text:"unknown error"));
    }
    static void CUPTIAPI request(uint8_t **buffer,size_t *size,size_t *maximum){
        *size=8*1024*1024;*maximum=0;
        *buffer=static_cast<uint8_t *>(hbflab::platform::aligned_allocate(*size,8));
        if(!*buffer){*size=0;if(active_)++active_->errors_;}
    }
    static void CUPTIAPI complete(CUcontext context,uint32_t stream,uint8_t *buffer,size_t,size_t valid){
        auto *s=active_;
        if(s)try{
            std::lock_guard<std::mutex> lock(s->mutex_);CUpti_Activity *activity=nullptr;CUptiResult result;
            while((result=cuptiActivityGetNextRecord(buffer,valid,&activity))==CUPTI_SUCCESS){
                using J=nlohmann::ordered_json;
                if(activity->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
                    const auto *k=reinterpret_cast<const CUpti_ActivityKernel10 *>(activity);
                    s->file_<<J({{"kind","kernel"},{"phase",s->phase_.load()},{"name",k->name},
                        {"start_ns",k->start},{"end_ns",k->end},{"stream",k->streamId},{"correlation",k->correlationId},
                        {"grid",{k->gridX,k->gridY,k->gridZ}},{"block",{k->blockX,k->blockY,k->blockZ}}}).dump()<<'\n';++s->kernels_;
                }else if(activity->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
                    const auto *c=reinterpret_cast<const CUpti_ActivityMemcpy6 *>(activity);
                    s->file_<<J({{"kind","copy"},{"phase",s->phase_.load()},{"start_ns",c->start},{"end_ns",c->end},
                        {"bytes",c->bytes},{"direction",c->copyKind},{"stream",c->streamId},{"correlation",c->correlationId}}).dump()<<'\n';++s->copies_;
                }else if(activity->kind==CUPTI_ACTIVITY_KIND_MEMSET){
                    const auto *m=reinterpret_cast<const CUpti_ActivityMemset4 *>(activity);
                    s->file_<<J({{"kind","memset"},{"phase",s->phase_.load()},{"start_ns",m->start},{"end_ns",m->end},
                        {"bytes",m->bytes},{"stream",m->streamId},{"correlation",m->correlationId}}).dump()<<'\n';++s->memsets_;
                }
            }
            if(result!=CUPTI_ERROR_MAX_LIMIT_REACHED)++s->errors_;
            size_t lost=0;if(cuptiActivityGetNumDroppedRecords(context,stream,&lost)!=CUPTI_SUCCESS)++s->errors_;
            s->dropped_+=lost;if(!s->file_)++s->errors_;
        }catch(...){++s->errors_;}
        hbflab::platform::aligned_release(buffer);
    }
public:
    explicit SampledGpuTrace(const std::filesystem::path &output):output_(output){
        file_.open(output/"gpu-activity.jsonl");if(!file_)throw std::runtime_error("GPU trace open failed");active_=this;
        try{
            check(cuptiActivityRegisterCallbacks(request,complete));
            check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
            check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
        }catch(...){cuptiActivityFlushAll(0);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
            cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET);active_=nullptr;throw;}
    }
    ~SampledGpuTrace(){
        if(!finished_){cuptiActivityFlushAll(0);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET);}active_=nullptr;
    }
    void phase(int value){check(cuptiActivityFlushAll(0));phase_=value;}
    void flush(){check(cuptiActivityFlushAll(0));}
    void finish(){
        check(cuptiActivityFlushAll(0));check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));file_.close();finished_=true;
        if(!file_ || errors_ || dropped_ || !kernels_)throw std::runtime_error("Incomplete sampled GPU activity trace");
        std::ofstream summary(output_/"gpu-trace-summary.json");summary<<nlohmann::ordered_json({{"status","pass"},
            {"kernel_records",kernels_.load()},{"copy_records",copies_.load()},{"dropped",dropped_.load()},{"errors",errors_.load()},
            {"memset_records",memsets_.load()},
            {"phase_labels",{{"-1","initialization"},{"0..7","generation step"},{"-2","final audit and drain"}}}}).dump(2)<<'\n';
        summary.close();if(!summary)throw std::runtime_error("GPU trace summary write failed");
    }
};
