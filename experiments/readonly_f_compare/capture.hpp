#pragma once
#include <cupti.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <cstdlib>
static int capture_step=0;static std::ofstream capture_file,mark_file;static std::mutex capture_lock;static size_t capture_dropped=0;
static void CUPTIAPI req(uint8_t**b,size_t*s,size_t*n){*s=8*1024*1024;*b=(uint8_t*)_aligned_malloc(*s,8);*n=0;}
static void CUPTIAPI done(CUcontext c,uint32_t stream,uint8_t*b,size_t,size_t valid){std::lock_guard<std::mutex> guard(capture_lock);CUpti_Activity*a=nullptr;while(cuptiActivityGetNextRecord(b,valid,&a)==CUPTI_SUCCESS){nlohmann::json r; if(a->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){auto*k=(CUpti_ActivityKernel10*)a;r={{"name",k->name},{"start",k->start},{"end",k->end},{"kind","kernel"},{"stream",k->streamId}};}else if(a->kind==CUPTI_ACTIVITY_KIND_MEMCPY){auto*m=(CUpti_ActivityMemcpy6*)a;r={{"start",m->start},{"end",m->end},{"kind","copy"},{"bytes",m->bytes},{"stream",m->streamId}};}else if(a->kind==CUPTI_ACTIVITY_KIND_MEMSET){auto*m=(CUpti_ActivityMemset4*)a;r={{"start",m->start},{"end",m->end},{"kind","memset"},{"stream",m->streamId}};}else continue;r["step"]=capture_step;capture_file<<r.dump()<<'\n';}size_t dropped=0;cuptiActivityGetNumDroppedRecords(c,stream,&dropped);capture_dropped+=dropped;_aligned_free(b);}
static void capture_begin(const std::filesystem::path&o){capture_file.open(o/"gpu.jsonl");mark_file.open(o/"batches.jsonl");cuptiActivityRegisterCallbacks(req,done);cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);}
static void capture_mark(const char*name,unsigned offset){uint64_t t;cuptiGetTimestamp(&t);mark_file<<nlohmann::json({{"step",capture_step},{"tensor",name},{"offset",offset},{"at",t}}).dump()<<'\n';}
static void capture_end(){cuptiActivityFlushAll(0);if(capture_dropped)throw std::runtime_error("CUPTI dropped");capture_file.flush();mark_file.flush();}
