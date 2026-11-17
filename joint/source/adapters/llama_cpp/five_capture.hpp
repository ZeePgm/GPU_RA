#pragma once
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <map>
#include <algorithm>
#include <vector>

// Reference-only capture. Every graph node is synchronized by cb_eval before
// the next node is examined, so input storage is captured before any in-place op.
struct FiveCapture {
    using J=nlohmann::ordered_json;
    std::filesystem::path root;
    std::map<const ggml_tensor *,J> pending;
    unsigned completed=0;
    static J descriptor(const ggml_tensor *t){
        return {{"type",int(t->type)},{"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},
            {"strides",{t->nb[0],t->nb[1],t->nb[2],t->nb[3]}},{"bytes",ggml_nbytes(t)}};
    }
    void bytes(const ggml_tensor *t,const std::string &name){
        std::vector<char> data(ggml_nbytes(t));ggml_backend_tensor_get(t,data.data(),0,data.size());
        std::ofstream f(root/name,std::ios::binary);f.write(data.data(),data.size());f.close();
        if(!f)throw std::runtime_error("five fixture capture write failed");
    }
    void before(ggml_tensor *node,int step,const std::map<const ggml_tensor *,std::string> &selected){
        if(root.empty() || step<0 || step>1)return;
        const std::vector<std::string> names={"blk.10.ffn_norm.weight","blk.10.attn_q.weight","blk.10.ffn_down.weight","token_embd.weight","output.weight"};
        for(int slot=0;slot<GGML_MAX_SRC;++slot){
            const auto *t=node->src[slot];if(!t || !selected.count(t))continue;
            auto it=std::find(names.begin(),names.end(),selected.at(t));if(it==names.end())throw std::runtime_error("capture scope");
            const auto id="w"+std::to_string(it-names.begin()+1)+(step?"-d":"-p");
            J row={{"scenario_id",id},{"step",step},{"tensor",t->name},{"op",ggml_op_name(node->op)},
                {"weight_slot",slot},{"weight",descriptor(t)},{"inputs",J::array()},{"output",descriptor(node)}};
            for(int k=0;k<GGML_MAX_SRC;++k)if(k!=slot && node->src[k]){
                auto input=descriptor(node->src[k]);const auto file=id+"-src"+std::to_string(k)+".bin";
                input["slot"]=k;input["file"]=file;bytes(node->src[k],file);row["inputs"].push_back(input);
            }
            row["output"]["file"]=id+"-output.bin";pending.emplace(node,std::move(row));
        }
    }
    void after(ggml_tensor *node){
        auto it=pending.find(node);if(it==pending.end())return;const auto row=it->second;
        bytes(node,row.at("output").at("file"));
        std::ofstream f(root/(row.at("scenario_id").get<std::string>()+".json"));f<<row.dump(2)<<'\n';f.close();
        if(!f)throw std::runtime_error("capture metadata write failed");pending.erase(it);++completed;
    }
};
