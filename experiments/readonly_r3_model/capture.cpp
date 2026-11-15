#include "llama.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
using json=nlohmann::ordered_json;
namespace fs=std::filesystem;
fs::path out; int phase=0; json records=json::array(); bool failed=false;
bool callback(ggml_tensor* t,bool ask,void*) {
 int wi=-1; for(int i=0;i<2;++i)if(t->src[i] && (std::string(t->src[i]->name)=="blk.29.ffn_norm.weight" || std::string(t->src[i]->name)=="blk.29.ffn_down.weight"))wi=i; const bool selected=wi>=0 && (t->op==GGML_OP_MUL || t->op==GGML_OP_MUL_MAT);
 if(ask)return selected;
 if(!selected)return true;
 try {
  const auto* x=t->src[1-wi];
  if(!x || x->type!=GGML_TYPE_F32 || x->ne[2]!=1 || x->ne[3]!=1 || x->ne[1]>32)throw std::runtime_error("activation shape");
  std::vector<char> raw(ggml_nbytes(x));ggml_backend_tensor_get(x,raw.data(),0,raw.size());
  std::vector<float> packed(x->ne[0]*x->ne[1]);
  for(int64_t j=0;j<x->ne[1];++j)for(int64_t i=0;i<x->ne[0];++i)std::memcpy(&packed[j*x->ne[0]+i],raw.data()+j*x->nb[1]+i*x->nb[0],4);
  const auto path=out/(std::to_string(records.size())+".f32");std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<char*>(packed.data()),packed.size()*4);if(!f)throw std::runtime_error("write activation");
  records.push_back({{"phase",phase==0?"prefill":"decode"},{"tensor",t->src[wi]->name},{"columns",x->ne[0]},{"tokens",x->ne[1]},{"activation_file",path.string()}});
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';failed=true;return false;}
 return true;
}
int main(int argc,char**argv)try {
 if(argc!=4)throw std::runtime_error("model context_length output");
 const int count=std::stoi(argv[2]);out=fs::absolute(argv[3]);if(fs::exists(out))throw std::runtime_error("output exists");fs::create_directories(out);
 llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=999;mp.split_mode=LLAMA_SPLIT_MODE_NONE;
 auto* model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("model load");
 auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=64;cp.n_ubatch=32;cp.n_seq_max=1;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.cb_eval=callback;
 auto* ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("context init");
 const auto* vocab=llama_model_get_vocab(model);std::string text="The memory system reads fixed weights and updates the cache. ";std::string prompt;for(int i=0;i<20;++i)prompt+=text;
 int n=-llama_tokenize(vocab,prompt.data(),int(prompt.size()),nullptr,0,true,true);std::vector<llama_token> tokens(n);llama_tokenize(vocab,prompt.data(),int(prompt.size()),tokens.data(),n,true,true);if(n<count)throw std::runtime_error("too few tokens");tokens.resize(count);
 auto* batch=llama_batch_ext_init(ctx);int pos=0;
 for(phase=0;phase<2;++phase){llama_batch_ext_clear(batch);for(auto token:tokens){int index=llama_batch_ext_add_token(batch,0,token);llama_pos p=pos++;if(index<0 || !llama_batch_ext_set_pos(batch,index,&p))throw std::runtime_error("batch");llama_batch_ext_set_output_logits(batch,index,true);}llama_batch_ext_set_output_logits(batch,int(tokens.size()-1),true);if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch)!=0 || failed)throw std::runtime_error("process");llama_synchronize(ctx);auto* logits=llama_get_logits_ith(ctx,-1);if(!logits)throw std::runtime_error("logits");int best=0;for(int i=1;i<llama_vocab_n_tokens(vocab);++i)if(logits[i]>logits[best])best=i;tokens={best};}
 std::ofstream(out/"manifest.json")<<json({{"context_length",count},{"ubatch",32},{"records",records}}).dump(2);
 llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}

