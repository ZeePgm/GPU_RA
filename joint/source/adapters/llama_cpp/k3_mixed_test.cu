#include "integrated_profile.hpp"
#include "paged_bridge.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <map>
#include <tuple>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
using Token=std::pair<int,int>;
using Key=std::tuple<int,int,int>;
using Results=std::map<Key,std::vector<float>>;
struct Run {
 int step=0;Results values;
 explicit Run(const std::filesystem::path& path){std::filesystem::create_directories(path);m4_begin(1,nullptr,path.string().c_str());}
 void batch(const std::vector<Token>& tokens){
  const auto& p=integrated_profile();const auto n=tokens.size();const auto qe=p.q_elements(),ke=p.kv_elements();
  m4_step_begin(step++);
  for(unsigned layer=0;layer<p.layers;++layer){
   std::vector<float> q(n*qe),k(n*ke),v(n*ke),out(n*qe);std::vector<int> ids,pos;
   for(std::size_t t=0;t<n;++t){auto [id,position]=tokens[t];ids.push_back(id);pos.push_back(position);
    for(std::size_t j=0;j<qe;++j)q[t*qe+j]=std::sin(float(j+position*3+id*101+layer*7)*.013f);
    for(std::size_t j=0;j<ke;++j){k[t*ke+j]=std::cos(float(j+position*5+id*107+layer*11)*.021f);v[t*ke+j]=std::sin(float(j+position*7+id*113+layer*13)*.031f);}
   }
   std::vector<void*> memory;auto upload=[&](const void* data,std::size_t bytes){void* ptr;ck(cudaMalloc(&ptr,bytes));memory.push_back(ptr);if(data)ck(cudaMemcpy(ptr,data,bytes,cudaMemcpyHostToDevice));return ptr;};
   ggml_tensor qt{},kt{},vt{},pt{},st{},dst{};
   for(auto* t:{&qt,&kt,&vt}){t->type=GGML_TYPE_F32;t->ne[0]=p.head_dim;t->ne[1]=t==&qt?p.q_heads:p.kv_heads;t->ne[2]=n;t->ne[3]=1;t->nb[0]=4;for(int j=1;j<4;++j)t->nb[j]=t->nb[j-1]*t->ne[j-1];}
   qt.data=upload(q.data(),q.size()*4);kt.data=upload(k.data(),k.size()*4);vt.data=upload(v.data(),v.size()*4);
   pt.type=st.type=GGML_TYPE_I32;pt.ne[0]=st.ne[0]=n;for(int j=1;j<4;++j)pt.ne[j]=st.ne[j]=1;pt.data=upload(pos.data(),n*4);st.data=upload(ids.data(),n*4);
   dst.type=GGML_TYPE_F32;dst.ne[0]=qe;dst.ne[1]=n;dst.ne[2]=dst.ne[3]=1;dst.nb[0]=4;dst.nb[1]=qe*4;dst.nb[2]=dst.nb[3]=n*qe*4;dst.data=upload(nullptr,out.size()*4);dst.op=GGML_OP_CUSTOM;
   dst.src[0]=&qt;dst.src[1]=&kt;dst.src[2]=&vt;dst.src[3]=&pt;dst.src[4]=&st;int tag=0x4d34564b;std::memcpy((char*)dst.op_params+56,&layer,4);std::memcpy((char*)dst.op_params+60,&tag,4);
   try{m4_attention(&dst,nullptr);ck(cudaMemcpy(out.data(),dst.data,out.size()*4,cudaMemcpyDeviceToHost));}catch(...){for(auto ptr:memory)cudaFree(ptr);throw;}
   for(std::size_t t=0;t<n;++t)values[{tokens[t].first,tokens[t].second,layer}]={out.begin()+t*qe,out.begin()+(t+1)*qe};
   for(auto ptr:memory)ck(cudaFree(ptr));
  }m4_step_end();
 }
 void finish(){m4_finish();}
};
std::vector<Token> prompt(int id,int n){std::vector<Token> r;for(int i=0;i<n;++i)r.push_back({id,i});return r;}
void same(const Results& a,const Results& b){if(a.size()!=b.size())throw std::runtime_error("oracle count");for(const auto& item:a){auto it=b.find(item.first);if(it==b.end() || item.second.size()!=it->second.size() || std::memcmp(item.second.data(),it->second.data(),item.second.size()*4))throw std::runtime_error("sequence logits byte mismatch");}}
int main(int argc,char** argv)try{
 if(argc!=3)throw std::runtime_error("profile and output required");std::filesystem::path root=argv[2];std::filesystem::create_directories(root);m6_load_profile(argv[1],argv[2]);Results reference;
 for(int id=0;id<2;++id){Run run(root/ ("independent"+std::to_string(id)));run.batch(prompt(id,id?7:17));run.batch({{id,id?7:17}});reference.insert(run.values.begin(),run.values.end());run.finish();}
 {Run run(root/"mixed");auto tokens=prompt(0,17);auto second=prompt(1,7);for(int i=0;i<7;++i)tokens.insert(tokens.begin()+2*i+1,second[i]);run.batch(tokens);run.batch({{1,7},{0,17}});same(reference,run.values);run.finish();}
 {Run run(root/"lifecycle");run.batch(prompt(0,17));run.batch(prompt(1,7));m4_sequence_truncate(0,5);run.batch({{1,7},{0,5}});m4_sequence_release(0);run.batch(prompt(0,17));run.batch({{0,17}});same(reference,run.values);run.finish();}
 {Run run(root/"budget");std::vector<Token> tokens;for(int id=0;id<16;++id)tokens.push_back({id,0});run.batch(tokens);for(int id=0;id<16;++id)m4_sequence_release(id);run.finish();}
 for(int invalid:{16,-1}){Run run(root/("invalid"+std::to_string(invalid)));bool rejected=false;try{run.batch({{invalid,0}});}catch(...){rejected=true;}m4_abort();if(!rejected)throw std::runtime_error("invalid sequence accepted");}
 {Run run(root/"position-gap");bool rejected=false;try{run.batch({{1,1}});}catch(...){rejected=true;}m4_abort();if(!rejected)throw std::runtime_error("sequence position gap accepted");}
 std::cout<<"K3 independent/mixed/lifecycle/16-sequence byte oracle passed\n";return 0;
}catch(const std::exception& e){m4_abort();std::cerr<<e.what()<<'\n';return 1;}
