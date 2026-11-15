#include "hbflab/kv.hpp"
#include <vector>
#include <iostream>
using namespace hbflab;
void check(bool ok){if(!ok)throw std::runtime_error("K3 capacity/identity oracle");}
template<class F>void reject(F f){bool caught=false;try{f();}catch(...){caught=true;}check(caught);}
int main()try{
 CapacityOptions options;options.logical_pages=16;options.frames=2;options.backend.geometry={1,1,1,8,8,1};CapacityStore store(options);
 KvOptions layout;layout.layers=2;layout.head_dimension=128;layout.element_bytes=2;layout.tokens_per_block=2;layout.max_tokens=8;layout.max_sequences=4;
 KvCache kv(store,layout);std::vector<KvSequence> sequences;for(int i=0;i<4;++i)sequences.push_back(kv.create());reject([&]{kv.create();});
 auto append=[&](int id){std::vector<unsigned char> key(512,17+id),value(512,91+id);kv.append(sequences[id],key.data(),value.data(),key.size());};
 for(int i=0;i<4;++i)append(i);append(0);check(store.free_pages()==0);
 reject([&]{append(0);});check(!kv.poisoned() && kv.length(sequences[0])==2);for(int i=1;i<4;++i)check(kv.length(sequences[i])==1);
 // Explicit bounded schedule: on pressure, release finished sequence 3, retry 0 once.
 const auto old=sequences[3];kv.release(old);append(0);check(kv.length(sequences[0])==3);
 kv.truncate(sequences[0],1);sequences[3]=kv.create();reject([&]{kv.length(old);});append(3);
 for(int id=0;id<4;++id)for(int layer=0;layer<2;++layer)for(auto kind:{KvKind::Key,KvKind::Value}){
  std::vector<unsigned char> bytes(256);kv.read(sequences[id],layer,0,kind,bytes.data(),bytes.size());
  for(auto value:bytes)check(value==(kind==KvKind::Key?17+id:91+id));
 }
 for(auto seq:sequences)kv.release(seq);auto stats=store.statistics();check(!stats.allocated_pages && !stats.pins && !stats.dirty_frames && stats.peak_pins<=2);
 std::cout<<"K3 configured sequence limit, exact bytes, pressure/retry, stale handle and release passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
