#include "hbflab/ptx.hpp"
#include "ptx_fixture.hpp"
#include <fstream>
#include <iostream>
using namespace hbflab;
int main(int argc,char** argv) try {
 require(argc==2,"raw PTX required");std::ifstream in(argv[1]);
 const std::string source{std::istreambuf_iterator<char>(in),{}};
 const auto rewritten=rewrite_async_ptx(source);
 require(rewritten.entries.size()==1 && rewritten.entries[0].loads==10 && rewritten.entries[0].stores==1,"down access manifest");
 for(const auto& change: {std::pair<const char*,const char*>{"nounroll","unknown"},
 {".align 4 .b8",".align 3 .b8"},{"sums[128]","sums[65536]"},
 {"bar.sync \t0;","bar.sync 1;"},{"bar.sync \t0;","bar.sync 0, 128;"},
 {"bar.sync \t0;","@%p1 bar.sync 0;"},{".f16 low,high;",".f16 low,low;"},
 {"%f15, low;","%f15, missing;"},{"%r40|%p7","%r40|%r1"},
 {"ld.shared.f32","ld.shared.u64"},{"fma.rn.f32","fma.rz.f32"}}) {
  auto bad=source;replace_one(bad,change.first,change.second);rejects([&]{rewrite_async_ptx(bad);});
 }
 rejects([&]{rewrite_ptx(source);});
 std::cout<<"down parser pass: manifest + 11 malformed variants + synchronous rejection\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
