#include "hbflab/ptx.hpp"
#include "plan_ptx.hpp"
#include <iostream>
#include <stdexcept>
void check(bool b){if(!b)throw std::runtime_error("policy contract");}
int main()try{
 for(unsigned width:{4,16})for(bool timed:{false,true}) {
  const auto source=plan_ptx(width,timed);const auto scalar=hbflab::rewrite_async_ptx(source,false),warp=hbflab::rewrite_async_ptx(source,true);
  check(warp.entries[0].loads==2 && scalar.entries[0].loads==2);
  check(warp.text.find("__hbflab_async_warp_gate")!=std::string::npos);
  for(const std::string bad:{"@%p0 ret;\n","bra L; L:\n","bar.sync 0;\n","ret;\n"}){
   auto variant=source;variant.insert(variant.find("ld.global.u32"),bad);bool rejected=false;try{hbflab::rewrite_async_ptx(variant,true);}catch(...){rejected=true;}check(rejected);
  }
  auto variant=source;variant.insert(variant.rfind("ret;"),"ld.global.u32 %r0,[%rd0];\n");bool rejected=false;try{hbflab::rewrite_async_ptx(variant,true);}catch(...){rejected=true;}check(rejected);
 }
 std::cout<<"uniform PTX policy: 4 positive and 20 negative variants passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
