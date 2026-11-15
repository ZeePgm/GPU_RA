#include "hbflab/capacity.hpp"
#include <iostream>
#include <vector>
using namespace hbflab;
void check(bool b){if(!b)throw std::runtime_error("K2 oracle mismatch");}
int main()try{
 for(auto frames:{1,8})for(auto pages:{1,3,7})for(bool merge:{false,true}){
  CapacityOptions o;o.logical_pages=8;o.frames=frames;o.backend.geometry={1,1,1,8,8,1};o.merge_flush=merge;
  CapacityStore s(o);auto a=s.allocate(pages*4096);auto other=s.allocate(17);
  std::vector<unsigned char> x(pages*4096),y(x.size());for(size_t i=0;i<x.size();++i)x[i]=i*13+7;
  unsigned char tag=93;s.write(other,3,&tag,1);s.flush_all();auto before=s.statistics().backend;
  s.write(a,0,x.data(),x.size());x[3]=191;s.write(a,3,x.data()+3,1);s.flush_all();auto after=s.statistics();
  s.read(a,0,y.data(),y.size());check(x==y);unsigned char got=0;s.read(other,3,&got,1);check(got==tag);
  check(!after.dirty_frames&&!after.pins && after.peak_batch_pages<=static_cast<size_t>(frames));
  if(frames==8){check(after.backend.writebacks-before.writebacks==(merge?1:pages));check(after.backend.replayed_pages-before.replayed_pages==(merge?8:pages*8));}
  std::cout<<frames<<","<<pages<<","<<merge<<","<<after.backend.writebacks-before.writebacks<<","<<after.backend.replayed_pages-before.replayed_pages<<"\n";
 }
 {
  CapacityOptions o;o.logical_pages=8;o.frames=4;o.backend.geometry={1,1,1,8,8,1};o.merge_flush=true;
  CapacityStore s(o);auto a=s.allocate(4096),b=s.allocate(4096),c=s.allocate(4096);unsigned char x=23;
  s.write(a,0,&x,1);s.write(b,0,&x,1);s.write(c,0,&x,1);s.flush_many({a,b});
  auto stats=s.statistics();check(stats.merged_snapshots==1 && stats.merged_pages==2 && stats.dirty_frames==1);
  s.flush_all();check(s.statistics().merged_snapshots==2 && !s.statistics().dirty_frames);
 }
 // Separate flush calls are visibility boundaries, even within one block.
 for(bool merge:{false,true}){
  CapacityOptions o;o.logical_pages=16;o.frames=16;o.backend.geometry={1,1,1,8,8,1};o.merge_flush=merge;
  CapacityStore s(o);auto a=s.allocate(9*4096);unsigned char x=41;
  auto before=s.statistics().backend;s.write(a,0,&x,1);s.flush_all();s.write(a,4096,&x,1);s.write(a,8*4096,&x,1);s.flush_all();
  auto after=s.statistics();check(after.backend.writebacks-before.writebacks==3);check(after.backend.replayed_pages-before.replayed_pages==24);
 }
 // A failed logical snapshot may not mark any of its dirty frames clean.
 {
  CapacityOptions o;o.logical_pages=8;o.frames=4;o.backend.geometry={1,1,1,8,8,1};o.merge_flush=true;o.backend.fail_program_after=9;
  CapacityStore s(o);auto a=s.allocate(3*4096);unsigned char x=51;for(unsigned i=0;i<3;++i)s.write(a,i*4096,&x,1);
  bool failed=false;try{s.flush_all();}catch(...){failed=true;}check(failed);auto stats=s.statistics();check(stats.poisoned && stats.dirty_frames==3 && stats.writebacks==0);
 }
 // Cross-allocation merge, partial updates and checkpoint reopen.
 {
  const auto path=std::filesystem::temp_directory_path()/"hbflab-k2-checkpoint.pages";
  CapacityOptions o;o.logical_pages=8;o.frames=8;o.backend.geometry={1,1,1,8,8,1};o.merge_flush=true;o.backing_mode=BackingMode::CreateFile;o.backing_path=path;
  {CapacityStore s(o);auto a=s.allocate(4096),b=s.allocate(4096);unsigned char x=19,y=73;s.write(a,4095,&x,1);s.write(b,1,&y,1);auto before=s.statistics().backend;s.flush_all();check(s.statistics().backend.writebacks-before.writebacks==1);}
  o.backing_mode=BackingMode::OpenFile;
  {CapacityStore s(o);auto a=s.allocations();check(a.size()==2);unsigned char x=0,y=0;s.read(a[0],4095,&x,1);s.read(a[1],1,&y,1);check(x==19 && y==73);std::vector<unsigned char> actual(4096),expected(4096);expected[4095]=19;s.read(a[0],0,actual.data(),4096);check(actual==expected);expected.assign(4096,0);expected[1]=73;s.read(a[1],0,actual.data(),4096);check(actual==expected);}
  std::filesystem::remove(path);
 }
 return 0;
}catch(const std::exception& e){std::cerr<<e.what();return 1;}

