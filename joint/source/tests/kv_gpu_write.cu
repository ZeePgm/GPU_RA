#include "hbflab/capacity_gpu.hpp"
#include "hbflab/kv.hpp"
#include "ptx_fixture.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
using namespace hbflab;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void range_and_replica_contract() {
    CapacityOptions options;options.logical_pages=8;options.frames=1;options.backend.geometry={1,1,1,8,8,1};
    CapacityGpu gpu(options);auto& store=gpu.store();auto a=store.allocate(4096),other=store.allocate(4096);
    std::vector<unsigned char> expected(4096),actual(4096);
    for(std::size_t i=0;i<expected.size();++i)expected[i]=static_cast<unsigned char>(i*23+11);
    store.write(a,0,expected.data(),expected.size());
    unsigned char* source=nullptr;ck(cudaMalloc(&source,4096));cudaStream_t stream;ck(cudaStreamCreate(&stream));
    ck(cudaMemsetAsync(source,93,4096,stream));
    gpu.write_device(a,113,source,71,stream);std::fill(expected.begin()+113,expected.begin()+184,93);
    auto stats=gpu.statistics();require(stats.h2d_bytes==0 && stats.d2h_bytes==71,"stale partial mirror: exact range download, no preservation upload");
    auto read=[&]{auto lease=gpu.acquire(a,0,PageAccess::Read,stream);ck(cudaMemcpyAsync(actual.data(),gpu.data(lease),4096,cudaMemcpyDeviceToHost,stream));gpu.finish(lease);require(actual==expected,"complete GPU page byte oracle");};
    read();require(gpu.statistics().h2d_bytes==4096,"stale partial mirror must reload untouched bytes");
    gpu.write_device(a,300,source,57,stream);std::fill(expected.begin()+300,expected.begin()+357,93);
    read();require(gpu.statistics().h2d_bytes==4096 && gpu.statistics().d2h_bytes==128,"current partial mirror remains valid");
    unsigned char cpu=201;store.write(a,500,&cpu,1);expected[500]=cpu;
    gpu.write_device(a,700,source,3,stream);std::fill(expected.begin()+700,expected.begin()+703,93);
    read();require(gpu.statistics().h2d_bytes==8192,"host revision change invalidates partial GPU mirror");
    auto evict=gpu.acquire(other,0,PageAccess::Read,stream);gpu.finish(evict);
    ck(cudaMemcpyAsync(source,expected.data(),4096,cudaMemcpyHostToDevice,stream));
    const auto before=gpu.statistics();
    auto lease=gpu.acquire_read_device(a,0,source,stream);
    ck(cudaMemcpyAsync(actual.data(),gpu.data(lease),4096,cudaMemcpyDeviceToHost,stream));gpu.finish(lease);
    require(actual==expected,"replica fill after frame generation change");
    auto after=gpu.statistics();require(after.h2d_bytes==before.h2d_bytes && after.read_source_d2d_bytes-before.read_source_d2d_bytes==4096,"replica replaces exactly one page upload");
    ck(cudaMemsetAsync(source,0,4096,stream));read(); // source is borrowed only through acquisition
    require(gpu.statistics().read_source_fills==1,"bounded synchronous source lifetime");
    store.read(a,0,actual.data(),4096);require(actual==expected,"host authority unchanged by replica reads");
    gpu.flush();require(!store.statistics().pins && !gpu.statistics().active_leases && !store.statistics().dirty_frames,"range/replica drain");
    rejects([&]{gpu.acquire_read_device(a,0,nullptr,stream);});
    ck(cudaStreamDestroy(stream));ck(cudaFree(source));
}
int main(int argc,char** argv) try {
    range_and_replica_contract();
    const std::string shape=argc>1?argv[1]:"full";
    const bool device=argc>2 && std::string(argv[2])=="device";
    const bool file=argc>3 && std::string(argv[3])=="file";
    const auto path=std::filesystem::temp_directory_path()/("k1-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".pages");
    struct Cleanup {std::filesystem::path p;~Cleanup(){std::error_code e;std::filesystem::remove(p,e);}} cleanup{path};
    CapacityOptions options;options.logical_pages=64;options.frames=shape=="full"?8:2;
    options.backend.geometry={1,1,1,8,8,1};

    KvOptions layout;layout.layers=2;layout.heads=1;layout.head_dimension=shape=="full"?1024:shape=="wide"?3200:192;
    layout.element_bytes=2;layout.tokens_per_block=shape=="full"?2:shape=="wide"?3:16;layout.max_tokens=64;
    const std::size_t tokens=shape=="full"?2:shape=="wide"?7:17;
    {
        CapacityGpu gpu(options);auto& store=gpu.store();KvCache kv(store,layout);auto seq=kv.create();
        const auto row=kv.row_bytes();std::vector<unsigned char> keys(tokens*row),values(keys.size());
        for(std::size_t i=0;i<keys.size();++i){keys[i]=static_cast<unsigned char>(i*13+7);values[i]=static_cast<unsigned char>(i*19+31);}
        unsigned char *dk=nullptr,*dv=nullptr;ck(cudaMalloc(&dk,keys.size()));ck(cudaMalloc(&dv,values.size()));
        ck(cudaMemcpy(dk,keys.data(),keys.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dv,values.data(),values.size(),cudaMemcpyHostToDevice));
        auto tx=kv.begin_append(seq,tokens);
        rejects([&]{kv.write_layer_with(tx,2,keys.size(),[](KvKind,KvSlice,std::size_t){});});
        rejects([&]{gpu.write_device({},0,dk,1);});
        std::uint64_t callback_bytes=0;
        for(std::size_t layer=0;layer<layout.layers;++layer){
            if(device)kv.write_layer_with(tx,layer,keys.size(),[&](KvKind kind,KvSlice slice,std::size_t offset){
                require(kv.length(seq)==0,"no early publication");
                gpu.write_device(slice.allocation,slice.offset,(kind==KvKind::Key?dk:dv)+offset,slice.bytes);
                callback_bytes+=slice.bytes;
            });
            else {
                ck(cudaMemcpy(keys.data(),dk,keys.size(),cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(values.data(),dv,values.size(),cudaMemcpyDeviceToHost));
                kv.write_layer(tx,layer,keys.data(),values.data(),keys.size());
            }
            require(kv.length(seq)==0,"layer does not publish length");
            rejects([&]{kv.write_layer_with(tx,layer,keys.size(),[](KvKind,KvSlice,std::size_t){});});
            if(layer+1<layout.layers)rejects([&]{kv.commit(tx);});
        }
        // Consume through GPU pages before publication, as paged attention does.
        for(std::size_t layer=0;layer<layout.layers;++layer)for(std::size_t token=0;token<tokens;++token)for(auto kind:{KvKind::Key,KvKind::Value}) {
            const auto slice=kv.locate_layer(tx,layer,token,kind);
            for(std::size_t at=0;at<slice.bytes;) {
                const auto pos=slice.offset+at;
                const auto count=(std::min)(slice.bytes-at,std::size_t{4096}-pos%4096);
                const auto lease=gpu.acquire(slice.allocation,pos/4096,PageAccess::Read);
                std::vector<unsigned char> actual(count);
                ck(cudaMemcpy(actual.data(),static_cast<const unsigned char*>(gpu.data(lease))+pos%4096,count,cudaMemcpyDeviceToHost));
                gpu.finish(lease);
                const auto& expected=kind==KvKind::Key?keys:values;
                require(std::equal(actual.begin(),actual.end(),expected.begin()+token*row+at),"GPU page consumption byte oracle");
                at+=count;
            }
        }
        auto transfers=gpu.statistics();
        require(transfers.device_bytes==options.frames*4096 && !transfers.active_leases && !store.statistics().pins,"bounded frames and released leases");
        if(device){require(callback_bytes==2*layout.layers*tokens*row && transfers.d2d_bytes==callback_bytes && transfers.d2h_bytes==callback_bytes,"exact GPU source and range collection bytes");
            if(shape=="full")require(transfers.h2d_bytes==0 && transfers.d2h_bytes==callback_bytes && transfers.overwrite_uploads_skipped==4,"full overwrite ledger");}
        if(shape=="full" && !device)require(transfers.h2d_bytes==2*layout.layers*tokens*row,"host reference page upload ledger");
        kv.commit(tx);require(kv.length(seq)==tokens,"atomic length publication");
        std::vector<unsigned char> actual(row);
        for(std::size_t layer=0;layer<layout.layers;++layer)for(std::size_t token=0;token<tokens;++token)for(auto kind:{KvKind::Key,KvKind::Value}){
            kv.read(seq,layer,token,kind,actual.data(),row);
            const auto& expected=kind==KvKind::Key?keys:values;
            require(std::equal(actual.begin(),actual.end(),expected.begin()+token*row),"eviction/host read byte oracle");
        }
        // Appending into the existing tail must preserve its committed prefix.
        kv.truncate(seq,1);auto next=kv.begin_append(seq,1);
        for(std::size_t layer=0;layer<layout.layers;++layer)kv.write_layer_with(next,layer,row,[&](KvKind kind,KvSlice slice,std::size_t offset){
            require(offset==0,"single row source offset");gpu.write_device(slice.allocation,slice.offset,kind==KvKind::Key?dv:dk,slice.bytes);
        });
        kv.commit(next);
        kv.read(seq,0,0,KvKind::Key,actual.data(),row);require(std::equal(actual.begin(),actual.end(),keys.begin()),"partial page preserves prefix");
        kv.read(seq,0,1,KvKind::Key,actual.data(),row);require(std::equal(actual.begin(),actual.end(),values.begin()),"tail append replacement");
        kv.flush(seq);gpu.flush();
        auto failed=kv.begin_append(seq,1);
        rejects([&]{kv.write_layer_with(failed,0,row,[&](KvKind kind,KvSlice slice,std::size_t){if(kind==KvKind::Value)throw std::runtime_error("injected writer failure");gpu.write_device(slice.allocation,slice.offset,dk,slice.bytes);});});
        require(kv.poisoned(),"failed writer poisons transaction");rejects([&]{kv.commit(failed);});
        ck(cudaFree(dk));ck(cudaFree(dv));
        std::cout<<"{\"shape\":\""<<shape<<"\",\"device\":"<<device<<",\"file\":"<<file<<",\"h2d\":"<<transfers.h2d_bytes<<",\"d2h\":"<<transfers.d2h_bytes<<",\"d2d\":"<<transfers.d2d_bytes<<",\"commit_d2h\":"<<(device?0:2*layout.layers*tokens*row)<<",\"status\":\"pass\"}\n";
    }
    if(file){
        options.backing_mode=BackingMode::CreateFile;options.backing_path=path;
        std::vector<unsigned char> expected(8207);
        for(std::size_t i=0;i<expected.size();++i)expected[i]=static_cast<unsigned char>(i*17+3);
        {
            CapacityGpu gpu(options);auto allocation=gpu.store().allocate(expected.size());
            unsigned char* source=nullptr;ck(cudaMalloc(&source,expected.size()));
            ck(cudaMemcpy(source,expected.data(),expected.size(),cudaMemcpyHostToDevice));
            gpu.write_device(allocation,0,source,expected.size());
            gpu.write_device(allocation,4093,source+100,19); // partial update across a page boundary
            std::copy_n(expected.begin()+100,19,expected.begin()+4093);
            gpu.flush();ck(cudaFree(source));
            require(gpu.statistics().active_leases==0 && gpu.store().statistics().pins==0,"file leases drained");
        }
        options.backing_mode=BackingMode::OpenFile;CapacityStore restored(options);
        auto allocations=restored.allocations();require(allocations.size()==1,"checkpoint directory");
        std::vector<unsigned char> bytes(restored.size(allocations[0]));restored.read(allocations[0],0,bytes.data(),bytes.size());
        require(bytes==expected,"checkpoint byte oracle including short tail");
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
