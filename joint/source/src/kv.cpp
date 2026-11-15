#include "hbflab/kv.hpp"
#include <atomic>
#include <array>
#include <map>
#include <thread>

namespace hbflab {
namespace {std::atomic<std::uint64_t> next_cache{1};}
struct KvCache::Impl {
    struct Pair {Allocation key,value;};
    struct Sequence {std::size_t length=0;std::uint64_t version=0;std::vector<std::vector<Pair>> blocks;
        std::uint64_t transaction=0;std::size_t pending=0;std::vector<bool> ready;
        std::vector<std::vector<Pair>> staged;
        std::size_t first_staged=0,staging_pages=0;bool local_failed=false,writing=false;};
    CapacityStore& store;
    KvOptions options;
    std::size_t row,block_bytes,block_pages;
    std::map<std::uint64_t,Sequence> sequences;
    const std::uint64_t identity=next_cache.fetch_add(1);
    std::uint64_t serial=0;
    const std::thread::id thread=std::this_thread::get_id();
    bool failed=false;
    KvTransactionStats stats;
    Impl(CapacityStore& s,KvOptions o):store(s),options(o) {
        if(!o.layers || o.layers>256 || !o.heads || !o.head_dimension ||
           (o.element_bytes!=1 && o.element_bytes!=2 && o.element_bytes!=4 && o.element_bytes!=8) ||
           !o.tokens_per_block || !o.max_tokens || !o.max_sequences || o.max_sequences>65536)
            throw std::invalid_argument("invalid KV layout");
        row=hbf_eventlab::checked_mul(hbf_eventlab::checked_mul(o.heads,o.head_dimension),o.element_bytes);
        block_bytes=hbf_eventlab::checked_mul(row,o.tokens_per_block);
        block_pages=hbf_eventlab::checked_mul(hbf_eventlab::checked_add(block_bytes,4095)/4096,o.layers*2);
        hbf_eventlab::checked_mul(row,o.layers);
        stats.staging_budget_pages=o.max_staging_pages?o.max_staging_pages:
            hbf_eventlab::checked_add(store.free_pages(),store.statistics().allocated_pages);
    }
    void current() const {
        if(thread!=std::this_thread::get_id())throw std::logic_error("KV host owner changed");
        if(failed || store.statistics().poisoned)throw std::runtime_error("KV cache poisoned by incomplete operation");
    }
    Sequence& sequence(KvSequence h) {
        current();const auto it=sequences.find(h.value);
        if(h.cache!=identity || it==sequences.end())throw std::invalid_argument("foreign/stale KV sequence");return it->second;
    }
    Sequence& transaction(KvAppend h,bool allow_failed=false) {
        auto& q=sequence({h.cache,h.sequence});
        if(!h.value || h.value!=q.transaction)throw std::invalid_argument("stale KV append transaction");
        if(q.writing)throw CapacityBusy("KV writer callback is active");
        if(q.local_failed && !allow_failed)throw KvRecoverableError("KV transaction failed; abort required");
        return q;
    }
    void idle(const Sequence& q) const {if(q.transaction)throw CapacityBusy("KV append transaction active");}
    void unpinned(const Sequence& s) const {
        for(const auto* blocks:{&s.blocks,&s.staged})for(const auto& b:*blocks)for(const auto& p:b)
            if(store.pin_count(p.key) || store.pin_count(p.value))throw CapacityBusy("KV mutation requires released sequence leases");
    }
    void free_block(const std::vector<Pair>& b){for(const auto& p:b){store.release(p.key);store.release(p.value);}}
    const std::vector<std::vector<Pair>>& working(const Sequence& q) const {
        return options.cancellable_append?q.staged:q.blocks;
    }
    void write_failed(Sequence& q,bool recoverable) {
        q.writing=false;
        if(options.cancellable_append && recoverable && !store.statistics().poisoned){q.local_failed=true;++stats.local_failures;}
        else {failed=true;store.abort();}
    }
    void clear_transaction(Sequence& q) {
        stats.staging_pages-=q.staging_pages;q.staging_pages=0;q.staged.clear();
        q.pending=0;q.transaction=0;q.ready.clear();q.local_failed=false;
    }
};
KvCache::KvCache(CapacityStore& store,KvOptions o):impl_(std::make_unique<Impl>(store,o)){}
KvCache::~KvCache(){
    // Teardown discards owned data, never performs implicit media writes. Callers
    // must finish leases first; a pinned allocation is preserved until store teardown.
    for(const auto& s:impl_->sequences)for(const auto& b:s.second.blocks)for(const auto& p:b) {
        try {impl_->store.release(p.key);}catch(...){}
        try {impl_->store.release(p.value);}catch(...){}
    }
    for(const auto& s:impl_->sequences)for(std::size_t i=s.second.first_staged;i<s.second.staged.size();++i)
        for(const auto& p:s.second.staged[i]){try{impl_->store.release(p.key);}catch(...){}try{impl_->store.release(p.value);}catch(...){}}
}
KvSequence KvCache::create() {
    auto& s=*impl_;s.current();if(s.sequences.size()>=s.options.max_sequences)throw CapacityBusy("KV sequence budget exhausted");
    s.serial=hbf_eventlab::checked_add(s.serial,1);s.sequences.emplace(s.serial,Impl::Sequence{});return {s.identity,s.serial};
}
void KvCache::release(KvSequence h) {
    auto& s=*impl_;const auto& sequence=s.sequence(h);s.idle(sequence);s.unpinned(sequence);
    for(const auto& b:sequence.blocks)s.free_block(b);s.sequences.erase(h.value);
}
void KvCache::truncate(KvSequence h,std::size_t tokens) {
    auto& s=*impl_;auto& sequence=s.sequence(h);s.idle(sequence);s.unpinned(sequence);
    if(tokens>sequence.length)throw std::invalid_argument("KV truncate cannot grow sequence");
    const auto version=tokens==sequence.length?sequence.version:hbf_eventlab::checked_add(sequence.version,1);
    const auto count=tokens/s.options.tokens_per_block+(tokens%s.options.tokens_per_block!=0);
    while(sequence.blocks.size()>count){s.free_block(sequence.blocks.back());sequence.blocks.pop_back();}sequence.length=tokens;sequence.version=version;
}
void KvCache::append(KvSequence h,const void* keys,const void* values,std::size_t bytes) {
    auto& s=*impl_;auto& sequence=s.sequence(h);s.idle(sequence);s.unpinned(sequence);
    if(!keys || !values || bytes!=s.row*s.options.layers || sequence.length>=s.options.max_tokens)
        throw std::invalid_argument("KV append requires one complete K/V token within token limit");
    if(s.options.cancellable_append){
        const auto tx=begin_append(h,1);
        try{for(std::size_t layer=0;layer<s.options.layers;++layer)
            write_layer(tx,layer,static_cast<const unsigned char*>(keys)+layer*s.row,
                        static_cast<const unsigned char*>(values)+layer*s.row,s.row);
            commit(tx);
        }catch(...){if(!poisoned())abort(tx);throw;}
        return;
    }
    // Avoid partial mutation caused by external pins exhausting the frame pool.
    if(s.store.statistics().pins)throw CapacityBusy("KV append requires an idle frame pool");
    const auto block=sequence.length/s.options.tokens_per_block;
    if(block==sequence.blocks.size()) {
        if(s.store.free_pages()<s.block_pages)throw CapacityBusy("KV block exceeds remaining logical capacity");
        std::vector<Impl::Pair> allocated;allocated.reserve(s.options.layers);
        sequence.blocks.reserve(sequence.blocks.size()+1);
        try {
            for(std::size_t layer=0;layer<s.options.layers;++layer) {
                const auto key=s.store.allocate(s.block_bytes);
                try {const auto value=s.store.allocate(s.block_bytes);allocated.push_back({key,value});}
                catch(...){s.store.release(key);throw;}
            }
        } catch(...){s.free_block(allocated);throw;}
        sequence.blocks.push_back(std::move(allocated));
    }
    try {
        const auto offset=(sequence.length%s.options.tokens_per_block)*s.row;
        for(std::size_t layer=0;layer<s.options.layers;++layer) {
            const auto& p=sequence.blocks[block][layer];
            s.store.write(p.key,offset,static_cast<const unsigned char*>(keys)+layer*s.row,s.row);
            s.store.write(p.value,offset,static_cast<const unsigned char*>(values)+layer*s.row,s.row);
        }
        ++sequence.length;++sequence.version;
    } catch(...){s.failed=true;throw;}
}
KvAppend KvCache::begin_append(KvSequence h,std::size_t tokens) {
    auto& s=*impl_;auto& q=s.sequence(h);s.idle(q);s.unpinned(q);
    if(!tokens || tokens>s.options.max_tokens-q.length)throw std::invalid_argument("KV batch token limit");
    if(s.store.statistics().pins)throw CapacityBusy("KV append requires idle frame pool");
    hbf_eventlab::checked_mul(tokens,s.row);
    const auto total=q.length+tokens;
    const auto needed=total/s.options.tokens_per_block+(total%s.options.tokens_per_block!=0);
    if(s.options.cancellable_append){
        const auto first=q.length/s.options.tokens_per_block;
        const auto pages=hbf_eventlab::checked_mul(needed-first,s.block_pages);
        if(pages>s.store.free_pages() || pages>s.stats.staging_budget_pages-s.stats.staging_pages)
            throw CapacityBusy("KV staging budget exhausted");
        // Prepare the entire replacement map before publishing a transaction.
        // Untouched prefix allocations are borrowed, not copied or owned twice.
        auto staged=q.blocks;staged.resize(first);staged.reserve(needed);
        std::vector<bool> ready(s.options.layers,false);
        s.serial=hbf_eventlab::checked_add(s.serial,1);
        try{
            while(staged.size()<needed){
                std::vector<Impl::Pair> block;block.reserve(s.options.layers);
                try{for(std::size_t layer=0;layer<s.options.layers;++layer){auto key=s.store.allocate(s.block_bytes);
                    try{auto value=s.store.allocate(s.block_bytes);block.push_back({key,value});}catch(...){s.store.release(key);throw;}}}
                catch(...){if(!s.store.statistics().poisoned)s.free_block(block);throw;}
                staged.push_back(std::move(block));
            }
            if(first<q.blocks.size()){
                std::array<unsigned char,4096> buffer{};
                for(std::size_t layer=0;layer<s.options.layers;++layer)for(auto kind:{KvKind::Key,KvKind::Value}){
                    const auto& old=q.blocks[first][layer];const auto& fresh=staged[first][layer];
                    for(std::size_t at=0;at<s.block_bytes;){const auto n=(std::min)(buffer.size(),s.block_bytes-at);
                        s.store.read(kind==KvKind::Key?old.key:old.value,at,buffer.data(),n);
                        s.store.write(kind==KvKind::Key?fresh.key:fresh.value,at,buffer.data(),n);
                        s.stats.copied_tail_bytes+=n;at+=n;
                    }
                }
            }
        }catch(...){if(!s.store.statistics().poisoned)for(std::size_t i=first;i<staged.size();++i)s.free_block(staged[i]);throw;}
        q.staged=std::move(staged);q.ready=std::move(ready);q.first_staged=first;
        q.staging_pages=pages;s.stats.staging_pages+=pages;
        s.stats.peak_staging_pages=(std::max)(s.stats.peak_staging_pages,s.stats.staging_pages);
        q.transaction=s.serial;q.pending=tokens;q.local_failed=false;
        return {s.identity,h.value,q.transaction};
    }
    if(needed-q.blocks.size()>s.store.free_pages()/s.block_pages)throw CapacityBusy("KV batch capacity exhausted");
    const auto old=q.blocks.size();q.blocks.reserve(needed);q.ready.assign(s.options.layers,false);
    try {
        while(q.blocks.size()<needed) {
            std::vector<Impl::Pair> b;b.reserve(s.options.layers);
            try {for(std::size_t i=0;i<s.options.layers;++i){auto k=s.store.allocate(s.block_bytes);
                try{auto v=s.store.allocate(s.block_bytes);b.push_back({k,v});}catch(...){s.store.release(k);throw;}}}
            catch(...){s.free_block(b);throw;}
            q.blocks.push_back(std::move(b));
        }
    } catch(...){while(q.blocks.size()>old){s.free_block(q.blocks.back());q.blocks.pop_back();}throw;}
    s.serial=hbf_eventlab::checked_add(s.serial,1);q.transaction=s.serial;q.pending=tokens;
    return {s.identity,h.value,q.transaction};
}
void KvCache::write_layer(KvAppend h,std::size_t layer,const void* keys,const void* values,std::size_t bytes) {
    auto& s=*impl_;auto& q=s.transaction(h);
    if(layer>=s.options.layers || q.ready[layer] || !keys || !values || bytes!=q.pending*s.row)
        throw std::invalid_argument("invalid or duplicate KV layer submission");
    s.unpinned(q);if(s.store.statistics().pins)throw CapacityBusy("KV layer write requires idle frame pool");
    q.writing=true;try {
        for(std::size_t i=0;i<q.pending;++i) {
            const auto token=q.length+i;const auto& p=s.working(q)[token/s.options.tokens_per_block][layer];
            const auto offset=(token%s.options.tokens_per_block)*s.row;
            s.store.write(p.key,offset,static_cast<const unsigned char*>(keys)+i*s.row,s.row);
            s.store.write(p.value,offset,static_cast<const unsigned char*>(values)+i*s.row,s.row);
        }
        q.ready[layer]=true;q.writing=false;
    }catch(const CapacityBusy&){s.write_failed(q,true);throw;}
    catch(...){s.write_failed(q,false);throw;}
}
void KvCache::write_layer_with(KvAppend h, std::size_t layer, std::size_t bytes, const LayerWriter& writer) {
    auto& s = *impl_;
    auto& q = s.transaction(h);
    if (layer >= s.options.layers || q.ready[layer] || !writer || bytes != q.pending * s.row)
        throw std::invalid_argument("invalid or duplicate KV layer writer");
    s.unpinned(q);
    if (s.store.statistics().pins) throw CapacityBusy("KV layer write requires idle frame pool");
    q.writing=true;try {
        for (std::size_t i = 0; i < q.pending;) {
            const auto token = q.length + i;
            const auto& pair = s.working(q)[token / s.options.tokens_per_block][layer];
            const auto count = (std::min)(q.pending - i, s.options.tokens_per_block - token % s.options.tokens_per_block);
            const auto offset = token % s.options.tokens_per_block * s.row;
            writer(KvKind::Key, {pair.key, offset, count * s.row}, i * s.row);
            writer(KvKind::Value, {pair.value, offset, count * s.row}, i * s.row);
            i += count;
        }
        s.unpinned(q);
        q.ready[layer] = true;q.writing=false;
    } catch (const KvRecoverableError&) {s.write_failed(q,true);throw;}
      catch (const CapacityBusy&) {s.write_failed(q,true);throw;}
      catch (...) {s.write_failed(q,false);throw;}
}
KvSlice KvCache::locate_layer(KvAppend h,std::size_t layer,std::size_t token,KvKind kind) const {
    auto& s=*impl_;const auto& q=s.transaction(h);
    if(layer>=s.options.layers || token>=q.length+(q.ready[layer]?q.pending:0) ||
       (kind!=KvKind::Key && kind!=KvKind::Value))throw std::invalid_argument("KV layer is not published");
    const auto& p=s.working(q)[token/s.options.tokens_per_block][layer];
    return {kind==KvKind::Key?p.key:p.value,(token%s.options.tokens_per_block)*s.row,s.row};
}
void KvCache::commit(KvAppend h) {
    auto& s=*impl_;auto& q=s.transaction(h);s.unpinned(q);
    for(bool ready:q.ready)if(!ready)throw CapacityBusy("KV batch has unpublished layers");
    const auto version=hbf_eventlab::checked_add(q.version,1);
    if(s.options.cancellable_append){
        // No callbacks or throwing allocations between map and length publication.
        q.blocks.swap(q.staged);q.length+=q.pending;q.version=version;
        try{for(std::size_t i=q.first_staged;i<q.staged.size();++i)s.free_block(q.staged[i]);}
        catch(...){s.failed=true;s.store.abort();throw;}
        s.clear_transaction(q);
    }else{q.length+=q.pending;q.version=version;q.pending=0;q.transaction=0;q.ready.clear();}
    ++s.stats.commits;
}
void KvCache::abort(KvAppend h){
    auto& s=*impl_;auto& q=s.transaction(h,true);
    if(!s.options.cancellable_append){s.failed=true;s.store.abort();return;}
    s.unpinned(q);
    try{for(std::size_t i=q.first_staged;i<q.staged.size();++i)s.free_block(q.staged[i]);}
    catch(...){s.failed=true;s.store.abort();throw;}
    s.clear_transaction(q);++s.stats.cancellations;
}
KvSlice KvCache::locate(KvSequence h,std::size_t layer,std::size_t token,KvKind kind) const {
    auto& s=*impl_;const auto& sequence=s.sequence(h);
    if(layer>=s.options.layers || token>=sequence.length || (kind!=KvKind::Key && kind!=KvKind::Value))
        throw std::invalid_argument("KV layer/token/kind out of range");
    const auto& p=sequence.blocks[token/s.options.tokens_per_block][layer];
    return {kind==KvKind::Key?p.key:p.value,(token%s.options.tokens_per_block)*s.row,s.row};
}
void KvCache::read(KvSequence h,std::size_t layer,std::size_t token,KvKind kind,void* out,std::size_t bytes) {
    const auto slice=locate(h,layer,token,kind);if(bytes!=slice.bytes || !out)throw std::invalid_argument("KV read row size");
    impl_->store.read(slice.allocation,slice.offset,out,bytes);
}
void KvCache::flush(KvSequence h){auto& s=*impl_;const auto& sequence=s.sequence(h);s.unpinned(sequence);std::vector<Allocation> allocations;for(const auto& b:sequence.blocks)for(const auto& p:b){allocations.push_back(p.key);allocations.push_back(p.value);}s.store.flush_many(allocations);}
std::size_t KvCache::length(KvSequence h) const{return impl_->sequence(h).length;}
std::uint64_t KvCache::version(KvSequence h) const{return impl_->sequence(h).version;}
std::size_t KvCache::row_bytes() const{impl_->current();return impl_->row;}
bool KvCache::poisoned() const{return impl_->failed || impl_->store.statistics().poisoned;}
KvTransactionStats KvCache::transaction_statistics() const{impl_->current();return impl_->stats;}
}
