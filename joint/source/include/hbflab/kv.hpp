#pragma once
#include "hbflab/capacity.hpp"
#include <functional>
namespace hbflab {
struct KvOptions {
    std::size_t layers=1,heads=1,head_dimension=64,element_bytes=2;
    std::size_t tokens_per_block=16,max_tokens=4096,max_sequences=16;
    // Opt in until commit overhead has qualified for default use. The staging
    // budget is shared by this cache's transactions; 0 derives it from store capacity.
    bool cancellable_append=false;
    std::size_t max_staging_pages=0;
};
struct KvSequence {std::uint64_t cache=0,value=0;};
struct KvAppend {std::uint64_t cache=0,sequence=0,value=0;};
enum class KvKind { Key, Value };
struct KvSlice {Allocation allocation;std::size_t offset=0,bytes=0;};
// A callback may declare a transaction-local error only after its submitted
// work has completed safely. Outstanding leases still prevent reclamation.
class KvRecoverableError:public std::runtime_error {public:using std::runtime_error::runtime_error;};
struct KvTransactionStats {
    std::size_t staging_pages=0,peak_staging_pages=0,staging_budget_pages=0;
    std::uint64_t copied_tail_bytes=0,commits=0,cancellations=0,local_failures=0;
};
// Layout: each block has separate K and V allocations per layer, each laid out
// [token_in_block][head][head_dimension][element_bytes]. Bytes are dtype-opaque.
class KvCache {
public:
    KvCache(CapacityStore&,KvOptions = {});
    ~KvCache(); // Discards owned unpinned allocations; finish all leases first.
    KvCache(const KvCache&)=delete;
    KvCache& operator=(const KvCache&)=delete;
    KvSequence create();
    void release(KvSequence);
    void truncate(KvSequence,std::size_t tokens);
    // K and V each contain [layer][head][head_dimension] for ONE token.
    // Publish length only after all layers complete. Cancellable mode stages writes.
    void append(KvSequence,const void* keys,const void* values,std::size_t bytes_each);
    // One batch transaction per sequence. A layer becomes visible only after
    // both K and V are written; sequence length publishes at commit.
    KvAppend begin_append(KvSequence,std::size_t tokens);
    void write_layer(KvAppend,std::size_t layer,const void* keys,const void* values,std::size_t bytes_each);
    // Synchronous writer must finish all leases before returning. Source offset
    // is relative to the submitted layer; readiness publishes after both kinds.
    using LayerWriter = std::function<void(KvKind, KvSlice, std::size_t)>;
    void write_layer_with(KvAppend, std::size_t layer, std::size_t bytes_each, const LayerWriter&);
    KvSlice locate_layer(KvAppend,std::size_t layer,std::size_t token,KvKind) const;
    void commit(KvAppend);
    // In cancellable mode discards only staging; requires finished leases. Repeat
    // abort/commit and all operations on a retired handle are rejected as stale.
    // Legacy in-place mode retains its terminal abort behavior.
    void abort(KvAppend);
    void read(KvSequence,std::size_t layer,std::size_t token,KvKind,void*,std::size_t bytes);
    KvSlice locate(KvSequence,std::size_t layer,std::size_t token,KvKind) const;
    void flush(KvSequence);
    std::size_t length(KvSequence) const;
    // Logical publication generation, independent of transient frame residency.
    std::uint64_t version(KvSequence) const;
    std::size_t row_bytes() const;
    bool poisoned() const;
    KvTransactionStats transaction_statistics() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
