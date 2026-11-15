#pragma once
#include "hbflab/capacity.hpp"
#include <cuda_runtime_api.h>
namespace hbflab {
struct GpuPageLease {std::uint64_t owner=0,value=0;};
enum class GpuLeaseState { Preparing, Ready, InUse, Writeback, Retiring };
struct CapacityGpuAsyncOptions {
    std::size_t staging_pages=0; // 0 retains the synchronous implementation.
    std::size_t demand_reserve=1; // Slots never consumed by speculation.
};
struct CapacityGpuStats {
    std::uint64_t uploads=0,downloads=0,mirror_hits=0,h2d_bytes=0,d2h_bytes=0,copy_wall_ns=0;
    std::uint64_t prefetch_uploads=0;
    std::uint64_t d2d_bytes=0, overwrite_uploads_skipped=0;
    std::uint64_t read_source_d2d_bytes=0, read_source_fills=0;
    std::size_t device_bytes=0,active_leases=0;
    std::size_t pinned_staging_bytes=0,peak_inflight=0;
    std::uint64_t async_prepared=0,async_retired=0,prefetch_issued=0,prefetch_used=0,prefetch_cancelled=0,pressure_waits=0;
};
// Explicit kernel-boundary paging, single host owner. Queue all pointer uses on
// the bound consumer stream before finish/finish_async (or drain/close). Async
// data access requires use(); poll may retire completed handles. No device fault.
class CapacityGpu {
public:
    explicit CapacityGpu(CapacityOptions = {},CapacityMedia* shared_media=nullptr,CapacityGpuAsyncOptions = {});
    ~CapacityGpu();
    CapacityGpu(const CapacityGpu&)=delete;
    CapacityGpu& operator=(const CapacityGpu&)=delete;
    CapacityStore& store();
    GpuPageLease acquire(Allocation,std::size_t page,PageAccess,cudaStream_t=nullptr);
    // Caller supplies an exact, current GPU replica of all valid bytes of this
    // page. Source remains alive until return; normal pin/media/lease rules apply.
    GpuPageLease acquire_read_device(Allocation,std::size_t page,const void* source,cudaStream_t=nullptr);
    // Explicit async replica ownership: source allocation stays alive until
    // retirement. After source_ready completes its bytes must exactly match
    // the current page and remain unchanged until retirement. Record the event
    // before this call; nullptr promises the source is already complete. No ownership
    // is inferred from the shorter acquire_read_device lifetime contract.
    GpuPageLease acquire_read_device_async(Allocation,std::size_t page,const void* source,cudaEvent_t source_ready,cudaStream_t consumer);
    const void* data(GpuPageLease) const;
    void* writable_data(GpuPageLease) const;
    std::size_t bytes(GpuPageLease) const;
    void finish(GpuPageLease); // Drain stream, collect writable bytes, then unpin.
    // Async page lifecycle. prepare still admits media through the existing
    // host owner; DMA completion and GPU consumption use explicit events.
    GpuPageLease prepare(Allocation,std::size_t page,PageAccess);
    void use(GpuPageLease,cudaStream_t); // Inserts upload dependency on this stream.
    GpuLeaseState lease_state(GpuPageLease);
    void finish_async(GpuPageLease); // Record last consumer; retain pin until retired.
    void wait(GpuPageLease); // Wait/retire a finish_async lease not yet retired by poll.
    std::size_t poll(); // Nonblocking retirement of completed copies/consumers.
    void cancel(GpuPageLease); // Only before use; waits upload before reclamation.
    bool try_prefetch(Allocation,std::size_t page); // Bounded, read-only, best effort.
    // If accepted, keep the source alive/unchanged through adoption+retirement,
    // cancellation, or drain. The same readiness contract as above applies.
    bool try_prefetch_device(Allocation,std::size_t page,const void* source,cudaEvent_t source_ready);
    void drain(); // Finish submitted uses, cancel unused preparations, keep reusable.
    void close(); // drain + flush, reject future leases; idempotent.
    bool async_enabled() const;
    // Copies GPU source into capacity pages, synchronously collecting each page
    // through range collection. Unmodified host bytes remain authoritative;
    // an incomplete GPU mirror is invalidated before any subsequent read.
    void write_device(Allocation, std::size_t offset, const void* source, std::size_t bytes, cudaStream_t=nullptr);
    void flush(); // All leases must be finished. Wait for modeled writeback duration.
    std::size_t prefetch(Allocation,std::size_t first_page,std::size_t count);
    std::size_t maintain(double gpu_watts=0);
    std::size_t device_bytes() const;
    CapacityGpuStats statistics() const;
private:
    GpuPageLease acquire_impl(Allocation, std::size_t, PageAccess, cudaStream_t, bool overwrite, const void* read_source=nullptr);
    GpuPageLease acquire_async(Allocation,std::size_t,PageAccess,cudaStream_t,const void* source=nullptr,cudaEvent_t source_ready=nullptr);
    GpuPageLease prepare_impl(Allocation,std::size_t,PageAccess,bool speculative,const void* source=nullptr,cudaEvent_t source_ready=nullptr);
    bool try_prefetch_impl(Allocation,std::size_t,const void*,cudaEvent_t);
    void retire(GpuPageLease);
    void finish_range(GpuPageLease,std::size_t offset,std::size_t bytes);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
