#include "five_media_api.hpp"
#include "integrated_geometry.hpp"
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" cudaError_t hbflab_q_auto_bind_convert(void*);
extern "C" cudaError_t hbflab_q_auto_bind_mmvf(void*);
extern "C" cudaError_t hbflab_q_auto_bind_binbcast(void*);
extern "C" cudaError_t hbflab_q_auto_bind_getrows(void*);

namespace {
using U64 = std::uint64_t;
using U32 = std::uint32_t;
using five_media::RangeSpec;
constexpr unsigned page_bytes = 4096, halfs = 2048, mask_words = 64, batch_limit = 16;
constexpr std::size_t sites_per_range = 4 * 32768;
constexpr U32 idle = 0, requested = 1, ready = 2, failed = 3;

#if defined(HBFLAB_ALL_MEDIA_BUILD)
constexpr bool all_media = true;
struct alignas(64) GpuRange { U64 base, bytes, first_page; U32* counts; U64 ordinal; };
static_assert(sizeof(GpuRange) == 64 && offsetof(GpuRange, ordinal) == 32);
#else
constexpr bool all_media = false;
struct GpuRange { U64 base, bytes, first_page; U32* counts; };
static_assert(sizeof(GpuRange) == 32);
#endif
std::size_t page_count(U64 bytes) { return (bytes-1)/page_bytes+1; }
struct alignas(64) Page { U64 request_ns, delay_ns; U32 status; U64 released_base; };
struct alignas(64) Audit { U64 ready_base, earliest_ns, latest_ns, loads, bytes; U32 claim; };
struct Fault { U64 code, address, page_index, time_ns; };
struct Root {
    GpuRange* ranges;
    U64 count;
    Page* pages;
    Audit* audits;
    U32* coverage;
    U64* sites;
    U32* queue;
    U32* tail;
    U64* errors;
    Fault* fault;
};
static_assert(offsetof(GpuRange, counts) == 24);
static_assert(sizeof(Page) == 64 && offsetof(Page, status) == 16 && offsetof(Page, released_base) == 24);
static_assert(sizeof(Audit) == 64 && offsetof(Audit, claim) == 40);
static_assert(sizeof(Fault) == 32 && offsetof(Fault, address) == 8 && offsetof(Fault, page_index) == 16 &&
              offsetof(Fault, time_ns) == 24);
static_assert(sizeof(Root) == 80 && offsetof(Root, count) == 8 && offsetof(Root, audits) == 24 &&
              offsetof(Root, queue) == 48 && offsetof(Root, errors) == 64 && offsetof(Root, fault) == 72);
using Receipt = std::array<U64, 10>;
using Bind = cudaError_t (*)(void*);
const Bind bindings[] = {hbflab_q_auto_bind_convert, hbflab_q_auto_bind_mmvf,
                         hbflab_q_auto_bind_binbcast, hbflab_q_auto_bind_getrows};

void need(bool okay, const char* reason) { if (!okay) throw std::runtime_error(reason); }
void check(cudaError_t err) { if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); }
U32 acquire(U32& word) { return cuda::atomic_ref<U32, cuda::thread_scope_system>(word).load(cuda::memory_order_acquire); }
void release(U32& word, U32 value) { cuda::atomic_ref<U32, cuda::thread_scope_system>(word).store(value, cuda::memory_order_release); }
template<class T> void allocate(T*& ptr, std::size_t count) { check(cudaMalloc(&ptr, count * sizeof(T))); }
template<class T> void mapped(T*& host, T*& device, std::size_t count) {
    check(cudaHostAlloc(reinterpret_cast<void**>(&host), count * sizeof(T), cudaHostAllocMapped));
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0));
}
template<class T> void clear(T* ptr, std::size_t count) { check(cudaMemset(ptr, 0, count * sizeof(T))); }
template<class T> void download(T* dest, const T* device, std::size_t count) {
    check(cudaMemcpy(dest, device, count * sizeof(T), cudaMemcpyDeviceToHost));
}
template<class T> void write_values(std::ofstream& out, const T* words, std::size_t count) {
    out.write(reinterpret_cast<const char*>(words), std::streamsize(count * sizeof(T)));
}
std::ofstream output(const std::filesystem::path& dir, const char* stem, unsigned step, const char* extension) {
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(dir/(std::string(stem)+"-"+std::to_string(step)+extension), std::ios::binary);
    return out;
}
__global__ void initialize_audits(Audit* audits, std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < count) { Audit a{}; a.earliest_ns = ~U64{0}; audits[i] = a; }
}

struct Session {
    Root root{};
    Root* device_root = nullptr;
    Page* host_pages = nullptr;
    U32* host_queue = nullptr;
    Fault* host_fault = nullptr;
    std::vector<RangeSpec> specs;
    std::vector<GpuRange> ranges;
    std::vector<unsigned> owners;
    std::vector<Receipt> receipts;
    std::unique_ptr<hbflab::TimingBackend> backend;
    std::thread worker;
    std::atomic<bool> stop{false};
    bool worker_failed = false, bound = false, phase_failed = false, cleaned = false, fault_reported = false;
    unsigned steps = 0;
    std::size_t consumed = 0;
    U64 total_requests = 0;

    void init(const RangeSpec* input, std::size_t count) {
        need(input && (all_media ? count > 0 && count < UINT32_MAX : count == 5), "invalid registered weight count");
        specs.assign(input, input + count);
        root.count = count;
        for (const auto& s : specs) {
            const U64 base = reinterpret_cast<U64>(s.base);
            need(base && s.bytes && !(s.bytes % (all_media ? 2 : page_bytes)) && !(s.logical % page_bytes) &&
                 s.bytes <= UINT64_MAX-base && s.retain_words <= 1, "invalid registered range");
            if (all_media && !ranges.empty())
                need(s.logical > specs[ranges.size()-1].logical, "expected registration in logical order");
            for (const auto& r : ranges)
                need(base >= r.base+r.bytes || r.base >= base+s.bytes, "registered GPU allocations overlap");
            ranges.push_back({base, s.bytes, owners.size(), nullptr});
            const auto pages = page_count(s.bytes);
            need(pages < UINT32_MAX-owners.size(), "too many registered pages");
            owners.insert(owners.end(), pages, unsigned(ranges.size()-1));
            if (s.retain_words) allocate(ranges.back().counts, pages*halfs);
        }
        const auto pages = owners.size();
        need(pages && pages < UINT32_MAX, "invalid demand queue capacity");
        receipts.resize(pages);
        allocate(root.ranges, count);
#if defined(HBFLAB_ALL_MEDIA_BUILD)
        auto by_gpu_address = ranges;
        for (std::size_t i=0; i<count; ++i) by_gpu_address[i].ordinal = i;
        std::sort(by_gpu_address.begin(), by_gpu_address.end(), [](const auto& a,const auto& b){return a.base < b.base;});
        check(cudaMemcpy(root.ranges, by_gpu_address.data(), count*sizeof(GpuRange), cudaMemcpyHostToDevice));
#else
        check(cudaMemcpy(root.ranges, ranges.data(), count*sizeof(GpuRange), cudaMemcpyHostToDevice));
#endif
        mapped(host_pages, root.pages, pages);
        mapped(host_queue, root.queue, pages);
        mapped(host_fault, root.fault, 1);
        allocate(root.audits, pages);
        allocate(root.coverage, pages*mask_words);
        allocate(root.sites, count*sites_per_range);
        allocate(root.tail, 1);
        allocate(root.errors, 1);
        allocate(device_root, 1);
        check(cudaMemcpy(device_root, &root, sizeof(root), cudaMemcpyHostToDevice));
        bound = true;
        for (const auto bind : bindings) check(bind(device_root));
    }

    void init_backend() {
        if (backend) return;
        const auto start = std::chrono::steady_clock::now();
        std::fprintf(stderr, "five media init_begin ranges=%zu registered_pages=%zu\n", specs.size(), owners.size());
        std::fflush(stderr);
        hbflab::BackendOptions options;
        options.geometry = m6_geometry();
        options.resident_fragments = 256;
        options.direct_device_transactions = all_media;
        options.coalesce_capacity_feedback = all_media;
        options.max_events_per_batch = integrated_profile().max_events;
        backend = std::make_unique<hbflab::TimingBackend>(options);
        U64 next = 0;
        const U64 group = U64(options.geometry.block_group_dlus()) * page_bytes;
        auto sorted = specs;
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){ return a.logical < b.logical; });
        for (const auto& s : sorted) {
            need(s.logical >= next && s.logical % group == 0, "unaligned or overlapping medium range");
            if (s.logical > next) backend->reserve_gap(s.logical-next);
            const auto padded = (s.bytes + group-1) / group * group;
            need(backend->reserve(padded) == s.logical, "medium reservation address mismatch");
            next = s.logical + padded;
        }
        need(backend->drained(), "medium initialization incomplete");
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::fprintf(stderr, "five media init_end ranges=%zu registered_pages=%zu wall_s=%.3f\n", specs.size(), owners.size(), elapsed);
        std::fflush(stderr);
    }

    U64 address(std::size_t page, bool logical) const {
        const auto owner = owners.at(page);
        return (logical ? specs[owner].logical : ranges[owner].base) +
               (page - ranges[owner].first_page)*page_bytes;
    }

    void progress(const char* event, double elapsed, double last_batch_seconds) const {
        const auto next_slot = consumed < owners.size() ? acquire(host_queue[consumed]) : 0;
        std::fprintf(stderr, "five media %s step=%u wall_s=%.3f served=%zu queue_slot=%zu queue_page_plus_one=%u last_batch_s=%.6f\n",
                     event, steps, elapsed, consumed, consumed, next_slot, last_batch_seconds);
        std::fflush(stderr);
    }
    bool first_device_fault() {
        const auto code = cuda::atomic_ref<U64, cuda::thread_scope_system>(host_fault->code).load(cuda::memory_order_acquire);
        if (code && !fault_reported) {
            // The device release-publishes code only after writing its first fault payload.
            std::fprintf(stderr, "five media device_fault step=%u code=%llu address=0x%llx page_index=%llu gpu_time_ns=%llu served=%zu\n",
                         steps, static_cast<unsigned long long>(code), static_cast<unsigned long long>(host_fault->address),
                         static_cast<unsigned long long>(host_fault->page_index), static_cast<unsigned long long>(host_fault->time_ns), consumed);
            std::fflush(stderr);
            fault_reported = true;
        }
        return code != 0;
    }
    void service() noexcept {
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        auto next_progress = start + std::chrono::seconds(5);
        double last_batch_seconds = 0;
        progress("phase_begin", 0, 0);
        try {
            std::vector<std::size_t> indices;
            std::vector<hbflab::TimedRead> reads;
            indices.reserve(batch_limit); reads.reserve(batch_limit);
            while (!stop.load(std::memory_order_acquire)) {
                need(!first_device_fault(), "GPU reported a five-weight instrumentation fault");
                const auto now = Clock::now();
                if (now >= next_progress) {
                    progress("phase_progress", std::chrono::duration<double>(now-start).count(), last_batch_seconds);
                    next_progress = now + std::chrono::seconds(5);
                }
                indices.clear(); reads.clear();
                // Streaming admission uses an exclusive floor. All demands already in
                // this FIFO cohort arrive together at the first admissible model tick.
                const auto arrival = backend->now_ps() + (all_media ? 1 : 0);
                for (std::size_t slot = consumed; slot < owners.size() && indices.size() < batch_limit; ++slot) {
                    const auto index_plus_one = acquire(host_queue[slot]);
                    if (!index_plus_one) break;
                    const auto page = std::size_t(index_plus_one-1);
                    need(page < owners.size() && !receipts[page][9] &&
                         std::find(indices.begin(), indices.end(), page) == indices.end(), "duplicate or invalid demand queue entry");
                    need(acquire(host_pages[page].status) == requested && host_pages[page].request_ns,
                         "demand queue references unpublished request");
                    indices.push_back(page);
                    reads.push_back({arrival, {{address(page, true), page_bytes}}});
                }
                if (reads.empty()) { std::this_thread::yield(); continue; }
                const auto batch_start = Clock::now();
                auto publish = [&](std::size_t page, const hbflab::ReadResult& result) {
                    need(result.status == hbflab::Status::Ok && result.sectors == 64 && result.end_ps > result.begin_ps,
                         "EventLab page read failed");
                    const auto duration = result.end_ps-result.begin_ps;
                    const auto delay = duration/1000 + (duration%1000 != 0);
                    need(delay < 1000000000 && acquire(host_pages[page].status) == requested, "invalid page deadline or ownership");
                    const auto backing = address(page, false);
                    receipts[page] = {page, address(page, true), backing, host_pages[page].request_ns, result.begin_ps,
                                      result.end_ps, delay, result.sectors, ++consumed, ready};
                    ++total_requests;
                    host_pages[page].delay_ns = delay;
                    host_pages[page].released_base = backing;
                    release(host_pages[page].status, ready);
                };
                if (all_media) {
                    // One stream per cohort preserves the configured event budget per
                    // batch. Reap after each model yield so winners can resume early.
                    backend->begin_stream(batch_limit);
                    for (std::size_t i=0; i<reads.size(); ++i) backend->submit_read(i+1, reads[i]);
                    std::size_t completed = 0;
                    bool advanced;
                    do {
                        advanced = backend->advance_before(UINT64_MAX);
                        for (const auto& item : backend->take_completed()) {
                            need(item.id && item.id <= indices.size(), "unknown EventLab completion id");
                            publish(indices[item.id-1], item.result);
                            ++completed;
                        }
                    } while (advanced);
                    need(completed == indices.size(), "missing EventLab stream completion");
                    backend->end_stream();
                } else {
                    const auto completed = backend->read_batch(reads);
                    need(completed.size() == reads.size(), "missing EventLab page completion");
                    for (std::size_t i=0; i<indices.size(); ++i) publish(indices[i], completed[i]);
                }
                last_batch_seconds = std::chrono::duration<double>(Clock::now()-batch_start).count();
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "five media service: %s\n", e.what());
            worker_failed = true;
            // A later CTA may publish after the first notification. Wake all
            // blocked and newly scheduled demand producers until the host joins.
            do {
                for (std::size_t i=0; i<owners.size(); ++i) release(host_pages[i].status, failed);
                std::this_thread::yield();
            } while (!stop.load());
        }
        // A fast device trap may cause the owner to join before the next poll.
        if (first_device_fault()) worker_failed = true;
        progress(worker_failed ? "phase_failed" : "phase_end", std::chrono::duration<double>(Clock::now()-start).count(), last_batch_seconds);
    }
    void begin_step() {
        need(!phase_failed && !worker_failed && !worker.joinable(), "five media phase lifecycle");
        init_backend();
        const auto pages = owners.size();
        std::memset(host_pages, 0, pages*sizeof(Page));
        std::memset(host_queue, 0, pages*sizeof(U32));
        *host_fault = {};
        fault_reported = false;
        std::fill(receipts.begin(), receipts.end(), Receipt{});
        for (const auto& r : ranges) if (r.counts) clear(r.counts, page_count(r.bytes)*halfs);
        initialize_audits<<<unsigned((pages+255)/256),256>>>(root.audits, pages);
        check(cudaGetLastError());
        clear(root.coverage, pages*mask_words);
        clear(root.sites, root.count*sites_per_range);
        clear(root.tail, 1); clear(root.errors, 1);
        check(cudaDeviceSynchronize());
        consumed = 0; stop = false;
        worker = std::thread([this]{service();});
    }
    void join() {
        stop.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
    }

    void dump(const std::filesystem::path& dir, unsigned step) {
        U32 tail; U64 errors;
        download(&tail, root.tail, 1); download(&errors, root.errors, 1);
        need(errors == 0 && tail == consumed && consumed > 0 && backend->drained(), "GPU audit error or incomplete page service");
        const auto pages = owners.size();
        std::vector<Audit> audits(pages);
        std::vector<U32> masks(pages*mask_words), repeat(halfs);
        std::vector<U64> sites((all_media ? 1 : root.count)*sites_per_range);
        download(audits.data(), root.audits, pages);
        download(masks.data(), root.coverage, masks.size());
        if (!all_media) download(sites.data(), root.sites, sites.size());
        auto media_file = output(dir, "media", step, ".u64");
        auto audit_file = output(dir, "audit", step, ".u64");
        auto mask_file = output(dir, "coverage", step, ".u32");
        auto repeat_file = output(dir, "repeats", step, ".u32");
        auto site_file = output(dir, "sites", step, ".u64");
        std::size_t written = 0;
        for (std::size_t page=0; page<pages; ++page) {
            const auto& a = audits[page];
            const auto& r = receipts[page];
            const U32* mask = masks.data()+page*mask_words;
            if (!r[9]) {
                need(!a.ready_base && !a.loads && !a.bytes && !a.claim && acquire(host_pages[page].status)==idle &&
                     std::all_of(mask, mask+mask_words, [](U32 x){ return x == 0; }), "unserved GPU accesses");
                continue;
            }
            need(a.ready_base == r[2] && a.earliest_ns >= r[3]+r[6] && a.earliest_ns <= a.latest_ns &&
                 a.loads && a.bytes && acquire(host_pages[page].status) == ready, "page gate dependency failed");
            write_values(media_file, r.data(), r.size());
            write_values(audit_file, &a.ready_base, 5);
            write_values(mask_file, mask, mask_words);
            const auto& range = ranges[owners[page]];
            if (range.counts) {
                download(repeat.data(), range.counts+(page-range.first_page)*halfs, halfs);
                write_values(repeat_file, repeat.data(), repeat.size());
            }
            ++written;
        }
        need(written == consumed, "incomplete receipt file");
        if (all_media) {
            for (std::size_t owner=0; owner<ranges.size(); ++owner) {
                download(sites.data(), root.sites+owner*sites_per_range, sites.size());
                for (std::size_t site=0; site<sites.size(); ++site) {
                    if (!sites[site]) continue;
                    const std::array<U64,4> row{owner, site/32768, site%32768, sites[site]};
                    write_values(site_file, row.data(), row.size());
                }
            }
        } else {
            write_values(site_file, sites.data(), sites.size());
        }
        media_file.close(); audit_file.close(); mask_file.close(); repeat_file.close(); site_file.close();
    }
    void end_step(const wchar_t* path, unsigned step) {
        need(path && step == steps && worker.joinable(), "five media step order");
        const auto gpu = cudaDeviceSynchronize();
        join();
        if (gpu != cudaSuccess) {
            std::fprintf(stderr, "five media gpu_sync_failed step=%u cuda=%d reason=%s device_fault_recorded=%d\n",
                         step, static_cast<int>(gpu), cudaGetErrorString(gpu), int(fault_reported));
            std::fflush(stderr);
        }
        check(gpu);
        need(!worker_failed, "five media worker failed");
        dump(path,step);
        ++steps;
    }
    void finish() {
        need(!worker.joinable() && !phase_failed && !worker_failed && steps && backend, "five media unfinished");
        backend->finish();
        const auto stats = backend->statistics();
        need(backend->drained() && stats.input_closed && !stats.host.logical_failed &&
             stats.requests == total_requests && stats.sectors == total_requests*64 &&
             stats.registered_globals == stats.completed_globals && stats.peak_resident_fragments <= 256,
             "five media backend failed to drain or conserved resources exceeded");
    }
    cudaError_t cleanup() {
        if (cleaned) return cudaSuccess;
        cleaned = true;
        cudaError_t error = cudaSuccess;
        auto take = [&](cudaError_t e){if (error==cudaSuccess && e!=cudaSuccess) error=e;};
        if (worker.joinable()) {
            // A normal or failing kernel can still need its CPU responder.
            take(cudaDeviceSynchronize()); join();
        }
        if (bound) for (const auto bind : bindings) take(bind(nullptr));
        for (auto& r : ranges) if (r.counts) take(cudaFree(r.counts));
        for (void* p : std::array<void*,7>{device_root, root.ranges, root.audits, root.coverage, root.sites, root.tail, root.errors})
            if (p) take(cudaFree(p));
        if (host_pages) take(cudaFreeHost(host_pages));
        if (host_queue) take(cudaFreeHost(host_queue));
        if (host_fault) take(cudaFreeHost(host_fault));
        return error;
    }
    ~Session() { cleanup(); }
};
std::unique_ptr<Session> session;

template<class Work> int call(Work work) {
    try { work(); return cudaSuccess; }
    catch (const std::exception& e) {
        std::fprintf(stderr, "five media: %s\n", e.what());
        if (session) session->phase_failed = true;
        return cudaErrorUnknown;
    }
}
} // namespace

extern "C" int hbflab_five_media_begin(const RangeSpec* specs, std::size_t count) {
    if (session) return cudaErrorInvalidValue;
    return call([&]{auto s = std::make_unique<Session>(); s->init(specs,count); session=std::move(s);});
}
extern "C" int hbflab_five_media_step_begin() {
    if (!session) return cudaErrorInvalidValue;
    return call([&]{session->begin_step();});
}
extern "C" int hbflab_five_media_step_end(const wchar_t* out, unsigned step) {
    if (!session) return cudaErrorInvalidValue;
    return call([&]{session->end_step(out,step);});
}
extern "C" int hbflab_five_media_finish() {
    if (!session) return cudaErrorInvalidValue;
    const int status = call([]{session->finish();});
    const auto cleanup = session->cleanup();
    session.reset();
    return status == cudaSuccess ? cleanup : status;
}
