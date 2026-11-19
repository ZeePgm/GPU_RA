#include "q_auto_device_api.hpp"
#include "integrated_geometry.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

extern "C" cudaError_t hbflab_q_auto_bind_convert(void*);
extern "C" cudaError_t hbflab_q_auto_bind_mmvf(void*);

namespace {

constexpr std::size_t summary_words = 32;
constexpr std::size_t site_count = 32768;
constexpr std::size_t page_bytes = 4096;
constexpr std::size_t expected_pages = 1024;
enum PageStatus : std::uint32_t { idle = 0, requested = 1, ready = 2, failed = 3 };

struct alignas(64) Page {
    std::uint64_t request_ns, delay_ns;
    std::uint32_t status;
    std::uint64_t released_base;
};
static_assert(sizeof(Page) == 64 && offsetof(Page, request_ns) == 0 &&
              offsetof(Page, delay_ns) == 8 && offsetof(Page, status) == 16 &&
              offsetof(Page, released_base) == 24);

std::uint32_t status(Page &page) {
    return cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>(page.status).load(cuda::memory_order_acquire);
}
void publish(Page &page, PageStatus next) {
    cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>(page.status).store(next, cuda::memory_order_release);
}

struct ProbeState {
    const void* base;
    std::uint64_t bytes;
    std::uint32_t* counts;
    std::uint64_t* summary;
    std::uint64_t* sites;
    std::uint32_t* claims;
    Page* pages;
    std::uint64_t* observed;
    std::uint64_t* resumed;
    std::uint64_t* ready_bases;
};
static_assert(offsetof(ProbeState, bytes) == 8);
static_assert(offsetof(ProbeState, counts) == 16);
static_assert(offsetof(ProbeState, summary) == 24);
static_assert(offsetof(ProbeState, sites) == 32);
static_assert(offsetof(ProbeState, claims) == 40 && offsetof(ProbeState, pages) == 48 &&
              offsetof(ProbeState, observed) == 56 && offsetof(ProbeState, resumed) == 64 &&
              offsetof(ProbeState, ready_bases) == 72);

ProbeState state{};
ProbeState* device_state = nullptr;
Page* host_pages = nullptr;

// A phase has a cold 4 KiB demand cache. The same medium remains initialized
// across phases. Only the worker touches the backend while a phase is active.
struct MediaService {
    std::uint64_t logical_base = 0;
    std::unique_ptr<hbflab::TimingBackend> backend;
    std::vector<q_auto::MediaReceipt> receipts;
    std::thread worker;
    std::atomic<bool> stop{false}, abort{false};
    bool fault = false, completed_phase = false;
    std::uint64_t serviced = 0;

    void initialize() {
        if (backend) return;
        hbflab::BackendOptions options;
        options.geometry = m6_geometry();
        options.resident_fragments = 256;
        options.max_events_per_batch = integrated_profile().max_events;
        backend = std::make_unique<hbflab::TimingBackend>(options);
        if (logical_base) backend->reserve_gap(logical_base);
        if (backend->reserve(state.bytes) != logical_base || !backend->drained())
            throw std::runtime_error("Q media initialization or logical base mismatch");
    }
    void poison_pages() {
        for (std::size_t i = 0; i < expected_pages; ++i) {
            if (status(host_pages[i]) != failed) publish(host_pages[i], failed);
        }
    }
    void serve() noexcept {
        try {
            std::vector<std::pair<std::uint64_t, std::size_t>> pending;
            pending.reserve(expected_pages);
            std::size_t completed = 0;
            while (!stop.load() && completed != expected_pages) {
                if (abort.load()) throw std::runtime_error("Q media phase cancelled");
                pending.clear();
                for (std::size_t i = 0; i < expected_pages; ++i) {
                    const auto observed_status = status(host_pages[i]);
                    if (observed_status == requested) pending.emplace_back(host_pages[i].request_ns, i);
                    else if (observed_status != idle && observed_status != ready)
                        throw std::runtime_error("Q device request failed");
                }
                if (pending.empty()) { std::this_thread::yield(); continue; }
                // One backend read at a time. Order concurrently visible requests
                // by GPU demand timestamp (with the page index as a stable tie).
                std::sort(pending.begin(), pending.end());
                for (const auto &entry : pending) {
                    if (stop.load() || abort.load()) throw std::runtime_error("Q media interrupted");
                    const auto i = entry.second;
                    const auto logical = logical_base + i * page_bytes;
                    if (!entry.first || receipts[i].status) throw std::runtime_error("Q duplicate or untimed page request");
                    const auto r = backend->read(logical, page_bytes);
                    if (r.status != hbflab::Status::Ok || r.sectors != page_bytes / 64 || r.end_ps < r.begin_ps)
                        throw std::runtime_error("Q medium returned an invalid page completion");
                    const auto duration = r.end_ps - r.begin_ps;
                    const auto delay = duration / 1000 + (duration % 1000 != 0);
                    if (!delay || delay >= 1000000000ull || status(host_pages[i]) != requested)
                        throw std::runtime_error("Q page completion expired or lost ownership");
                    const auto backing = reinterpret_cast<std::uintptr_t>(state.base) + i * page_bytes;
                    receipts[i] = {logical, entry.first, r.begin_ps, r.end_ps, delay, r.sectors, ++completed, ready, backing};
                    ++serviced;
                    host_pages[i].delay_ns = delay;
                    host_pages[i].released_base = backing;
                    publish(host_pages[i], ready);
                }
            }
            if (completed != expected_pages || !backend->drained())
                throw std::runtime_error("Q media phase lacks a complete drained set of pages");
            completed_phase = true;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "Q media service: %s\n", e.what());
            fault = true;
        } catch (...) { fault = true; }
        // An as-yet unscheduled warp can still publish its first request after
        // an error. Keep waking these waiters until all device work has stopped.
        if (fault) do { poison_pages(); std::this_thread::yield(); } while (!stop.load());
    }
    void begin_step() {
        if (worker.joinable() || fault) throw std::runtime_error("Q media phase already active or failed");
        initialize();
        receipts.assign(expected_pages, {});
        std::memset(host_pages, 0, expected_pages * sizeof(Page));
        completed_phase = false;
        stop = false; abort = false;
        worker = std::thread([this] { serve(); });
    }
    void join() {
        stop = true;
        if (worker.joinable()) worker.join();
    }
    bool finish() {
        if (!backend || fault || !completed_phase) return false;
        backend->finish();
        const auto stats = backend->statistics();
        return backend->drained() && stats.input_closed && stats.requests == serviced &&
            stats.sectors == serviced * (page_bytes / 64) &&
            stats.registered_globals == stats.completed_globals && !stats.host.logical_failed &&
            stats.peak_resident_fragments <= 256;
    }
};
std::unique_ptr<MediaService> media;

cudaError_t destroy() {
    const auto convert = hbflab_q_auto_bind_convert(nullptr);
    const auto mmvf = hbflab_q_auto_bind_mmvf(nullptr);
    const auto device = cudaFree(device_state);
    const auto counts = cudaFree(state.counts);
    const auto summary = cudaFree(state.summary);
    const auto sites = cudaFree(state.sites);
    const auto claims = cudaFree(state.claims);
    const auto observed = cudaFree(state.observed);
    const auto resumed = cudaFree(state.resumed);
    const auto ready_bases = cudaFree(state.ready_bases);
    const auto pages = host_pages ? cudaFreeHost(host_pages) : cudaSuccess;
    state = {};
    device_state = nullptr;
    host_pages = nullptr;
    media.reset();
    for (auto result : {convert, mmvf, device, counts, summary, sites, claims, observed, resumed, ready_bases, pages}) {
        if (result != cudaSuccess) return result;
    }
    return cudaSuccess;
}

}

extern "C" int hbflab_q_auto_device_begin(const void* base, std::size_t bytes, std::uint64_t logical_base, int timed) {
    if (device_state || !base || bytes < 32 || bytes % 2 || (timed != 0 && timed != 1) ||
        (timed && (bytes != expected_pages * page_bytes || logical_base % page_bytes))) return cudaErrorInvalidValue;
    state.base = base;
    state.bytes = bytes;
    auto result = cudaMalloc(&state.counts, bytes / 2 * sizeof(*state.counts));
    if (result == cudaSuccess) result = cudaMalloc(&state.summary, summary_words * sizeof(*state.summary));
    if (result == cudaSuccess) result = cudaMalloc(&state.sites, site_count * sizeof(*state.sites));
    if (timed) {
        try { media = std::make_unique<MediaService>(); media->logical_base = logical_base; }
        catch (...) { result = cudaErrorMemoryAllocation; }
        if (result == cudaSuccess) result = cudaMalloc(&state.claims, expected_pages * sizeof(*state.claims));
        if (result == cudaSuccess) result = cudaMalloc(&state.observed, bytes / 2 * sizeof(*state.observed));
        if (result == cudaSuccess) result = cudaMalloc(&state.resumed, bytes / 2 * sizeof(*state.resumed));
        if (result == cudaSuccess) result = cudaMalloc(&state.ready_bases, expected_pages * sizeof(*state.ready_bases));
        if (result == cudaSuccess) result = cudaHostAlloc(reinterpret_cast<void **>(&host_pages),
                                                          expected_pages * sizeof(Page), cudaHostAllocMapped);
        if (result == cudaSuccess) result = cudaHostGetDevicePointer(reinterpret_cast<void **>(&state.pages), host_pages, 0);
    }
    if (result == cudaSuccess) result = cudaMalloc(&device_state, sizeof(state));
    if (result == cudaSuccess) result = cudaMemcpy(device_state, &state, sizeof(state), cudaMemcpyHostToDevice);
    if (result == cudaSuccess) result = hbflab_q_auto_bind_convert(device_state);
    if (result == cudaSuccess) result = hbflab_q_auto_bind_mmvf(device_state);
    if (result != cudaSuccess) destroy();
    return result;
}

extern "C" int hbflab_q_auto_device_step_begin() {
    if (!device_state || (media && (media->worker.joinable() || media->fault))) return cudaErrorInvalidValue;
    auto result = cudaMemset(state.counts, 0, state.bytes / 2 * sizeof(*state.counts));
    if (result == cudaSuccess) result = cudaMemset(state.summary, 0, summary_words * sizeof(*state.summary));
    if (result == cudaSuccess) result = cudaMemset(state.sites, 0, site_count * sizeof(*state.sites));
    if (result == cudaSuccess && media) result = cudaMemset(state.claims, 0, expected_pages * sizeof(*state.claims));
    if (result == cudaSuccess && media) result = cudaMemset(state.observed, 0, state.bytes / 2 * sizeof(*state.observed));
    if (result == cudaSuccess && media) result = cudaMemset(state.resumed, 0, state.bytes / 2 * sizeof(*state.resumed));
    if (result == cudaSuccess && media) result = cudaMemset(state.ready_bases, 0, expected_pages * sizeof(*state.ready_bases));
    if (result == cudaSuccess) result = cudaDeviceSynchronize();
    if (result == cudaSuccess && media) {
        try { media->begin_step(); } catch (const std::exception &e) {
            std::fprintf(stderr, "Q media setup: %s\n", e.what()); media->fault = true; return cudaErrorUnknown;
        }
    }
    return result;
}

extern "C" int hbflab_q_auto_device_step_end(std::uint32_t* counts, std::size_t count,
                                              std::uint64_t* summary, std::size_t words) {
    if (!device_state || !counts || !summary || count != state.bytes / 2 || words != summary_words) {
        return cudaErrorInvalidValue;
    }
    auto result = cudaDeviceSynchronize();
    if (media) {
        if (!media->worker.joinable()) return cudaErrorInvalidValue;
        media->join();
        if (result == cudaSuccess && (media->fault || !media->completed_phase)) result = cudaErrorUnknown;
    }
    if (result == cudaSuccess) result = cudaMemcpy(counts, state.counts, count * sizeof(*counts), cudaMemcpyDeviceToHost);
    if (result == cudaSuccess) result = cudaMemcpy(summary, state.summary, words * sizeof(*summary), cudaMemcpyDeviceToHost);
    return result;
}

extern "C" int hbflab_q_auto_device_sites(std::uint64_t* counts, std::size_t count) {
    if (!device_state || !counts || count != site_count) return cudaErrorInvalidValue;
    return cudaMemcpy(counts, state.sites, count * sizeof(*counts), cudaMemcpyDeviceToHost);
}

extern "C" int hbflab_q_auto_device_timing(std::uint64_t* observed, std::uint64_t* resumed, std::size_t words,
                                            q_auto::MediaReceipt* pages, std::size_t count) {
    if (!media || media->worker.joinable() || media->fault || !media->completed_phase ||
        !observed || !resumed || !pages || words != state.bytes / 2 || count != expected_pages)
        return cudaErrorInvalidValue;
    auto result = cudaMemcpy(observed, state.observed, words * sizeof(*observed), cudaMemcpyDeviceToHost);
    if (result == cudaSuccess) result = cudaMemcpy(resumed, state.resumed, words * sizeof(*resumed), cudaMemcpyDeviceToHost);
    if (result == cudaSuccess) std::copy(media->receipts.begin(), media->receipts.end(), pages);
    return result;
}

extern "C" int hbflab_q_auto_device_finish() {
    if (!device_state) return cudaErrorInvalidValue;
    if (media && media->worker.joinable()) media->abort = true;
    auto result = cudaDeviceSynchronize();
    if (media) {
        media->join();
        try {
            if (!media->finish() && result == cudaSuccess) result = cudaErrorUnknown;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "Q media finish: %s\n", e.what()); if (result == cudaSuccess) result = cudaErrorUnknown;
        }
    }
    const auto cleanup = destroy();
    return result == cudaSuccess ? cleanup : result;
}
