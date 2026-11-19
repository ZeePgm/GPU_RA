#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32) && defined(HBFLAB_M2_BUILD)
#define HBFLAB_Q_DEVICE_API __declspec(dllexport)
#elif defined(_WIN32)
#define HBFLAB_Q_DEVICE_API __declspec(dllimport)
#else
#define HBFLAB_Q_DEVICE_API
#endif

namespace q_auto {
constexpr std::size_t summary_words = 32;
constexpr std::size_t sites_per_translation_unit = 16384;
constexpr std::size_t site_count = 2 * sites_per_translation_unit;
enum SummaryIndex : std::size_t {
    convert_loads = 0, convert_bytes, decode_loads, decode_bytes, partial_or_misaligned
};
struct MediaReceipt {
    std::uint64_t logical, request_ns, begin_ps, end_ps, delay_ns, sectors, completion_order, status;
    std::uint64_t released_base;
};
static_assert(sizeof(MediaReceipt) == 9 * sizeof(std::uint64_t));
}

extern "C" {
HBFLAB_Q_DEVICE_API int hbflab_q_auto_device_begin(const void *base, std::size_t bytes,
                                                   std::uint64_t logical_base, int timed);
HBFLAB_Q_DEVICE_API int hbflab_q_auto_device_step_begin();
HBFLAB_Q_DEVICE_API int hbflab_q_auto_device_step_end(
    std::uint32_t *counts, std::size_t count, std::uint64_t *summary, std::size_t summary_words);
HBFLAB_Q_DEVICE_API int hbflab_q_auto_device_sites(std::uint64_t *counts, std::size_t count);
HBFLAB_Q_DEVICE_API int hbflab_q_auto_device_timing(std::uint64_t *observed, std::uint64_t *resumed, std::size_t words,
                                                     q_auto::MediaReceipt *pages, std::size_t count);
HBFLAB_Q_DEVICE_API int hbflab_q_auto_device_finish();
}
