#pragma once
#include <cstddef>
#include <cstdint>

#if defined(_WIN32) && defined(HBFLAB_M2_BUILD)
#define HBFLAB_FIVE_MEDIA_API __declspec(dllexport)
#elif defined(_WIN32)
#define HBFLAB_FIVE_MEDIA_API __declspec(dllimport)
#else
#define HBFLAB_FIVE_MEDIA_API
#endif

namespace five_media {
struct RangeSpec {
    const void* base;
    std::uint64_t bytes, logical;
    std::uint32_t retain_words;
};
}

extern "C" {
HBFLAB_FIVE_MEDIA_API int hbflab_five_media_begin(const five_media::RangeSpec*, std::size_t);
HBFLAB_FIVE_MEDIA_API int hbflab_five_media_step_begin();
HBFLAB_FIVE_MEDIA_API int hbflab_five_media_step_end(const wchar_t* out_path, unsigned step);
HBFLAB_FIVE_MEDIA_API int hbflab_five_media_finish();
}
