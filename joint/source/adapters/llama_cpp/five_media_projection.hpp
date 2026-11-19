#pragma once

#include "instrumented_bridge.hpp"
#include <cstdint>

constexpr unsigned five_media_output_tile_rows = 512;
extern "C" HBFLAB_M2_API std::uint64_t hbflab_five_projection_tiles();
