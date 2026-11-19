#pragma once

#include "instrumented_bridge.hpp"
#include <cstdint>

constexpr unsigned five_media_input_tile_bytes = 512 * 1024;
constexpr unsigned five_media_output_tile_rows = five_media_input_tile_bytes / (1024 * sizeof(std::uint16_t));
extern "C" HBFLAB_M2_API std::uint64_t hbflab_five_projection_tiles();
