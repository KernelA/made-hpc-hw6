#pragma once

#include <stdafx.h>
#include <utils.h>
#include <median_filter_kernel.h>

namespace median
{
    void median_filter(const utils::Byte * input_image, std::vector<utils::Byte> & output_image, size_t width, size_t height, size_t num_c, size_t window_size);
}
