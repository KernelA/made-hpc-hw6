#pragma once

#include <stdafx.h>
#include <utils.h>

namespace gpu
{
    __device__ void insert_sort(utils::Byte * arary, size_t size);

    __global__ void median_filter(const utils::Byte * input_image, utils::Byte * output_image, size_t image_width, size_t image_height, size_t num_c, size_t window_size);
}
