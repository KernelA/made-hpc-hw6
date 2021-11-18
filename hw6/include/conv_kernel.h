#pragma once

#include <stdio.h>

namespace gpu
{
    __global__ void conv2d_kernel(const float * input_image, float * output_image, size_t image_width, size_t image_height, size_t num_c, const float * kernel, float normalization, size_t kernel_size);
}
