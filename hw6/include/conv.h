#pragma once

#include <stdafx.h>
#include <conv_kernel.h>

namespace conv
{
    void conv2d(const float * input_image, std::vector<float> & output_image, size_t width, size_t height, size_t num_c, const std::vector<float> & kernel_buffer, float normalization, size_t kernel_size);
}
