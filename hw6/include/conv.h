#pragma once

#include <cuda.h>
#include <conv_kernel.h>
#include <stdafx.h>

namespace conv
{
    void conv2d(const float * input_image, std::vector<float> & output_image, size_t width, size_t height, size_t num_c, const std::vector<float> & kernel_buffer, float normalization, size_t kernel_size);

    size_t num_blocks(size_t image_size, size_t num_thread_per_block);

    float * cuda_allocate(size_t size_in_bytes);

    struct CudaDeleter {
        CudaDeleter() = default;
        CudaDeleter(const CudaDeleter &) = default;
        CudaDeleter(CudaDeleter &&) = default;

        void operator()(void * p) const;
    };
}
