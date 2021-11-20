#pragma once

#include <stdafx.h>

namespace utils
{
    using Byte = uchar;

    bool check_and_print_error(cudaError_t error);

    size_t num_blocks(size_t image_size, size_t num_thread_per_block);

    template <typename T>
    T *cuda_allocate(size_t size_in_bytes)
    {
        T *buffer = nullptr;
        cudaMalloc(&buffer, size_in_bytes);
        cudaMemset(buffer, 0, size_in_bytes);
        return buffer;
    }

    struct CudaDeleter
    {
        CudaDeleter() = default;
        CudaDeleter(const CudaDeleter &) = default;
        CudaDeleter(CudaDeleter &&) = default;

        void operator()(void *p) const;
    };
}
