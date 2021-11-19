#include <utils.h>

bool utils::check_and_print_error(cudaError_t error)
{
    if (error != cudaError::cudaSuccess)
    {
        std::cerr << cudaGetErrorString(error) << std::endl;
        return true;
    }

    return false;
}

size_t utils::num_blocks(size_t image_size, size_t num_thread_per_block)
{
    return (image_size + num_thread_per_block - 1) / num_thread_per_block;
}

void utils::CudaDeleter::operator()(void * p) const
{
    cudaFree(p);
}
