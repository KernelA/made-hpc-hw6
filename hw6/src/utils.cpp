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
