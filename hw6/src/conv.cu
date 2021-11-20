#include <conv.h>
#include <utils.h>

void conv::conv2d(const float *input_image, std::vector<float> &output_image, size_t width, size_t height, size_t num_c, const std::vector<float> &kernel_buffer, float normalization, size_t kernel_size)
{
    const int THREAD_PER_BLOCK = 16;

    int num_width_blocks = utils::num_blocks(width, THREAD_PER_BLOCK);
    int num_height_blocks = utils::num_blocks(height, THREAD_PER_BLOCK);

    size_t num_bytes = width * height * num_c * sizeof(float);

    std::unique_ptr<float[], utils::CudaDeleter> device_image_buffer(utils::cuda_allocate<float>(num_bytes));
    std::unique_ptr<float[], utils::CudaDeleter> device_image_out_buffer(utils::cuda_allocate<float>(num_bytes));
    std::unique_ptr<float[], utils::CudaDeleter> device_kernel_buffer(utils::cuda_allocate<float>(num_bytes));

    auto error = cudaMemcpy(device_image_buffer.get(), input_image, num_bytes, cudaMemcpyHostToDevice);

    utils::check_and_print_error(error);

    error = cudaMemcpy(device_kernel_buffer.get(), kernel_buffer.data(), kernel_buffer.size() * sizeof(float), cudaMemcpyHostToDevice);

    utils::check_and_print_error(error);

    gpu::conv2d_kernel<<<dim3(num_width_blocks, num_height_blocks), dim3(THREAD_PER_BLOCK, THREAD_PER_BLOCK, num_c)>>>(device_image_buffer.get(), device_image_out_buffer.get(), width, height, num_c, device_kernel_buffer.get(), normalization, kernel_size);

    error = cudaPeekAtLastError();

    utils::check_and_print_error(error);

    error = cudaMemcpy(output_image.data(), device_image_out_buffer.get(), num_bytes, cudaMemcpyDeviceToHost);

    utils::check_and_print_error(error);
}
