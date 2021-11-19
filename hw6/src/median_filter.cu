#include <median_filter.h>

void median::median_filter(const utils::Byte * input_image, std::vector<utils::Byte> & output_image, size_t width, size_t height, size_t num_c, size_t window_size)
{
    using utils::Byte;

    const int THREAD_PER_BLOCK = 8;

    int num_width_blocks = utils::num_blocks(width, THREAD_PER_BLOCK);
    int num_height_blocks = utils::num_blocks(height, THREAD_PER_BLOCK);

    size_t num_bytes = width * height * num_c * sizeof(Byte);

    std::unique_ptr<Byte[], utils::CudaDeleter> device_image_buffer(utils::cuda_allocate<Byte>(num_bytes));
    std::unique_ptr<Byte[], utils::CudaDeleter> device_image_out_buffer(utils::cuda_allocate<Byte>(num_bytes));

    auto error = cudaMemcpy(device_image_buffer.get(), input_image, num_bytes, cudaMemcpyHostToDevice);

    utils::check_and_print_error(error);


    utils::check_and_print_error(error);

    gpu::median_filter<< <dim3(num_width_blocks, num_height_blocks), dim3(THREAD_PER_BLOCK, THREAD_PER_BLOCK, num_c), THREAD_PER_BLOCK * THREAD_PER_BLOCK * sizeof(Byte) * window_size * window_size * num_c>> > (device_image_buffer.get(), device_image_out_buffer.get(), width, height, num_c, window_size);

    error = cudaPeekAtLastError();

    utils::check_and_print_error(error);

    error = cudaMemcpy(output_image.data(), device_image_out_buffer.get(), num_bytes, cudaMemcpyDeviceToHost);

    utils::check_and_print_error(error);
}
