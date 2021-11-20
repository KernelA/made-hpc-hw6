#include <hist.h>
#include <utils.h>

void hist::hist(const utils::Byte *input_image, std::vector<gpu::HistType> &hist, size_t width, size_t height)
{
    using gpu::HistType;
    using utils::Byte;

    static_assert(gpu::LOCAL_HIST_SIZE == std::numeric_limits<Byte>::max() - std::numeric_limits<Byte>::min() + 1);

    const int THREAD_PER_BLOCK = 32;

    int num_width_blocks = utils::num_blocks(width, THREAD_PER_BLOCK);
    int num_height_blocks = utils::num_blocks(height, THREAD_PER_BLOCK);

    const size_t image_num_bytes = sizeof(Byte) * width * height;

    std::unique_ptr<Byte[], utils::CudaDeleter> device_image_buffer(utils::cuda_allocate<Byte>(image_num_bytes));

    const size_t hist_num_bytes = sizeof(HistType) * gpu::LOCAL_HIST_SIZE;

    std::unique_ptr<HistType[], utils::CudaDeleter> device_hist(utils::cuda_allocate<HistType>(hist_num_bytes));

    auto error = cudaMemcpy(device_image_buffer.get(), input_image, image_num_bytes, cudaMemcpyHostToDevice);

    utils::check_and_print_error(error);

    gpu::hist_kernel<<<dim3(num_width_blocks, num_height_blocks), dim3(THREAD_PER_BLOCK, THREAD_PER_BLOCK)>>>(device_image_buffer.get(), device_hist.get(), width, height);

    error = cudaPeekAtLastError();

    utils::check_and_print_error(error);

    error = cudaMemcpy(hist.data(), device_hist.get(), hist_num_bytes, cudaMemcpyDeviceToHost);

    utils::check_and_print_error(error);
}
