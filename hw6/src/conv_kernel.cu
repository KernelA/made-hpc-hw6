#include <conv_kernel.h>

__global__ void gpu::conv2d_kernel(const float *input_image, float *output_image, size_t image_width, size_t image_height, size_t num_c, const float *kernel, float normalization, size_t kernel_size)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int column = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = threadIdx.z;

	if (column < image_width && row < image_height)
	{
		float result = 0.0f;

		int half_kernel_size = kernel_size / 2;

		for (int kernelRow = 0; kernelRow < kernel_size; kernelRow++)
		{
			int imageRowIdx = row - half_kernel_size + kernelRow;

			for (int kernelColumn = 0; kernelColumn < kernel_size; kernelColumn++)
			{
				int imageColumnIdx = column - half_kernel_size + kernelColumn;

				if (imageRowIdx >= 0 && imageRowIdx < image_height && imageColumnIdx >= 0 && imageColumnIdx < image_width)
				{
					int index = imageColumnIdx * num_c + imageRowIdx * image_width * num_c + channel;
					result += kernel[kernelColumn + kernelRow * kernel_size] * input_image[index];
				}
			}
		}

		result /= normalization;

		int index = column * num_c + row * image_width * num_c + channel;

		output_image[index] = fminf(fmaxf(result, 0.0f), 1.0f);
	}
}