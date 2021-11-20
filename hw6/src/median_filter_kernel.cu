#include <median_filter_kernel.h>

__device__ void gpu::insert_sort(utils::Byte *array_buffer, size_t size)
{
	for (size_t i = 1; i < size; ++i)
	{
		size_t j = i;

		while (j > 0 && array_buffer[j] < array_buffer[j - 1])
		{
			utils::Byte temp = array_buffer[j];
			array_buffer[j] = array_buffer[j - 1];
			array_buffer[j - 1] = temp;
			j--;
		}
	}
}

__global__ void gpu::median_filter(const utils::Byte *input_image, utils::Byte *output_image, size_t image_width, size_t image_height, size_t num_c, size_t window_size)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int column = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = threadIdx.z;

	int half_window_size = window_size / 2;

	extern __shared__ utils::Byte window[];

	const size_t WINDOW_ARRAY_SIZE = window_size * window_size;

	const size_t start_index = (blockDim.x * threadIdx.y + threadIdx.x + channel * blockDim.x * blockDim.y) * WINDOW_ARRAY_SIZE;

	if (column < image_width && row < image_height)
	{
		int index = column * num_c + row * image_width * num_c + channel;

		if (column < image_width - half_window_size && column >= half_window_size && row >= half_window_size && row < image_height - half_window_size)
		{

			for (int kernelRow = 0; kernelRow < window_size; kernelRow++)
			{
				int imageRowIdx = row - half_window_size + kernelRow;

				for (int kernelColumn = 0; kernelColumn < window_size; kernelColumn++)
				{
					int imageColumnIdx = column - half_window_size + kernelColumn;

					if (imageRowIdx >= 0 && imageRowIdx < image_height && imageColumnIdx >= 0 && imageColumnIdx < image_width)
					{
						int index = imageColumnIdx * num_c + imageRowIdx * image_width * num_c + channel;
						window[kernelColumn + kernelRow * window_size + start_index] = input_image[index];
					}
				}
			}

			insert_sort(&window[start_index], WINDOW_ARRAY_SIZE);
			output_image[index] = window[start_index + WINDOW_ARRAY_SIZE / 2];
		}
		else
		{
			output_image[index] = input_image[index];
		}
	}
}