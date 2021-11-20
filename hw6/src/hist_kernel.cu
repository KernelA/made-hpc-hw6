#include <hist_kernel.h>
#include <stdio.h>

__global__ void gpu::hist_kernel(const utils::Byte *input_image, gpu::HistType *hist, size_t image_width, size_t image_height)
{
	__shared__ gpu::HistType local_hist[LOCAL_HIST_SIZE];

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int column = threadIdx.x + blockIdx.x * blockDim.x;

	int localIndex = blockDim.x * threadIdx.y + threadIdx.x;

	if (localIndex < LOCAL_HIST_SIZE)
	{
		local_hist[localIndex] = 0;
	}

	__syncthreads();

	if (column < image_width && row < image_height)
	{
		atomicAdd(&local_hist[input_image[row * image_width + column]], 1);
	}

	__syncthreads();

	if (localIndex == 0)
	{
		for (size_t i{}; i < LOCAL_HIST_SIZE; i++)
		{
			atomicAdd_block(&hist[i], local_hist[i]);
		}
	}
}