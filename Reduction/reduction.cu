#include "reduction.cuh"

const int TILE_DIM = 32;
const int BLOCK_ROWS = 16;

// reduction
__global__ void reduction_0_kernel(float* output, float* input, int size, int num_elements)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) return;

}

void reduction_0_gpu(std::vector<float>& output_gpu, std::vector<float>& input)
{
	printf("transpose_naive_gpu() \n");
	//device-side data
	float *dev_i = 0;
	float *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_i, input.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_gpu.size() * sizeof(float)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_i, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_a=a;

	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = input.size();
	int block = 256;
	int grid = (thread_cnt - 1) / block + 1;

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	reduction_0_kernel << <dimGrid, dimBlock >> > (dev_o, dev_i, input.size(), thread_cnt);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gpu.data(), dev_o, output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 출력
	printf("dur_time(gpu) = %6.3f [msec] (with data transfer time)\n", (start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) = %6.3f [msec] (without data transfer time)\n", (start_time3 - start_time2) / 1000.f);

	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_i));
}
