#include "reduction.cuh"

// reduction
__global__ void reduction_0_kernel(long* output, long* input, int num_elements)
{
	extern __shared__ long shared_data[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) return;

	shared_data[tid] = input[index];

	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {

		if (tid % (2 * s) == 0) {
			shared_data[tid] += shared_data[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) output[blockIdx.x] = shared_data[0];

	//if (index == 0) {
	//	for (unsigned int s = 1; s < gridDim.x; s++) {
	//		output[0] += output[s];
	//	}
	//}
}

void reduction_0_gpu(std::vector<long>& output_gpu, std::vector<long>& input)
{
	printf("reduction_0_gpu() \n");
	//device-side data
	long *dev_i = 0;
	long *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_i, input.size() * sizeof(long)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_gpu.size() * sizeof(long)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_i, input.data(), input.size() * sizeof(long), cudaMemcpyHostToDevice));//dev_a=a;
	//CUDA_CHECK(cudaMemcpyAsync(dev_i, input.data(), input.size() * sizeof(long), cudaMemcpyHostToDevice));
	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = input.size();
	int block = 512;
	int grid = ((thread_cnt - 1) / block + 1);

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	reduction_0_kernel << <dimGrid, dimBlock, block * sizeof(long) >> > (dev_o, dev_i, thread_cnt);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());


	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gpu.data(), dev_o, output_gpu.size() * sizeof(long), cudaMemcpyDeviceToHost));
	//CUDA_CHECK(cudaMemcpyAsync(output_gpu.data(), dev_o, output_gpu.size() * sizeof(long), cudaMemcpyDeviceToHost));
	
	for (unsigned int s = 1; s < grid; s++) {
		output_gpu[0] += output_gpu[s];
	}

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 출력
	printf("dur_time(gpu) = %6.3f [msec] (with data transfer time)\n", (start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) = %6.3f [msec] (without data transfer time)\n", (start_time3 - start_time2) / 1000.f);

	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_i));
}

// reduction 1
__global__ void reduction_1_kernel(long* output, long* input, int num_elements)
{
	extern __shared__ long shared_data[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) return;
	shared_data[tid] = input[index];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		unsigned int index_s = 2 * s * tid;
		if (index < blockDim.x) {
			shared_data[index_s] += shared_data[index_s + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) output[blockIdx.x] = shared_data[0];
}

void reduction_1_gpu(std::vector<long>& output_gpu, std::vector<long>& input)
{
	printf("reduction_1_kernel() \n");
	//device-side data
	long *dev_i = 0;
	long *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_i, input.size() * sizeof(long)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_gpu.size() * sizeof(long)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_i, input.data(), input.size() * sizeof(long), cudaMemcpyHostToDevice));//dev_a=a;
	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = input.size();
	int block = 512;
	int grid = ((thread_cnt - 1) / block + 1);

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	reduction_1_kernel << <dimGrid, dimBlock, block * sizeof(long) >> > (dev_o, dev_i, thread_cnt);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gpu.data(), dev_o, output_gpu.size() * sizeof(long), cudaMemcpyDeviceToHost));

	for (unsigned int s = 1; s < grid; s++) {
		output_gpu[0] += output_gpu[s];
	}

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 출력
	printf("dur_time(gpu) = %6.3f [msec] (with data transfer time)\n", (start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) = %6.3f [msec] (without data transfer time)\n", (start_time3 - start_time2) / 1000.f);

	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_i));
}