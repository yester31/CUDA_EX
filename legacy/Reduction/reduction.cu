#include "reduction.cuh"

// reduction
__global__ void reduction_0_kernel(int* output, int* input, int num_elements)
{
	extern __shared__ int shared_data[];
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
}

void reduction_0_gpu(std::vector<int>& output_gpu, std::vector<int>& input)
{
	printf("reduction_0_gpu() \n");
	//device-side data
	int *dev_i = 0;
	int *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_i, input.size() * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_gpu.size() * sizeof(int)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_i, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));//dev_a=a;
	//CUDA_CHECK(cudaMemcpyAsync(dev_i, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));
	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = input.size();
	int block = 512;
	int grid = ((thread_cnt - 1) / block + 1);

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	reduction_0_kernel << <dimGrid, dimBlock, block * sizeof(int) >> > (dev_o, dev_i, thread_cnt);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());


	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gpu.data(), dev_o, output_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));
	//CUDA_CHECK(cudaMemcpyAsync(output_gpu.data(), dev_o, output_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));
	
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
__global__ void reduction_1_kernel(int* output, int* input, int num_elements)
{
	extern __shared__ int shared_data[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) return;
	shared_data[tid] = input[index];

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		unsigned int index_s = 2 * s * tid;
		if (index_s < blockDim.x) {
			shared_data[index_s] += shared_data[index_s + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) output[blockIdx.x] = shared_data[0];
}

void reduction_1_gpu(std::vector<int>& output_gpu, std::vector<int>& input)
{
	printf("reduction_1_kernel() \n");
	//device-side data
	int *dev_i = 0;
	int *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_i, input.size() * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_gpu.size() * sizeof(int)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_i, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));//dev_a=a;
	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = input.size();
	int block = 512;
	int grid = ((thread_cnt - 1) / block + 1);

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	reduction_1_kernel << <dimGrid, dimBlock, block * sizeof(int) >> > (dev_o, dev_i, thread_cnt);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gpu.data(), dev_o, output_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));

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


// reduction 2
__global__ void reduction_2_kernel(int* output, int* input, int num_elements)
{
	extern __shared__ int shared_data[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) return;
	shared_data[tid] = input[index];

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			shared_data[tid] += shared_data[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) output[blockIdx.x] = shared_data[0];
}

void reduction_2_gpu(std::vector<int>& output_gpu, std::vector<int>& input)
{
	printf("reduction_2_kernel() \n");
	//device-side data
	int *dev_i = 0;
	int *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_i, input.size() * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_gpu.size() * sizeof(int)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_i, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));//dev_a=a;
	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = input.size();
	int block = 512;
	int grid = ((thread_cnt - 1) / block + 1);

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	reduction_2_kernel << <dimGrid, dimBlock, block * sizeof(int) >> > (dev_o, dev_i, thread_cnt);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gpu.data(), dev_o, output_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));

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

// reduction 3
__global__ void reduction_3_kernel(int* output, int* input, int num_elements)
{
	extern __shared__ int shared_data[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * (blockDim.x * 2);
	if (index >= num_elements) return;
	shared_data[tid] = input[index] + input[index + blockDim.x];

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared_data[tid] += shared_data[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) output[blockIdx.x] = shared_data[0];
}

void reduction_3_gpu(std::vector<int>& output_gpu, std::vector<int>& input)
{
	printf("reduction_3_kernel() \n");
	//device-side data
	int *dev_i = 0;
	int *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_i, input.size() * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_gpu.size() * sizeof(int)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_i, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));//dev_a=a;
	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = input.size();
	int block = 512;
	int grid = ((thread_cnt - 1) / block + 1);

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	reduction_3_kernel << <dimGrid, dimBlock, block * sizeof(int) >> > (dev_o, dev_i, thread_cnt);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gpu.data(), dev_o, output_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));

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

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// reduction 4
__global__ void reduction_4_kernel(int* output, int* input, int num_elements)
{
    extern __shared__ int shared_data[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + (blockDim.x * 2) * blockIdx.x;
    if (index >= num_elements) return;
    shared_data[tid] = input[index] + input[index + blockDim.x];

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) warpReduce<512>(shared_data, tid);

    // write result for this block to global mem
    if (tid == 0) output[blockIdx.x] = shared_data[0];
}

void reduction_4_gpu(std::vector<int>& output_gpu, std::vector<int>& input)
{
    printf("reduction_4_kernel() \n");
    //device-side data
    int *dev_i = 0;
    int *dev_o = 0;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&dev_i, input.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_o, output_gpu.size() * sizeof(int)));

    uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    //copy from host to device 
    CUDA_CHECK(cudaMemcpy(dev_i, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));//dev_a=a;
    //launch a kernel on the GPU with one thread for each element.
    int thread_cnt = input.size();
    int block = 512;
    int grid = ((thread_cnt - 1) / block + 1);

    dim3 dimGrid(grid, 1, 1);
    dim3 dimBlock(block, 1, 1);//x,y,z

    uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    reduction_4_kernel << <dimGrid, dimBlock, block * sizeof(int) >> > (dev_o, dev_i, thread_cnt);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    //copy from device to host
    CUDA_CHECK(cudaMemcpy(output_gpu.data(), dev_o, output_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));

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


