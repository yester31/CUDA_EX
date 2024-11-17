#include "transpose.cuh"

const int TILE_DIM = 32;
const int BLOCK_ROWS = 16;

// naive transpose
__global__ void transpose_naive_kernel(float* output, float* input, int H, int W, int num_elements)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) return;

	int idx_h = index / W;
	int idx_w = index % W;

	output[idx_w * H + idx_h] = input[idx_h * W + idx_w];
}

void transpose_naive_gpu(std::vector<float>& output_gpu, std::vector<float>& input, int H, int W)
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
	int thread_cnt = H * W;
	int block = 256;
	int grid = (thread_cnt - 1) / block + 1;

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	transpose_naive_kernel << <dimGrid, dimBlock >> > (dev_o, dev_i, H, W, thread_cnt);

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

// tiled transpose
// doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transpose_tiled_kernel(float *odata, const float *idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[x*width + (y + j)] = idata[(y + j)*width + x];
}

void transpose_tiled_gpu(std::vector<float>& output_gpu, std::vector<float>& input, int H, int W)
{
	printf("transpose_naive_tiled_gpu() \n");
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
	dim3 dimGrid(W / TILE_DIM, H / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	transpose_tiled_kernel << <dimGrid, dimBlock >> > (dev_o, dev_i);

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

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transpose_tiled_coalesced_kernel(float *odata, const float *idata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y + j][threadIdx.x] = idata[(y + j)*width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

void transpose_tiled_coalesced_gpu(std::vector<float>& output_gpu, std::vector<float>& input, int H, int W)
{
	printf("transpose_tiled_coalesced_gpu() \n");
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
	dim3 dimGrid(W / TILE_DIM, H / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	transpose_tiled_coalesced_kernel << <dimGrid, dimBlock >> > (dev_o, dev_i);

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

// No bank-conflict transpose
// the first tile dimension is padded 
// to avoid shared memory bank conflicts.
__global__ void transpose_tiled_coalesced_padded_kernel(float *odata, const float *idata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y + j][threadIdx.x] = idata[(y + j)*width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

void transpose_tiled_coalesced_padded_gpu(std::vector<float>& output_gpu, std::vector<float>& input, int H, int W)
{
	printf("transpose_tiled_coalesced_padded_gpu() \n");
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
	dim3 dimGrid(W / TILE_DIM, H / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	transpose_tiled_coalesced_padded_kernel << <dimGrid, dimBlock >> > (dev_o, dev_i);

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