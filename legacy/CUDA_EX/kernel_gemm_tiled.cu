#include "util_cuda.cuh"

const int WIDTH = 1024;
const int TILE_WIDTH = 32;

//kernel program for the device (GPU): compiled by NVCC
__global__ void matMul_kernel_shared_memory(float* g_C, const float* g_A, const float* g_B)
{
	extern __shared__ float s_A[];
	extern __shared__ float s_B[];

	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= WIDTH * WIDTH) return;

	int gx = pos % WIDTH;
	int gy = pos / WIDTH;
	int tx = pos % TILE_WIDTH;
	int ty = gx / TILE_WIDTH;

	float sum = 0.0F;
	for (register int m = 0; m < WIDTH / TILE_WIDTH; ++m) {
		//read into the shared memory blocks
		s_A[ty * TILE_WIDTH + tx] = g_A[gy * WIDTH + (m * TILE_WIDTH + tx)];
		s_B[ty * TILE_WIDTH + tx] = g_B[(m * TILE_WIDTH + ty) * WIDTH + gx];
		__syncthreads();
		//use the shared memory blocks to get the partial sum
		for (register int k = 0; k < TILE_WIDTH; ++k) {
			sum += s_A[ty * TILE_WIDTH + k] * s_B[k * TILE_WIDTH + tx];
		}
		__syncthreads();
	}
	g_C[gy * WIDTH + gx] = sum;
}

// shared memory 48KB
int main(void) {
	// A[M, K] * B[K, N] = C[M, N]
	const int M = 1024;
	const int K = 1024;
	const int N = 1024;

	std::vector<float> input_a(M * K);
	std::vector<float> input_b(K * N);
	std::vector<float> output(M * N);
	std::vector<float> output_cpu(M * N);

	// input data 초기화
	generate_data_f(input_a.data(), input_a.size());
	generate_data_f(input_b.data(), input_b.size());

	//device-side data
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_a, input_a.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_b, input_b.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output.size() * sizeof(float)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_a, input_a.data(), input_a.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_a=a;
	CUDA_CHECK(cudaMemcpy(dev_b, input_b.data(), input_b.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_b=b;

	//launch a kernel on the GPU with one thread for each element.
	const int GRID_WIDTH = (WIDTH * WIDTH - 1) / (TILE_WIDTH * TILE_WIDTH) + 1;	
	dim3 dimGrid(GRID_WIDTH, 1, 1);
	dim3 dimBlock(TILE_WIDTH*TILE_WIDTH, 1, 1);

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	
	matMul_kernel_shared_memory << <dimGrid, dimBlock, TILE_WIDTH * TILE_WIDTH * sizeof(float) >> > (dev_o, dev_a, dev_b);
	cudaDeviceSynchronize();
	
	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output.data(), dev_o, output.size() * sizeof(float), cudaMemcpyDeviceToHost));//c=dev_c;

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_a));
	CUDA_CHECK(cudaFree(dev_b));

	uint64_t start_time5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	//validate gpu kernel function
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			float sum = 0.f;
			for (int k = 0; k < K; ++k) {
				sum += input_a[m * K + k] * input_b[k * N + n];
			}
			output_cpu[m * N + n] = sum;
		}
	}
	uint64_t start_time6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 검증
	valid_results_f(output, output_cpu);
	printf("dur_time(gpu) w = %6.3f [msec] \n", (start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) wo = %6.3f [msec] \n", (start_time3 - start_time2) / 1000.f);
	printf("dur_time(cpu) = %6.3f [msec] \n", (start_time6 - start_time5) / 1000.f);

	return 0;
}

// dur_time(gpu) w  = 6.130 [msec]
// dur_time(gpu) wo = 3.028 [msec]
// dur_time(cpu)    = 2046.027 [msec]