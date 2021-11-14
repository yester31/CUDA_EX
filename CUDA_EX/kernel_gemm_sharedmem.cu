#include "util_cuda.cuh"

#define ID2INDEX(_row,_col, _width) (((_row)*(_width))+(_col))
#define BLOCK_SIZE 16

__global__ void matMul_kernel_shared_memory_2(float* matC, float* matA, float* matB,  int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	float val = 0;
	//__shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
	//__shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];
	extern __shared__ float subA[];
	extern __shared__ float subB[];

	int localRow = threadIdx.x;
	int localCol = threadIdx.y;

	for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
		int offset = bID * BLOCK_SIZE;

		// load A and B
		if (row >= m || offset + localCol >= k)
			subA[localCol*BLOCK_SIZE + localRow] = 0;
		else
			subA[localCol*BLOCK_SIZE + localRow] = matA[ID2INDEX(row, offset + localCol, k)];

		if (col >= n || offset + localRow >= k)
			subB[localRow*BLOCK_SIZE + localCol] = 0;
		else
			subB[localRow*BLOCK_SIZE + localCol] = matB[ID2INDEX(offset + localRow, col, n)];

		__syncthreads();

		// compute
		for (int i = 0; i < BLOCK_SIZE; i++) {
			val += __fmul_rn(subA[i * BLOCK_SIZE + localRow], subB[i * BLOCK_SIZE + localCol]);
		}
		__syncthreads();
	}

	if (row >= m || col >= n)
		return;

	matC[ID2INDEX(row, col, n)] = val;
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
	dim3 gridDim(ceil((float)M / BLOCK_SIZE), ceil((float)N / BLOCK_SIZE));
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	matMul_kernel_shared_memory_2 << <gridDim, blockDim, BLOCK_SIZE * BLOCK_SIZE * sizeof(float) >> > (dev_o, dev_a, dev_b, M, N, K);
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