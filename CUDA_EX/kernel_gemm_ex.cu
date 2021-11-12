#include "util_cuda.cuh"

//kernel program for the device (GPU): compiled by NVCC
__global__ void matrixMulKernel_2d(
	float* output, const float* input_a, const float* input_b,
	int M, int K, int N, const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;
	
	int w_idx = pos % N;
	int h_idx = pos / N;
	int sum = 0;
	for (int k = 0; k < K; ++k) {
		sum += input_a[h_idx * K + k] * input_b[k * N + w_idx];
	}
	output[h_idx * N + w_idx] = sum;
}


int main(void) {
	// A[M, K] * B[K, N] = C[M, N]
	const int M = 128;
	const int K = 256;
	const int N = 128;

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
	int thread_cnt = output.size();
	int block = 256;
	int grid = (thread_cnt - 1) / block + 1;

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	matrixMulKernel_2d << <dimGrid, dimBlock >> > (dev_o, dev_a, dev_b, M, K, N, thread_cnt);
	CUDA_CHECK(cudaPeekAtLastError());

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

	printf("dur_time(gpu) w = %6.3f [msec] \n",	(start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) wo = %6.3f [msec] \n",(start_time3 - start_time2) / 1000.f);
	printf("dur_time(cpu) = %6.3f [msec] \n",	(start_time6 - start_time5) / 1000.f);

	return 0;
}