#include "util_cuda.cuh"
#include <cublas_v2.h>
#include "opencv2/opencv.hpp"
#include "cublasXt.h"

const int WIDTH = 1024;
const int TILE_WIDTH = 32;

//kernel program for the device (GPU): compiled by NVCC
__global__ void matrixMulKernel_2d_sharedMemory(float* g_C, const float* g_A, const float* g_B)
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

cublasStatus_t Sgemm(
	cublasHandle_t Blas,
	cublasOperation_t AOp, cublasOperation_t BOp,
	const float* dev_A, int WidthA, int HeightA,
	const float* dev_B, int WidthB, int HeightB,
	float *dev_C,
	float Alpha = 1.0f, float Beta = 0.0f)
{
	int lda = WidthA;
	int ldb = WidthB;

	if (AOp != CUBLAS_OP_N) {
		int tmp = WidthA;
		WidthA = HeightA;
		HeightA = tmp;
	}
	if (BOp != CUBLAS_OP_N) {
		int tmp = WidthB;
		WidthB = HeightB;
		HeightB = tmp;
	}
	int m = WidthB;
	int n = HeightA;
	int k = WidthA;

	return cublasSgemm_v2(Blas, BOp, AOp, m, n, k, &Alpha, dev_B, ldb, dev_A, lda, &Beta, dev_C, m);
}

//kernel program for the device (GPU): compiled by NVCC
__global__ void matrixMulKernel_2d_f(
	float* output, const float* input_a, const float* input_b,
	int M, int K, int N, const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;

	int w_idx = pos % N;
	int h_idx = pos / N;
	float sum = 0.f;
	for (int k = 0; k < K; ++k) {
		sum += input_a[h_idx * K + k] * input_b[k * N + w_idx];
	}
	output[h_idx * N + w_idx] = sum;
}


int main(void) {

	cudaSetDevice(0);
	cublasHandle_t cublasHandle;
	CUDA_CHECK(cublasCreate(&cublasHandle));

	// A[M, K] * B[K, N] = C[M, N]
	const int M = 1024;
	const int K = 1024;
	const int N = 1024;

	std::vector<float> input_a(M * K);
	std::vector<float> input_b(K * N);
	std::vector<float> output_kernel(M * N);
	std::vector<float> output_kernels(M * N);
	std::vector<float> output_cublas(M * N);
	std::vector<float> output_cpu(M * N);
	std::vector<float> output_cv(M * N);

	// input data 초기화
	generate_data_f(input_a.data(), input_a.size());
	generate_data_f(input_b.data(), input_b.size());

	//device-side data
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_o = 0;
	float *dev_os = 0;
	float *dev_cb = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_a, input_a.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_b, input_b.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_kernel.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_os, output_kernels.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_cb, output_cublas.size() * sizeof(float)));

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_a, input_a.data(), input_a.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_a=a;
	CUDA_CHECK(cudaMemcpy(dev_b, input_b.data(), input_b.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_b=b;

	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = output_kernel.size();
	int block = 128;
	int grid = (thread_cnt - 1) / block + 1;
	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	Sgemm(cublasHandle, (cublasOperation_t)0, (cublasOperation_t)0, dev_a, K, M, dev_b, N, K, dev_cb);
	cudaDeviceSynchronize();
	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	matrixMulKernel_2d_f << <dimGrid, dimBlock >> > (dev_o, dev_a, dev_b, M, K, N, thread_cnt);
	cudaDeviceSynchronize();
	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	const int GRID_WIDTH = (WIDTH * WIDTH - 1) / (TILE_WIDTH * TILE_WIDTH) + 1;
	dim3 dimGrid2(GRID_WIDTH, 1, 1);
	dim3 dimBlock2(TILE_WIDTH*TILE_WIDTH, 1, 1);
	uint64_t start_time_s1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	matrixMulKernel_2d_sharedMemory << <dimGrid2, dimBlock2, TILE_WIDTH * TILE_WIDTH * sizeof(float) >> > (dev_os, dev_a, dev_b);
	cudaDeviceSynchronize();
	uint64_t start_time_s2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_cublas.data(), dev_cb, output_cublas.size() * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(output_kernel.data(), dev_o, output_kernel.size() * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(output_kernels.data(), dev_os, output_kernels.size() * sizeof(float), cudaMemcpyDeviceToHost));
	
	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_os));
	CUDA_CHECK(cudaFree(dev_cb));
	CUDA_CHECK(cudaFree(dev_a));
	CUDA_CHECK(cudaFree(dev_b));
	cublasDestroy(cublasHandle);

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	//validate gpu kernel function
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			float sum = 0;
			for (int k = 0; k < K; ++k) {
				sum += input_a[m * K + k] * input_b[k * N + n];
			}
			output_cpu[m * N + n] = sum;
		}
	}
	uint64_t start_time5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	cv::Mat A = cv::Mat(M, K, CV_32FC1, input_a.data());
	cv::Mat B = cv::Mat(K, N, CV_32FC1, input_b.data());
	cv::Mat O;
	uint64_t start_time6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	cv::gemm(A, B, 1, cv::Mat(), 0, O, 0);
	uint64_t start_time7 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	memcpy(output_cv.data(), O.data, M * N * sizeof(float));

	// 결과 검증
	valid_results_f(output_kernel, output_cpu);
	valid_results_f(output_cublas, output_cpu);
	valid_results_f(output_kernel, output_cublas);
	valid_results_f(output_kernels, output_cpu);
	valid_results_f(output_cv, output_cpu); 

	printf("dur_time(cublas)    wo = %6.3f [msec] \n", (start_time2 - start_time1) / 1000.f);
	printf("dur_time(kernel)    wo = %6.3f [msec] \n", (start_time3 - start_time2) / 1000.f);
	printf("dur_time(kernel sm) wo = %6.3f [msec] \n", (start_time_s2 - start_time_s1) / 1000.f);
	printf("dur_time(cpu)          = %6.3f [msec] \n", (start_time5 - start_time4) / 1000.f);
	printf("dur_time(cv)           = %6.3f [msec] \n", (start_time7 - start_time6) / 1000.f);

	return 0;
}

// A[1024, 1024] * B[1024, 1024] = C[1024, 1024]
//dur_time(cublas)    wo = 0.984 [msec]
//dur_time(kernel)    wo = 4.879 [msec]
//dur_time(kernel sm) wo = 2.645 [msec]
//dur_time(cpu)          = 2005.481 [msec]
//dur_time(cv)           = 439.728  [msec]