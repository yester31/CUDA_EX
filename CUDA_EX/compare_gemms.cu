#include "util_cuda.cuh"
#include <cublas_v2.h>
#include "opencv2/opencv.hpp"

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
	cudaStream_t stream;
	cublasHandle_t cublasHandle;
	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cublasCreate(&cublasHandle));
	CUDA_CHECK(cublasSetStream(cublasHandle, stream));

	// A[M, K] * B[K, N] = C[M, N]
	const int M = 1024;
	const int K = 1024;
	const int N = 1024;

	std::vector<float> input_a(M * K);
	std::vector<float> input_b(K * N);
	std::vector<float> output_kernel(M * N);
	std::vector<float> output_gemm(M * N);
	std::vector<float> output_cpu(M * N);
	std::vector<float> output_cv(M * N);

	// input data 초기화
	generate_data_f(input_a.data(), input_a.size());
	generate_data_f(input_b.data(), input_b.size());

	//device-side data
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_o = 0;
	float *dev_go = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_a, input_a.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_b, input_b.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_kernel.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_go, output_gemm.size() * sizeof(float)));

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_a, input_a.data(), input_a.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_a=a;
	CUDA_CHECK(cudaMemcpy(dev_b, input_b.data(), input_b.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_b=b;

	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = output_kernel.size();
	int block = 256;
	int grid = (thread_cnt - 1) / block + 1;

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	Sgemm(cublasHandle, (cublasOperation_t)0, (cublasOperation_t)0, dev_a, K, M, dev_b, N, K, dev_go);
	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	matrixMulKernel_2d_f << <dimGrid, dimBlock >> > (dev_o, dev_a, dev_b, M, K, N, thread_cnt);
	CUDA_CHECK(cudaPeekAtLastError());
	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	
	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_kernel.data(), dev_o, output_kernel.size() * sizeof(float), cudaMemcpyDeviceToHost));//c=dev_c;
	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gemm.data(), dev_go, output_gemm.size() * sizeof(float), cudaMemcpyDeviceToHost));//c=dev_c;
	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_go));
	CUDA_CHECK(cudaFree(dev_a));
	CUDA_CHECK(cudaFree(dev_b));
	CUDA_CHECK(cudaStreamDestroy(stream));

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
	valid_results_f(output_gemm, output_cpu);
	valid_results_f(output_kernel, output_gemm);
	valid_results_f(output_cv, output_cpu);
	
	printf("dur_time(cublas) wo = %6.3f [msec] \n", (start_time2 - start_time1) / 1000.f);
	printf("dur_time(kernel) wo = %6.3f [msec] \n", (start_time3 - start_time2) / 1000.f);
	printf("dur_time(cpu) = %6.3f [msec] \n", (start_time5 - start_time4) / 1000.f);
	printf("dur_time(cv) = %6.3f [msec] \n", (start_time7 - start_time6) / 1000.f);

	return 0;
}

// A[1024, 1024] * B[1024, 1024] = C[1024, 1024]
// dur_time(cublas) wo = 0.120		[msec]
// dur_time(kernel) wo = 0.006		[msec]
// dur_time(cpu)       = 2192.705	[msec]
// dur_time(cv)        = 450.568	[msec]