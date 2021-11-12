#include "util_cuda.cuh"
#include <cublas_v2.h>

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


int main(void) {

	cudaSetDevice(0);
	cudaStream_t stream;
	cublasHandle_t cublasHandle;

	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cublasCreate(&cublasHandle));
	CUDA_CHECK(cublasSetStream(cublasHandle, stream));

	// A[M, K] * B[K, N] = C[M, N]
	const int M = 128;
	const int K = 256;
	const int N = 128;

	std::vector<float> input_a(M * K);
	std::vector<float> input_b(K * N);
	std::vector<float> output_gemm(M * N);
	std::vector<float> output_cpu(M * N);

	// input data 초기화
	generate_data_f(input_a.data(), input_a.size());
	generate_data_f(input_b.data(), input_b.size());

	//device-side data
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_go = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_a, input_a.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_b, input_b.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_go, output_gemm.size() * sizeof(float)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_a, input_a.data(), input_a.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_a=a;
	CUDA_CHECK(cudaMemcpy(dev_b, input_b.data(), input_b.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_b=b;

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	Sgemm(cublasHandle, (cublasOperation_t)0, (cublasOperation_t)0, dev_a, K, M, dev_b, N, K, dev_go);

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gemm.data(), dev_go, output_gemm.size() * sizeof(float), cudaMemcpyDeviceToHost));//c=dev_c;

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//free device memory
	CUDA_CHECK(cudaFree(dev_go));
	CUDA_CHECK(cudaFree(dev_a));
	CUDA_CHECK(cudaFree(dev_b));
	CUDA_CHECK(cudaStreamDestroy(stream));
	uint64_t start_time5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

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
	uint64_t start_time6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 검증
	valid_results_f(output_gemm, output_cpu);

	//print_results(output_gemm, M, N);
	//print_results(output_cpu, M, N);

	printf("dur_time(gpu) w = %6.3f [msec] \n", (start_time4 - start_time1) / 1000.f);
	printf("dur_time(cublas) wo = %6.3f [msec] \n", (start_time3 - start_time2) / 1000.f);
	printf("dur_time(cpu) = %6.3f [msec] \n", (start_time6 - start_time5) / 1000.f);


	return 0;
}