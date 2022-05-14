#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "util_cuda.cuh"

void transpose(std::vector<float> &AI, std::vector<float> &AO, int n) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			AO[j*n + i] = AI[i*n + j];
		}
	}
}

void gemm(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, int n)
{
	int i, j, k;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			float dot = 0;
			for (k = 0; k < n; k++) {
				dot += A[i*n + k] * B[k*n + j];
			}
			C[i*n + j] = dot;
		}
	}
}

void gemm_omp(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, int n)
{
#pragma omp parallel
	{
		int i, j, k;
#pragma omp for
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				float dot = 0;
				for (k = 0; k < n; k++) {
					dot += A[i*n + k] * B[k*n + j];
				}
				C[i*n + j] = dot;
			}
		}

	}
}

void gemmT(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, int n)
{
	int i, j, k;
	std::vector<float> B2(n*n);
	transpose(B, B2, n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			float dot = 0;
			for (k = 0; k < n; k++) {
				dot += A[i*n + k] * B2[j*n + k];
			}
			C[i*n + j] = dot;
		}
	}
}

void gemmT_omp(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, int n)
{
	std::vector<float> B2(n*n);
	transpose(B, B2, n);
#pragma omp parallel
	{
		int i, j, k;
#pragma omp for
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				float dot = 0;
				for (k = 0; k < n; k++) {
					dot += A[i*n + k] * B2[j*n + k];
				}
				C[i*n + j] = dot;
			}
		}

	}
}

int main() {

	// A[M, K] * B[K, N] = C[M, N]
	const int M = 1024;
	const int K = 1024;
	const int N = 1024;

	std::vector<float> input_a(M * K);
	std::vector<float> input_b(K * N);
	std::vector<float> output_cpu(M * N);
	std::vector<float> output_opneMP(M * N);
	std::vector<float> output_cpu_Transpose(M * N);
	std::vector<float> output_openMP_Transpose(M * N);

	// input data √ ±‚»≠
	generate_data_f(input_a.data(), input_a.size());
	generate_data_f(input_b.data(), input_b.size());

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	gemm(input_a, input_b, output_cpu, K);
	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();


	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	gemm_omp(input_a, input_b, output_opneMP, K);
	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();


	uint64_t start_time5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	gemmT(input_a, input_b, output_cpu_Transpose, K);
	uint64_t start_time6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();


	uint64_t start_time7 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	gemmT_omp(input_a, input_b, output_openMP_Transpose, K);
	uint64_t start_time8 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	valid_results(output_opneMP, output_cpu);
	valid_results(output_cpu_Transpose, output_cpu);
	valid_results(output_openMP_Transpose, output_cpu);

	printf("dur_time(cpu)          = %6.3f [msec] \n", (start_time2 - start_time1) / 1000.f);
	printf("dur_time(openMP)       = %6.3f [msec] \n", (start_time4 - start_time3) / 1000.f);
	printf("dur_time(cpu w T)      = %6.3f [msec] \n", (start_time6 - start_time5) / 1000.f);
	printf("dur_time(openMP w t)   = %6.3f [msec] \n", (start_time8 - start_time7) / 1000.f);


	return 0;

}