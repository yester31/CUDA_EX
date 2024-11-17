#include "util_cuda.cuh"
#include "opencv2/opencv.hpp"

int main(void) {

	// A[M, K] * B[K, N] = C[M, N]
	const int M = 128;
	const int K = 256;
	const int N = 128;

	std::vector<float> input_a(M * K);
	std::vector<float> input_b(K * N);
	std::vector<float> output_cpu(M * N);
	std::vector<float> output_cv(M * N);

	// input data 초기화
	generate_data_f(input_a.data(), input_a.size());
	generate_data_f(input_b.data(), input_b.size());

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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
	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	
	cv::Mat A = cv::Mat(M, K, CV_32FC1, input_a.data());
	cv::Mat B = cv::Mat(K, N, CV_32FC1, input_b.data());
	cv::Mat O;

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	cv::gemm(A, B, 1, cv::Mat(), 0, O, 0);
	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	memcpy(output_cv.data(), O.data, M * N * sizeof(float));

	// 결과 검증
	valid_results_f(output_cpu, output_cv);

	//print_results(output_cv, M, N);
	//print_results(output_cpu, M, N);

	printf("dur_time(cpu) = %6.3f [msec] \n", (start_time2 - start_time1) / 1000.f);
	printf("dur_time(cv) = %6.3f [msec] \n", (start_time4 - start_time3) / 1000.f);

	return 0;
}