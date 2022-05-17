#include "reduction.cuh"
// reference : https://github.com/jeonggunlee/CUDATeaching


static void generate_data_f(float* ptr, unsigned int size, int offset = 255) {
	int tt = size;
	while (size--) {
		*ptr++ = rand() % offset; //  0 ~ offset 사이 난수 생성
		//*ptr++ = 10 * (tt - size); // 10, 20, 30 ...
	}
}

static void valid_results(std::vector<float> &gpu, std::vector<float> &cpu) {
	bool result = true;
	for (int i = 0; i < gpu.size(); i++) {
		if ((gpu[i]) != cpu[i]) {
			//printf("[%d] The results is not matched! (%d, %d)\n", i, gpu[i], cpu[i]);
			//printf("[%d] The results is not matched! \n", i);
			result = false;
		}
	}
	if (result)printf("Both values is same. works well! \n");
	else printf("Both values is not matched! \n");
}

int main(void) {
	// Reduction
	// Input [size] -> Ouput [1]
	int size = 1024;
	std::vector<float> input(size);
	std::vector<float> output_gpu(size);
	std::vector<float> output_cpu(size);

	// input data 초기화
	generate_data_f(input.data(), input.size());

	// 결과 검증 수행
	uint64_t start_time5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	//transpose_cpu(output_cpu.data(), input.data(), H, W);
	uint64_t start_time6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	printf("dur_time(cpu)    = %6.3f [msec] \n", (start_time6 - start_time5) / 1000.f);
	valid_results(output_gpu, output_cpu);

	return 0;
}
