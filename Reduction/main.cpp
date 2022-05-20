#include "reduction.cuh"
// reference : https://github.com/jeonggunlee/CUDATeaching
// reference : https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf

static void reduction_cpu(long long int* output, long* input, int size)
{
	long long int sum = 0;
	for (int idx = 0; idx < size; idx++) {
		sum += input[idx];
	}
	*output = sum;
}

static void generate_data_f(long* ptr, long size, int offset = 255) {
	long  tt = size;
	while (size--) {
		//*ptr++ = rand() % offset; //  0 ~ offset 사이 난수 생성
		//*ptr++ = 10 * (tt - size); // 10, 20, 30 ...
		*ptr++ = 1 * (tt - size); // 1, 2, 3 ...
	}
}


int main(void) {
	// Reduction
	// Input [size] -> Ouput [1]
	std::cout << "Size of long long int : " << sizeof(long long int) << std::endl;
	std::cout << "Size of long long : " << sizeof(long long) << std::endl;
	std::cout << "Size of long int : " << sizeof(long int) << std::endl;
	std::cout << "Size of long : " << sizeof(long) << std::endl;
	std::cout << "Size of int : " << sizeof(int) << std::endl;
	long size = 1 << 22; // 2^10
	std::cout << "size : " << size << std::endl;

	//int size = 10;
	std::vector<long> input(size);
	long long int sum = 0;
	std::vector<long> output_gpu0(size);
	std::vector<long> output_gpu1(size);
	std::vector<long> output_gpu2(size);
	std::vector<long> output_gpu3(size);

	// input data 초기화
	generate_data_f(input.data(), input.size());

	reduction_0_gpu(output_gpu0, input);

	reduction_1_gpu(output_gpu1, input);

	reduction_2_gpu(output_gpu2, input);

	reduction_3_gpu(output_gpu3, input);

	// 결과 검증 수행
	uint64_t start_time5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	reduction_cpu(&sum, input.data(), size);
	uint64_t start_time6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	printf("dur_time(cpu) = %6.3f [msec] \n", (start_time6 - start_time5) / 1000.f);
	std::cout << "output_cpu0 : " << sum <<std::endl;
	std::cout << "output_gpu0 : " << output_gpu0[0] << std::endl;
	std::cout << "output_gpu1 : " << output_gpu1[0] << std::endl;
	std::cout << "output_gpu2 : " << output_gpu2[0] << std::endl;
	std::cout << "output_gpu3 : " << output_gpu2[0] << std::endl;

	//valid_results(output_gpu, output_cpu);

	return 0;
}
