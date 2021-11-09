#include "util_cuda.cuh"

//kernel program for the device (GPU): compiled by NVCC
__global__ void matrixMulKernel(
	int* output, const int* input_a, const int* input_b, 
	int AROW, int K, int BCOL, const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;
	
	int w_idx = pos % BCOL;
	int h_idx = pos / BCOL;
	int sum = 0;
	for (int k = 0; k < K; ++k) {
		sum += input_a[h_idx * K + k] * input_b[k * BCOL + w_idx];
	}
	output[h_idx * BCOL + w_idx] = sum;
}


int main(void) {
	// MA[AROW, K] * MB[K, BCOL] = MC[AROW, BCOL]
	const int AROW = 128;
	const int K = 256;
	const int BCOL = 128;

	std::vector<int> input_a(AROW * K);
	std::vector<int> input_b(K * BCOL);
	std::vector<int> output(AROW * BCOL);
	std::vector<int> output_cpu(AROW * BCOL);

	// input data 초기화
	generate_data(input_a.data(), input_a.size());
	generate_data(input_b.data(), input_b.size());

	//device-side data
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_a, input_a.size() * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_b, input_b.size() * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output.size() * sizeof(int)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_a, input_a.data(), input_a.size() * sizeof(int), cudaMemcpyHostToDevice));//dev_a=a;
	CUDA_CHECK(cudaMemcpy(dev_b, input_b.data(), input_b.size() * sizeof(int), cudaMemcpyHostToDevice));//dev_b=b;

	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = output.size();
	int block = 256;
	int grid = (thread_cnt - 1) / block + 1;

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	matrixMulKernel << <dimGrid, dimBlock >> > (dev_o, dev_a, dev_b, AROW, K, BCOL, thread_cnt);
	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output.data(), dev_o, output.size() * sizeof(int), cudaMemcpyDeviceToHost));//c=dev_c;

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_a));
	CUDA_CHECK(cudaFree(dev_b));

	uint64_t start_time5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	
	//validate gpu kernel function
	for (int y = 0; y < AROW; ++y) {
		for (int x = 0; x < BCOL; ++x) {
			int sum = 0;
			for (int k = 0; k < K; ++k) {
				sum += input_a[y * K + k] * input_b[k * BCOL + x];
			}
			output_cpu[y * BCOL + x] = sum;
		} 
	}
	uint64_t start_time6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 검증
	valid_results(output, output_cpu);

	printf("dur_time(gpu) w = %6.3f [msec] \n",	(start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) wo = %6.3f [msec] \n",(start_time3 - start_time2) / 1000.f);
	printf("dur_time(cpu) = %6.3f [msec] \n",	(start_time6 - start_time5) / 1000.f);


	return 0;
}