// Matrix Transpose kernel
#include "../CUDA_EX/util_cuda.cuh"

void transpose_cpu(float* output, float* input, int H, int W) {
	for (int idx_h = 0; idx_h < H; idx_h++) {
		for (int idx_w = 0; idx_w < W; idx_w++) {
			output[idx_w * H + idx_h] = input[idx_h * W + idx_w];
		}
	}
}

__global__ void transpose_naive_kernel(float* output, float* input, int H, int W, int num_elements)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) return;

	int idx_h = index / W;
	int idx_w = index % W;

	output[idx_w * H + idx_h] = input[idx_h * W + idx_w];
}


int main(void) {
	// Matrix Transpose
	// Input [H, W] -> Ouput [W, H]
	int H = 1024;
	int W = 2048;
	std::vector<float> input(H * W);
	std::vector<float> output_gpu(H * W);
	std::vector<float> output_cpu(H * W);

	// input data 초기화
	generate_data_f(input.data(), input.size());
	//print_results(input, H, W);
	//device-side data
	float *dev_i = 0;
	float *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_i, input.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output_gpu.size() * sizeof(float)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_i, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_a=a;

	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = H * W;
	int block = 256;
	int grid = (thread_cnt - 1) / block + 1;

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	transpose_naive_kernel << <dimGrid, dimBlock >> > (dev_o, dev_i, H, W, thread_cnt);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output_gpu.data(), dev_o, output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 출력
	//print_results(input, H, W);
	//print_results(output_gpu, W, H);
	printf("dur_time(gpu) = %6.3f [msec] (with data transfer time)\n", (start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) = %6.3f [msec] (without data transfer time)\n", (start_time3 - start_time2) / 1000.f);

	// 결과 검증 수행
	uint64_t start_time5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	transpose_cpu(output_cpu.data(), input.data(), H, W);
	uint64_t start_time6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	printf("dur_time(cpu)    = %6.3f [msec] \n", (start_time6 - start_time5) / 1000.f);
	valid_results(output_gpu, output_cpu);

	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_i));

	return 0;
}
