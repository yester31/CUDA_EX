// bicubic interpolation kernel (from torch upsample algorithm)
#include "../CUDA_EX/util_cuda.cuh"

__device__ __forceinline__ static float cubic1d(float x0, float x1, float x2, float x3, float t) {
	float A = -0.75f;
	float coeffs_0 = ((A * (t + 1.f) - 5.f * A) * (t + 1.f) + 8.f * A) * (t + 1.f) - 4.f * A;
	float coeffs_1 = ((A + 2.f) * t - (A + 3.f)) * t * t + 1.f;
	float coeffs_2 = ((A + 2.f) * (1.f - t) - (A + 3.f)) * (1.f - t) * (1.f - t) + 1.f;
	float coeffs_3 = ((A * (2.f - t) - 5.f * A) * (2.f - t) + 8.f * A) * (2.f - t) - 4.f * A;
	return x0 * coeffs_0 + x1 * coeffs_1 + x2 * coeffs_2 + x3 * coeffs_3;
}

__global__ void bicubic2d(
	float* output, float* input,
	float height_scale,
	float width_scale,
	int N, int C, int H, int W, int P, int Q,
	int num_elements)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) return;

	int q_idx = index % Q; // Q 
	int idx = index / Q;
	int p_idx = idx % P; // P 
	idx /= P;
	int c_idx = idx % C; // C
	int n_idx = idx / C; // N
	const int g_idx = n_idx * C * P * Q + c_idx * P * Q + p_idx * Q + q_idx;

	// just copy
	if (H == P && W == Q) {
		output[g_idx] = input[g_idx];
		return;
	}

	// Interpolation kernel
	float real_x = width_scale * q_idx;
	int in_x = floorf(real_x);
	float t_x = real_x - in_x;

	float real_y = height_scale * p_idx;
	int in_y = floorf(real_y);
	float t_y = real_y - in_y;

	float coeff[4];
	int access_x0 = max(min((in_x - 1), W - 1), 0);
	int access_x1 = max(min((in_x + 0), W - 1), 0);
	int access_x2 = max(min((in_x + 1), W - 1), 0);
	int access_x3 = max(min((in_x + 2), W - 1), 0);
	int cu_idx = n_idx * C * H * W + c_idx * H * W;

	for (int k = 0; k < 4; k++) {
		int access_y = max(min((in_y - 1 + k), H - 1), 0);
		coeff[k] = cubic1d(
			input[cu_idx + access_y * W + access_x0],
			input[cu_idx + access_y * W + access_x1],
			input[cu_idx + access_y * W + access_x2],
			input[cu_idx + access_y * W + access_x3],
			t_x);
	}

	output[g_idx] = static_cast<float>(cubic1d(coeff[0], coeff[1], coeff[2], coeff[3], t_y));
}

int main(void) {
	// Bicubic Interpolation
	// Input [N, C, H, W] -> Ouput [N, C, P, Q]
	float rescale_factor = 2.f;
	int N = 1;
	int C = 3;
	int H = 1080;
	int W = 1920;
	int P = H * rescale_factor;
	int Q = W * rescale_factor;

	std::vector<float> input(N * C * H * W);
	std::vector<float> output(N * C * P * Q);

	// input data 초기화
	generate_data_f(input.data(), input.size());
	//print_results(input, H, W);
	//device-side data
	float *dev_a = 0;
	float *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_a, input.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output.size() * sizeof(float)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_a, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));//dev_a=a;

	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = N * C * P * Q;
	int block = 256;
	int grid = (thread_cnt - 1) / block + 1;

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z
	float h_scale = float(H - 1) / (P - 1);
	float w_scale = float(W - 1) / (Q - 1);

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	bicubic2d << <dimGrid, dimBlock >> > (dev_o, dev_a, h_scale, w_scale, N, C, H, W, P, Q, thread_cnt);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output.data(), dev_o, output.size() * sizeof(float), cudaMemcpyDeviceToHost));//c=dev_c;

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 출력
	//print_results(output, P, Q);

	printf("dur_time(gpu) w = %6.3f [msec] \n", (start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) wo = %6.3f [msec] \n", (start_time3 - start_time2) / 1000.f);

	tofile(output, "../Validation_py/Output_C");
	tofile(input, "../Validation_py/Input_C");

	// python 검증 스크립트 수행
	printf("\n *Validation with python \n");
	std::string command = "python ../Validation_py/bicubic.py --N=" + std::to_string(N) + " --C=" + std::to_string(C) + " --H=" + std::to_string(H) + " --W=" + std::to_string(W);
	const char *cmd = command.c_str();
	system(cmd); //터미널에 명령어 전달 

	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_a));

	return 0;
}
//int N = 1;
//int C = 3;
//int H = 1080;
//int W = 1920;

//upsample_bicubic2d_opti
// dur_time(gpu) w = 67.234 [msec]
// dur_time(gpu) wo = 36.090 [msec]
// dur time(pytorch) : 153.073 [msec]