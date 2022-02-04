// bicubic interpolation kernel (torch upsample algorithm)
#include "../CUDA_EX/util_cuda.cuh"
#include "opencv2/opencv.hpp"

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_interp1d(scalar_t x0, scalar_t x1, scalar_t x2, scalar_t x3, accscalar_t t) {
	accscalar_t coeffs[4];

	accscalar_t A = -0.75f;
	coeffs[0] = ((A * (t + 1.f) - 5.f * A) * (t + 1.f) + 8.f * A) * (t + 1.f) - 4.f * A;
	coeffs[1] = ((A + 2.f) * t - (A + 3.f)) * t * t + 1.f;

	// opposite coefficients
	coeffs[2] = ((A + 2.f) * (1.f - t) - (A + 3.f)) * (1.f - t) * (1.f - t) + 1.f;
	coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
	//coeffs[3] = ((A * (2.f - t) - 5.f * A) * (2.f - t) + 8.f * A) * (2.f - t) - 4.f * A;

	return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t area_pixel_compute_source_index(accscalar_t scale, int dst_index, bool align_corners, bool cubic) {

	if (align_corners) {
		return scale * dst_index;
	}
	else {
		accscalar_t src_idx = scale * (dst_index + static_cast<accscalar_t>(0.5)) - static_cast<accscalar_t>(0.5);
		// See Note[Follow Opencv resize logic]
		return (!cubic && src_idx < static_cast<accscalar_t>(0)) ? static_cast<accscalar_t>(0) : src_idx;
	}
}

template <typename scalar_t, typename accscalar_t>
__global__ void upsample_bicubic2d(
	scalar_t* output, scalar_t* input,
	accscalar_t height_scale,
	accscalar_t width_scale,
	bool align_corners,
	int N, int C, int H, int W, int P, int Q,
	int num_elements)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) return;

	// Special case: input and output are the same size, just copy
	const int output_x = index % Q;
	const int output_y = index / Q;

	if (H == P && W == Q) {
		int g_idx = 0;
		for (int n = 0; n < N; n++) {
			for (int c = 0; c < C; c++) {
				g_idx = n * C * P * Q + c * P * Q + output_y * Q + output_x;
				const scalar_t val = input[g_idx];
				output[g_idx] = val;
			}
		}
		return;
	}

	// Interpolation kernel
	accscalar_t real_x = area_pixel_compute_source_index(width_scale, output_x, align_corners, /*cubic=*/true);
	int in_x = floorf(real_x);
	accscalar_t t_x = real_x - in_x;

	accscalar_t real_y = area_pixel_compute_source_index(height_scale, output_y, align_corners, /*cubic=*/true);
	int in_y = floorf(real_y);
	accscalar_t t_y = real_y - in_y;

	//printf("in_x : %d, in_y : %d \n", in_x, in_y);

	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			accscalar_t coefficients[4];

			for (int k = 0; k < 4; k++) {
				int access_y = max(min((in_y - 1 + k), H - 1), 0);
				int access_x0 = max(min((in_x - 1), W - 1), 0);
				int access_x1 = max(min((in_x + 0), W - 1), 0);
				int access_x2 = max(min((in_x + 1), W - 1), 0);
				int access_x3 = max(min((in_x + 2), W - 1), 0);

				//printf("y : %d, x0 : %d, x1 : %d, x2 : %d, x3 : %d\n", access_y, access_x0, access_x1, access_x2, access_x3);
				coefficients[k] = cubic_interp1d(
					input[n * C * H * W + c * H * W + access_y * W + access_x0],
					input[n * C * H * W + c * H * W + access_y * W + access_x1],
					input[n * C * H * W + c * H * W + access_y * W + access_x2],
					input[n * C * H * W + c * H * W + access_y * W + access_x3],
					t_x);
			}

			output[n * C * P * Q + c * P * Q + output_y * Q + output_x] = static_cast<scalar_t>(cubic_interp1d(
				coefficients[0],
				coefficients[1],
				coefficients[2],
				coefficients[3],
				t_y));
		}
	}
}

int main(void) {
	// Bicubic Interpolation
	// Input [N, C, H, W] -> Ouput [N, C, P, Q]
	float rescale_factor = 2.f;
	int N = 1;
	int C = 3;
	int H = 4;
	int W = 4;
	int P = H * rescale_factor;
	int Q = W * rescale_factor;

	std::vector<float> input(N * C * H * W);
	std::vector<float> output(N * C * P * Q);

	// input data 초기화
	generate_data_f(input.data(), input.size());
	print_results(input, H, W);
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

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	bool align_corners = true;
	float h_scale = float(H - 1) / (P - 1);
	float w_scale = float(W - 1) / (Q - 1);
	upsample_bicubic2d << <dimGrid, dimBlock >> > (dev_o, dev_a, h_scale, w_scale, align_corners, N, C, H, W, P, Q, thread_cnt);

	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output.data(), dev_o, output.size() * sizeof(float), cudaMemcpyDeviceToHost));//c=dev_c;

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 출력
	print_results(output, P, Q);

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