// bicubic interpolation kernel 
#include "../CUDA_EX/util_cuda.cuh"
#include "opencv2/opencv.hpp"

//kernel program for the device (GPU): compiled by NVCC
__global__ void bicubic(
	float* output, 
	const float* input_a, 
	const int height_scale,
	const int width_scale,
	int N, int C, int H, int W, int P, int Q, 
	const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;

	int w_idx = pos % N;
	int h_idx = pos / N;

}

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_convolution1(accscalar_t x, accscalar_t A) {
	return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_convolution2(accscalar_t x, accscalar_t A) {
	return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename accscalar_t>
__device__ __forceinline__ static void get_cubic_upsampling_coefficients(accscalar_t coeffs[4], accscalar_t t) {
	accscalar_t A = -0.75;

	accscalar_t x1 = t;
	coeffs[0] = cubic_convolution2<accscalar_t>(x1 + 1.0, A);
	coeffs[1] = cubic_convolution1<accscalar_t>(x1, A);

	// opposite coefficients
	accscalar_t x2 = 1.0 - t;
	coeffs[2] = cubic_convolution1<accscalar_t>(x2, A);
	coeffs[3] = cubic_convolution2<accscalar_t>(x2 + 1.0, A);
}

template <typename scalar_t>
__device__ __forceinline__ static scalar_t upsample_get_value_bounded(
	const scalar_t* data, int n, int c, int C, int H, int W, int y, int x) {
	int access_y = max(min(y, H - 1), 0);
	int access_x = max(min(x, W - 1), 0);

	int g_idx = n * C * H * W + c * H * W + access_y * H + access_x;

	return data[g_idx];
}

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_interp1d(scalar_t x0, scalar_t x1, scalar_t x2, scalar_t x3, accscalar_t t) {
	accscalar_t coeffs[4];
	get_cubic_upsampling_coefficients<accscalar_t>(coeffs, t);

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

	if (index >= num_elements) {
		return;
	}

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

	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			accscalar_t coefficients[4];

			for (int k = 0; k < 4; k++) {
				coefficients[k] = cubic_interp1d(
					upsample_get_value_bounded<scalar_t>(input, n, c, C, H, W, in_y - 1 + k, in_x - 1),
					upsample_get_value_bounded<scalar_t>(input, n, c, C, H, W, in_y - 1 + k, in_x + 0),
					upsample_get_value_bounded<scalar_t>(input, n, c, C, H, W, in_y - 1 + k, in_x + 1),
					upsample_get_value_bounded<scalar_t>(input, n, c, C, H, W, in_y - 1 + k, in_x + 2),
					t_x);
			}

			int g_idx = n * C * P * Q + c * P * Q + output_y * Q + output_x;
			output[g_idx] = static_cast<scalar_t>(cubic_interp1d(
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
	// A[H, W] x 2 = C[P, Q]
	int rescale_factor = 2;
	int N = 1;
	int C = 3;
	int H = 128;
	int W = 128;
	int P = H * rescale_factor;
	int Q = W * rescale_factor;

	std::vector<uint8_t> input(N * C * H * W);
	std::vector<uint8_t> output(N * C * P * Q);
	std::vector<uint8_t> output_cpu(N * C * P * Q);

	// input data 초기화
	generate_data_i8(input.data(), input.size());

	//device-side data
	uint8_t *dev_a = 0;
	uint8_t *dev_o = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_a, input.size() * sizeof(uint8_t)));
	CUDA_CHECK(cudaMalloc((void**)&dev_o, output.size() * sizeof(uint8_t)));

	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_a, input.data(), input.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));//dev_a=a;

	//launch a kernel on the GPU with one thread for each element.
	int thread_cnt = output.size();
	int block = 256;
	int grid = (thread_cnt - 1) / block + 1;

	dim3 dimGrid(grid, 1, 1);
	dim3 dimBlock(block, 1, 1);//x,y,z

	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	bool align_corners = true;
	upsample_bicubic2d << <dimGrid, dimBlock >> > (dev_o, dev_a, rescale_factor, rescale_factor, align_corners, N, C, H, W, P, Q, thread_cnt);

	CUDA_CHECK(cudaPeekAtLastError());

	uint64_t start_time3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(output.data(), dev_o, output.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));//c=dev_c;

	uint64_t start_time4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//free device memory
	CUDA_CHECK(cudaFree(dev_o));
	CUDA_CHECK(cudaFree(dev_a));

	uint64_t start_time5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	//validate gpu kernel function

	cv::Mat src = cv::Mat(H, W, CV_8UC3, input.data());
	cv::Mat dst(P, Q, CV_8UC3);
	cv::resize(src, dst, src.size(), cv::INTER_CUBIC); // 정합성 일치
	memcpy(output_cpu.data(), dst.data, N * C * H * W);

	//validate gpu kernel function

	uint64_t start_time6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	// 결과 검증
	valid_results(output, output_cpu);

	printf("dur_time(gpu) w = %6.3f [msec] \n", (start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) wo = %6.3f [msec] \n", (start_time3 - start_time2) / 1000.f);
	printf("dur_time(cpu) = %6.3f [msec] \n", (start_time6 - start_time5) / 1000.f);

	return 0;
}