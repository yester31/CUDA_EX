# CUDA_EX
CUDA 예제 

** 완성
* 1d_VectorAdd_ex.cu		(벡터 합 예제)
* cublas_gemm_ex.cu			(Cublas lib 행렬 곱 예제)
* opencv_gemm_ex.cpp		(OpenCV lib 행렬 곱 예제)
* kernel_gemm_ex.cu			(CUDA Kernel 함수로 행렬 곱 예제)
* kernel_gemm_sharedmem.cu	(CUDA Kernel 함수(shared memory 사용)로 행렬 곱 예제)
* kernel_gemm_tiled.cu		(CUDA Kernel 함수(shared memory 사용)로 tiled 방식 정사각형 행렬 곱 예제)
* kernel_gemm_tiled_rect.cu	(CUDA Kernel 함수(shared memory 사용)로 tiled 방식 직사각형 행렬 곱 예제)
* compare_gemms.cu			(6가지 행렬곱 함수 성능 비교 예제)
* kernel_gemm_int8.cu		(CUDA Kernel 함수로 int8 data 행렬 곱 예제)

** gemm 성능 비교 (wo data transfer time for GPU)
* A[1024, 1024] * B[1024, 1024] = C[1024, 1024]
* cublas     wo = 1.075 ms
* kernel     wo = 4.844 ms
* kernel sm  wo = 2.678 ms (tiled)
* kernel sm2 wo = 3.014 ms (tiled_rect)
* cpu           = 2592.317 ms
* OpenCV        = 585.577 ms

** 준비중
* openMP gemm (예정)
* kernel_im2col(예정)
* kernel_col2im(예정)
* kernel_gemm_conv2d(예정)

