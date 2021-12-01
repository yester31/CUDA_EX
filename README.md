# CUDA_EX
 
## Completed
* 1d_VectorAdd_ex.cu		(Vector Sum)
* cublas_gemm_ex.cu			(GEMM from Cublas lib)
* opencv_gemm_ex.cpp		(GEMM from OpenCV lib)
* kernel_gemm_ex.cu			(CUDA Kernel GEMM)
* kernel_gemm_sharedmem.cu	(CUDA Kernel GEMM (using shared memory))
* kernel_gemm_tiled.cu		(CUDA Kernel GEMM (using shared memory)tiled method for square)
* kernel_gemm_tiled_rect.cu	(CUDA Kernel GEMM (using shared memory)tiled method for rectangle)
* compare_gemms.cu			(Performace Evaluation Code)
* kernel_gemm_int8.cu		(CUDA Kernel GEMM for int8 data type)

## Performace Evaluation of GEMM (wo data transfer time for GPU)
* A[1024, 1024] * B[1024, 1024] = C[1024, 1024]
* cublas     wo = 1.075 [ms]
* kernel     wo = 4.844 [ms]
* kernel sm  wo = 2.678 [ms] (tiled)
* kernel sm2 wo = 3.014 [ms] (tiled_rect)
* cpu           = 2592.317 [ms]
* OpenCV        = 585.577 [ms]

## Preparing
* openMP gemm (Preparing)
* kernel_im2col(Preparing)
* kernel_col2im(Preparing)
* kernel_gemm_conv2d(Preparing)

