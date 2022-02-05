# CUDA_EX
 
## CUDA Vector Add(Completed)
* 1d_VectorAdd_ex.cu		(Vector Sum)

## CUDA Matrix Multiplication(Completed)
  * cublas_gemm_ex.cu			(GEMM from Cublas lib)
  * kernel_gemm_ex.cu			(CUDA Kernel GEMM)
  * kernel_gemm_sharedmem.cu	(CUDA Kernel GEMM (using shared memory))
  * kernel_gemm_tiled.cu		(CUDA Kernel GEMM (using shared memory)tiled method for square)
  * kernel_gemm_tiled_rect.cu	(CUDA Kernel GEMM (using shared memory)tiled method for rectangle)
  * compare_gemms.cu			(Performace Evaluation Code)
  * kernel_gemm_int8.cu		(CUDA Kernel GEMM for int8 data type)
  * openMP_gemm.cpp           (openMP GEMM with Transpose for cache friendly)
  * opencv_gemm_ex.cpp		(GEMM from OpenCV lib)
  * Performace Evaluation for A[1024, 1024] * B[1024, 1024] = C[1024, 1024]
    * on GPU(wo : without data transfer time for Device)
        - cublas     wo = 1.075 [ms]
        - kernel     wo = 4.844 [ms]
        - kernel sm  wo = 2.678 [ms] (tiled)
        - kernel sm2 wo = 3.014 [ms] (tiled_rect)
    * on CPU(T : Transpose for cache friendly)
        - OpenCV        = 585.577  [ms]
        - cpu wo T      = 2426.978 [ms]
        - openMP wo T   = 549.573  [ms]
        - cpu w T       = 1318.835 [ms]
        - openMP w t    = 214.211  [ms]
        
## CUDA Bicubic Interpolation(Completed)
  * bicubic.cu
      - based on pytorch bicubic algorithm
      - validation results with pytorch up-sample


