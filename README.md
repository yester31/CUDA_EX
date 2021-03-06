# CUDA_EX

## Enviroments
* Windows 10 laptop
* CPU 11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz (cpu)
* NVIDIA GeForce RTX 3060 Laptop GPU (gpu)
***
## CUDA Vector Add(Completed)
* 1d_VectorAdd_ex.cu		(Vector Sum)
***
## CUDA Matrix Multiplication(need debug)
  * cublas_gemm_ex.cu			(GEMM from Cublas lib)
  * kernel_gemm_ex.cu			(CUDA Kernel GEMM)
  * kernel_gemm_sharedmem.cu	(CUDA Kernel GEMM (using shared memory))
  * kernel_gemm_tiled.cu		(CUDA Kernel GEMM (using shared memory)tiled method for square)
  * kernel_gemm_tiled_rect.cu	(CUDA Kernel GEMM (using shared memory)tiled method for rectangle)
  * compare_gemms.cu			(Performace Evaluation Code)
  * kernel_gemm_int8.cu		    (CUDA Kernel GEMM for int8 data type)
  * openMP_gemm.cpp             (openMP GEMM with Transpose for cache friendly)
  * opencv_gemm_ex.cpp		    (GEMM from OpenCV lib)
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
***     
## CUDA Bicubic Interpolation(Completed)
  * bicubic.cu
      - based on pytorch bicubic algorithm
      - validation results with pytorch up-sample
***
## CUDA Matrix Transpose(Completed)
   * Matrix Transpose kernel
       - naive transpose
       - tiled transpose
       - tilde coalesced transpose using by shared memory 
       - tilde coalesced padded transpose using by shared memory 
***
## CUDA Reduction(on progress)
   * Reduction kernel
       - interleaved addressing with divergent branching
       - interleaved addressing with bank conflicts
       - sequential addressing
       - first add during global load
       - unroll last warp
***
       
## reference   
* transpose : <https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/>
* reduction : <https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>