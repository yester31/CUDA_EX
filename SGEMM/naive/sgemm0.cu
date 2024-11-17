#include <cuda.h>
#include <cuda_fp16.h>
#include "sgemm0.h"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

using half = __half;

// kernel program for the device (GPU): compiled by NVCC
template <typename T>
__global__ void sgemm_kernel_navie(
    T const *A,
    T const *B,
    T *C,
    int32_t const M,
    int32_t const K,
    int32_t const N,
    T const alpha,
    T const beta)
{
    // compute position in C that this thread is responsible for
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N)
        return;

    T sum = 0;
    for (int k = 0; k < K; ++k)
    {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

template <typename T>
cudaError_t SGEMM_Naive_Impl(cudaStream_t stream,
                             T const *A,
                             T const *B,
                             T *C,
                             int32_t const M,
                             int32_t const K,
                             int32_t const N,
                             T const alpha,
                             T const beta)
{
    // row-> x, col-> y
    // launch a kernel on the GPU with one thread for each element.
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32, 1);

    sgemm_kernel_navie<T><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N, alpha, beta);
    return cudaGetLastError();
}

#define SPECIALIZED_IMPL(T) \
    template cudaError_t SGEMM_Naive_Impl<T>(cudaStream_t stream, T const *A, T const *B, T *C, int32_t const M, int32_t const K, int32_t const N, T const alpha, T const beta);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(half)