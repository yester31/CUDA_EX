#include <cuda.h>
#include <cuda_fp16.h>
#include "sgemm.h"

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
    T const beta,
    const int tcount)
{
    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos >= tcount)
        return;

    int x_idx = pos % N;
    int y_idx = pos / N;
    T sum = 0;
    for (int k = 0; k < K; ++k)
    {
        sum += A[y_idx * K + k] * B[k * N + x_idx];
    }
    C[y_idx * N + x_idx] = alpha * sum + beta * C[y_idx * N + x_idx];
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
    // launch a kernel on the GPU with one thread for each element.
    int thread_cnt = M * N;
    int block = 256;
    int grid = (thread_cnt - 1) / block + 1;
    dim3 dimGrid(grid, 1, 1);
    dim3 dimBlock(block, 1, 1); // x,y,z
    sgemm_kernel_navie<T><<<dimGrid, dimBlock, 0, stream>>>(A, B, C, M, K, N, alpha, beta, thread_cnt);
    return cudaGetLastError();
}

#define SPECIALIZED_IMPL(T) \
    template cudaError_t SGEMM_Naive_Impl<T>(cudaStream_t stream, T const *A, T const *B, T *C, int32_t const M, int32_t const K, int32_t const N, T const alpha, T const beta);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(half)