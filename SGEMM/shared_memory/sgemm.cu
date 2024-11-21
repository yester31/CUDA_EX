#include <cuda.h>
#include <cuda_fp16.h>
#include "sgemm.h"

#include <iostream>
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define BLOCK_SIZE 16
using half = __half;

// kernel program for the device (GPU): compiled by NVCC
template <typename T>
__global__ void sgemm_kernel_shared_memory(
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

    __shared__ T subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T subB[BLOCK_SIZE][BLOCK_SIZE];

    int localRow = threadIdx.x;
    int localCol = threadIdx.y;

    for (int bID = 0; bID < ceil((float)K / BLOCK_SIZE); bID++)
    {
        int offset = bID * BLOCK_SIZE;

        // load A and B
        if (row >= M || offset + localCol >= K)
            subA[localCol][localRow] = 0;
        else
            subA[localCol][localRow] = A[row * K + (offset + localCol)];

        if (col >= N || offset + localRow >= K)
            subB[localRow][localCol] = 0;
        else
            subB[localRow][localCol] = B[(offset + localRow) * N + col];

        __syncthreads();

        // compute
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            sum += subA[i][localRow] * subB[i][localCol];
        }
        __syncthreads();
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

template <typename T>
cudaError_t SGEMM_Shared_Memory_Impl(cudaStream_t stream,
                                     T const *A,
                                     T const *B,
                                     T *C,
                                     int32_t const M,
                                     int32_t const K,
                                     int32_t const N,
                                     T const alpha,
                                     T const beta)
{
    // row-> y, col-> x
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    sgemm_kernel_shared_memory<T><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N, alpha, beta);
    return cudaGetLastError();
}

#define SPECIALIZED_IMPL1(T) \
    template cudaError_t SGEMM_Shared_Memory_Impl<T>(cudaStream_t stream, T const *A, T const *B, T *C, int32_t const M, int32_t const K, int32_t const N, T const alpha, T const beta);

SPECIALIZED_IMPL1(float)
SPECIALIZED_IMPL1(half)