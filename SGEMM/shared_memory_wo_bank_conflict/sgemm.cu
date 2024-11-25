#include <cuda.h>
#include <cuda_fp16.h>
#include "sgemm.h"

#include <iostream>
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define BLOCK_SIZE 32
using half = __half;

// kernel program for the device (GPU): compiled by NVCC
template <typename T>
__global__ void sgemm_kernel_shared_memory_wo_bank_conflict(
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
    const uint h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint w_idx = blockIdx.y * blockDim.y + threadIdx.y;

    T sum = 0;

    __shared__ T subA[BLOCK_SIZE][BLOCK_SIZE]; // w h
    __shared__ T subB[BLOCK_SIZE][BLOCK_SIZE]; // h w

    int local_h_idx = threadIdx.x;
    int local_w_idx = threadIdx.y;

    for (int bID = 0; bID < ceil((float)K / BLOCK_SIZE); bID++)
    {
        int offset = bID * BLOCK_SIZE;

        // load A and B
        if (h_idx >= M || offset + local_w_idx >= K)
            subA[local_w_idx][local_h_idx] = 0;
        else
            subA[local_w_idx][local_h_idx] = A[h_idx * K + (offset + local_w_idx)];

        if (w_idx >= N || offset + local_h_idx >= K)
            subB[local_h_idx][local_w_idx] = 0;
        else
            subB[local_h_idx][local_w_idx] = B[(offset + local_h_idx) * N + w_idx];

        __syncthreads();

        // compute
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            sum += subA[i][local_h_idx] * subB[i][local_w_idx];
        }
        __syncthreads();
    }

    C[h_idx * N + w_idx] = alpha * sum + beta * C[h_idx * N + w_idx];
}

template <typename T>
cudaError_t SGEMM_Shared_Memory_WO_Bank_Conflict_Impl(cudaStream_t stream,
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
    cudaFuncSetAttribute(sgemm_kernel_shared_memory_wo_bank_conflict<T>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    sgemm_kernel_shared_memory_wo_bank_conflict<T><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N, alpha, beta);
    return cudaGetLastError();
}

#define SPECIALIZED_IMPL1(T) \
    template cudaError_t SGEMM_Shared_Memory_WO_Bank_Conflict_Impl<T>(cudaStream_t stream, T const *A, T const *B, T *C, int32_t const M, int32_t const K, int32_t const N, T const alpha, T const beta);

SPECIALIZED_IMPL1(float)
SPECIALIZED_IMPL1(half)