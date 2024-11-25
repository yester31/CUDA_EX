#include <cuda.h>
#include <cuda_fp16.h>
#include "sgemm.h"

#include <iostream>
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define BLOCK_SIZE 32
using half = __half;

// kernel program for the device (GPU): compiled by NVCC
template <typename T>
__global__ void sgemm_kernel_shared_memory_1_(
    T const *A,
    T const *B,
    T *C,
    int32_t const M,
    int32_t const K,
    int32_t const N,
    T const alpha,
    T const beta)
{
    // the output block that we want to compute in this threadblock
    const uint c_h_idx = blockIdx.x;
    const uint c_w_idx = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // the inner row & col that we're accessing in this thread
    const uint thread_w_idx = threadIdx.x % BLOCK_SIZE;
    const uint thread_h_idx = threadIdx.x / BLOCK_SIZE;

    // advance pointers to the starting positions
    A += c_h_idx * BLOCK_SIZE * K;                        // row=cRow, col=0
    B += c_w_idx * BLOCK_SIZE;                            // row=0, col=cCol
    C += c_h_idx * BLOCK_SIZE * N + c_w_idx * BLOCK_SIZE; // row=cRow, col=cCol

    T tmp = 0.0;
    // the outer loop advances A along the columns and B along
    // the rows until we have fully calculated the result in C.
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE)
    {
        // Have each thread load one of the elements in A & B from
        // global memory into shared memory.
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[thread_w_idx * BLOCK_SIZE + thread_h_idx] = A[thread_h_idx * K + thread_w_idx];
        Bs[thread_h_idx * BLOCK_SIZE + thread_w_idx] = B[thread_h_idx * N + thread_w_idx];

        // block threads in this block until cache is fully populated
        __syncthreads();

        // advance pointers onto next chunk
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx)
        {
            tmp += As[dotIdx * BLOCK_SIZE + thread_h_idx] * Bs[dotIdx * BLOCK_SIZE + thread_w_idx];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[thread_h_idx * N + thread_w_idx] = alpha * tmp + beta * C[thread_h_idx * N + thread_w_idx];
}

// kernel program for the device (GPU): compiled by NVCC
template <typename T>
__global__ void sgemm_kernel_shared_memory_1__(
    T const *A,
    T const *B,
    T *C,
    int32_t const M,
    int32_t const K,
    int32_t const N,
    T const alpha,
    T const beta)
{
    // the output block that we want to compute in this threadblock
    const uint c_h_idx = blockIdx.x;
    const uint c_w_idx = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // the inner row & col that we're accessing in this thread
    const uint thread_w_idx = threadIdx.x % BLOCK_SIZE;
    const uint thread_h_idx = threadIdx.x / BLOCK_SIZE;

    // advance pointers to the starting positions
    A += c_h_idx * BLOCK_SIZE * K;                        // row=cRow, col=0
    B += c_w_idx * BLOCK_SIZE;                            // row=0, col=cCol
    C += c_h_idx * BLOCK_SIZE * N + c_w_idx * BLOCK_SIZE; // row=cRow, col=cCol

    T tmp = 0.0;
    // the outer loop advances A along the columns and B along
    // the rows until we have fully calculated the result in C.
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE)
    {
        // Have each thread load one of the elements in A & B from
        // global memory into shared memory.
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[thread_h_idx * BLOCK_SIZE + thread_w_idx] = A[thread_h_idx * K + thread_w_idx];
        Bs[thread_h_idx * BLOCK_SIZE + thread_w_idx] = B[thread_h_idx * N + thread_w_idx];

        // block threads in this block until cache is fully populated
        __syncthreads();

        // advance pointers onto next chunk
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx)
        {
            tmp += As[thread_h_idx * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + thread_w_idx];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[thread_h_idx * N + thread_w_idx] = alpha * tmp + beta * C[thread_h_idx * N + thread_w_idx];
}

// kernel program for the device (GPU): compiled by NVCC
template <typename T>
__global__ void sgemm_kernel_shared_memory_1(
    T const *A,
    T const *B,
    T *C,
    int32_t const M,
    int32_t const K,
    int32_t const N,
    T const alpha,
    T const beta)
{
    // the output block that we want to compute in this threadblock
    const uint c_h_idx = blockIdx.y; // M
    const uint c_w_idx = blockIdx.x; // N

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // the inner row & col that we're accessing in this thread
    const uint thread_h_idx = threadIdx.x / BLOCK_SIZE;
    const uint thread_w_idx = threadIdx.x % BLOCK_SIZE;

    // advance pointers to the starting positions
    A += c_h_idx * BLOCK_SIZE * K;                        // row=cRow, col=0
    B += c_w_idx * BLOCK_SIZE;                            // row=0, col=cCol
    C += c_h_idx * BLOCK_SIZE * N + c_w_idx * BLOCK_SIZE; // row=cRow, col=cCol

    T tmp = 0.0;
    // the outer loop advances A along the columns and B along
    // the rows until we have fully calculated the result in C.
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE)
    {
        // Have each thread load one of the elements in A & B from
        // global memory into shared memory.
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[thread_h_idx * BLOCK_SIZE + thread_w_idx] = A[thread_h_idx * K + thread_w_idx];
        Bs[thread_h_idx * BLOCK_SIZE + thread_w_idx] = B[thread_h_idx * N + thread_w_idx];

        // block threads in this block until cache is fully populated
        __syncthreads();

        // advance pointers onto next chunk
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx)
        {
            tmp += As[thread_h_idx * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + thread_w_idx];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[thread_h_idx * N + thread_w_idx] = alpha * tmp + beta * C[thread_h_idx * N + thread_w_idx];
}

template <typename T>
cudaError_t SGEMM_Shared_Memory_1_Impl(cudaStream_t stream,
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
    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE), 1);
    // dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    cudaFuncSetAttribute(sgemm_kernel_shared_memory_1<T>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    sgemm_kernel_shared_memory_1<T><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N, alpha, beta);
    return cudaGetLastError();
}

#define SPECIALIZED_IMPL1(T) \
    template cudaError_t SGEMM_Shared_Memory_1_Impl<T>(cudaStream_t stream, T const *A, T const *B, T *C, int32_t const M, int32_t const K, int32_t const N, T const alpha, T const beta);

SPECIALIZED_IMPL1(float)
SPECIALIZED_IMPL1(half)