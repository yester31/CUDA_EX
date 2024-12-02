#include <cuda.h>
#include <cuda_fp16.h>
#include "sgemm.h"
#include "assert.h"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

using half = __half;

// kernel program for the device (GPU): compiled by NVCC
template <typename T, const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_kernel_2d_block_tiling(
    T const *A,
    T const *B,
    T *C,
    int32_t const M,
    int32_t const K,
    int32_t const N,
    T const alpha,
    T const beta)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN; // 128 * 128 = 16384 = 블록마다 계산되는 결과 수
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN); // 16384/(8*8) = 256 = 블록에 속한 스레드 수

    // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
    assert(numThreadsBlocktile == blockDim.x);

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN); // 0-255 % (128 / 8) = 0-15
    const int threadRow = threadIdx.x / (BN / TN); // 0-255 / (128 / 8) = 0-15

    // allocate space for the current blocktile in smem
    __shared__ T As[BM * BK]; // 128 * 8 = 1024
    __shared__ T Bs[BK * BN]; // 8 * 128 = 1024

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    const uint innerRowA = threadIdx.x / BK; // 0-255 / 8 = 0-31
    const uint innerColA = threadIdx.x % BK; // 0-255 % 8 = 0-7
    // calculates the number of rows of As that are being loaded in a single step
    // by a single block
    const uint strideA = numThreadsBlocktile / BK; // 256 / 8 = 32
    const uint innerRowB = threadIdx.x / BN;       // 0-255 / 128 = 0-1
    const uint innerColB = threadIdx.x % BN;       // 0-255 % 128 = 0-127
    // for both As and Bs we want each load to span the full column-width, for
    // better GMEM coalescing (as opposed to spanning full row-width and iterating
    // across columns)
    const uint strideB = numThreadsBlocktile / BN; // 256 / 128 = 2

    // allocate thread-local cache for results in registerfile
    T threadResults[TM * TN] = {0.0}; // 8 * 8
    // register caches for As and Bs
    T regM[TM] = {0.0}; // 8
    T regN[TN] = {0.0}; // 8

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) // k=4096, 4096/8=512, 0-511
    {
        // populate the SMEM caches
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) // strideA=32, 128/32=4, 0-3
        {
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) // strideB=2, 8/2=4, 0-3
        {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results 64
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) // BK=8, dotIdx=1, 0-7
        {
            // block into registers
            for (uint i = 0; i < TM; ++i) // TM=8, 0-7
            {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) // TN=8, 0-7
            {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) // TM=8, 0-7
            {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) // TN=8, 0-7
                {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
    {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
        {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}

template <typename T>
cudaError_t SGEMM_2D_Block_Tiling_Impl(cudaStream_t stream,
                                       T const *A,
                                       T const *B,
                                       T *C,
                                       int32_t const M,
                                       int32_t const K,
                                       int32_t const N,
                                       T const alpha,
                                       T const beta)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm_kernel_2d_block_tiling<T, BM, BN, BK, TM, TN><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N, alpha, beta);
    return cudaGetLastError();
}

#define SPECIALIZED_IMPL(T) \
    template cudaError_t SGEMM_2D_Block_Tiling_Impl<T>(cudaStream_t stream, T const *A, T const *B, T *C, int32_t const M, int32_t const K, int32_t const N, T const alpha, T const beta);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(half)