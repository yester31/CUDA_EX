#include <cuda.h>
#include <cuda_fp16.h>
#include "sgemm.h"
#include "assert.h"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

using half = __half;

template <typename T>
struct __device_builtin__ __builtin_align__(sizeof(T) * 4) typeT
{
    T x, y, z, w;
};

// kernel program for the device (GPU): compiled by NVCC
template <typename T, const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_kernel_vectorize(
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

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    // allocate thread-local cache for results in registerfile
    T threadResults[TM * TN] = {0.0};
    T regM[TM] = {0.0};
    T regN[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate the SMEM caches
        // transpose A while loading it
        // typeT<T> tmp = reinterpret_cast<typeT<T> *>(&A[innerRowA * K + innerColA * 4])[0];
        typeT<T> tmp = *reinterpret_cast<typeT<T> *>(const_cast<T *>(&A[innerRowA * K + innerColA * 4]));

        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        // reinterpret_cast<typeT<T> *>(&Bs[innerRowB * BN + innerColB * 4])[0] = reinterpret_cast<typeT<T> *>(&B[innerRowB * N + innerColB * 4])[0];
        reinterpret_cast<typeT<T> *>(&Bs[innerRowB * BN + innerColB * 4])[0] = *reinterpret_cast<typeT<T> *>(const_cast<T *>(&B[innerRowB * N + innerColB * 4]));
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // block into registers
            for (uint i = 0; i < TM; ++i)
            {
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            for (uint i = 0; i < TN; ++i)
            {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
            {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
    {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4)
        {
            // load C vector into registers
            // typeT<T> tmp = reinterpret_cast<typeT<T> *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            typeT<T> tmp = *reinterpret_cast<typeT<T> *>(const_cast<T *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]));
            // perform GEMM update in reg
            tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
            tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
            // write back
            // reinterpret_cast<typeT<T> *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
            *reinterpret_cast<typeT<T> *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]) = tmp;
        }
    }
}

template <typename T>
cudaError_t SGEMM_Vectorized_Mem_Access_Impl(cudaStream_t stream,
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
    sgemm_kernel_vectorize<T, BM, BN, BK, TM, TN><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N, alpha, beta);
    return cudaGetLastError();
}

#define SPECIALIZED_IMPL(T) \
    template cudaError_t SGEMM_Vectorized_Mem_Access_Impl<T>(cudaStream_t stream, T const *A, T const *B, T *C, int32_t const M, int32_t const K, int32_t const N, T const alpha, T const beta);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(half)