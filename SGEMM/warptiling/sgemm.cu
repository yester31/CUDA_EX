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

const int WARPSIZE = 32; // warpSize is not constexpr

namespace wt
{
    template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
    __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                                 float *As, float *Bs, int innerRowA, int innerColA,
                                 int innerRowB, int innerColB)
    {
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA)
        {
            const float4 tmp = reinterpret_cast<const float4 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
            // float4 tmp;
            // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
            //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
            //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB)
        {
            reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] = reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
            // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
            //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
            //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
            //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
            //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
            //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
        }
    }

    template <const int BM, const int BN, const int BK, const int WM, const int WN,
              const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
              const int TM, const int TN>
    __device__ void
    processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                    const float *Bs, const uint warpRow, const uint warpCol,
                    const uint threadRowInWarp, const uint threadColInWarp)
    {
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // populate registers for whole warptile
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
            {
                for (uint i = 0; i < TM; ++i)
                {
                    regM[wSubRowIdx * TM + i] = As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                }
            }
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
            {
                for (uint i = 0; i < TN; ++i)
                {
                    regN[wSubColIdx * TN + i] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
                }
            }

            // execute warptile matmul
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
            {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                {
                    // calculate per-thread results
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                    {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                        {
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] +=
                                regM[wSubRowIdx * TM + resIdxM] * regN[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }
    }

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <typename T, const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_kernel_warp_tiling(
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

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/2=32
    constexpr uint WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    // we cache into registers on the warptile level
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
    {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
        {
            // move C pointer to current warp subtile
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
            {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4)
                {
                    // load C vector into registers
                    float4 tmp = reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    // write back
                    reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

template <typename T>
cudaError_t SGEMM_Warp_Tiling_Impl(cudaStream_t stream,
                                   T const *A,
                                   T const *B,
                                   T *C,
                                   int32_t const M,
                                   int32_t const K,
                                   int32_t const N,
                                   T const alpha,
                                   T const beta)
{
    // Settings for A100
    // const uint K10_NUM_THREADS = 128;
    // const uint K10_BN = 128;
    // const uint K10_BM = 64;
    // const uint K10_BK = 16;
    // const uint K10_WN = 64;
    // const uint K10_WM = 32;
    // const uint K10_WNITER = 1;
    // const uint K10_TN = 4;
    // const uint K10_TM = 4;
    // Settings for A6000
    const uint K10_NUM_THREADS = 128;
    const uint K10_BN = 128;
    const uint K10_BM = 128;
    const uint K10_BK = 16;
    const uint K10_WN = 64;
    const uint K10_WM = 64;
    const uint K10_WNITER = 4;
    const uint K10_TN = 4;
    const uint K10_TM = 8;
    dim3 blockDim(K10_NUM_THREADS);

    constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

    // warptile in threadblocktile
    static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
    static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

    // threads in warpsubtile
    static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) == 0);
    constexpr uint K10_WMITER = (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
    // warpsubtile in warptile
    static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

    static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                  "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of Bs during each iteraion)");
    static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                  "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of As during each iteration)");
    static_assert(K10_BN % (16 * K10_TN) == 0, "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(K10_BM % (16 * K10_TM) == 0, "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0, "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0, "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
    sgemm_kernel_warp_tiling<T, K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS>
        <<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N, alpha, beta);
    return cudaGetLastError();
}

#define SPECIALIZED_IMPL(T) \
    template cudaError_t SGEMM_Warp_Tiling_Impl<T>(cudaStream_t stream, T const *A, T const *B, T *C, int32_t const M, int32_t const K, int32_t const N, T const alpha, T const beta);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(half)