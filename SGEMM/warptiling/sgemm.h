#ifndef SGEMM_WARP_TILING_H
#define SGEMM_WARP_TILING_H

#include <cuda_runtime.h>
#include <stdint.h>

template <typename T>
cudaError_t SGEMM_Warp_Tiling_Impl(
    cudaStream_t stream,
    T const *A,
    T const *B,
    T *C,
    int32_t const M,
    int32_t const K,
    int32_t const N,
    T const alpha,
    T const beta);

#endif // SGEMM_WARP_TILING_H
