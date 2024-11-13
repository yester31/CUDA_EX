#ifndef SGEMM_KERNEL_H
#define SGEMM_KERNEL_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <chrono>
#include <string>
#include <iostream>

template <typename T>
cudaError_t SGEMM_Naive_Impl(
    cudaStream_t stream,
    T const *A,
    T const *B,
    T *C,
    int32_t const M,
    int32_t const K,
    int32_t const N,
    T const alpha,
    T const beta);

class Timer
{
public:
    Timer(const std::string &name = "Timer") : name_(name)
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        // Stop();
    }

    void Stop()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cout << name_ << ": " << duration.count() / 1000.0f << " ms" << std::endl;
    }

    float ElapsedMilliseconds() const
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0f;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// ERROR CHECK
#if defined(NDEBUG) // release mode
#define CUDA_CHECK(x) (x)
#else // debug mode
// error check
#define CUDA_CHECK(x)                             \
    do                                            \
    {                                             \
        (x);                                      \
        cudaError_t e = cudaGetLastError();       \
        if (e != cudaSuccess)                     \
        {                                         \
            printf("cuda failure %s at %s:%d \n", \
                   cudaGetErrorString(e),         \
                   __FILE__, __LINE__);           \
            exit(0);                              \
        }                                         \
    } while (0)
#endif

#endif // SGEMM_KERNEL_H