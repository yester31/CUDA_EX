#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

// ERROR CHECK
#if defined(NDEBUG)     //release mode
#define CUDA_CHECK(x) (x)   
#else                   // debug mode
//error check 
#define CUDA_CHECK(x)   do{\
    (x); \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("cuda failure %s at %s:%d \n", \
        cudaGetErrorString(e), \
            __FILE__, __LINE__); \
        exit(0); \
    } \
}while(0)
#endif


void transpose_naive_gpu(std::vector<float>& output_gpu, std::vector<float>& input, int H, int W);

void transpose_tiled_gpu(std::vector<float>& output_gpu, std::vector<float>& input, int H, int W);

void transpose_tiled_coalesced_gpu(std::vector<float>& output_gpu, std::vector<float>& input, int H, int W);

void transpose_tiled_coalesced_padded_gpu(std::vector<float>& output_gpu, std::vector<float>& input, int H, int W);