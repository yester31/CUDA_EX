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

void reduction_0_gpu(std::vector<int>& output_gpu, std::vector<int>& input);
void reduction_1_gpu(std::vector<int>& output_gpu, std::vector<int>& input);
void reduction_2_gpu(std::vector<int>& output_gpu, std::vector<int>& input);
void reduction_3_gpu(std::vector<int>& output_gpu, std::vector<int>& input);
void reduction_4_gpu(std::vector<int>& output_gpu, std::vector<int>& input);