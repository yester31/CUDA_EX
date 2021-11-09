#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <chrono>
#include <vector>
#include <iostream>

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

// 0 - 100 사이 정수로 데이터 초기화
void generate_data(int* ptr, unsigned int size) {
	int tt = 0;
	while (size--) {
		*ptr++ = rand() % 100;
		//*ptr++ = tt++;
	}
}

void valid_results(std::vector<int> &gpu, std::vector<int> &cpu) {
	bool result = true;
	for (int i = 0; i < gpu.size(); i++) {
		if ((gpu[i]) != cpu[i]) {
			printf("[%d] The results is not matched! (%d, %d)\n", i, gpu[i], cpu[i]);
			result = false;
		}
	}
	if (result)printf("GPU works well! \n");
	else printf("GPU and CPU results is not matched! \n");
}