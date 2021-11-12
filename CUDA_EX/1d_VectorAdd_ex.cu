#include "util_cuda.cuh"

using namespace std;
using namespace chrono;
#define NUM_DATA 10240000
#define BLOCK_SIZE 1024

// kernel 함수
__global__ void vecAdd(int *_a, int *_b, int *_c) {
	int tID = blockIdx.x * blockDim.x + threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int main(void) {

	// start the timer
	uint64_t total_time = 0;

	// 포인터 변수 선언
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	// 할당할 메모리공간 사이즈 계산 
	int memSize = sizeof(int) * NUM_DATA;
	printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

	// 메모리 공간을 생성후 그 주소를 메모리 포인터에 할당 후 메모리의 내용을 0으로 memSize 크기만큼 세팅
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);

	// 할당한 메모리 공간에 연산에 사용할 데이터 할당
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Device에 memSize 만큼의 공간을 생성 후 포인터 변수에 Device에서 할당된 공간의 주소를 전달
	cudaMalloc(&d_a, memSize);
	cudaMalloc(&d_b, memSize);
	cudaMalloc(&d_c, memSize);

	uint64_t start_time1 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

	// Host -> Device 데이터 전달 (data transfer overhead)
	cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice); // 동기로 작동함.
	cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE);
	dim3 grid((NUM_DATA + block.x - 1) / block.x);

	uint64_t start_time2 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

	// cuda kernel 함수 호출
	vecAdd << <grid, block >> > (d_a, d_b, d_c);
	cudaDeviceSynchronize();

	uint64_t start_time3 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

	// Device -> Host 데이터 전달 (data transfer overhead)
	cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

	//end the timer
	uint64_t start_time4 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

	printf("dur_time(gpu) w = %6.3f [msec] \n", (start_time4 - start_time1) / 1000.f);
	printf("dur_time(gpu) wo = %6.3f [msec] \n", (start_time3 - start_time2) / 1000.f);

	// 결과 검증
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if ((a[i] + b[i]) != c[i]) {
			printf("[%d] The results is not matched! (%d, %d)\n", i, a[i] + b[i], c[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well! \n");

	uint64_t start_time5 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

	for (int i = 0; i < NUM_DATA; i++) {
		c[i] = a[i] + b[i];
	}
	//end the timer
	uint64_t start_time6 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	printf("dur_time(cpu) = %6.3f [msec] \n", (start_time6 - start_time5) / 1000.f);

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	delete[] a;	delete[] b;	delete[] c;
}