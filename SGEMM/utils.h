#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdint.h>

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

// generate random data (0.0~1.0)
void generate_random_data(float *ptr, unsigned int size)
{
    int tt = size;
    while (size--)
    {
        *ptr++ = (float)rand() / RAND_MAX;
    }
};

void print_matrix(const float *matrix, int rows, int cols, const char *name)
{
    std::cout << "\n"
              << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++)
    {
        std::cout << "[ ";
        for (int j = 0; j < cols; j++)
        {
            printf("%7.2f ", matrix[i * cols + j]);
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
};

void sgemm_cpu(const float *A, const float *B, float *C,
               const int M, const int K, const int N,
               const float alpha = 1.0f, const float beta = 1.0f)
{
    // C = alpha * (A @ B) + beta * C
    // A[M, K] * B[K, N] = C[M, N]
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = alpha * sum + beta * C[m * N + n];
        }
    }
};

float get_max_diff(std::vector<float> matrix_1, std::vector<float> matrix_2)
{
    if (matrix_1.size() != matrix_2.size())
    {
        std::cerr << "Matrix sizes do not match" << std::endl;
        return -1;
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < matrix_1.size(); ++i)
    {
        float diff = std::abs(matrix_1[i] - matrix_2[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
};