#include "sgemm.h"
#include <iostream>
#include <vector>

void generate_data_f(float *ptr, unsigned int size, int offset = 255)
{
    int tt = size;
    while (size--)
    {
        *ptr++ = (rand() % offset) / 100.0f;
    }
}

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
}

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
}

int main(void)
{
    // GEMM
    // C = α * (A @ B) + β *C
    // C[M, N] = alpha * (A[M, K] @ B[K, N]) + beta * C[M, N]

    // GEMM parameters
    const int M = 3;
    const int K = 4;
    const int N = 5;
    const float alpha = 1.0f;
    const float beta = 1.0f;

    // matrix initialization
    std::vector<float> matrix_a(M * K);
    std::vector<float> matrix_b(K * N);
    std::vector<float> matrix_c(M * N);
    std::vector<float> matrix_c_cpu(M * N);

    // generate random data
    generate_data_f(matrix_a.data(), matrix_a.size());
    generate_data_f(matrix_b.data(), matrix_b.size());
    generate_data_f(matrix_c.data(), matrix_c.size());
    matrix_c_cpu = matrix_c;

    // Print the matrix matrices
    // print_matrix(matrix_a.data(), M, K, "Matrix A");
    // print_matrix(matrix_b.data(), K, N, "Matrix B");
    // print_matrix(matrix_c.data(), M, N, "Matrix C");
    // print_matrix(matrix_c_cpu.data(), M, N, "Matrix C_cpu");

    // CPU sgemm
    Timer timer("cpu sgemm");
    sgemm_cpu(matrix_a.data(), matrix_b.data(), matrix_c_cpu.data(), M, K, N, alpha, beta);
    timer.Stop();
    // device-side data
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&dev_a, matrix_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, matrix_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c, matrix_c.size() * sizeof(float)));

    // copy from host to device
    CUDA_CHECK(cudaMemcpy(dev_a, matrix_a.data(), matrix_a.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_a=a;
    CUDA_CHECK(cudaMemcpy(dev_b, matrix_b.data(), matrix_b.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_b=b;

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // GPU sgemm
    Timer timer2("gpu sgemm");
    CUDA_CHECK(
        SGEMM_Naive_Impl<float>(
            stream,
            dev_a,
            dev_b,
            dev_c,
            M,
            K,
            N,
            alpha,
            beta));

    // Synchronize and destroy stream
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer2.Stop();

    // copy from device to host
    CUDA_CHECK(cudaMemcpy(matrix_c.data(), dev_c, matrix_c.size() * sizeof(float), cudaMemcpyDeviceToHost)); // c=dev_c;

    // Compare CPU and GPU results
    float max_diff = 0.0f;
    for (size_t i = 0; i < matrix_c.size(); ++i)
    {
        float diff = std::abs(matrix_c[i] - matrix_c_cpu[i]);
        max_diff = std::max(max_diff, diff);
    }
    std::cout << "Max difference between CPU and GPU results: " << max_diff << std::endl;

    print_matrix(matrix_c_cpu.data(), M, N, "cpu sgemm C");
    print_matrix(matrix_c.data(), M, N, "gpu sgemm C");

    // free device memory
    CUDA_CHECK(cudaFree(dev_c));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));

    // free stream
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}