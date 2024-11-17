#include "../timer.h"
#include "../utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(void)
{
    // GEMM
    // C = α * (A @ B) + β *C
    // C[M, N] = alpha * (A[M, K] @ B[K, N]) + beta * C[M, N]

    // GEMM parameters
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;
    const float alpha = 3.0f;
    const float beta = 2.0f;

    // matrix initialization
    std::vector<float> matrix_a(M * K);
    std::vector<float> matrix_b(K * N);
    std::vector<float> matrix_c(M * N);
    std::vector<float> matrix_c_cpu(M * N);

    // generate random data
    generate_random_data(matrix_a.data(), matrix_a.size());
    generate_random_data(matrix_b.data(), matrix_b.size());
    generate_random_data(matrix_c.data(), matrix_c.size());
    matrix_c_cpu = matrix_c;

    // CPU sgemm
    Timer timer("cpu sgemm");
    sgemm_cpu(matrix_a.data(), matrix_b.data(), matrix_c_cpu.data(), M, K, N, alpha, beta);
    timer.Stop();

    cudaSetDevice(0);
    cudaStream_t stream;
    cublasHandle_t cublasHandle;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));

    // device-side data
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&dev_a, matrix_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, matrix_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c, matrix_c.size() * sizeof(float)));

    // warmup
    CUBLAS_CHECK(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K, &alpha, dev_b, CUDA_R_32F,
                              N, dev_a, CUDA_R_32F, K,
                              &beta, dev_c, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy from host to device
    CUDA_CHECK(cudaMemcpy(dev_a, matrix_a.data(), matrix_a.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_a=a;
    CUDA_CHECK(cudaMemcpy(dev_b, matrix_b.data(), matrix_b.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c, matrix_c.data(), matrix_c.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_c=c;

    // GPU sgemm
    Timer timer2("cublass cublasSgemm");
    // row-major sgemm
    // ! cuBLAS uses column-major order.
    // So we change the order of our row-major A & B, since (B^T*A^T)^T = (A*B)
    CUBLAS_CHECK(cublasSgemm(cublasHandle,
                             CUBLAS_OP_N, // B matrix not transposed
                             CUBLAS_OP_N, // A matrix not transposed
                             N,           // Number of columns of C and B
                             M,           // Number of rows of C and A
                             K,           // Number of columns of A, rows of B
                             &alpha,      // Scalar alpha
                             dev_b,       // Matrix B
                             N,           // Leading dimension of B
                             dev_a,       // Matrix A
                             K,           // Leading dimension of A
                             &beta,       // Scalar beta
                             dev_c,       // Matrix C
                             N));         // Leading dimension of C
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer2.Stop();

    // copy from device to host
    CUDA_CHECK(cudaMemcpy(matrix_c.data(), dev_c, matrix_c.size() * sizeof(float), cudaMemcpyDeviceToHost)); // c=dev_c;

    // Compare CPU and GPU results
    get_max_diff(matrix_c, matrix_c_cpu);

    // free device memory
    CUDA_CHECK(cudaFree(dev_c));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));

    // free stream
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    return 0;
}