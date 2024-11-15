#include "../naive/sgemm.h"
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
    std::vector<float> matrix_c2(M * N);

    // generate random data
    generate_random_data(matrix_a.data(), matrix_a.size());
    generate_random_data(matrix_b.data(), matrix_b.size());
    generate_random_data(matrix_c.data(), matrix_c.size());

    cudaSetDevice(0);
    cudaStream_t stream;
    cublasHandle_t cublasHandle;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cublasCreate(&cublasHandle));
    CUDA_CHECK(cublasSetStream(cublasHandle, stream));

    // device-side data
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    float *dev_a2 = 0;
    float *dev_b2 = 0;
    float *dev_c2 = 0;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&dev_a, matrix_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, matrix_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c, matrix_c.size() * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void **)&dev_a2, matrix_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b2, matrix_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c2, matrix_c.size() * sizeof(float)));

    // warmup
    CUDA_CHECK(SGEMM_Naive_Impl<float>(stream, dev_a, dev_b, dev_c, M, K, N, alpha, beta));
    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, // B matrix not transposed
                CUBLAS_OP_N, // A matrix not transposed
                N,           // Number of columns of C and B
                M,           // Number of rows of C and A
                K,           // Number of columns of A, rows of B
                &alpha,      // Scalar alpha
                dev_b2,      // Matrix B
                N,           // Leading dimension of B
                dev_a2,      // Matrix A
                K,           // Leading dimension of A
                &beta,       // Scalar beta
                dev_c2,      // Matrix C
                N);          // Leading dimension of C
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy from host to device
    CUDA_CHECK(cudaMemcpy(dev_a, matrix_a.data(), matrix_a.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_a=a;
    CUDA_CHECK(cudaMemcpy(dev_b, matrix_b.data(), matrix_b.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c, matrix_c.data(), matrix_c.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_b=b;

    CUDA_CHECK(cudaMemcpy(dev_a2, matrix_a.data(), matrix_a.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_a=a;
    CUDA_CHECK(cudaMemcpy(dev_b2, matrix_b.data(), matrix_b.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c2, matrix_c.data(), matrix_c.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_b=b;

    // GPU sgemm
    Timer timer2("cublasSgemm");
    // row-major sgemm
    // ! cuBLAS uses column-major order.
    // So we change the order of our row-major A & B, since (B^T*A^T)^T = (A*B)
    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, // B matrix not transposed
                CUBLAS_OP_N, // A matrix not transposed
                N,           // Number of columns of C and B
                M,           // Number of rows of C and A
                K,           // Number of columns of A, rows of B
                &alpha,      // Scalar alpha
                dev_b2,      // Matrix B
                N,           // Leading dimension of B
                dev_a2,      // Matrix A
                K,           // Leading dimension of A
                &beta,       // Scalar beta
                dev_c2,      // Matrix C
                N);          // Leading dimension of C
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer2.Stop();

    // GPU sgemm
    Timer timer1("cuda naive sgemm");
    CUDA_CHECK(SGEMM_Naive_Impl<float>(stream, dev_a, dev_b, dev_c, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer1.Stop();

    // copy from device to host
    CUDA_CHECK(cudaMemcpy(matrix_c.data(), dev_c, matrix_c.size() * sizeof(float), cudaMemcpyDeviceToHost));    // c=dev_c;
    CUDA_CHECK(cudaMemcpy(matrix_c2.data(), dev_c2, matrix_c2.size() * sizeof(float), cudaMemcpyDeviceToHost)); // c=dev_c;

    // Compare CPU and GPU results
    Timer timer3("compare time");
    auto max_diff = get_max_diff(matrix_c, matrix_c2);
    timer3.Stop();
    if (max_diff == -1)
    {
        std::cerr << "Matrix size mismatch: " << matrix_c.size() << " vs " << matrix_c2.size() << std::endl;
        return -1;
    }

    const float tolerance = 1e-3f;
    std::cout << "Max difference between CPU and GPU results: " << max_diff << std::endl
              << (max_diff <= tolerance ? "MATCH" : "MISMATCH") << " (tolerance: " << tolerance << ")" << std::endl;

    // free device memory
    CUDA_CHECK(cudaFree(dev_c));
    CUDA_CHECK(cudaFree(dev_c2));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));

    // free stream
    CUDA_CHECK(cudaStreamDestroy(stream));
    cublasDestroy(cublasHandle);

    return 0;
}