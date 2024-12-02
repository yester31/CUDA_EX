#include "../sgemm.h"
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
    const int M = 4097;
    const int K = 4096;
    const int N = 4096;
    const float alpha = 1.0f;
    const float beta = 2.0f;
    std::cout << "GEMM Parameters" << std::endl;
    std::cout << "M: " << M << ", K: " << K << ", N: " << N << ", alpha: " << alpha << ", beta: " << beta << std::endl;
    std::cout << "C[" << M << "," << N << "] = " << alpha << " * (A[" << M << "," << K << "] @ B[" << K << "," << N << "]) + " << beta << " * C[" << M << "," << N << "]" << std::endl;

    // matrix initialization
    std::vector<float> matrix_a(M * K);
    std::vector<float> matrix_b(K * N);
    std::vector<float> matrix_c_cublas(M * N);
    std::vector<float> matrix_c0(M * N);
    std::vector<float> matrix_c1(M * N);
    std::vector<float> matrix_c2(M * N);
    std::vector<float> matrix_c3(M * N);
    std::vector<float> matrix_c4(M * N);
    std::vector<float> matrix_c5(M * N);
    std::vector<float> matrix_c6(M * N);
    std::vector<float> matrix_c7(M * N);
    std::vector<float> matrix_c8(M * N);
    // generate random data
    generate_random_data(matrix_a.data(), matrix_a.size());
    generate_random_data(matrix_b.data(), matrix_b.size());
    generate_random_data(matrix_c_cublas.data(), matrix_c_cublas.size());

    cudaSetDevice(0);
    cudaStream_t stream;
    cublasHandle_t cublasHandle;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));

    // device-side data
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c0 = 0;
    float *dev_c1 = 0;
    float *dev_c2 = 0;
    float *dev_c3 = 0;
    float *dev_c4 = 0;
    float *dev_c5 = 0;
    float *dev_c6 = 0;
    float *dev_c7 = 0;
    float *dev_c8 = 0;
    float *dev_c_cublas = 0;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&dev_a, matrix_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, matrix_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c0, matrix_c0.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c1, matrix_c1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c2, matrix_c2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c3, matrix_c3.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c4, matrix_c4.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c5, matrix_c5.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c6, matrix_c6.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c7, matrix_c7.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c8, matrix_c8.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c_cublas, matrix_c_cublas.size() * sizeof(float)));

    // copy from host to device
    CUDA_CHECK(cudaMemcpy(dev_a, matrix_a.data(), matrix_a.size() * sizeof(float), cudaMemcpyHostToDevice));                      // dev_a=a;
    CUDA_CHECK(cudaMemcpy(dev_b, matrix_b.data(), matrix_b.size() * sizeof(float), cudaMemcpyHostToDevice));                      // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c0, matrix_c_cublas.data(), matrix_c0.size() * sizeof(float), cudaMemcpyHostToDevice));             // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c1, matrix_c_cublas.data(), matrix_c1.size() * sizeof(float), cudaMemcpyHostToDevice));             // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c2, matrix_c_cublas.data(), matrix_c2.size() * sizeof(float), cudaMemcpyHostToDevice));             // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c3, matrix_c_cublas.data(), matrix_c3.size() * sizeof(float), cudaMemcpyHostToDevice));             // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c4, matrix_c_cublas.data(), matrix_c4.size() * sizeof(float), cudaMemcpyHostToDevice));             // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c5, matrix_c_cublas.data(), matrix_c5.size() * sizeof(float), cudaMemcpyHostToDevice));             // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c6, matrix_c_cublas.data(), matrix_c6.size() * sizeof(float), cudaMemcpyHostToDevice));             // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c7, matrix_c_cublas.data(), matrix_c7.size() * sizeof(float), cudaMemcpyHostToDevice));             // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c8, matrix_c_cublas.data(), matrix_c8.size() * sizeof(float), cudaMemcpyHostToDevice));             // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c_cublas, matrix_c_cublas.data(), matrix_c_cublas.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_b=b;

    // warmup & verify
    CUDA_CHECK(SGEMM_Naive_Impl<float>(stream, dev_a, dev_b, dev_c0, M, K, N, alpha, beta));
    CUDA_CHECK(SGEMM_Global_Memory_Coalescing_Impl<float>(stream, dev_a, dev_b, dev_c1, M, K, N, alpha, beta));
    CUDA_CHECK(SGEMM_Shared_Memory_Impl<float>(stream, dev_a, dev_b, dev_c2, M, K, N, alpha, beta));
    CUDA_CHECK(SGEMM_Shared_Memory_WO_Bank_Conflict_Impl<float>(stream, dev_a, dev_b, dev_c3, M, K, N, alpha, beta));
    CUDA_CHECK(SGEMM_Shared_Memory_GMC_Impl<float>(stream, dev_a, dev_b, dev_c4, M, K, N, alpha, beta));
    CUDA_CHECK(SGEMM_Shared_Memory_1_Impl<float>(stream, dev_a, dev_b, dev_c5, M, K, N, alpha, beta));
    CUDA_CHECK(SGEMM_1D_Block_Tiling_Impl<float>(stream, dev_a, dev_b, dev_c6, M, K, N, alpha, beta));
    CUDA_CHECK(SGEMM_2D_Block_Tiling_Impl<float>(stream, dev_a, dev_b, dev_c7, M, K, N, alpha, beta));
    CUDA_CHECK(SGEMM_Vectorized_Mem_Access_Impl<float>(stream, dev_a, dev_b, dev_c8, M, K, N, alpha, beta));
    CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dev_b, N, dev_a, K, &beta, dev_c_cublas, N));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy from device to host
    CUDA_CHECK(cudaMemcpy(matrix_c0.data(), dev_c0, matrix_c0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(matrix_c1.data(), dev_c1, matrix_c1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(matrix_c2.data(), dev_c2, matrix_c2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(matrix_c3.data(), dev_c3, matrix_c3.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(matrix_c4.data(), dev_c4, matrix_c4.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(matrix_c5.data(), dev_c5, matrix_c5.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(matrix_c6.data(), dev_c6, matrix_c6.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(matrix_c7.data(), dev_c7, matrix_c7.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(matrix_c8.data(), dev_c8, matrix_c8.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(matrix_c_cublas.data(), dev_c_cublas, matrix_c_cublas.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare two results results
    std::cout << "[Accuracy Verification]" << std::endl;
    get_max_diff(matrix_c_cublas, matrix_c0); // naive
    get_max_diff(matrix_c_cublas, matrix_c1); // GMC
    get_max_diff(matrix_c_cublas, matrix_c2); // shared memory
    get_max_diff(matrix_c_cublas, matrix_c3); // shared memory wo bank conflict
    get_max_diff(matrix_c_cublas, matrix_c4); // shared memory GMC
    get_max_diff(matrix_c_cublas, matrix_c5); // shared memory 1
    get_max_diff(matrix_c_cublas, matrix_c6); // 1d block tiling
    get_max_diff(matrix_c_cublas, matrix_c7); // 2d block tiling
    get_max_diff(matrix_c_cublas, matrix_c8); // Vectorized_Mem_Access
    std::cout << "[Comparison of Latency]" << std::endl;

    // GPU sgemm
    Timer timer1("=> cublas sgemm");
    // row-major sgemm
    // ! cuBLAS uses column-major order.
    // So we change the order of our row-major A & B, since (B^T*A^T)^T = (A*B)
    CUBLAS_CHECK(cublasSgemm(cublasHandle,
                             CUBLAS_OP_N,  // B matrix not transposed
                             CUBLAS_OP_N,  // A matrix not transposed
                             N,            // Number of columns of C and B
                             M,            // Number of rows of C and A
                             K,            // Number of columns of A, rows of B
                             &alpha,       // Scalar alpha
                             dev_b,        // Matrix B
                             N,            // Leading dimension of B
                             dev_a,        // Matrix A
                             K,            // Leading dimension of A
                             &beta,        // Scalar beta
                             dev_c_cublas, // Matrix C
                             N));          // Leading dimension of C
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer1.Stop();

    // GPU sgemm
    Timer timer2("=> naive sgemm");
    CUDA_CHECK(SGEMM_Naive_Impl<float>(stream, dev_a, dev_b, dev_c0, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer2.Stop();

    // GPU sgemm
    Timer timer3("=> global memory coalescing sgemm");
    CUDA_CHECK(SGEMM_Global_Memory_Coalescing_Impl<float>(stream, dev_a, dev_b, dev_c1, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer3.Stop();

    // GPU sgemm
    Timer timer4("=> shared memory w bank conflict sgemm");
    CUDA_CHECK(SGEMM_Shared_Memory_Impl<float>(stream, dev_a, dev_b, dev_c2, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer4.Stop();

    // GPU sgemm
    Timer timer5("=> shared memory wo bank conflict sgemm");
    CUDA_CHECK(SGEMM_Shared_Memory_WO_Bank_Conflict_Impl<float>(stream, dev_a, dev_b, dev_c3, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer5.Stop();

    // GPU sgemm
    Timer timer6("=> shared memory GMC sgemm");
    CUDA_CHECK(SGEMM_Shared_Memory_GMC_Impl<float>(stream, dev_a, dev_b, dev_c4, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer6.Stop();

    // GPU sgemm
    Timer timer7("=> shared memory 1 sgemm");
    CUDA_CHECK(SGEMM_Shared_Memory_1_Impl<float>(stream, dev_a, dev_b, dev_c5, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer7.Stop();

    // GPU sgemm
    Timer timer8("=> 1d block tiling sgemm");
    CUDA_CHECK(SGEMM_1D_Block_Tiling_Impl<float>(stream, dev_a, dev_b, dev_c6, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer8.Stop();

    // GPU sgemm
    Timer timer9("=> 2d block tiling sgemm");
    CUDA_CHECK(SGEMM_2D_Block_Tiling_Impl<float>(stream, dev_a, dev_b, dev_c7, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer9.Stop();

    // GPU sgemm
    Timer timer10("=> Vectorized_Mem_Access sgemm");
    CUDA_CHECK(SGEMM_Vectorized_Mem_Access_Impl<float>(stream, dev_a, dev_b, dev_c8, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer10.Stop();

    // free device memory
    CUDA_CHECK(cudaFree(dev_c0));
    CUDA_CHECK(cudaFree(dev_c1));
    CUDA_CHECK(cudaFree(dev_c2));
    CUDA_CHECK(cudaFree(dev_c3));
    CUDA_CHECK(cudaFree(dev_c4));
    CUDA_CHECK(cudaFree(dev_c5));
    CUDA_CHECK(cudaFree(dev_c6));
    CUDA_CHECK(cudaFree(dev_c7));
    CUDA_CHECK(cudaFree(dev_c8));
    CUDA_CHECK(cudaFree(dev_c_cublas));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));

    // free stream
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasHandle));

    return 0;
}