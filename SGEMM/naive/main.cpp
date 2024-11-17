#include "sgemm0.h"
#include "../timer.h"
#include "../utils.h"

int main(void)
{
    // GEMM
    // C = α * (A @ B) + β *C
    // C[M, N] = alpha * (A[M, K] @ B[K, N]) + beta * C[M, N]

    // GEMM parameters
    const int M = 512;
    const int K = 512;
    const int N = 512;
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
    CUDA_CHECK(cudaStreamCreate(&stream));

    // device-side data
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&dev_a, matrix_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, matrix_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c, matrix_c.size() * sizeof(float)));

    // warmup
    CUDA_CHECK(SGEMM_Naive_Impl<float>(stream, dev_a, dev_b, dev_c, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy from host to device
    CUDA_CHECK(cudaMemcpy(dev_a, matrix_a.data(), matrix_a.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_a=a;
    CUDA_CHECK(cudaMemcpy(dev_b, matrix_b.data(), matrix_b.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_b=b;
    CUDA_CHECK(cudaMemcpy(dev_c, matrix_c.data(), matrix_c.size() * sizeof(float), cudaMemcpyHostToDevice)); // dev_b=b;

    // GPU sgemm
    Timer timer2("gpu sgemm");
    CUDA_CHECK(SGEMM_Naive_Impl<float>(stream, dev_a, dev_b, dev_c, M, K, N, alpha, beta));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer2.Stop();

    // copy from device to host
    CUDA_CHECK(cudaMemcpy(matrix_c.data(), dev_c, matrix_c.size() * sizeof(float), cudaMemcpyDeviceToHost)); // c=dev_c;

    // Compare CPU and GPU results
    Timer timer3("compare");
    auto max_diff = get_max_diff(matrix_c, matrix_c_cpu);
    timer3.Stop();
    if (max_diff == -1)
    {
        std::cerr << "Matrix size mismatch: " << matrix_c.size() << " vs " << matrix_c_cpu.size() << std::endl;
        return -1;
    }

    const float tolerance = 1e-2f;
    std::cout << "Max difference between CPU and GPU results: " << max_diff << std::endl
              << (max_diff <= tolerance ? "MATCH" : "MISMATCH") << " (tolerance: " << tolerance << ")" << std::endl;

    // free device memory
    CUDA_CHECK(cudaFree(dev_c));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));

    // free stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}