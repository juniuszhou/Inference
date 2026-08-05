#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUBLAS(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

void test_gemm() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // ===== 2. 准备数据 =====
    const int M = 2, N = 3, K = 4;   // C(MxN) = A(MxK) * B(KxN)
    float alpha = 1.0f, beta = 0.0f;

    // 主机数据（列优先存储，cuBLAS 默认是列优先！）
    float h_A[M*K] = {1, 2, 3, 4, 5, 6, 7, 8};   // 示例数据
    float h_B[K*N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float h_C[M*N] = {0};

    // 设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    // 拷贝到 GPU
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // ===== 3. 调用 cuBLAS 函数 =====
    // C = alpha * A * B + beta * C
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,   // 不转置
        M, N, K,                    // 矩阵维度
        &alpha,
        d_A, M,                     // A, lda = M（列优先）
        d_B, K,                     // B, ldb = K
        &beta,
        d_C, M                      // C, ldc = M
    ));

    // ===== 4. 结果拷回并清理 =====
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("结果矩阵 C:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%8.2f ", h_C[j * M + i]);  // 列优先读取
        }
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    CHECK_CUBLAS(cublasDestroy(handle));
}

int main() {
    printf("Testing cublas SGEMM...\n");
    test_gemm();
    printf("cublas test complete.\n");
    return 0;
}