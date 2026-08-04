// Test for multiply.cu kernel: matmul
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/multiply.cuh"

#define BLOCK_SIZE 16
#define EPSILON 1e-3

int test_matmul(int m, int k, int n) {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    for (int i = 0; i < m * k; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < k * n; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    memset(h_C, 0, size_C);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    matmul<<<grid, block>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    float *h_C_ref = (float*)malloc(size_C);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += h_A[i * k + p] * h_B[p * n + j];
            }
            h_C_ref[i * n + j] = sum;
        }
    }

    int passed = 1;
    int max_errors = 10;
    int errors = 0;
    for (int i = 0; i < m && errors < max_errors; i++) {
        for (int j = 0; j < n && errors < max_errors; j++) {
            float expected = h_C_ref[i * n + j];
            if (fabsf(h_C[i * n + j] - expected) > EPSILON) {
                if (errors == 0) {
                    printf("  First few errors (m=%d, k=%d, n=%d):\n", m, k, n);
                }
                printf("    C[%d][%d]: got %f, expected %f\n", i, j, h_C[i * n + j], expected);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("PASSED\n");
    } else {
        printf("FAILED (%d errors)\n", errors);
        passed = 0;
    }

    free(h_A); free(h_B); free(h_C);
    free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return passed;
}

int main() {
    printf("Testing multiply.cu matmul kernel...\n");

    printf("  test_matmul(64, 64, 64)... ");
    test_matmul(64, 64, 64);

    printf("  test_matmul(128, 32, 128)... ");
    test_matmul(128, 32, 128);

    printf("  test_matmul(32, 128, 32)... ");
    test_matmul(32, 128, 32);

    printf("multiply.cu tests complete.\n");
    return 0;
}
