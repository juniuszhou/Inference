// Test for add.cu kernels: add and addSlice
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/add.cuh"

#define BLOCK_SIZE 256
#define EPSILON 1e-5

int test_add(int N) {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x);

    add<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_c[i] - expected) > EPSILON) {
            printf("  FAIL at idx %d: got %f, expected %f\n", i, h_c[i], expected);
            passed = 0;
            break;
        }
    }

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return passed;
}

int test_addSlice(int N) {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    int a_off = 10, b_off = 20, c_off = 5;
    int test_size = N - max(max(a_off, b_off), c_off) - 100;
    if (test_size <= 0) test_size = 100;

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    memset(h_a, 0, size);
    memset(h_b, 0, size);
    memset(h_c, 0, size);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((test_size + block.x - 1) / block.x);

    addSlice<<<grid, block>>>(d_a, d_b, d_c, a_off, b_off, c_off, test_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < test_size; i++) {
        float expected = h_a[a_off + i] + h_b[b_off + i];
        if (fabsf(h_c[c_off + i] - expected) > EPSILON) {
            printf("  FAIL at slice idx %d: got %f, expected %f\n", i, h_c[c_off + i], expected);
            passed = 0;
            break;
        }
    }

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return passed;
}

int main() {
    printf("Testing add.cu kernels...\n");

    printf("  test_add(1000000)... ");
    if (test_add(1000000)) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }

    printf("  test_addSlice(1000000)... ");
    if (test_addSlice(1000000)) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }

    printf("add.cu tests complete.\n");
    return 0;
}
