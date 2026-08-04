// Test for basic.cu kernels: add_global, add_device, add_host, add_host_device
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/basic.cuh"

#define BLOCK_SIZE 256
#define EPSILON 1e-5

int test_add_global(int N) {
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

    add_global<<<grid, block>>>(d_a, d_b, d_c, N);
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

int test_add_host(int N) {
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    add_host(h_a, h_b, h_c, N);

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

    return passed;
}

int test_add_host_device(int N) {
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    add_host_device(h_a, h_b, h_c, N);

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

    return passed;
}

int main() {
    printf("Testing basic.cu kernels...\n");

    printf("  test_add_global(1000000)... ");
    if (test_add_global(1000000)) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }

    printf("  test_add_host(1000000)... ");
    if (test_add_host(1000000)) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }

    printf("  test_add_host_device(1000000)... ");
    if (test_add_host_device(1000000)) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }

    printf("basic.cu tests complete.\n");
    return 0;
}
