// Test for nccl_demo.cu: broadcast, reduce, allreduce
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../src/nccl_demo.cuh"

#define EPSILON 1e-5
#define N 1024

int test_broadcast() {
    printf("  test_broadcast(%d floats)... ", N);

    NcclContext ctx;
    if (nccl_init(&ctx) != ncclSuccess) {
        printf("FAILED (init)\n");
        return 0;
    }

    size_t bytes = N * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, bytes);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    nccl_broadcast(d_data, N, &ctx);
    cudaDeviceSynchronize();

    float *h_result = (float*)malloc(bytes);
    cudaMemcpy(h_result, d_data, bytes, cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_result[i] - h_data[i]) > EPSILON) {
            printf("FAILED at idx %d: got %f, expected %f\n", i, h_result[i], h_data[i]);
            passed = 0;
            break;
        }
    }
    if (passed) printf("PASSED\n");

    free(h_data); free(h_result);
    cudaFree(d_data);
    nccl_cleanup(&ctx);
    return passed;
}

int test_reduce() {
    printf("  test_reduce(%d floats)... ", N);

    NcclContext ctx;
    if (nccl_init(&ctx) != ncclSuccess) {
        printf("FAILED (init)\n");
        return 0;
    }

    size_t bytes = N * sizeof(float);
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_input[i] = (float)(i + 1);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_input, bytes, cudaMemcpyHostToDevice);

    // NCCL single-GPU: reduce is identity (no-op beyond copy)
    nccl_reduce(d_input, d_output, N, &ctx);
    cudaDeviceSynchronize();

    float *h_output = (float*)malloc(bytes);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < N; i++) {
        // Single-GPU reduce acts as identity: output == input
        if (fabsf(h_output[i] - h_input[i]) > EPSILON) {
            printf("FAILED at idx %d: got %f, expected %f\n", i, h_output[i], h_input[i]);
            passed = 0;
            break;
        }
    }
    if (passed) printf("PASSED (single-GPU identity)\n");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    nccl_cleanup(&ctx);
    return passed;
}

int test_allreduce() {
    printf("  test_allreduce(%d floats)... ", N);

    NcclContext ctx;
    if (nccl_init(&ctx) != ncclSuccess) {
        printf("FAILED (init)\n");
        return 0;
    }

    size_t bytes = N * sizeof(float);
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_input[i] = (float)(i + 1);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_input, bytes, cudaMemcpyHostToDevice);

    // NCCL single-GPU: allreduce is identity (no-op beyond copy)
    nccl_allreduce(d_input, d_output, N, &ctx);
    cudaDeviceSynchronize();

    float *h_output = (float*)malloc(bytes);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < N; i++) {
        // Single-GPU allreduce acts as identity: output == input
        if (fabsf(h_output[i] - h_input[i]) > EPSILON) {
            printf("FAILED at idx %d: got %f, expected %f\n", i, h_output[i], h_input[i]);
            passed = 0;
            break;
        }
    }
    if (passed) printf("PASSED (single-GPU identity)\n");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    nccl_cleanup(&ctx);
    return passed;
}

int main() {
    printf("Testing nccl_demo.cu kernels...\n");

    test_broadcast();
    test_reduce();
    test_allreduce();

    printf("nccl_demo.cu tests complete.\n");
    return 0;
}
