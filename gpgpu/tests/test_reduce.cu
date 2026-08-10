// Test for reduce.cu kernel: reduceMax (parallel max reduction)
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "../src/reduce.cuh"

#define BLOCK_SIZE 256

/*
    Host-side helper that drives reduceMax to a single final value.

    reduceMax emits ONE max per block, so with a big array we launch it
    repeatedly: each launch shrinks the problem by blockDim.x. Two buffers
    alternate as source/destination so a kernel never reads a slot that another
    block is about to overwrite in the same pass.
*/
int reduceMax_host(const float* h_in, int n, float* h_out) {
    float *d_a, *d_buf;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_buf, n * sizeof(float));
    cudaMemcpy(d_a, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    float* src = d_a;
    float* dst = d_buf;
    int elems = n;
    while (elems > BLOCK_SIZE) {
        int grid = (elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduceMax<<<grid, BLOCK_SIZE>>>(src, dst, elems);
        cudaDeviceSynchronize();
        elems = grid;

        float* tmp = src; src = dst; dst = tmp; // swap roles for next pass
    }

    // Final pass: one block over the remaining partials -> result in dst[0].
    reduceMax<<<1, BLOCK_SIZE>>>(src, dst, elems);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, dst, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_buf);
    return 0;
}

float reference_max(const float* a, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (a[i] > m) m = a[i];
    }
    return m;
}

// Verifies reduceMax2D against a host reference for an M x N matrix.
int test_reduce2D(int M, int N, unsigned seed) {
    int len = M * N;
    float* h_a = (float*)malloc(len * sizeof(float));
    float* h_c = (float*)malloc(M * sizeof(float));
    float* h_ref = (float*)malloc(M * sizeof(float));

    srand(seed);
    for (int i = 0; i < len; i++) {
        h_a[i] = (float)(rand() % 2000) / 3.0f - 100.0f; // include negatives
    }
    for (int r = 0; r < M; r++) {
        h_ref[r] = reference_max(h_a + r * N, N);
    }

    float *d_a, *d_c;
    cudaMalloc(&d_a, len * sizeof(float));
    cudaMalloc(&d_c, M * sizeof(float));
    cudaMemcpy(d_a, h_a, len * sizeof(float), cudaMemcpyHostToDevice);

    reduceMax2D<<<M, BLOCK_SIZE>>>(d_a, d_c, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, M * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int r = 0; r < M; r++) {
        if (h_c[r] != h_ref[r]) {
            printf("  FAIL row %d: got %f, expected %f\n", r, h_c[r], h_ref[r]);
            passed = 0;
            break;
        }
    }

    free(h_a); free(h_c); free(h_ref);
    cudaFree(d_a); cudaFree(d_c);
    return passed;
}

int test_reduce_rand(int n, unsigned seed) {
    float* h_a = (float*)malloc(n * sizeof(float));
    srand(seed);
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)(rand() % 1000) / 7.0f; // non-trivial floats, incl. repeats
    }

    float got, expected = reference_max(h_a, n);
    reduceMax_host(h_a, n, &got);

    free(h_a);

    if (got != expected) {
        printf("  FAIL n=%d (seed=%u): got %f, expected %f\n", n, seed, got, expected);
        return 0;
    }
    return 1;
}

int test_reduce_fixed() {
    // Known-value cases: negatives, max in the middle, single element, size 1..N.
    float a1[] = {-3.0f, 1.5f, 7.25f, -0.5f, 4.0f};
    float a2[] = {-5.0f, -4.0f, -9.0f, -1.0f}; // all negative
    float a3[] = {42.0f};                      // single element
    float a4[] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f}; // max in the middle

    struct {
        float* data;
        int n;
        float expect;
    } cases[] = {
        {a1, 5, 7.25f},
        {a2, 4, -1.0f},
        {a3, 1, 42.0f},
        {a4, 5, 3.0f},
    };

    int passed = 1;
    for (int i = 0; i < 4; i++) {
        float got;
        reduceMax_host(cases[i].data, cases[i].n, &got);
        if (got != cases[i].expect) {
            printf("  FAIL fixed case %d: got %f, expected %f\n", i, got, cases[i].expect);
            passed = 0;
        }
    }
    return passed;
}

int main() {
    printf("Testing reduce.cu kernel...\n");

    int failed = 0;

    struct { int n; unsigned seed; } cases[] = {
        {1, 1}, {256, 2}, {257, 3}, {1024, 4}, {4096, 5}, {1000000, 6}, {1000003, 7},
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);

    for (int i = 0; i < n_cases; i++) {
        printf("  test_reduce_rand(n=%d)... ", cases[i].n);
        if (test_reduce_rand(cases[i].n, cases[i].seed)) {
            printf("PASSED\n");
        } else {
            printf("FAILED\n");
            failed++;
        }
    }

    printf("  test_reduce_fixed()... ");
    if (test_reduce_fixed()) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
        failed++;
    }

    struct { int M, N, seed; } cases2d[] = {
        {1, 1, 11}, {10, 10, 12}, {3, 300, 13}, {300, 3, 14},
        {100, 1000, 15}, {128, 500, 16}, {1000, 42, 17}, {16, 100000, 18},
    };
    int n2d = sizeof(cases2d) / sizeof(cases2d[0]);
    for (int i = 0; i < n2d; i++) {
        printf("  test_reduce2D(M=%d, N=%d)... ", cases2d[i].M, cases2d[i].N);
        if (test_reduce2D(cases2d[i].M, cases2d[i].N, cases2d[i].seed)) {
            printf("PASSED\n");
        } else {
            printf("FAILED\n");
            failed++;
        }
    }

    if (failed) {
        printf("reduce.cu tests FAILED (%d failed)\n", failed);
        return 1;
    }
    printf("reduce.cu tests complete.\n");
    return 0;
}
