#ifndef MAIN_CUH
#define MAIN_CUH

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add1D(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

#endif // MAIN_CUH
