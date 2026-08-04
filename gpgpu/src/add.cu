#include "add.cuh"
#include <stdio.h>

extern "C" __global__ void add(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void addSlice(float* a, float* b, float* c, int a_off, int b_off, int c_off, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[c_off + idx] = a[a_off + idx] + b[b_off + idx];
    }
}

#ifndef NO_MAIN
int main() {
    printf("add.cu standalone: kernels in add.cu\n");
    return 0;
}
#endif
