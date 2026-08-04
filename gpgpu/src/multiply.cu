#include "multiply.cuh"
#include <stdio.h>

extern "C" __global__ void matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

#ifndef NO_MAIN
int main() {
    printf("multiply.cu standalone: kernels in multiply.cu\n");
    return 0;
}
#endif
