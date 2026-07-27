#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        // C[row * n + col] = sum; the tricky part is here.
        // there are two dimensions here. so row * n is the start index of the row in the C matrix.
        C[row * n + col] = sum;
    }
}
