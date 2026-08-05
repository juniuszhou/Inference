#include "multiply.cuh"

extern "C" __global__ void matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }

        // Synchronize threads in the block
        __syncthreads();

        // Synchronize threads in the warp
        __syncwarp();

        // after warp synchronization, we can use __shfl_sync to shuffle data between threads
        // __shfl_sync

        C[row * n + col] = sum;
    }
}
