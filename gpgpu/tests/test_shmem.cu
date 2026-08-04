// Test for shmem.cu kernel: cuda_put_kernel
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declaration (implementation in src/shmem.cu)
__global__ void cuda_put_kernel(int* my_buffer, int* neighbor_buffer);

int main() {
    printf("Testing shmem.cu cuda_put_kernel...\n");

    int *d_buffer0, *d_buffer1;
    int h_data0 = 0, h_data1 = 0;
    int h_result0, h_result1;

    cudaMalloc(&d_buffer0, sizeof(int));
    cudaMalloc(&d_buffer1, sizeof(int));

    cudaMemcpy(d_buffer0, &h_data0, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer1, &h_data1, sizeof(int), cudaMemcpyHostToDevice);

    cuda_put_kernel<<<1, 32>>>(d_buffer0, d_buffer1);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result0, d_buffer0, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_result1, d_buffer1, sizeof(int), cudaMemcpyDeviceToHost);

    int passed = 1;

    printf("  GPU 0 buffer: %d (expected 0)... ", h_result0);
    if (h_result0 == 0) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
        passed = 0;
    }

    printf("  GPU 1 buffer: %d (expected 8888)... ", h_result1);
    if (h_result1 == 8888) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
        passed = 0;
    }

    cudaFree(d_buffer0);
    cudaFree(d_buffer1);

    printf("shmem.cu tests complete.\n");
    return passed ? 0 : 1;
}
