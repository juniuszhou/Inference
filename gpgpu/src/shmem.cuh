#ifndef SHMEM_CUH
#define SHMEM_CUH

#include <cuda_runtime.h>
#include <iostream>

__global__ void cuda_put_kernel(int* my_buffer, int* neighbor_buffer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        *neighbor_buffer = 8888;
    }
}

#endif // SHMEM_CUH
