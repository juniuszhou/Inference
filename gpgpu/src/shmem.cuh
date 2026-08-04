#ifndef SHMEM_CUH
#define SHMEM_CUH

#include <cuda_runtime.h>

__global__ void cuda_put_kernel(int* my_buffer, int* neighbor_buffer);

#endif // SHMEM_CUH
