#ifndef BASIC_CUH
#define BASIC_CUH

#include <cuda_runtime.h>

__global__ void add_global(float* a, float* b, float* c, int N);
__device__ void add_device(float* a, float* b, float* c, int N);
__host__ void add_host(float* a, float* b, float* c, int N);
__host__ __device__ void add_host_device(float* a, float* b, float* c, int N);

#endif // BASIC_CUH
