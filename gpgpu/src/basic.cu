#include "basic.cuh"
#include <cuda_bf16.h>

__global__ void add_global(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__device__ void add_device(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__host__ void add_host(float* a, float* b, float* c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

__host__ __device__ void add_host_device(float* a, float* b, float* c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

__host__ __device__ void data_type() {
    __nv_bfloat16 bfloat16_val = __float2bfloat16(1.0f);
}
