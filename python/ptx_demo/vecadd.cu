// vecadd.cu — Vector addition kernel (CUDA C source for vecadd.ptx)
//
// Compile to PTX:
//   nvcc --ptx -arch=sm_86 vecadd.cu -o vecadd.ptx
//
// Computes: c[i] = a[i] + b[i]  for i in [0, n)
//
// Each thread handles exactly one element.
// Launch config: 1-D grid of 1-D blocks.
//   grid_size  = ceil(n / block_size)
//   block_size = number of threads per block

#include <cuda_runtime.h>

extern "C" __global__ void vecadd(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
