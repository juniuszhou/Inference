#ifndef REDUCE_CUH
#define REDUCE_CUH

#include <cuda_runtime.h>

// Maximum threads per block this kernel supports (shared memory size).
#define MAX_BLOCK 1024

// Launch contract:
//   a: input array (N floats)
//   c: one float written per block -> c[blockIdx.x]
//   N: number of input elements
extern "C" __global__ void reduceMax(const float* a, float* c, int N);

// Per-row max of an M x N row-major matrix.
//   a: M*N input floats
//   c: M output floats, c[row] = max over columns of row `row`
//   M: number of rows (gridDim.x must equal M: one block per row)
//   N: number of columns (one row is reduced by one block)
extern "C" __global__ void reduceMax2D(const float* a, float* c, int M, int N);

#endif // REDUCE_CUH