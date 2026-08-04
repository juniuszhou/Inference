#ifndef MULTIPLY_CUH
#define MULTIPLY_CUH

#include <cuda_runtime.h>

extern "C" __global__ void matmul(const float* A, const float* B, float* C, int m, int k, int n);

#endif // MULTIPLY_CUH
