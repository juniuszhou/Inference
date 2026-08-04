#ifndef BASIC_CUH
#define BASIC_CUH

#include <cuda_runtime.h>

__global__ void add1D(float* a, float* b, float* c, int N);

#endif // BASIC_CUH
