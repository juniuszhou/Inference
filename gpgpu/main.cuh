#ifndef MAIN_CUH
#define MAIN_CUH

#include <cuda_runtime.h>

__global__ void add1D(float* a, float* b, float* c, int N);

#endif // MAIN_CUH
