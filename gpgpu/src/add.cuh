#ifndef ADD_CUH
#define ADD_CUH

#include <cuda_runtime.h>

// Kernel declarations only
extern "C" __global__ void add(float* a, float* b, float* c, int N);
extern "C" __global__ void addSlice(float* a, float* b, float* c, int a_off, int b_off, int c_off, int N);

#endif // ADD_CUH
