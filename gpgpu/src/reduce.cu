#include "reduce.cuh"
#include <float.h>

/*
    Parallel max reduction of the array `a` (N floats).

    The kernel is written as a TWO-PHASE reduction so that it is both correct
    and coalesced:

      Phase 1 - grid-stride loop
          Every thread folds a strided slice of the array into a PRIVATE
          register (`m`). We stride by `gridDim.x * blockDim.x`, so consecutive
          threads always read consecutive addresses -> memory coalescing, and
          each element is visited exactly once. Keeping the accumulator in a
          register (instead of writing back to global memory each step) slashes
          global traffic.

      Phase 2 - intra-block tree reduction
          Threads publish their running max into shared memory, then combine
          pairwise: each step halves the number of active threads. `__syncthreads()`
          guards both the publish and every combine level so no thread reads a
          stale slot. Thread 0 writes the block's max to c[blockIdx.x].

    Result location:
          If gridDim.x == 1 the whole array fits one block and c[0] holds the
          final max. With a multi-block grid each block emits a partial max into
          c[blockIdx.x]; the caller must reduce those partials (e.g. with a
          second single-block launch) to get the overall max into c[0].
*/
extern "C" __global__ void reduceMax(const float* a, float* c, int N) {
    // Phase 1: private running max. -FLT_MAX guarantees real values always win.
    float m = -FLT_MAX;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockDim.x * gridDim.x; // grid-stride step

    for (; index < N; index += total) {
        m = fmaxf(m, a[index]);
    }

    // Phase 2: tree reduction over this block in shared memory.
    __shared__ float sRed[MAX_BLOCK];
    sRed[threadIdx.x] = m;
    __syncthreads(); // all threads must publish before any thread combines

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sRed[threadIdx.x] = fmaxf(sRed[threadIdx.x], sRed[threadIdx.x + s]);
        }
        __syncthreads(); // don't let faster threads read stale slots next level
    }

    if (threadIdx.x == 0) {
        c[blockIdx.x] = sRed[0];
    }
}

// the length of c should be equal to the number of blocks in the grid
extern "C" __global__ void reduceMin(const float* a, float* c, int N) {
    // Phase 1: private running max. -FLT_MAX guarantees real values always win.
    float m = FLT_MAX;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // total threads in the grid
    int total = blockDim.x * gridDim.x; // grid-stride step

    // If the total length is more than total threads in the grid, we need to iterate over the array in a loop.
    for (; index < N; index += total) {
        m = fminf(m, a[index]);
    }

    // Phase 2: tree reduction over this block in shared memory.
    __shared__ float sRed[MAX_BLOCK];
    sRed[threadIdx.x] = m;
    __syncthreads(); // all threads must publish before any thread combines

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sRed[threadIdx.x] = fminf(sRed[threadIdx.x], sRed[threadIdx.x + s]);
        }
        __syncthreads(); // don't let faster threads read stale slots next level
    }

    // write the result to the output array by the first thread in the block
    if (threadIdx.x == 0) {
        c[blockIdx.x] = sRed[0];
    }
}

/*
    Per-row max reduction of an M x N row-major matrix.
    Output `c` has length M: c[row] = max over columns of row `row`.

    Unlike reduceMax/reduceMin (which flatten the whole array into ONE output),
    here there is one independent reduction per ROW. So the grid must provide
    one BLOCK per row: block index x == row index, gridDim.x must equal M.
    Within each block the threads stride across the row's N columns, then the
    usual shared-memory tree collapses the block into a single value written by
    thread 0 to c[row].
*/
extern "C" __global__ void reduceMax2D(const float* a, float* c, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) {
        return; // grid launched larger than M; block unused.
    }

    // Phase 1: each thread folds its strided slice of THIS row into `m`.
    // Consecutive threads read consecutive columns -> coalesced.
    float m = -FLT_MAX;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        m = fmaxf(m, a[row * N + col]);
    }

    // Phase 2: intra-block tree reduction (identical to reduceMax).
    __shared__ float sRed[MAX_BLOCK];
    sRed[threadIdx.x] = m;
    __syncthreads(); // all threads must publish before any thread combines

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sRed[threadIdx.x] = fmaxf(sRed[threadIdx.x], sRed[threadIdx.x + s]);
        }
        __syncthreads(); // don't let faster threads read stale slots next level
    }

    if (threadIdx.x == 0) {
        c[row] = sRed[0];
    }
}
