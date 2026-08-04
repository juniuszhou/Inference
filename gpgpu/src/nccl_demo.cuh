#ifndef NCCL_DEMO_CUH
#define NCCL_DEMO_CUH

#include <cuda_runtime.h>
#include <nccl.h>

// NCCL communicator (single GPU context)
typedef struct {
    ncclComm_t comm;
} NcclContext;

// Initialize NCCL for single GPU
ncclResult_t nccl_init(NcclContext* ctx);

// Cleanup NCCL
void nccl_cleanup(NcclContext* ctx);

// Broadcast: data[0] (root) -> all ranks
void nccl_broadcast(float* data, int count, NcclContext* ctx);

// Reduce: sum all ranks into data[0]
void nccl_reduce(float* input, float* output, int count, NcclContext* ctx);

// AllReduce: sum all ranks, result to all
void nccl_allreduce(float* input, float* output, int count, NcclContext* ctx);

#endif // NCCL_DEMO_CUH
