#include "nccl_demo.cuh"
#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <string.h>

ncclResult_t nccl_init(NcclContext* ctx) {
    int deviceId = 0;
    cudaSetDevice(deviceId);

    ncclUniqueId id;
    ncclResult_t ret = ncclGetUniqueId(&id);
    if (ret != ncclSuccess) {
        printf("ncclGetUniqueId failed: %s\n", ncclGetErrorString(ret));
        return ret;
    }

    ret = ncclCommInitRank(&ctx->comm, 1, id, 0);
    if (ret != ncclSuccess) {
        printf("ncclCommInitRank failed: %s\n", ncclGetErrorString(ret));
    }
    return ret;
}

void nccl_cleanup(NcclContext* ctx) {
    if (ctx->comm) ncclCommDestroy(ctx->comm);
}

void nccl_broadcast(float* data, int count, NcclContext* ctx) {
    ncclBroadcast(data, data, count, ncclFloat, 0, ctx->comm, 0);
}

void nccl_reduce(float* input, float* output, int count, NcclContext* ctx) {
    ncclReduce(input, output, count, ncclFloat, ncclSum, 0, ctx->comm, 0);
}

void nccl_allreduce(float* input, float* output, int count, NcclContext* ctx) {
    ncclAllReduce(input, output, count, ncclFloat, ncclSum, ctx->comm, 0);
}
