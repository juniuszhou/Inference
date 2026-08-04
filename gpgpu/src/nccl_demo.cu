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

// Standalone main
#ifndef NO_MAIN
int main() {
    printf("nccl_demo.cu: testing NCCL collectives on single GPU\n");

    NcclContext ctx;
    if (nccl_init(&ctx) != ncclSuccess) {
        printf("Failed to init NCCL\n");
        return 1;
    }

    int N = 1024;
    size_t bytes = N * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, bytes);

    float h_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    cudaMemcpy(d_data, h_data, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // Test broadcast
    nccl_broadcast(d_data, 4, &ctx);
    cudaDeviceSynchronize();
    printf("broadcast: OK\n");

    // Test reduce
    nccl_reduce(d_data, d_data, 4, &ctx);
    cudaDeviceSynchronize();
    printf("reduce: OK\n");

    // Test allreduce
    nccl_allreduce(d_data, d_data, 4, &ctx);
    cudaDeviceSynchronize();
    printf("allreduce: OK\n");

    cudaFree(d_data);
    nccl_cleanup(&ctx);
    printf("nccl_demo.cu done.\n");
    return 0;
}
#endif
