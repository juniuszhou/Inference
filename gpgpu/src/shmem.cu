#include <cuda_runtime.h>
#include <iostream>

// --- 1. CUDA GPU 内核函数（简化版本，没有 nvshmem）---
__global__ void cuda_put_kernel(int* my_buffer, int* neighbor_buffer) {
    // 获取当前线程在 Grid 中的全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 假设我们只让第 0 号线程演示向邻居 GPU 发送数据
    if (tid == 0) {
        // 直接将数据写入邻居 GPU 的内存
        *neighbor_buffer = 8888;
    }
}

// --- 2. Host 端驱动与初始化代码 ---
int main() {
    // 为两个 GPU 分配设备显存
    int* d_buffer0;
    int* d_buffer1;
    cudaMalloc(&d_buffer0, sizeof(int));
    cudaMalloc(&d_buffer1, sizeof(int));

    // 初始化本地内存为 0
    int host_data0 = 0;
    int host_data1 = 0;
    cudaMemcpy(d_buffer0, &host_data0, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer1, &host_data1, sizeof(int), cudaMemcpyHostToDevice);

    // 拉起内核，只有 GPU 0 负责向 GPU 1 发送数据
    cuda_put_kernel<<<1, 32>>>(d_buffer0, d_buffer1);
    cudaDeviceSynchronize();

    // GPU 0 打印检查自己的内存
    cudaMemcpy(&host_data0, d_buffer0, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU 0: Data in my buffer is: " << host_data0 << std::endl;

    // GPU 1 打印检查邻居的内存，看有没有被 GPU 0 直接修改
    cudaMemcpy(&host_data1, d_buffer1, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU 1: Verification! Data in my buffer is: " << host_data1 << std::endl;

    // 清理内存
    cudaFree(d_buffer0);
    cudaFree(d_buffer1);
    return 0;
}