#include "shmem.cuh"

// Standalone executable for shmem kernel
int main() {
    int *d_buffer0, *d_buffer1;
    int host_data0 = 0, host_data1 = 0;
    
    cudaMalloc(&d_buffer0, sizeof(int));
    cudaMalloc(&d_buffer1, sizeof(int));
    
    cudaMemcpy(d_buffer0, &host_data0, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer1, &host_data1, sizeof(int), cudaMemcpyHostToDevice);
    
    cuda_put_kernel<<<1, 32>>>(d_buffer0, d_buffer1);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&host_data0, d_buffer0, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_data1, d_buffer1, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "GPU 0: Data in my buffer is: " << host_data0 << std::endl;
    std::cout << "GPU 1: Verification! Data in my buffer is: " << host_data1 << std::endl;
    
    cudaFree(d_buffer0);
    cudaFree(d_buffer1);
    return 0;
}
