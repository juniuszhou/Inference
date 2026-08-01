use crate::mvshmem::*;

fn main() -> anyhow::Result<()> {
    // Initialize the GPU memory sharing environment
    init_mvshmem(0)?;

    // Write some data to a shared key
    let key1 = MvshmemKey::new("my_shared_data", 1024);
    let data = vec![1i32, 2i32, 3i32, 4i32, 5i32];
    write_to_mvshmem(&key1, &data)?;

    // Read data back from the shared key
    let read_data = read_from_mvshmem(&key1)?;
    println!("Read data: {:?}", read_data);

    // Test with a different key to demonstrate independence
    let key2 = MvshmemKey::new("another_key", 2048);
    let data2 = vec![10i32, 20i32, 30i32];
    write_to_mvshmem(&key2, &data2)?;

    let read_data2 = read_from_mvshmem(&key2)?;
    println!("Read data2: {:?}", read_data2);

    // Execute a GPU kernel that uses the shared memory
    let kernel_ptx = r"
        __global__ void shared_memory_kernel(int* data, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                data[idx] = idx * 2;
            }
        }
    ";

    execute_gpu_kernel(&key1, kernel_ptx, 256)?;
    let kernel_result = read_from_mvshmem(&key1)?;
    println!("Kernel result: {:?}", kernel_result);

    Ok(())
}