/*

This module is to demonstrate the mvshmem in the cuda.
It provides the following functions:
1. init the mvshmem environment in the cuda.
2. function to read / write data from / to mvshmem.
*/

pub fn init_mvshmem() {}

pub fn read_from_mvshmem(key: &str) -> Vec<u8> {
    let data = mvshmem::read(key);
    data
}

pub fn write_to_mvshmem(key: &str, data: &[u8]) {
    mvshmem::write(key, data);
}
