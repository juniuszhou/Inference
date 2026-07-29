use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

/// Threads per block; also the size of the shared-memory reduction buffer.
/// Must be a power of two for the tree reduction.
pub const BLOCK_SIZE: usize = 256;

#[cuda_module]
pub mod norm {
    use super::*;

    /// RMSNorm over rows of an (n_rows, d) matrix:
    /// `y = x / sqrt(mean(x^2) + epsilon)`
    ///
    /// Launch with one block per row and `BLOCK_SIZE` threads per block;
    /// each thread strides over the row's columns.
    #[kernel]
    pub fn rmsnorm(x: &[f32], d: i32, epsilon: f32, mut y: DisjointSlice<f32>) {
        /// d is second dimension of the matrix
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as i32;
        let row = thread::blockIdx_x() as i32;

        // block size
        let block = thread::blockDim_x() as i32;

        let base = (row * d) as usize;

        // Each thread writes strided indices of its own row, so y is
        // accessed through a raw pointer (indices are disjoint by launch
        // geometry, not by a per-thread witness).
        let y_ptr = y.as_mut_ptr();

        // Per-thread partial sum of squares over the row
        let mut sum = 0.0f32;
        let mut col = tid;

        while col < d {
            let v = x[base + col as usize];
            sum += v * v;
            col += block;
        }
        unsafe {
            PARTIAL[tid as usize] = sum;
        }

        // sync threads to ensure all threads have written their partial sums
        thread::sync_threads();

        // Tree reduction in shared memory
        let mut stride = block / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid as usize] += PARTIAL[(tid + stride) as usize];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        // mean(x^2) = sum / d, shared by every thread in the block
        let inv_rms = unsafe { 1.0f32 / libm::sqrtf(PARTIAL[0] / d as f32 + epsilon) };

        let mut col = tid;
        while col < d {
            unsafe {
                *y_ptr.add(base + col as usize) = x[base + col as usize] * inv_rms;
            }
            col += block;
        }
    }

    /* LayerNorm over rows of an (n_rows, d) matrix:
    Formula:

        μ  = mean(x)
        σ² = mean((x − μ)²)          # population variance (÷ n)
        y  = (x − μ) / sqrt(σ² + ε)

    Launch with one block per row and `BLOCK_SIZE` threads per block;
    each thread strides over the row's columns.
     */
    #[kernel]
    pub fn layernorm(x: &[f32], d: i32, epsilon: f32, mut y: DisjointSlice<f32>) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as i32;
        let row = thread::blockIdx_x() as i32;
        let block = thread::blockDim_x() as i32;
        let base = (row * d) as usize;

        // Strided writes within the row are disjoint by launch geometry,
        // so y is accessed through a raw pointer (see rmsnorm).
        let y_ptr = y.as_mut_ptr();

        // --- Reduction 1: sum(x) -> mean ---
        let mut sum = 0.0f32;
        let mut col = tid;
        while col < d {
            sum += x[base + col as usize];
            col += block;
        }
        unsafe {
            PARTIAL[tid as usize] = sum;
        }
        thread::sync_threads();

        let mut stride = block / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid as usize] += PARTIAL[(tid + stride) as usize];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let mu = unsafe { PARTIAL[0] } / d as f32;

        // Every thread has read PARTIAL[0]; barrier before the buffer is
        // overwritten by the second reduction.
        thread::sync_threads();

        // --- Reduction 2: sum((x - mu)^2) -> variance ---
        let mut sum_sq = 0.0f32;
        let mut col = tid;
        while col < d {
            let v = x[base + col as usize] - mu;
            sum_sq += v * v;
            col += block;
        }
        unsafe {
            PARTIAL[tid as usize] = sum_sq;
        }
        thread::sync_threads();

        let mut stride = block / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid as usize] += PARTIAL[(tid + stride) as usize];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let variance = unsafe { PARTIAL[0] } / d as f32;
        let inv_std = 1.0f32 / libm::sqrtf(variance + epsilon);

        // --- Normalize ---
        let mut col = tid;
        while col < d {
            unsafe {
                *y_ptr.add(base + col as usize) = (x[base + col as usize] - mu) * inv_std;
            }
            col += block;
        }
    }
}
