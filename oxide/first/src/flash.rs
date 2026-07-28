use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
#[allow(non_snake_case)]
pub mod flash {
    use super::*;

    /// FlashAttention forward pass (1 thread per query row, Bc threads per block).
    /// Port of flash-attention-minimal's `forward_kernel`.
    #[kernel]
    #[allow(non_snake_case)]
    pub fn flash_attention(
        Q: &[f32],
        K: &[f32],
        V: &[f32],
        N: i32,
        d: i32,
        Tc: i32,
        Tr: i32,
        Bc: i32,
        Br: i32,
        softmax_scale: f32,
        mut l: DisjointSlice<f32>,
        mut m: DisjointSlice<f32>,
        mut O: DisjointSlice<f32>,
    ) {
        let tx = thread::threadIdx_x() as i32;
        let bx = thread::blockIdx_x() as i32; // batch index
        let by = thread::blockIdx_y() as i32; // head index
        let nh = thread::gridDim_y() as i32;

        // Offsets into Q,K,V,O and l,m for this (batch, head)
        let qkv_offset = (bx * nh * N * d) + (by * N * d);
        let lm_offset = (bx * nh * N) + (by * N);
        let tile_size = Bc * d; // size of Qi, Kj, Vj

        // l, m, O are written at raw indices computed from the launch
        // geometry (each thread owns row `tx` of each tile), so access them
        // through raw pointers like the CUDA original.
        let l_ptr = l.as_mut_ptr();
        let m_ptr = m.as_mut_ptr();
        let o_ptr = O.as_mut_ptr();

        // Partition dynamic shared memory into Qi, Kj, Vj, S
        let smem = DynamicSharedArray::<f32>::get();
        let Qi: *mut f32 = smem;
        let Kj: *mut f32;
        let Vj: *mut f32;
        let S: *mut f32;
        unsafe {
            Kj = smem.add(tile_size as usize);
            Vj = smem.add((tile_size * 2) as usize);
            S = smem.add((tile_size * 3) as usize);
        }

        for j in 0..Tc {
            // Load Kj, Vj tiles from HBM to SRAM
            for x in 0..d {
                unsafe {
                    let src = (qkv_offset + tile_size * j + tx * d + x) as usize;
                    let dst = (tx * d + x) as usize;
                    *Kj.add(dst) = K[src];
                    *Vj.add(dst) = V[src];
                }
            }
            thread::sync_threads();

            for i in 0..Tr {
                // Load Qi tile from HBM to SRAM
                for x in 0..d {
                    unsafe {
                        let src = (qkv_offset + tile_size * i + tx * d + x) as usize;
                        let dst = (tx * d + x) as usize;
                        *Qi.add(dst) = Q[src];
                    }
                }

                // Read previous m and l
                let lm_idx = (lm_offset + Br * i + tx) as usize;
                let row_m_prev: f32;
                let row_l_prev: f32;
                unsafe {
                    row_m_prev = *m_ptr.add(lm_idx);
                    row_l_prev = *l_ptr.add(lm_idx);
                }

                // S = Q * K^T * scale, row_m = rowmax(S)
                let mut row_m = -f32::INFINITY;
                for y in 0..Bc {
                    let mut sum = 0.0f32;
                    for x in 0..d {
                        unsafe {
                            sum += *Qi.add((tx * d + x) as usize) * *Kj.add((y * d + x) as usize);
                        }
                    }
                    sum *= softmax_scale;
                    unsafe {
                        *S.add((Bc * tx + y) as usize) = sum;
                    }
                    if sum > row_m {
                        row_m = sum;
                    }
                }

                // P = exp(S - row_m), row_l = rowsum(P)
                let mut row_l = 0.0f32;
                for y in 0..Bc {
                    unsafe {
                        let exp_val = libm::expf(*S.add((Bc * tx + y) as usize) - row_m);
                        *S.add((Bc * tx + y) as usize) = exp_val;
                        row_l += exp_val;
                    }
                }

                // Online softmax update
                let row_m_new = if row_m_prev > row_m {
                    row_m_prev
                } else {
                    row_m
                };
                let row_l_new = libm::expf(row_m_prev - row_m_new) * row_l_prev
                    + libm::expf(row_m - row_m_new) * row_l;

                // Write O with rescaling
                for x in 0..d {
                    let mut pv = 0.0f32;
                    for y in 0..Bc {
                        unsafe {
                            pv += *S.add((Bc * tx + y) as usize) * *Vj.add((y * d + x) as usize);
                        }
                    }
                    let o_idx = (qkv_offset + tile_size * i + tx * d + x) as usize;
                    unsafe {
                        let old_o = *o_ptr.add(o_idx);
                        *o_ptr.add(o_idx) = (1.0 / row_l_new)
                            * (row_l_prev * libm::expf(row_m_prev - row_m_new) * old_o
                                + libm::expf(row_m - row_m_new) * pv);
                    }
                }

                // Write updated l and m
                unsafe {
                    *m_ptr.add(lm_idx) = row_m_new;
                    *l_ptr.add(lm_idx) = row_l_new;
                }
            }
            thread::sync_threads();
        }
    }
}
