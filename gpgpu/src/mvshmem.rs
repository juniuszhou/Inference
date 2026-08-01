use anyhow::{Result, bail};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::ptr;
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MvshmemKey {
    name: String,
    size: usize,
}

impl MvshmemKey {
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

struct MvshmemState {
    gpu_slices: HashMap<MvshmemKey, (CudaSlice<i32>, CudaSlice<i32>)>,
    ctx: Option<CudaContext>,
    device_index: usize,
    initialized: bool,
}

static mut MVSHMEM_STATE: Option<Mutex<MvshmemState>> = None;

pub fn init_mvshmem(device_index: usize) -> Result<()> {
    if unsafe { MVSHMEM_STATE.is_none() } {
        let ctx = CudaContext::new(device_index)?;
        let state = MvshmemState {
            gpu_slices: HashMap::new(),
            ctx: Some(ctx),
            device_index,
            initialized: true,
        };
        unsafe {
            MVSHMEM_STATE = Some(Mutex::new(state));
        }
    }
    Ok(())
}

fn get_state() -> &'static mut MvshmemState {
    unsafe {
        MVSHMEM_STATE
            .as_mut()
            .expect("MVSHMEM state not initialized. Call init_mvshmem() first.")
            .get_mut()
    }
}

pub fn read_from_mvshmem(key: &MvshmemKey) -> Result<Vec<i32>> {
    let mut state = get_state();

    if !state.gpu_slices.contains_key(key) {
        let slice = state
            .ctx
            .as_ref()
            .unwrap()
            .alloc_zeros::<i32>(key.size / std::mem::size_of::<i32>())?;
        let _neighbor_slice = state
            .ctx
            .as_ref()
            .unwrap()
            .alloc_zeros::<i32>(key.size / std::mem::size_of::<i32>())?;
        state
            .gpu_slices
            .insert(key.clone(), (slice, _neighbor_slice));
    }

    let (gpu_slice, _neighbor_slice) = state.gpu_slices.get_mut(key).unwrap();

    let mut host_data = vec![0i32; gpu_slice.len()];
    state
        .ctx
        .as_ref()
        .unwrap()
        .default_stream()
        .clone_dtoh_into(gpu_slice, &mut host_data)?;

    Ok(host_data)
}

pub fn write_to_mvshmem(key: &MvshmemKey, data: &[i32]) -> Result<()> {
    let mut state = get_state();

    if !state.gpu_slices.contains_key(key) {
        let slice = state
            .ctx
            .as_ref()
            .unwrap()
            .alloc_zeros::<i32>(key.size / std::mem::size_of::<i32>())?;
        let _neighbor_slice = state
            .ctx
            .as_ref()
            .unwrap()
            .alloc_zeros::<i32>(key.size / std::mem::size_of::<i32>())?;
        state
            .gpu_slices
            .insert(key.clone(), (slice, _neighbor_slice));
    }

    let (gpu_slice, _neighbor_slice) = state.gpu_slices.get_mut(key).unwrap();

    state
        .ctx
        .as_ref()
        .unwrap()
        .default_stream()
        .clone_htod_into(data, gpu_slice)?;

    Ok(())
}

pub fn execute_gpu_kernel(
    key: &MvshmemKey,
    kernel_ptx: &str,
    shared_mem_size: usize,
) -> Result<()> {
    let state = get_state();

    let ptx = Ptx::from_src(kernel_ptx)?;
    let module = state.ctx.as_ref().unwrap().load_module(ptx)?;
    let kernel = module.load_function("shared_memory_kernel")?;

    let (gpu_slice, _neighbor_slice) = state
        .gpu_slices
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("Key not found in GPU slices"))?;

    let block_size = 256u32;
    let grid_size = ((key.size as u32 + block_size - 1) / block_size, 1, 1);

    unsafe {
        state
            .ctx
            .as_ref()
            .unwrap()
            .default_stream()
            .launch_builder(&kernel)
            .arg(gpu_slice)
            .arg(&(key.size as i32))
            .launch(LaunchConfig {
                grid_dim: grid_size,
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_mem_size as u32,
            })?;
    }

    state.ctx.as_ref().unwrap().default_stream().synchronize()?;

    Ok(())
}
