use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

// Device: generic kernel that applies any function to each element.
// F can be a closure with captures — rustc monomorphizes it to a concrete type.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[i]);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let input = DeviceBuffer::from_host(&stream, &data).unwrap();
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    let module = kernels::load(&ctx).unwrap();

    // Launch with a closure — factor is captured and passed to the GPU automatically
    let factor = 2.5f32;
    module
        .map::<f32, _>(
            &stream,
            LaunchConfig::for_num_elems(1024),
            move |x: f32| x * factor,
            &input,
            &mut output,
        )
        .unwrap();

    let result = output.to_host_vec(&stream).unwrap();
    assert!((result[1] - 2.5).abs() < 1e-5);
}

