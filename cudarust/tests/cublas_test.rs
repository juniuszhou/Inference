use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{cublasSaxpy_v2, cublasStatus_t};
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};

/*
cudarc 0.19.8's safe `CudaBlas` only exposes gemm/gemv/asum — there is no safe
`saxpy` wrapper. The raw `cublasSaxpy_v2` sys call is still available, so this
test drives it directly through `blas.handle()`.

saxpy computes `y := alpha*x + y` (in place on y). To get a plain vector add
`c = a + b`, seed c_dev with a copy of b and pass a as x.
*/
#[test]
fn test_add() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // need a blas instance to call the cublas function
    let blas = CudaBlas::new(stream.clone()).unwrap();

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 2.0, 3.0, 4.0];
    let a_dev: CudaSlice<f32> = stream.clone_htod(&a).unwrap();
    let b_dev: CudaSlice<f32> = stream.clone_htod(&b).unwrap();
    // y starts as a copy of b, so y := alpha*x + y becomes c = a + b
    let mut c_dev: CudaSlice<f32> = b_dev.try_clone().unwrap();

    let n: i32 = a.len() as i32;
    let alpha: f32 = 1.0;
    unsafe {
        let (a_ptr, _sa) = a_dev.device_ptr(&stream);
        let (c_ptr, _sc) = c_dev.device_ptr_mut(&stream);
        // c := alpha*a + c = a + b
        let status = cublasSaxpy_v2(
            *blas.handle(),
            n,
            &alpha,
            a_ptr as *const f32,
            1,
            c_ptr as *mut f32,
            1,
        );
        assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }

    // The raw call is enqueued on the handle's stream (set by CudaBlas::new);
    // wait for it before copying the result to the host.
    stream.synchronize().unwrap();

    let c_host = stream.clone_dtoh(&c_dev).unwrap();

    // c = a + b = [2, 4, 6, 8]
    let expected = [2.0f32, 4.0, 6.0, 8.0];
    for (i, (&got, &exp)) in c_host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "element {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }

    println!("SAXPY test passed. Result: {:?}", c_host);
}
