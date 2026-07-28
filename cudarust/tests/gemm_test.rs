use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaSlice};

#[test]
fn test_gemm() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // need a blas instance to call the gemm function
    let blas = CudaBlas::new(stream.clone()).unwrap();

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_data_b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let a_dev: CudaSlice<f32> = stream.clone_htod(&input_data).unwrap();
    let b_dev: CudaSlice<f32> = stream.clone_htod(&input_data_b).unwrap();
    let mut c_dev: CudaSlice<f32> = stream.clone_htod(&vec![0.0; 4]).unwrap();

    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: 2,
        n: 2,
        k: 2,
        alpha: 1.0f32,
        lda: 2,
        ldb: 2,
        beta: 0.0f32,
        ldc: 2,
    };

    // call it via unsafe
    unsafe {
        blas.gemm(config, &a_dev, &b_dev, &mut c_dev).unwrap();
    }

    let c_host = stream.clone_dtoh(&c_dev).unwrap();

    // A = [[1,2],[3,4]], B = [[1,2],[3,4]] (row-major)
    // C = A * B = [[7,10],[15,22]]
    let expected = vec![7.0f32, 10.0, 15.0, 22.0];
    for (i, (&got, &exp)) in c_host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "element {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }

    println!("GEMM test passed. Result: {:?}", c_host);
}
