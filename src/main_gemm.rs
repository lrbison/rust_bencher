extern crate libc;

use bencher::{GemmTest, run_gemm_in_rust, run_gemm_external, run_gemv_external};



fn main() {
    let mut config: GemmTest = GemmTest::new(1, 512, 32768);
    run_gemm_in_rust(&mut config);
    println!("Basic Rust: {:?}", config.c[0]);

    let mut config: GemmTest = GemmTest::new(1, 10, 32768);
    run_gemv_external(&mut config);
    println!("BLAS gemv: {:?}", config.c[0]);

    let mut config: GemmTest = GemmTest::new(1, 10, 32768);
    run_gemm_external(&mut config);
    println!("BLAS gemm: {:?}", config.c[0]);
}
