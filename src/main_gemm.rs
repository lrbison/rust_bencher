extern crate libc;

use bencher::{GemmTest, run_gemm_in_rust, run_gemm_external};



fn main() {
    let mut config: GemmTest = GemmTest::new(2, 10, 8);
    run_gemm_in_rust(&mut config);
    println!("Basic Rust: {:?}", config.c[0]);

    run_gemm_external(&mut config);
    println!("External BLAS: {:?}", config.c[0]);
}
