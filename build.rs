fn main ()
{
    let blas_path =
        ::std::env::var("BLAS_PATH")
            .expect("Please provide the `BLAS_PATH` env var")
    ;
    let blas_lib =
    ::std::env::var("BLAS_LIB")
        .expect("Please provide the `BLAS_LIB` env var");
    println!("cargo:rustc-link-search={}", blas_path); // -L $BLAS_PATH
    println!("cargo:rustc-link-lib={}", blas_lib);
    println!("cargo:rustc-env=LD_LIBRARY_PATH={}", blas_path);
}