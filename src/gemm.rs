extern crate libc;

use libc::c_float;
use vrd::random::Random;

extern "C" {
    #[link_name = "sgemm_"]
    fn sgemm_external(
        transa: &libc::c_char,
        transb: &libc::c_char,
        m: *const libc::c_int,
        n: *const libc::c_int,
        k: *const libc::c_int,
        alpha: *const libc::c_float,
        a: *const libc::c_float,
        lda: *const libc::c_int,
        b: *const libc::c_float,
        ldb: *const libc::c_int,
        beta: *const libc::c_float,
        c: *mut libc::c_float,
        ldc: *const libc::c_int,
    );
}

extern "C" {
    #[link_name = "sgemv_"]
    fn sgemv_external(
        transa: &libc::c_char,
        m: *const libc::c_int,
        n: *const libc::c_int,
        alpha: *const libc::c_float,
        a: *const libc::c_float,
        lda: *const libc::c_int,
        x: *const libc::c_float,
        incx: *const libc::c_int,
        beta: *const libc::c_float,
        y: *mut libc::c_float,
        incy: *const libc::c_int,
    );
}

pub struct GemmTest {
    pub m: libc::c_int,
    pub k: libc::c_int,
    pub n: libc::c_int,

    pub a: Vec<c_float>,
    pub b: Vec<c_float>,
    pub c: Vec<c_float>,
}

impl GemmTest {
    pub fn new(m: usize, k: usize, n: usize) -> Self{
        let mut a = Vec::with_capacity(m*k);
        let mut b = Vec::with_capacity(k*n);
        let mut c = Vec::with_capacity(m*n);

        let big_num: c_float = 2147483647 as c_float;
        let big_numi: i32 = 2147483647;

        let mut rng = Random::new();
        rng.seed(12345);

        for _ in 0..m*k {
            a.push( (rng.int(0,big_numi) as c_float)/big_num );
        }
        for _ in 0..k*n {
            b.push( (rng.int(0,big_numi) as c_float)/big_num );
        }
        for _ in 0..m*n {
            c.push( 0.0 );
        }
        let mc = m as libc::c_int;
        let kc = k as libc::c_int;
        let nc = n as libc::c_int;
        return GemmTest{m:mc,k:kc,n:nc,a:a,b:b,c:c};
    }
}

pub fn run_gemm_in_rust(config: &mut GemmTest) {
    gemm_in_rust(config.a.as_ptr(), config.b.as_ptr(), config.c.as_mut_ptr(), config.m, config.n, config.k);
}

pub fn run_gemm_external(config: &mut GemmTest) {
    // C := alpha*A*B + beta*C,
    let not_transposed: libc::c_char = 'N' as libc::c_char;
    let one: libc::c_float = 1.0f32;
    let zero: libc::c_float = 0.0f32;
    unsafe {
        sgemm_external(
            &not_transposed,
            &not_transposed,
            &config.m as *const libc::c_int,
            &config.n as *const libc::c_int,
            &config.k as *const libc::c_int,
            &one,
            config.a.as_ptr(),
            &config.m as *const libc::c_int,
            config.b.as_ptr(),
            &config.k as *const libc::c_int,
            &zero,
            config.c.as_mut_ptr(),
            &config.m as *const libc::c_int );
    }
}

pub fn run_gemv_external(config: &mut GemmTest) {
    // y := alpha*A*x + beta*y.
    // For our config: A[m x k], B[k x n], C[m x n], when m==1:
    // C[n] := (B[k x n]])' * A[k] *
    /* so...
        gemv's m := config.k
        gemv's n := config.n
     */


    let transposed: libc::c_char = 'T' as libc::c_char;
    let one: libc::c_float = 1.0f32;
    let zero: libc::c_float = 0.0f32;
    let incxy: libc::c_int = 1;
    unsafe {
        sgemv_external(
            &transposed,
            &config.k as *const libc::c_int,
            &config.n as *const libc::c_int,
            &one,
            config.b.as_ptr(),
            &config.k as *const libc::c_int,
            config.a.as_ptr(),
            &incxy as *const libc::c_int,
            &zero,
            config.c.as_mut_ptr(),
            &incxy as *const libc::c_int );
    }
}


// a : m x k
// b : k x n
// c : m x n
pub fn gemm_in_rust(aptr: *const c_float, bptr: *const c_float, cptr: *mut c_float, m: i32, n: i32, k: i32) {
    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let a = unsafe { std::slice::from_raw_parts(aptr, mu*ku) };
    let b = unsafe { std::slice::from_raw_parts(bptr, ku*nu) };
    let c = unsafe { std::slice::from_raw_parts_mut(cptr, mu*nu) };
    for mm in 0..mu {
        for nn in 0..nu {
            let mut sum = 0.0;
            for kk in 0..ku {
                sum += a[mm * ku + kk] * b[kk * nu + nn];
            }
            c[mm * nu + nn] = sum;
        }
    }
}