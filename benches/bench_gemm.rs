
use bencher::run_gemm_external;
use bencher::run_gemv_external;
use criterion::BenchmarkId;
use criterion::Criterion;

use criterion::{criterion_group, criterion_main};

use bencher::{GemmTest, run_gemm_in_rust};


fn bench_gemm(c: &mut Criterion) {

    let mut group = c.benchmark_group("gemm");

    for k in [512, 2048].iter() {
        let mut config: GemmTest = GemmTest::new(1, *k as usize, 512);

        group.bench_with_input(BenchmarkId::new("simple", k), k,
            |b, _c| b.iter(|| run_gemm_in_rust(&mut config)));
        group.bench_with_input(BenchmarkId::new("external", k), k,
            |b, _c| b.iter(|| run_gemm_external(&mut config)));
    }
    group.finish();
}

fn bench_gemm2(c: &mut Criterion) {

    // https://github.com/OpenMathLib/OpenBLAS/issues/4580
    let mut group = c.benchmark_group("openblas ticket");

    let mut config: GemmTest = GemmTest::new(1, 2048, 512);
    let name = format!("sgemm m={},k={},n={}",config.m,config.k,config.n);
    group.bench_function(BenchmarkId::new(name, 1),
        |b| b.iter(|| run_gemm_external(&mut config)));
    let name = format!("sgemv m={},n={}",config.k,config.n);
    group.bench_function(BenchmarkId::new(name, 1),
        |b| b.iter(|| run_gemv_external(&mut config)));

    let mut config: GemmTest = GemmTest::new(1, 512, 512);
    let name = format!("sgemm m={},k={},n={}",config.m,config.k,config.n);
    group.bench_function(BenchmarkId::new(name, 2),
        |b| b.iter(|| run_gemm_external(&mut config)));
    let name = format!("sgemv m={},n={}",config.k,config.n);
    group.bench_function(BenchmarkId::new(name, 1),
        |b| b.iter(|| run_gemv_external(&mut config)));

    let mut config: GemmTest = GemmTest::new(1, 512, 2048);
    let name = format!("sgemm m={},k={},n={}",config.m,config.k,config.n);
    group.bench_function(BenchmarkId::new(name, 3),
        |b| b.iter(|| run_gemm_external(&mut config)));
    let name = format!("sgemv m={},n={}",config.k,config.n);
    group.bench_function(BenchmarkId::new(name, 1),
        |b| b.iter(|| run_gemv_external(&mut config)));

    let mut config: GemmTest = GemmTest::new(1, 512, 32128);
    let name = format!("sgemm m={},k={},n={}",config.m,config.k,config.n);
    group.bench_function(BenchmarkId::new(name, 3),
        |b| b.iter(|| run_gemm_external(&mut config)));
    let name = format!("sgemv m={},n={}",config.k,config.n);
    group.bench_function(BenchmarkId::new(name, 1),
        |b| b.iter(|| run_gemv_external(&mut config)));

    group.finish();
}

// criterion_group!(benches, bench_gemm);
criterion_group!(benches, bench_gemm2);
criterion_main!(benches);