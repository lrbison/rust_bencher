
export BLAS_PATH=$(spack location -i armpl-gcc)/armpl_23.10_gcc-12.2/lib
export BLAS_LIB=armpl
cargo bench > result_armpl.log

export BLAS_PATH=$(spack location -i armpl-gcc)/armpl_23.10_gcc-12.2/lib
export BLAS_LIB=armpl_mp
OMP_NUM_THREADS=1 cargo bench > result_armplmp_1thread.log
OMP_NUM_THREADS=8 cargo bench > result_armplmp_8thread.log
OMP_NUM_THREADS=16 cargo bench > result_armplmp_16thread.log
OMP_NUM_THREADS=32 cargo bench > result_armplmp_32thread.log

export BLAS_PATH=$(spack location -i openblas threads=none)/lib
export BLAS_LIB=openblas
cargo bench > result_openblas_nothreads.log

export BLAS_PATH=$(spack location -i openblas threads=openmp)/lib
export BLAS_LIB=openblas
OMP_NUM_THREADS=1 cargo bench > result_openblas_1thread.log
OMP_NUM_THREADS=8 cargo bench > result_openblas_8thread.log
OMP_NUM_THREADS=16 cargo bench > result_openblas_16thread.log
OMP_NUM_THREADS=32 cargo bench > result_openblas_32thread.log