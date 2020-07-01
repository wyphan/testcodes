`ZGEMM` test
============

Tests the BLAS routine `ZGEMM` (double-precision complex matrix-matrix multiply) using a variety of compilers and libraries.

`ibm`:
- `ibmessl` - IBM XL Fortran compiler and IBM ESSL (on Summit)

`pgi`:
- `pgibuiltin` - PGI Fortran compiler and PGI-bundled BLAS
- `pgiessl` - PGI Fortran compiler and IBM ESSL (on Summit)
- `pgicublas` - PGI Fortran compiler, OpenACC, and cuBLAS
- `pgimagma` - PGI Fortran compiler, OpenACC,and [MAGMA](https://icl.utk.edu/magma/software/index.html)
