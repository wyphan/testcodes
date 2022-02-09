`ZGEMM` test
============

Tests the BLAS routine `ZGEMM` (double-precision complex matrix-matrix multiply) using a variety of compilers and libraries.

`ibm`:
- `make ibmessl` - IBM XL Fortran compiler and IBM [ESSL](https://www.ibm.com/support/knowledgecenter/en/SSFHY8_6.2/navigation/welcome.html) (on Summit)

`pgi`:
- `make pgibuiltin` - PGI Fortran compiler and PGI-bundled OpenBLAS
- `make pgiessl` - PGI Fortran compiler and IBM [ESSL](https://www.ibm.com/support/knowledgecenter/en/SSFHY8_6.2/navigation/welcome.html) (on Summit)
- `make pgicublas` - PGI Fortran compiler, OpenACC, and [cuBLAS](https://developer.nvidia.com/cublas)
- `make pgimagma` - PGI Fortran compiler, OpenACC, and [MAGMA](https://icl.utk.edu/magma/software/index.html)

`nv`:
- `make nvbuiltin` - NVIDIA HPC SDK and its bundled OpenBLAS
- `make nvessl` - NVIDIA HPC SDK and IBM [ESSL](https://www.ibm.com/support/knowledgecenter/en/SSFHY8_6.2/navigation/welcome.html) (on Summit)
- `make nvcublas` - NVIDIA HPC SDK, OpenACC, and [cuBLAS](https://developer.nvidia.com/cublas)
- `make nvmagma` - NVIDIA HPC SDK, OpenACC, and [MAGMA](https://icl.utk.edu/magma/software/index.html)

`llvm`:
- `make flangaocl` - [AMD AOCC](https://developer.amd.com/amd-aocc/) `flang` compiler and [AMD AOCL](https://developer.amd.com/amd-aocl/)
