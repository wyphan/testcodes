PROGRAM compiler_ifdefs
  IMPLICIT NONE

#ifdef __GFORTRAN__
  CHARACTER(LEN=*), PARAMETER :: compiler = 'GCC GFortran'
#endif /* __GFORTRAN__ */

#ifdef __INTEL_COMPILER
  CHARACTER(LEN=*), PARAMETER :: compiler = 'Intel Fortran'
#endif /* __INTEL_COMPILER */

#ifdef __PGI
#ifdef __NVCOMPILER
  ! Note: NVHPC also defines __PGI for backwards compatibility
  CHARACTER(LEN=*), PARAMETER :: compiler = 'NVIDIA HPC Fortran'
#else
  CHARACTER(LEN=*), PARAMETER :: compiler = 'PGI Fortran'
#endif /* __NVCOMPILER */
#endif /* __PGI */

#ifdef __FLANG
  CHARACTER(LEN=*), PARAMETER :: compiler = 'LLVM Flang'
#endif /* __FLANG */

#ifdef __IBMC__
  CHARACTER(LEN=*), PARAMETER :: compiler = 'IBM XL Fortran'
#endif /* __IBMC__ */

#ifdef _CRAYFTN
  CHARACTER(LEN=*), PARAMETER :: compiler = 'Cray Fortran'
#endif /* _CRAYFTN */

  PRINT *, 'Compiled with ', compiler

  STOP
END PROGRAM compiler_ifdefs
