MODULE mod_prec

  IMPLICIT NONE

  ! Precision constants
  INTEGER, PARAMETER :: dl = SELECTED_INT_KIND(18)      ! 64-bit (long)
  INTEGER, PARAMETER :: dd = SELECTED_REAL_KIND(14,100) ! 64-bit (double)

  ! Dummy variables for sizeof() check
  INTEGER(KIND=dl), PRIVATE :: dummy_l(1)
  REAL(KIND=dd), PRIVATE :: dummy_d(1)

#if defined(__GFORTRAN__) || defined(__INTEL_COMPILER) || defined(__IBMC__) || defined(_PGI) || defined(_NVCOMPILER)
  ! GCC, Intel, IBM, PGI/NVHPC include SIZEOF() as an extension
  INTEGER(KIND=dl), PARAMETER :: sz_l = SIZEOF(dummy_l(1))
  INTEGER(KIND=dl), PARAMETER :: sz_d = SIZEOF(dummy_d(1))
#endif /* compiler */

END MODULE
