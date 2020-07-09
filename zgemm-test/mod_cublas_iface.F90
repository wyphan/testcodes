MODULE cublas_iface

#ifdef _OPENACC

  USE cudafor
  USE cublas_v2
  TYPE(cublasHandle) :: mycublas

#else

  USE ISO_C_BINDING

!==============================================================================
! These values are taken from "Fortran CUDA Library Interfaces" by PGI
!==============================================================================

  ! cublasStatus_t
  ENUM, BIND(C)
     ENUMERATOR :: CUBLAS_STATUS_SUCCESS          = 0
     ENUMERATOR :: CUBLAS_STATUS_NOT_INITIALIZED  = 1
     ENUMERATOR :: CUBLAS_STATUS_ALLOC_FAILED     = 3
     ENUMERATOR :: CUBLAS_STATUS_INVALID_VALUE    = 7
     ENUMERATOR :: CUBLAS_STATUS_ARCH_MISMATCH    = 8
     ENUMERATOR :: CUBLAS_STATUS_MAPPING_ERROR    = 11
     ENUMERATOR :: CUBLAS_STATUS_EXECUTION_FAILED = 13
     ENUMERATOR :: CUBLAS_STATUS_INTERNAL_ERROR   = 14
  END ENUM

  ! cublasOperation_t
  ENUM, BIND(C)
     ENUMERATOR :: CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
  END ENUM

  ! cublasFillMode_t
  ENUM, BIND(C)
     ENUMERATOR :: CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
  END ENUM

  ! cublasDiagType_t
  ENUM, BIND(C)
     ENUMERATOR :: CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT
  END ENUM

  ! cublasSideMode_t
  ENUM, BIND(C)
     ENUMERATOR :: CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT
  END ENUM

  ! cublasPointerMode_t
  ENUM, BIND(C)
     ENUMERATOR :: CUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_DEVICE
  END ENUM

  ! cublasHandle_t
  TYPE cublasHandle
     TYPE(C_PTR) :: handle
  END type cublasHandle
  TYPE(cublasHandle) :: cublashandle

!==============================================================================
! These interfaces are taken from "Fortran CUDA Library Interfaces" by PGI`
!==============================================================================

  INTERFACE

     INTEGER(KIND=C_INT) FUNCTION cublasCreate( handle ) &
                         BIND(C, NAME='cublasCreate_v2')
       USE ISO_C_BINDING
       TYPE(C_PTR), VALUE :: handle
     END FUNCTION cublasCreate

     INTEGER(KIND=C_INT) FUNCTION cublasDestroy( handle ) &
                         BIND(C, NAME='cublasDestroy_v2')
       USE ISO_C_BINDING
       TYPE(C_PTR), VALUE :: handle
     END FUNCTION cublasDestroy

     INTEGER(KIND=C_INT) FUNCTION cublasZgemm_v2( handle, transA, transB, &
                                                  m, n, k, &
                                                  alpha, dA, lda, &
                                                         dB, ldb, &
                                                  beta,  dC, ldc ) &
                         BIND(C, NAME='cublasZgemm_v2')
       USE ISO_C_BINDING
       TYPE(C_PTR), VALUE :: handle
       TYPE(C_PTR), INTENT(IN) :: dA, dB
       TYPE(C_PTR), INTENT(INOUT) :: dC
       INTEGER(KIND=C_INT), INTENT(IN) :: transA, transB, m, n, k, lda, ldb, ldc
       ! Technically, alpha & beta can be already on device (instead of host)
       COMPLEX(KIND=C_DOUBLE_COMPLEX), INTENT(IN) :: alpha, beta
     END FUNCTION cublasZgemm_v2

  END INTERFACE

#endif /* _OPENACC */

!==============================================================================
! For helper function cublas_char
  INTEGER, PARAMETER :: &
       trans = 0, &
       uplo  = 1, &
       diag  = 2, &
       side  = 3, &
       ptr   = 10

!==============================================================================

CONTAINS

!==============================================================================

  SUBROUTINE cublas_init_f
    IMPLICIT NONE

    ! Local variables
    INTEGER :: ierr

    ! Only the master thread has access to cuBLAS
    !$OMP MASTER
    ierr = cublasCreate( mycublas )
    IF( ierr /= CUBLAS_STATUS_SUCCESS ) CALL cublas_error('cublasCreate', ierr)
    !$OMP END MASTER

    RETURN
  END SUBROUTINE cublas_init_f

!==============================================================================

  SUBROUTINE cublas_fin_f
    IMPLICIT NONE

    ! Local variables
    INTEGER :: ierr

    ! Only the master thread has access to cuBLAS
    !$OMP MASTER
    ierr = cublasDestroy( mycublas )
    IF( ierr /= CUBLAS_STATUS_SUCCESS ) CALL cublas_error('cublasDestroy', ierr)
    !$OMP END MASTER

    RETURN
  END SUBROUTINE cublas_fin_f

!==============================================================================

  SUBROUTINE cublas_zgemm_f( transA, transB, m, n, k, &
                             alpha, dA, lda, &
                                    dB, ldb, &
                             beta,  dC, ldc )
#ifdef _OPENACC
    USE cublas_v2
    USE openacc
#else
    USE ISO_C_BINDING
#endif

    IMPLICIT NONE

    ! Arguments
    CHARACTER(LEN=1), INTENT(IN) :: transA, transB
    INTEGER(KIND=C_INT), VALUE :: m, n, k, lda, ldb, ldc
    COMPLEX(KIND=C_DOUBLE_COMPLEX), VALUE :: alpha, beta

#ifdef _OPENACC
    ! The device pointers will be extracted
    COMPLEX(KIND=C_DOUBLE_COMPLEX), DIMENSION(lda,*), INTENT(IN) :: dA
    COMPLEX(KIND=C_DOUBLE_COMPLEX), DIMENSION(ldb,*), INTENT(IN) :: dB
    COMPLEX(KIND=C_DOUBLE_COMPLEX), DIMENSION(ldc,*), INTENT(INOUT) :: dC
#else
    ! These are already device pointers
    TYPE(C_PTR), INTENT(IN) :: dA, dB
    TYPE(C_PTR), INTENT(INOUT) :: dC
#endif /* _OPENACC */

    ! Internal variables
    INTEGER :: ierr
    INTEGER(KIND=C_INT) :: op_a, op_b
#ifdef _OPENACC
    TYPE(C_PTR) :: dptr_a, dptr_b, dptr_c
#endif

    ! Only the master thread has access to cuBLAS
    !$OMP MASTER

    ! Map transA and transB to enum using helper function
    op_a = cublas_char( 'cublas_zgemm_f', transA, trans )
    op_b = cublas_char( 'cublas_zgemm_f', transB, trans )

#ifdef _OPENACC

    ! Expose device pointers
    !$ACC DATA PRESENT( dA, dB, dC )
    !$ACC HOST_DATA USE_DEVICE( dA, dB, dC )

    ! Call cuBLAS with exposed device pointers
    ierr = cublasZgemm_v2( mycublas, op_a, op_b, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc )
    IF( ierr /= CUBLAS_STATUS_SUCCESS ) CALL cublas_error('cublasZgemm_v2', ierr)

    !$ACC END HOST_DATA
    !$ACC END DATA

#else

    ! Call cuBLAS with device pointers passed directly
    ierr = cublasZgemm_v2( mycublas, op_a, op_b, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc )
    IF( ierr /= CUBLAS_STATUS_SUCCESS ) CALL cublas_error('cublasZgemm_v2', ierr)

#endif /* _OPENACC */

    !$OMP END MASTER

    RETURN
  END SUBROUTINE cublas_zgemm_f

!==============================================================================
! Helper function to print error message
!==============================================================================
  SUBROUTINE cublas_error( fname, ierr )
    IMPLICIT NONE

    CHARACTER(LEN=*), INTENT(IN) :: fname
    INTEGER, INTENT(IN) :: ierr

    SELECT CASE( ierr )
    CASE(CUBLAS_STATUS_NOT_INITIALIZED)
       WRITE(*,*) fname, ': error 1 (cuBLAS not yet initialized)'
    CASE(CUBLAS_STATUS_ALLOC_FAILED)
       WRITE(*,*) fname, ': error 3 (Memory allocation failed)'
    CASE(CUBLAS_STATUS_INVALID_VALUE)
       WRITE(*,*) fname, ': error 7 (Invalid argument value)'
    CASE(CUBLAS_STATUS_ARCH_MISMATCH)
       WRITE(*,*) fname, ': error 8 (Unavailable feature)'
    CASE(CUBLAS_STATUS_MAPPING_ERROR)
       WRITE(*,*) fname, ': error 11 (Failed to map device memory / bind texture)'
    CASE(CUBLAS_STATUS_EXECUTION_FAILED)
       WRITE(*,*) fname, ': error 13 (Failed to launch kernel)'
    CASE(CUBLAS_STATUS_INTERNAL_ERROR)
       WRITE(*,*) fname, ': error 14 (Internal cuBLAS error)'
    CASE DEFAULT
       WRITE(*,*) fname, ': error ', ierr
    END SELECT

    RETURN
  END SUBROUTINE cublas_error

!==============================================================================
! Helper function to translate chars to enums
!==============================================================================
  INTEGER(C_INT) FUNCTION cublas_char( fname, char, type ) RESULT(num)
    IMPLICIT NONE

    CHARACTER(LEN=*), INTENT(IN) :: fname
    CHARACTER(LEN=1), INTENT(IN) :: char
    INTEGER, INTENT(IN) :: type

    SELECT CASE( type )

    CASE( trans )
       SELECT CASE( char )
       CASE( 'N', 'n' )
          num = CUBLAS_OP_N
       CASE( 'T', 't' )
          num = CUBLAS_OP_T
       CASE( 'C', 'c' )
          num = CUBLAS_OP_C
       CASE DEFAULT
          WRITE(*,*) fname, ': unrecognized option for trans (N/T/C): ', char
       END SELECT

    CASE( uplo )
       SELECT CASE( char )
       CASE( 'L', 'l' )
          num = CUBLAS_FILL_MODE_LOWER
       CASE( 'U', 'u' )
          num = CUBLAS_FILL_MODE_UPPER
       CASE DEFAULT
          WRITE(*,*) fname, ': unrecognized option for uplo (L/U): ', char
       END SELECT

    CASE( diag )
       SELECT CASE( char )
       CASE( 'N', 'n' )
          num = CUBLAS_DIAG_NON_UNIT
       CASE( 'U', 'u' )
          num = CUBLAS_DIAG_UNIT
       CASE DEFAULT
          WRITE(*,*) fname, ': unrecognized option for diag (N/U): ', char
       END SELECT

    CASE( side )
       SELECT CASE( char )
       CASE( 'L', 'l' )
          num = CUBLAS_SIDE_LEFT
       CASE( 'R', 'r' )
          num = CUBLAS_SIDE_RIGHT
       CASE DEFAULT
          WRITE(*,*) fname, ': unrecognized option for side (L/R): ', char
       END SELECT

    CASE( ptr )
       SELECT CASE( char )
       CASE( 'D', 'd' )
          num = CUBLAS_POINTER_MODE_DEVICE
       CASE( 'H', 'h' )
          num = CUBLAS_POINTER_MODE_HOST
       CASE DEFAULT
          WRITE(*,*) fname, ': unrecognized option for pointer mode (D/H): ', char
       END SELECT

    CASE DEFAULT
       WRITE(*,*) fname, ': unrecognized option ', char
    END SELECT

    RETURN
  END FUNCTION cublas_char

!==============================================================================

END MODULE cublas_iface
