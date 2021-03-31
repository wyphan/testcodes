MODULE mod_blas

  IMPLICIT NONE

  INTERFACE

     ! BLAS Level 1 function
     ! Double precision Euclidean norm
     DOUBLE PRECISION FUNCTION DNRM2( n, X, incx )
       INTEGER n, incx
       DOUBLE PRECISION X(*)
     END FUNCTION DNRM2

     ! BLAS Level 2 subroutine
     ! Double precision matrix-vector multiply C = alpha * A x + beta * y
     SUBROUTINE DGEMV( trans, m, n, &
                       alpha, A, lda, &
                              x, incx, &
                       beta,  y, incy )
       CHARACTER*1 trans
       INTEGER m, n, lda, incx, incy
       DOUBLE PRECISION alpha, beta
       DOUBLE PRECISION A(lda,*), X(*), Y(*)
     END SUBROUTINE DGEMV

  END INTERFACE

END MODULE mod_blas
