PROGRAM loop
  USE ISO_FORTRAN_ENV, ONLY: compiler_version, real64
  IMPLICIT NONE

  INTEGER, PARAMETER :: N = 10
  INTEGER :: i
  REAL(real64) :: a(N)

  PRINT *, 'Compiled using ', compiler_version()

  a(:) = [( 0.1_real64 * i, i = 1, N )]
  DO i = 1, N
    PRINT *, 'a(', i, ') = ', a(i)
  END DO

  STOP
END PROGRAM loop
