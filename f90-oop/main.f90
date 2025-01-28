PROGRAM classical_mechanics

  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: compiler_version, dp => real64
  USE body_m, ONLY: body_t
  USE vec_m, ONLY: vec_t
  IMPLICIT NONE

  TYPE(vec_t) :: zero, g

  PRINT '(2A)', 'Compiled with ', compiler_version()
  PRINT *

  zero = vec_t(0._dp, 0._dp, 0._dp)
  g = vec_t(0._dp, 0._dp, -9.81_dp)

  ! Free fall
  BLOCK
    TYPE(body_t) :: falling
    TYPE(vec_t) :: x0
    x0 = vec_t(0._dp, 0._dp, 100._dp)
    falling = body_t(1._dp, x0, a0=g)
    PRINT '(A)', '1. Free fall from ', x0%z()
    CALL print_time_evolution(falling, 0.25_dp, 18)
  END BLOCK

  PRINT *

  ! Cannonball
  BLOCK
    INTRINSIC :: SIN, ASIN, COS
    REAL(dp), PARAMETER :: pi = 2._dp * ASIN(1._dp)
    TYPE(body_t) :: cannonball
    REAL(dp) :: v = 20._dp
    REAL(dp) :: angle = 45._dp * pi / 180._dp
    CHARACTER(*), PARAMETER :: fmt = '(A,F12.7,2A,DT(12,7),A,DT(12,7))'

    cannonball = body_t(2._dp, zero, v0=vec_t(0._dp, v*SIN(angle), v*COS(angle)), a0=g)
    PRINT('(A)'), '2. Cannonball'
    CALL print_time_evolution(cannonball, 0.25_dp, 12)
  END BLOCK

CONTAINS

  SUBROUTINE print_time_evolution(body, dt, n)
    IMPLICIT NONE
    TYPE(body_t), INTENT(IN) :: body
    REAL(dp), INTENT(IN) :: dt
    INTEGER, INTENT(IN) :: n

    REAL(dp) :: t = 0._dp
    INTEGER :: i

    PRINT '(A,F12.7,2(A,DT(12,7)))', 't=', 0._dp, ' x=', body%position(), ' v=', body%velocity()    
    DO i = 1, n
      t = t + dt
      PRINT '(A,F12.7,2(A,DT(12,7)))', 't=', t, ' x=', body%position(t), ' v=', body%velocity(t)
    END DO

  END SUBROUTINE print_time_evolution

END PROGRAM classical_mechanics