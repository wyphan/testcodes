MODULE body_m

  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: dp => real64
  USE vec_m, ONLY: vec_t
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: body_t

  TYPE body_t
    PRIVATE
    REAL(dp) :: m
    TYPE(vec_t) :: r, v, a
  CONTAINS
    PRIVATE
    PROCEDURE :: x0 => body_position_initial
    PROCEDURE :: x1 => body_position_at_time
    PROCEDURE :: v0 => body_velocity_initial
    PROCEDURE :: v1 => body_velocity_at_time
    PROCEDURE, PUBLIC :: mass => body_mass
    GENERIC, PUBLIC :: position => x0, x1
    GENERIC, PUBLIC :: velocity => v0, v1
    PROCEDURE, PUBLIC :: acceleration => body_acceleration
  END TYPE body_t

  ! Constructors from body_construct_s submodule
  INTERFACE body_t
    MODULE FUNCTION body_from_values(m0, x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0) RESULT(newbody)
      IMPLICIT NONE
      REAL(dp), INTENT(IN) :: m0, x0, y0, z0
      REAL(dp), INTENT(IN), OPTIONAL :: vx0, vy0, vz0, ax0, ay0, az0
      TYPE(body_t) :: newbody
    END FUNCTION body_from_values
    MODULE FUNCTION body_from_arrays(m0, r0, v0, a0) RESULT(newbody)
      IMPLICIT NONE
      REAL(dp), INTENT(IN) :: m0, r0(3)
      REAL(dp), INTENT(IN), OPTIONAL :: v0(3), a0(3)
      TYPE(body_t) :: newbody
    END FUNCTION body_from_arrays
    MODULE FUNCTION body_from_vectors(m0, r0, v0, a0) RESULT(newbody)
      IMPLICIT NONE
      REAL(dp), INTENT(IN) :: m0
      TYPE(vec_t), INTENT(IN) :: r0
      TYPE(vec_t), INTENT(IN), OPTIONAL :: v0, a0
      TYPE(body_t) :: newbody
    END FUNCTION body_from_vectors
  END INTERFACE body_t

  ! Functions from body_motion_s submodule
  INTERFACE
    MODULE FUNCTION body_position_at_time(body, time) RESULT(pos) ! from body_motion_s submodule
      IMPLICIT NONE
      CLASS(body_t), INTENT(IN) :: body
      REAL(dp), INTENT(IN) :: time
      TYPE(vec_t) :: pos
    END FUNCTION body_position_at_time
    MODULE FUNCTION body_velocity_at_time(body, time) RESULT(vel) ! from body_motion_s submodule
      IMPLICIT NONE
      CLASS(body_t), INTENT(IN) :: body
      REAL(dp), INTENT(IN) :: time
      TYPE(vec_t) :: vel
    END FUNCTION body_velocity_at_time
  END INTERFACE

CONTAINS

  ! Get function for body mass
  FUNCTION body_mass(body) RESULT(mas)

    CLASS(body_t), INTENT(IN) :: body
    REAL(dp) :: mas

    mas = body%m

  END FUNCTION body_mass

  ! Get function for body position
  FUNCTION body_position_initial(body) RESULT(pos)

    CLASS(body_t), INTENT(IN) :: body
    TYPE(vec_t) :: pos

    pos = body%r

  END FUNCTION body_position_initial

  ! Get function for body velocity
  FUNCTION body_velocity_initial(body) RESULT(vel)

    CLASS(body_t), INTENT(IN) :: body
    TYPE(vec_t) :: vel

    vel = body%v

  END FUNCTION body_velocity_initial

  ! Get function for body acceleration
  FUNCTION body_acceleration(body) RESULT(acc)

    CLASS(body_t), INTENT(IN) :: body
    TYPE(vec_t) :: acc

    acc = body%a

  END FUNCTION body_acceleration

END MODULE body_m
