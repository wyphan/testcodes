MODULE vec_m

  USE ISO_FORTRAN_ENV, ONLY: dp => real64
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: vec_t, OPERATOR(.dot.), OPERATOR(.cross.)

  TYPE vec_t
    PRIVATE
    REAL(dp) :: v(3)
  CONTAINS
    PRIVATE
    PROCEDURE :: vec_add
    PROCEDURE, PASS(vec) :: vec_scale_sv, vec_scale_vs
    PROCEDURE :: vec_print_fmt
    PROCEDURE, PUBLIC :: x => vec_x
    PROCEDURE, PUBLIC :: y => vec_y
    PROCEDURE, PUBLIC :: z => vec_z
    GENERIC, PUBLIC :: OPERATOR(+) => vec_add
    GENERIC, PUBLIC :: OPERATOR(*) => vec_scale_sv, vec_scale_vs
    GENERIC, PUBLIC :: WRITE(FORMATTED) => vec_print_fmt
  END TYPE vec_t

  ! Constructors for vec_t
  INTERFACE vec_t    
    MODULE FUNCTION vec_from_vals(x, y, z) RESULT(newvec)
      IMPLICIT NONE
      REAL(dp), INTENT(IN) :: x, y, z
      TYPE(vec_t) :: newvec
    END FUNCTION vec_from_vals
    MODULE FUNCTION vec_from_array(arr) RESULT(newvec)
      IMPLICIT NONE
      REAL(dp), INTENT(IN) :: arr(3)
      TYPE(vec_t) :: newvec
    END FUNCTION vec_from_array
  END INTERFACE vec_t

  ! Dot product operator
  INTERFACE OPERATOR(.dot.)
    MODULE PROCEDURE vec_dot_prod
  END INTERFACE

  ! Cross product operator
  INTERFACE OPERATOR(.cross.)
    MODULE PROCEDURE vec_cross_prod
  END INTERFACE

CONTAINS

  ! Constructor for vec_t from three REAL values
  MODULE PROCEDURE vec_from_vals
    newvec%v(1) = x
    newvec%v(2) = y
    newvec%v(3) = z
  END PROCEDURE vec_from_vals

  ! Constructor for vec_t from array of REALs
  MODULE PROCEDURE vec_from_array
    newvec%v = arr
  END PROCEDURE vec_from_array

  ! Get x component
  FUNCTION vec_x(vec) RESULT(x)
    CLASS(vec_t), INTENT(IN) :: vec
    REAL(dp) :: x

    x = vec%v(1)
  END FUNCTION vec_x

  ! Get y component
  FUNCTION vec_y(vec) RESULT(y)
    CLASS(vec_t), INTENT(IN) :: vec
    REAL(dp) :: y

    y = vec%v(2)
  END FUNCTION vec_y

  ! Get z component
  FUNCTION vec_z(vec) RESULT(z)
    CLASS(vec_t), INTENT(IN) :: vec
    REAL(dp) :: z

    z = vec%v(3)
  END FUNCTION vec_z

  ! Vector addition
  ELEMENTAL FUNCTION vec_add(a, b) RESULT(r)
    CLASS(vec_t), INTENT(IN) :: a, b
    TYPE(vec_t) :: r

    r%v = a%v + b%v

  END FUNCTION vec_add

  ! Scalar times vector
  FUNCTION vec_scale_sv(scale, vec) RESULT(res)
    REAL(dp), INTENT(IN) :: scale
    CLASS(vec_t), INTENT(IN) :: vec
    TYPE(vec_t) :: res

    res%v = scale * vec%v

  END FUNCTION vec_scale_sv

  ! Vector times scalar
  FUNCTION vec_scale_vs(vec, scale) RESULT(res)
    CLASS(vec_t), INTENT(IN) :: vec
    REAL(dp), INTENT(IN) :: scale
    TYPE(vec_t) :: res

    res%v = scale * vec%v

  END FUNCTION vec_scale_vs

  ! Dot product operator
  FUNCTION vec_dot_prod(a, b) RESULT(r)
    CLASS(vec_t), INTENT(IN) :: a, b
    REAL(dp) :: r

    ASSOCIATE( &
      ax => a%v(1), ay => a%v(2), az => a%v(3), &
      bx => b%v(1), by => b%v(2), bz => b%v(3)  &
    )
      r = ax*bx + ay*by + az*bz
    END ASSOCIATE

  END FUNCTION vec_dot_prod

  ! Cross product operator
  FUNCTION vec_cross_prod(a, b) RESULT(c)
    CLASS(vec_t), INTENT(IN) :: a, b
    TYPE(vec_t) :: c

    ASSOCIATE( &
      ax => a%v(1), ay => a%v(2), az => a%v(3), &
      bx => b%v(1), by => b%v(2), bz => b%v(3), &
      cx => c%v(1), cy => c%v(2), cz => c%v(3)  &
    )
      cx = ay*bz - az*by
      cy = az*bx - ax*bz
      cz = ax*by - ay*bx
    END ASSOCIATE

  END FUNCTION vec_cross_prod

  ! Print function
  SUBROUTINE vec_print_fmt(vec, unit, iotype, v_list, iostat, iomsg)
    CLASS(vec_t), INTENT(IN) :: vec
    INTEGER, INTENT(IN) :: unit, v_list(:)
    CHARACTER(*), INTENT(IN) :: iotype
    INTEGER, INTENT(OUT) :: iostat
    CHARACTER(*), INTENT(INOUT) :: iomsg

    IF(iotype == 'LISTDIRECTED') THEN
      WRITE(unit, FMT='("(",2(G0,", "),G0,")")', IOSTAT=iostat) vec%v(1), vec%v(2), vec%v(3)
    ELSE IF(SIZE(v_list,1) == 2) THEN
      BLOCK
        CHARACTER(LEN=30) :: fmtstr
        CHARACTER(LEN=6) :: field
        WRITE(field, FMT='("F",I2,".",I2)') v_list(1), v_list(2)
        WRITE(fmtstr, FMT='(5A)') '("[",2(', field, ',", ")', field, ',"]")'
        WRITE(unit, FMT=fmtstr, IOSTAT=iostat) vec%v(1), vec%v(2), vec%v(3)
      END BLOCK
    END IF

  END SUBROUTINE vec_print_fmt

END MODULE vec_m
