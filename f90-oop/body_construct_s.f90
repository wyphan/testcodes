SUBMODULE(body_m) body_construct_s
  IMPLICIT NONE

CONTAINS

  MODULE PROCEDURE body_from_values
    newbody%m = m0
    newbody%r = vec_t(x0, y0, z0)
    BLOCK
      REAL(dp) :: temp(3) = [0._dp, 0._dp, 0._dp]
      IF (PRESENT(vx0)) temp(1) = vx0
      IF (PRESENT(vy0)) temp(2) = vy0
      IF (PRESENT(vz0)) temp(3) = vz0
      newbody%v = vec_t(temp)
    END BLOCK
    BLOCK
      REAL(dp) :: temp(3) = [0._dp, 0._dp, 0._dp]
      IF (PRESENT(ax0)) temp(1) = ax0
      IF (PRESENT(ay0)) temp(2) = ay0
      IF (PRESENT(az0)) temp(3) = az0
      newbody%a = vec_t(temp)
    END BLOCK
  END PROCEDURE body_from_values

  MODULE PROCEDURE body_from_arrays
    newbody%m = m0
    newbody%r = vec_t(r0)
    IF (PRESENT(v0)) THEN
      newbody%v = vec_t(v0)
    ELSE
      newbody%v = vec_t(0._dp,0._dp,0._dp)
    END IF
    IF (PRESENT(a0)) THEN
      newbody%a = vec_t(a0)
    ELSE
      newbody%a = vec_t(0._dp,0._dp,0._dp)
    END IF
  END PROCEDURE body_from_arrays

  MODULE PROCEDURE body_from_vectors
    newbody%m = m0
    newbody%r = r0
    IF (PRESENT(v0)) THEN
      newbody%v = v0
    ELSE
      newbody%v = vec_t(0._dp,0._dp,0._dp)
    END IF
    IF (PRESENT(a0)) THEN
      newbody%a = a0
    ELSE
      newbody%a = vec_t(0._dp,0._dp,0._dp)
    END IF
  END PROCEDURE body_from_vectors

END SUBMODULE body_construct_s