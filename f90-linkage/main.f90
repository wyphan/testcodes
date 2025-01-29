PROGRAM fortran_linkage

  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: compiler_version
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_int
  USE my_module, ONLY: my_module_fn
  IMPLICIT NONE

  EXTERNAL :: my_external_f77_sub

  INTERFACE
    FUNCTION my_explicit_iface_fn(arg) RESULT(ret)
      IMPLICIT NONE
      INTEGER, INTENT(in) :: arg
      INTEGER :: ret
    END FUNCTION my_explicit_iface_fn
    FUNCTION my_c_fn(arg) BIND(C)
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_int
      IMPLICIT NONE
      INTEGER(c_int), INTENT(IN), VALUE :: arg
      INTEGER(c_int) :: my_c_fn
    END FUNCTION my_c_fn
  END INTERFACE

  INTEGER :: dummy
  INTEGER(c_int) :: dummy_c

  PRINT *, 'Compiled with ', compiler_version()

  dummy = my_contained_fn(11)
  PRINT *, 'my_contained_fn returned ', dummy

  dummy = my_module_fn(22)
  PRINT *, 'my_module_fn returned ', dummy

  CALL my_external_f77_sub(33)

  CALL my_implicit_iface_sub(44)

  dummy = my_explicit_iface_fn(55)

  dummy_c = my_c_fn(66)
  PRINT *, 'my_c_fn returned ', dummy_c

CONTAINS

  INTEGER FUNCTION my_contained_fn(arg) RESULT(ret)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: arg
    PRINT *, "my_contained_fn is a function CONTAINed in the main program", &
             " and called with argument ", arg
    ret = 0
  END FUNCTION my_contained_fn

END PROGRAM fortran_linkage

SUBROUTINE my_implicit_iface_sub(arg)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: arg
  PRINT *, "my_implicit_iface_sub is a subroutine in the same file as the main program", &
           " and called with argument ", arg
END SUBROUTINE my_implicit_iface_sub
