MODULE my_module
  IMPLICIT NONE

CONTAINS

  INTEGER FUNCTION my_module_fn(arg) RESULT(ret)
    IMPLICIT NONE
    INTEGER, INTENT(in) :: arg
    PRINT *, 'my_contained_fn is a module function from my_module', &
             ' and called with argument ', arg
    ret = 0
  END FUNCTION my_module_fn

END MODULE my_module
