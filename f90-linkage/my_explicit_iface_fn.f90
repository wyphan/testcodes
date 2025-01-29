INTEGER FUNCTION my_explicit_iface_fn(arg) RESULT(ret)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: arg
  PRINT *, 'my_explicit_iface_fn is a function with an interface block in the main program', &
           ' and called with argument ', arg
  ret = 0
END FUNCTION my_explicit_iface_fn
