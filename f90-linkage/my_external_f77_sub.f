      SUBROUTINE MY_EXTERNAL_F77_SUB(IARG)
        PRINT *, 'my_external_f77_sub is a subroutine declared as',
     &           ' EXTERNAL in the main program and',
     &           ' called with argument ', IARG
      END SUBROUTINE
