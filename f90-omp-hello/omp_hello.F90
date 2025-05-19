PROGRAM omp_hello

#ifdef _OPENMP
  USE omp_lib
#endif /* _OPENMP */

  ! Internal variables
  INTEGER :: nthreads = 0
  INTEGER :: tid = 0

  !$OMP PARALLEL PRIVATE(tid)

#ifdef _OPENMP
  nthreads = OMP_GET_NUM_THREADS()
  tid = OMP_GET_THREAD_NUM()
#endif /* _OPENMP */

  PRINT *, 'Hello from thread ', tid, ' of ', nthreads

  !$OMP END PARALLEL

  STOP
END PROGRAM omp_hello
