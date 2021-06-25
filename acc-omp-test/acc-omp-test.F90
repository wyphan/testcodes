PROGRAM accomp

#ifdef _OPENACC
  USE openacc
#endif /* _OPENACC */

#ifdef _OPENMP
  USE omp_lib
#endif /* _OPENMP */

  IMPLICIT NONE

  ! OpenMP variables
  INTEGER :: nthreads, tid

  ! Vector dimension
  INTEGER, PARAMETER :: N = 2**20

  ! Device variables
  REAL, ALLOCATABLE :: matA(:,:), vecAsum(:)

  ! Result variable
  REAL :: matAsum

  ! Loop indices
  INTEGER :: i, j

  ! Other internal variables
  REAL :: tmp

  !$OMP PARALLEL PRIVATE(tid)
#ifdef _OPENMP
  nthreads = omp_get_num_threads()
  tid = omp_get_thread_num()
#else
  nthreads = 1
  tid = 0
#endif

  PRINT *, 'Hello from thread ', tid, ' of ', nthreads

  !$OMP END PARALLEL
  
  ! Allocate arrays
  ALLOCATE( matA( N, N ))
  ALLOCATE( vecAsum( N ))

  ! Note: Gfortran complains if ENTER/EXIT DATA is within OpenMP PARALLEL
  !$ACC ENTER DATA CREATE( matA, vecAsum )

  !$ACC PARALLEL

  ! Initialize arrays on device using OpenACC
  !$ACC LOOP COLLAPSE(2)
  DO j = 1, N
     DO i = 1, N
        matA(i,j) = REAL( i+j )
     END DO ! j
  END DO ! i
  !$ACC LOOP
  DO j = 1, N
     vecAsum(j) = 0.E0
  END DO ! j

  ! Sum over the columns of A on device using OpenACC
  !$ACC LOOP GANG PRIVATE(tmp)
  DO j = 1, N
     tmp = 0.E0
     !$ACC LOOP VECTOR REDUCTION(+:tmp)
     DO i = 1, N
        tmp = tmp + matA(i,j)
     END DO !j
     vecAsum(j) = tmp
  END DO

  !$ACC END PARALLEL

  ! Transfer vector to host
  !$ACC UPDATE HOST(vecAsum)
  !$ACC WAIT

  ! Sum the vector using OpenMP
  matAsum = 0
  !$OMP PARALLEL DO REDUCTION(+:matAsum)
  DO j = 1, N
     matAsum = matAsum + vecAsum(j)
  END DO ! j
  !$OMP END PARALLEL DO

  ! Print results to stdout
  PRINT *, 'Sum of matrix = ', matAsum

  ! Clean up
  !$ACC EXIT DATA DELETE( matA, vecAsum )
  DEALLOCATE( matA )
  DEALLOCATE( vecAsum )

  STOP
END PROGRAM accomp
