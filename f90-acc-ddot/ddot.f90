PROGRAM dot_product

  ! Precision constants (Fortran 2008)
  USE iso_fortran_env, ONLY: dp => real64, longlong => int64

  IMPLICIT NONE

  ! Vector dimension
  INTEGER(KIND=longlong) :: N

  ! Loop index
  INTEGER(KIND=longlong) :: i

  ! Vectors to take the dot product with
  REAL(KIND=dp), DIMENSION(:), ALLOCATABLE :: vecA, vecB

  ! Result validation
  INTRINSIC :: LOG
  REAL(KIND=dp) :: result, check
  REAL(KIND=dp), PARAMETER :: tol = 1.0E-7_dp

  ! Timing subroutine
  INTRINSIC :: DATE_AND_TIME

  ! Timing vars
  INTEGER, DIMENSION(8) :: t0, t1, t2
  REAL(KIND=dp) :: time_init, time_ddot

  ! argc, argv (Fortran 2003)
  INTRINSIC :: COMMAND_ARGUMENT_COUNT, GET_COMMAND_ARGUMENT
  INTEGER :: argc
  CHARACTER(LEN=12) :: argv ! We only need one

  ! Check argc
  IF( COMMAND_ARGUMENT_COUNT() > 0 ) THEN

     ! Read n from argv[1]
     CALL GET_COMMAND_ARGUMENT( 1, VALUE=argv )
     READ( argv, * ) N

  ELSE

     ! Read n from standard input
     WRITE(*,*) 'Input vector length N:'
     READ(*,*) N

  END IF

  ! Echo n to standard output
  WRITE(*,*) 'Using N = ', N

  CALL DATE_AND_TIME( values=t0 )

  ! Allocate vectors on CPU
  ALLOCATE( vecA(N) )
  ALLOCATE( vecB(N) )

  ! Allocate vectors and result variable on device (GPU)
  !$ACC DATA CREATE( vecA, vecB, result )

  ! Initialize vectors on device
  !$ACC PARALLEL LOOP PRESENT( vecA, vecB ) COPYIN( N )
  DO i = 1, N
     vecA(i) = REAL( i,   KIND=dp )
     vecB(i) = REAL( 2*i, KIND=dp )
  END DO ! i
  !$ACC END PARALLEL LOOP

  ! Initialize result variable
  !$ACC KERNELS PRESENT( result )
  result = 0._dp
  !$ACC END KERNELS

  ! Barrier to ensure initialization completes
  !$ACC WAIT

  CALL DATE_AND_TIME( values=t1 )

  ! Perform dot product on device
  !$ACC PARALLEL LOOP PRESENT( vecA, vecB, result ) COPYIN( N ) &
  !$ACC   REDUCTION(+:result)
  DO i = 1, N
     result = result + vecA(i) * vecB(i)
  END DO ! i
  !$ACC END PARALLEL LOOP
  !$ACC WAIT

  CALL DATE_AND_TIME( values=t2 )

  ! Fetch result from device
  !$ACC UPDATE HOST( result )

  ! Validate result using logarithmic error
  check = LOG(REAL(N, KIND=dp)) + LOG(REAL(N, KIND=dp)+1) &
          + LOG(2*REAL(N, KIND=dp)+1) - LOG(3._dp)
  IF( ABS( LOG(result) - check ) > tol ) THEN
     WRITE(*,*) 'Error! log(result) = ', LOG(result), ' when it should be ', check
  ELSE
     WRITE(*,*) 'Success! Result = ', result
  END IF

  ! Close off data region
  !$ACC END DATA

  ! Calculate and display timers
  time_init = REAL( 3600*t1(5) + 60*t1(6) + t1(7), KIND=dp ) &
              + 0.001_dp*REAL( t1(8), KIND=dp ) &
              - REAL( 3600*t0(5) + 60*t0(6) + t0(7), KIND=dp ) &
              - 0.001_dp*REAL( t0(8), KIND=dp )
  time_ddot = REAL( 3600*t2(5) + 60*t2(6) + t2(7), KIND=dp ) &
              + 0.001_dp*REAL( t2(8), KIND=dp ) &
              - REAL( 3600*t1(5) + 60*t1(6) + t1(7), KIND=dp ) &
              - 0.001_dp*REAL( t1(8), KIND=dp )
  WRITE(*,'(A,F8.3,A)') 'Initialization took ', time_init, ' seconds.'
  WRITE(*,'(A,F8.3,A)') 'Computation took ', time_ddot, ' seconds.'
  
  ! Clean up
  DEALLOCATE( vecA )
  DEALLOCATE( vecB )

  STOP
END PROGRAM dot_product
