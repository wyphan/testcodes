PROGRAM dot_product
  IMPLICIT NONE

  ! Precision constants
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14,100) ! 64-bit (double)

  ! Vector dimension
  INTEGER :: N

  ! Loop index
  INTEGER :: i

  ! Vectors to take the dot product with
  REAL(KIND=dp), DIMENSION(:), ALLOCATABLE :: vecA, vecB

  ! Result and check value
  REAL(KIND=dp) :: result, check

  ! Tolerance for error checking
  REAL(KIND=dp), PARAMETER :: tol = 10._dp**(-10)
  ! Timing subroutine
  INTRINSIC :: DATE_AND_TIME

  ! Timing vars
  INTEGER, DIMENSION(8) :: t0, t1, t2
  REAL(KIND=dp) :: time_init, time_ddot

  ! Read n from standard input
  WRITE(*,*) 'Input vector length n:'
  READ(*,*) n

  ! Echo n to standard output
  WRITE(*,*) 'Using n = ', n

  CALL DATE_AND_TIME( values=t0 )

  ! Allocate vectors on CPU
  ALLOCATE( vecA(n) )
  ALLOCATE( vecB(n) )

  ! Allocate vectors and result variable on device (GPU)
  !$ACC DATA CREATE( vecA, vecB, result )

  ! Initiqlize vectors on device
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

  CALL DATE_AND_TIME( values=t2 )

  ! Fetch result from device
  !$ACC UPDATE HOST( result )

  ! Check value ( using relative error ) and print result to standard output
  check = REAL( N, KIND=dp ) * REAL( N+1, KIND=dp ) * REAL( 2*N+1, KIND=dp) &
          / 3._dp
  IF( ABS( result/check - 1 ) > tol ) THEN
     WRITE(*,*) 'Error! Result = ', result, ' when it should be ', check
  ELSE
     WRITE(*,*) 'Success! Result = ', result
  END IF

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
  !$ACC END DATA
  DEALLOCATE( vecA )
  DEALLOCATE( vecB )

  STOP
END PROGRAM dot_product
