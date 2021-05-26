PROGRAM mpi_acc_magma

  USE mpi
  USE ISO_C_BINDING
  USE mod_prec
  USE mod_blas
  USE mod_magma

  ! Input variable
  INTEGER :: N

  ! argc, argv (Fortran 2003)
  INTRINSIC :: COMMAND_ARGUMENT_COUNT, GET_COMMAND_ARGUMENT
  INTEGER :: argc
  CHARACTER(LEN=8) :: argv ! We only need one

  ! MPI variables
  INTEGER :: ierr, myid, nranks
  INTEGER, PARAMETER :: root = 0

  ! MAGMA variables
  INTEGER(C_INT) :: stat ! cudaError_t
  TYPE(C_PTR) :: queue ! queue_t
  INTEGER :: devnum

  ! Arrays
  REAL(KIND=dd), DIMENSION(:,:), ALLOCATABLE :: matA
  REAL(KIND=dd), DIMENSION(:), ALLOCATABLE :: vecX, vecY, vecY1, check

  ! Scalars
  REAL(KIND=dd) :: alpha, beta, error

  ! Loop indices
  INTEGER :: i, j

  ! Initialize MPI
  CALL MPI_Init( ierr )
  CALL MPI_Comm_size( MPI_COMM_WORLD, nranks, ierr )
  CALL MPI_Comm_rank( MPI_COMM_WORLD, myid, ierr )

  ! Initialize MAGMA
  ! Note: this assumes the MPI ranks are on different nodes
  CALL magma_init()
  devnum = 0
  CALL magma_set_device( devnum )
  CALL magma_queue_create( devnum, queue )

  ! Check argc
  IF( COMMAND_ARGUMENT_COUNT() > 0 ) THEN

     ! Read N from argv[1]
     IF( myid == root ) THEN
        CALL GET_COMMAND_ARGUMENT( 1, VALUE=argv )
        READ( argv, *, IOSTAT=ierr ) N
        IF( ierr /= 0 ) THEN
           WRITE(*,*) 'Error: cannot parse N from first argument ', argv
           CALL MPI_Abort( MPI_COMM_WORLD, 1, ierr )
           STOP
        END IF ! ierr
        WRITE(*,*) 'Using N = ', N
     END IF ! root

  ELSE

     ! Read N from stdin
     IF( myid == root ) THEN
        READ(*,*) N
        WRITE(*,*) 'Using N = ', N
     END IF

  END IF

  ! Broadcast array dimension
  CALL MPI_Bcast( N, 1, MPI_INTEGER, root, MPI_COMM_WORLD, ierr )

  ! Initialize arrays
  ALLOCATE( matA(N,N) )
  ALLOCATE( vecX(N) )
  ALLOCATE( vecY(N) )
  ALLOCATE( vecY1(N) )
  ALLOCATE( check(N) )
  !$ACC DATA CREATE( matA, vecX, vecY, alpha, beta ) COPYIN( N )

  ! Fill in arrays
  DO j = 1, N
     DO i = 1, N
        matA(i,j) = REAL( i + j, KIND=dd )
     END DO ! i
     vecX(j) = REAL( j, KIND=dd )
  END DO ! j

  ! Set scalars
  alpha = 1._dd
  beta  = 0._dd

  ! Copy variables to device
  !$ACC UPDATE DEVICE( matA, vecX, alpha, beta )
  !$ACC WAIT

  ! Call MAGMA to perform DGEMV on device
  !$ACC HOST_DATA USE_DEVICE( matA, vecX, vecY )
  CALL magmablas_dgemv( MagmaNoTrans, N, N, &
                        alpha, C_LOC(matA), N, &
                               C_LOC(vecX), 1, &
                        beta,  C_LOC(vecY), 1, &
                        queue )
  !$ACC END HOST_DATA
  CALL magma_queue_sync( queue )

  ! Copy results to device
  !$ACC UPDATE HOST( vecY )
  !$ACC WAIT

  ! Check results with BLAS
  CALL DGEMV( 'N', N, N, &
              alpha, matA, N, &
                     vecX, 1, &
              beta,  vecY1, 1 )
  check(:) = vecY(:) - vecY1(:)
  error = DNRM2( N, check, 1 )
  WRITE(*,*) 'Rank ', myid, ': error = ', error

  ! Clean up
  !$ACC END DATA
  DEALLOCATE( matA )
  DEALLOCATE( vecX )
  DEALLOCATE( vecY )
  DEALLOCATE( vecY1 )
  DEALLOCATE( check )

  ! Finalize MAGMA
  CALL magma_queue_destroy( queue )
  CALL magma_finalize()

  ! Finalize MPI
  CALL MPI_Finalize( ierr )

  STOP
END PROGRAM mpi_acc_magma
