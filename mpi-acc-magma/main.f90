PROGRAM mpi_acc_magma

  USE mpi
  USE ISO_C_BINDING
  USE mod_prec
  USE mod_blas
  USE mod_magma

  ! Input variable
  INTEGER :: N

  ! MPI variables
  INTEGER :: ierr, myid, nranks
  INTEGER, PARAMETER :: root = 0

  ! MAGMA variables
  INTEGER(C_INT) :: stat ! cudaError_t
  TYPE(C_PTR) :: queue ! queue_t
  INTEGER :: devnum

  ! Arrays
  REAL(KIND=dd), DIMENSION(:,:), ALLOCATABLE :: matA
  REAL(KIND=dd), DIMENSION(:), ALLOCATABLE :: vecX, vecY, vecYcheck

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

  ! Read N from stdin
  IF( myid == root ) THEN
     READ(*,*) N
     WRITE(*,*) 'Using N = ', N
  END IF

  ! Broadcast array dimension
  CALL MPI_Bcast( N, 1, MPI_INTEGER, root, MPI_COMM_WORLD, ierr )

  ! Initialize arrays
  ALLOCATE( matA(N,N) )
  ALLOCATE( vecX(N) )
  ALLOCATE( vecY(N) )
  ALLOCATE( vecYcheck(N) )
  !$ACC DATA CREATE( matA, vecX, vecY, alpha, beta ) COPYIN( N )

  ! Fill in arrays
  DO j = 1, N
     DO i = 1, N
        matA(i,j) = REAL( i + j, KIND=dd )
     END DO ! i
     vecX(i,j) = REAL( j, KIND=dd )
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
                        alpha, C_PTR(matA), N, &
                               C_PTR(vecX), 1, &
                        beta,  C_PTR(vecY), 1, &
                        queue )
  CALL magma_sync_queue( queue )

  ! Copy results to device
  !$ACC UPDATE HOST( vecY )
  !$ACC WAIT

  ! Check results with BLAS
  CALL DGEMV( 'N', N, N, &
              alpha, matA, N, &
                     vecX, 1, &
              beta,  vecYcheck, 1 )
  error = DNRM2( N, ABS( vecYcheck - vecY ), 1 )
  WRITE(*,*) 'Rank ', myid, ': error = ', error

  ! Clean up
  !$ACC END DATA
  DEALLOCATE( matA )
  DEALLOCATE( vecX )
  DEALLOCATE( vecY )
  DEALLOCATE( vecYcheck )

  ! Finalize MAGMA
  CALL magma_queue_destroy( queue )
  CALL magma_finalize()

  ! Finalize MPI
  CALL MPI_Finalize( ierr )

  STOP
END PROGRAM mpi_acc_magma
