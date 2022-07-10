PROGRAM mpi_hello

  USE mpi
  IMPLICIT NONE

  ! Internal variables
  INTEGER :: nranks, myid, ierr

  CALL MPI_INIT( ierr )

  CALL MPI_COMM_SIZE( MPI_COMM_WORLD, nranks, ierr )
  CALL MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )

  PRINT *, 'Hello from rank ', myid, ' of ', nranks

  CALL MPI_FINALIZE( ierr )

  STOP
END PROGRAM mpi_hello
