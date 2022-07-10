#include <mpi.h>
#include <stdio.h>

int main (int argc, char *argv[]) {

  int ierr, nranks, myid;

  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  printf("Hello from rank %i of %i\n", \
         myid, nranks);
  ierr = MPI_Finalize();

  return 0;
}
