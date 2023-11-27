#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include "common.h"

int main (int argc, char *argv[]) {

  // Initialize MPI
  MPI_Init(&argc, &argv);
  init_mpi_vars(&myid, &nranks);
  if (myid == 0) {
    printf("Rank 0: Using %i ranks\n", nranks);
  }

  // Enumerate GPUs for each rank
  char** gpu_luid = NULL;
  char** gpu_name = NULL;
  int ngpus = find_GPUs(&gpu_luid, &gpu_name);

  // Collect GPU info at rank 0 and print it to stdout
  int ngpus_total;
  int* ngpus_all = NULL;
  char** gpu_luid_all = NULL;
  char** gpu_name_all = NULL;
  ngpus_total = collect_GPUs(ngpus, gpu_luid, gpu_name, &ngpus_all, &gpu_luid_all, &gpu_name_all);
  if (myid == 0) {
    if (ngpus_total > 0) {
      printf("Rank 0: Detected %i GPUs in total.\n", ngpus_total);
      unsigned int idx = 0;
      for (unsigned int r = 0; r < nranks; r++) {
	print_gpus(r, ngpus_all[r], idx, gpu_luid_all, gpu_name_all);
	idx += ngpus_all[r];
      } // r
    } else {
      parabort(ERR_TYPE_CUSTOM, 0, "No GPUs detected. Exiting...");
    } // ngpus_total

    free(ngpus_all);
    free(gpu_luid_all);
    free(gpu_name_all);
    
  } // myid

  free(gpu_luid);
  free(gpu_name);
  

  MPI_Finalize();
  return 0;
}

