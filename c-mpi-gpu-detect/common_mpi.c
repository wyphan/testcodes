#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "common.h"

// Helper macros for error checking
#define MPI_CHECK(command) {              \
  int status = command;                   \
  if ( status != MPI_SUCCESS ) {          \
    parabort(ERR_TYPE_MPI, status, NULL); \
  }                                       \
}

void parabort
( enum errtype_t errtype, int errcode, const char* errmsg ) {

  switch (errtype) {
  case ERR_TYPE_CUSTOM:
    fprintf(stderr, "Rank %i: %s\n", myid, errmsg);
    break;
  case ERR_TYPE_MPI:
    fprintf(stderr, "Rank %i: MPI error %i\n", myid, errcode);
    break;
  case ERR_TYPE_GPU:
    fprintf(stderr, "Rank %i: %s error %i\n", myid, get_gpu_api_name(), errcode);
    break;
  }
  MPI_Abort(MPI_COMM_WORLD, errcode);
}

void init_mpi_vars
(int* rank, int* nranks) {
  MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, nranks) );
  MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, rank) );
}

int collect_GPUs
( const int ngpus, char** luid, char** name,
  int* *ngpu_list, char** *luid_list, char** *name_list ) {

  MPI_Datatype MPI_SIZE_T;
  MPI_CHECK( MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(size_t), &MPI_SIZE_T) );

  int ngpus_total = 0;
  MPI_CHECK ( MPI_Allreduce(&ngpus, &ngpus_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) );
  if (myid == 0) {
    *ngpu_list = (int*) malloc( nranks * sizeof(int) );
  } // myid
  MPI_CHECK ( MPI_Gather(&ngpus, 1, MPI_INT, &(*ngpu_list)[myid], 1, MPI_INT, 0, MPI_COMM_WORLD) );

  if (myid == 0 && ngpus_total > 0) {

    size_t* namelen = (size_t*)malloc( ngpus_total * sizeof(size_t) );
    *luid_list = (char**)malloc( ngpus_total * sizeof(char*) );
    *name_list = (char**)malloc( ngpus_total * sizeof(char*) );
    for (unsigned int g = 0; g < ngpus; g++) {
      (*luid_list)[g] = strdup(luid[g]);
      (*name_list)[g] = strdup(name[g]);
      namelen[g] = strlen(name[g]);
    }

    if (nranks > 1 && ngpus_total > ngpus) {
      MPI_Request* req = (MPI_Request*)malloc( 2 * (ngpus_total-ngpus) * sizeof(MPI_Request) );

      unsigned int i = ngpus;
      for (unsigned int r = 1; r < nranks; r++) {
	unsigned int rank_ngpus = (*ngpu_list)[r];
	MPI_CHECK( MPI_Irecv(&namelen[i], rank_ngpus, MPI_SIZE_T, r, r, MPI_COMM_WORLD, &req[r-1]) );
	i += rank_ngpus;
      } // r
      MPI_CHECK( MPI_Waitall(nranks-1, req, MPI_STATUSES_IGNORE) );

      for (unsigned int g = 0; g < ngpus_total; g++) {
	printf("namelen[%i] = %i\n", g, namelen[g]);
      }

      i = 0;
      for (unsigned int r = 1; r < nranks; r++) {
	unsigned int rank_ngpus = (*ngpu_list)[r];
	if (rank_ngpus > 0) {
	  for (unsigned int g = 0; g < rank_ngpus; g++) {
	    (*luid_list)[ngpus+i] = (char*)malloc( 8 * sizeof(char) );
	    (*name_list)[ngpus+i] = (char*)malloc( namelen[ngpus+i] * sizeof(char) );
	    MPI_CHECK( MPI_Irecv((*luid_list)[ngpus+i], 8, MPI_CHAR, r, 2*r, MPI_COMM_WORLD, &req[2*i]) );
	    MPI_CHECK( MPI_Irecv((*name_list)[ngpus+i], namelen[ngpus+i], MPI_CHAR, r, 2*r+1, MPI_COMM_WORLD, &req[2*i+1]) );
	    i++;
	  }
	} // rank_ngpus
      } // r
      MPI_CHECK( MPI_Waitall(i, req, MPI_STATUSES_IGNORE) );

    } // nranks

    unsigned int i = 0;
    for (unsigned int r = 0; r < nranks; r++) {
      unsigned int rank_ngpus = (*ngpu_list)[r];
      for (unsigned int g = 0; g < rank_ngpus; g++) {
	printf("Rank %i GPU %i: %i %s %s\n", r, g, i, (*luid_list)[i], (*name_list)[i]);
	i++;
      }
    }
    
  } else if (myid != 0 && ngpus > 0) {

    MPI_Request* req = (MPI_Request*)malloc( 2 * ngpus * sizeof(MPI_Request) );    
    size_t* namelen = (size_t*)malloc( ngpus * sizeof(size_t) );
    for (unsigned int g = 0; g < ngpus; g++) {
      namelen[g] = strlen(name[g]);
    }
    MPI_CHECK( MPI_Isend(namelen, ngpus, MPI_SIZE_T, 0, myid, MPI_COMM_WORLD, &req[0]) );
    MPI_CHECK( MPI_Waitall(1, req, MPI_STATUSES_IGNORE) );

    for (unsigned int g = 0; g < ngpus; g++) {
      MPI_CHECK( MPI_Isend(luid[g], 8, MPI_CHAR, 0, 2*myid, MPI_COMM_WORLD, &req[2*g]) );
      MPI_CHECK( MPI_Isend(name[g], namelen[g], MPI_CHAR, 0, 2*myid+1, MPI_COMM_WORLD, &req[2*g+1]) );
    }
    MPI_CHECK( MPI_Waitall(2*ngpus, req, MPI_STATUSES_IGNORE) );

  } // myid

  return ngpus_total;
}

void print_gpus
( const int rank, const int ngpus, const int start_idx,
  char** luid, char** name ) {

  printf("Rank %i detects %i GPUs.\n", rank, ngpus);
  for (unsigned int g = 0; g < ngpus; g++) {
    const char* this_gpu_luid = strdup(luid[start_idx+g]);
    const char* this_gpu_name = strdup(name[start_idx+g]);
    printf("  GPU %i: %s %s\n", g, this_gpu_luid, this_gpu_name);
  } // g

}
