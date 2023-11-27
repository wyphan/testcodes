#ifndef COMMON_H
#define COMMON_H

// Global variables
int myid, nranks;

// Error types for parabort
typedef enum errtype_t
  { ERR_TYPE_MPI,
    ERR_TYPE_GPU,
    ERR_TYPE_CUSTOM }
  errtype_t;

// Wrapper for MPI_Abort
//
// Inputs:
// - enum errtype_t = error type
// - int errcode = error code to relay into MPI_Abort
// - char* errmsg = error message for ERR_TYPE_CUSTOM error type
//
void parabort
( enum errtype_t errtype, int errcode, const char* errmsg );

// Get MPI current rank and the total number of ranks.
//
// Outputs:
// - int *rank = current MPI rank
// - int *nranks = total number of MPI ranks
//
void init_mpi_vars
(int* rank, int* nranks);

// Get name of selected GPU API.
//
// Returns: string of the GPU API name.
//
const char* get_gpu_api_name
(void);

// Find GPUs. Implementation depends on selected GPU API.
//
// Returns: number of detected GPUs for the current rank,
//
// Outputs:
// - char** *luid = GPU local IDs for the current rank
// - char** *name = GPU names for the current rank
//
int find_GPUs
( char*** luid, char*** name );

// Collect GPU info from each rank to rank 0.
//
// Inputs:
// - int ngpus = number of GPUs for the current rank
// - char** luid = GPU local IDs for the current rank
// - char** name = GPU names for the current rank
//
// Returns: total number of detected GPUs,
//
// Outputs (only allocated and filled at rank 0):
// - int* ngpu_list = number of GPUs for each rank
// - char** *luid_list = List of all GPU local IDs
// - char** *name_list = List of all GPU names
//
int collect_GPUs
( const int ngpus, char** luid, char** name,
  int** ngpu_list, char*** luid_list, char*** name_list );

// Print GPU info to stdout.
//
// Inputs:
// - int rank = current MPI rank
// - int ngpus = number of GPUs
// - int start_idx = starting index in luid and name arrays
// - char** luid = GPU local IDs
// - char** name = GPU names
//
void print_gpus
( const int rank, const int ngpus, const int start_idx,
  char** luid, char** name );

#endif /* COMMON_H */
