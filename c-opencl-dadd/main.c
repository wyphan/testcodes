#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 100
#include <CL/cl.h>

// Structs
typedef struct devmap_t {
  cl_uint ngpus;
  cl_device_id* gpus;
} devmap_t;

typedef struct devlist_t {
  struct devlist_t* next;
  cl_device_id* gpu;
} devlist_t;

// Helper function for error checking
#define CL_CHECK(command) {                                \
  cl_int status = command;                                 \
  if( status != CL_SUCCESS ) {                             \
    fprintf(stderr, "OpenCL API error %d\n", (int)status); \
    abort();                                               \
  }                                                        \
}

// Global variables
const double tol = 1.0e-10;

int main (int argc, char* argv[]) {
  
  // Detect OpenCL platforms
  cl_uint nplats;
  CL_CHECK( clGetPlatformIDs(0, NULL, &nplats) );
  printf("Detected %i OpenCL platforms\n", nplats);

  // Set up data structures for tracking GPU devices
  cl_uint ngpus = 0;
  unsigned int igpu = 0;
  devmap_t* plat_gpus = (devmap_t*)malloc( nplats * sizeof(devmap_t) );
  devlist_t* gpulist = (devlist_t*)malloc( sizeof(devlist_t) );
  gpulist->next = NULL;

  // Dump platform vendors and names, then find which platforms has GPUs
  cl_platform_id* cl_plats = (cl_platform_id*) malloc( nplats * sizeof(cl_platform_id) );
  CL_CHECK( clGetPlatformIDs(nplats, cl_plats, NULL) );

  for (cl_uint p = 0; p < nplats; p++) {

    char plat_ven[128];   size_t sz_ven = 128 * sizeof(char);
    char plat_name[1024]; size_t sz_name = 1024 * sizeof(char);
    size_t len_ven, len_name;
    CL_CHECK( clGetPlatformInfo(cl_plats[p], CL_PLATFORM_VENDOR, sz_ven, &plat_ven, NULL) );
    CL_CHECK( clGetPlatformInfo(cl_plats[p], CL_PLATFORM_NAME, sz_name, &plat_name, NULL) );
    printf("%i. %s: %s\n", p, plat_ven, plat_name);

    cl_uint plat_ngpus = 0;
    devlist_t* prev_item = gpulist;
    CL_CHECK( clGetDeviceIDs(cl_plats[p], CL_DEVICE_TYPE_GPU, 0, NULL, &plat_ngpus) );
    if (plat_ngpus > 0) {

      plat_gpus[p].ngpus = plat_ngpus;
      plat_gpus[p].gpus = (cl_device_id*)malloc( plat_ngpus * sizeof(cl_device_id) );
      for (cl_uint d = 0; d < plat_ngpus; p++) {

	cl_device_id* this_gpu = (cl_device_id*)malloc( sizeof(cl_device_id) );
	CL_CHECK( clGetDeviceIDs(cl_plats[p], CL_DEVICE_TYPE_GPU, plat_ngpus, this_gpu, NULL) );

	devlist_t* cur_item = (devlist_t*)malloc( sizeof(devlist_t) );
	cur_item->next = NULL;
	prev_item->next = cur_item;
	prev_item->gpu = this_gpu;

	char gpu_name[1024]; sz_name = 1024 * sizeof(char);
	CL_CHECK( clGetDeviceInfo(*this_gpu, CL_DEVICE_NAME, sz_name, &gpu_name, NULL) );

	printf("   GPU %i: %s\n", igpu, gpu_name);

	prev_item = cur_item;
	igpu++;
      }
      ngpus += plat_ngpus;

    } else {
      printf("   No GPUs in this platform.\n");
    }
  }

  /*
  unsigned long N;
  // Check argc and read N from argv
  if (argc > 1 && atol(argv[1]) > 0L) {
    N = atol(argv[1]);
  } else {
    // Read N from standard input
    printf("Input vector length N:\n");
    scanf("%lu", &N);
  } // argc, argv

  // Echo N to standard output
  printf("Using N = %lu\n", N);

  time_t t0 = time(NULL);

  // Allocate vectors on CPU
  double* vecA = (double*)malloc( (size_t)N * sizeof(double) );
  double* vecB = (double*)malloc( (size_t)N * sizeof(double) );
  double* vecC = (double*)malloc( (size_t)N * sizeof(double) );
  */

  return 0;
}
