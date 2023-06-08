#include <iostream>
#include <sstream>
#include <cmath>

#include <cuda_runtime_api.h>

#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif /* USE_NVTX */

#include "globals.hpp"
#include "kernel.cuh"

using namespace std;

// Helper function for error checking
#define CUDA_CHECK(command) {                                     \
  cudaError_t status = command;                                   \
  if ( status != cudaSuccess ) {                                  \
    cerr << "CUDA Error: " << cudaGetErrorString(status) << endl; \
  abort();                                                        \
  }                                                               \
}

#ifdef USE_NVTX
#define NVTX_CHECK(command) {               \
  int stat = command;                       \
  if (stat < 0) {                           \
    cerr << "NVTX Error: " << stat << endl; \
  }                                         \
}
#else
#define NVTX_CHECK(command)
#endif /* USE_NVTX */

#define NBLOCKS(N, threadsPerBlock) (N + threadsPerBlock - 1) / threadsPerBlock  

// Main function
int main( int argc, char* argv[] ) {

  // Set CUDA device to device 0
  CUDA_CHECK( cudaSetDevice( 0 ) );

  unsigned long N;
  // Check argc and read N from argv
  if (argc > 1) {
    std::string str = argv[1];
    std::istringstream s(str);
    s >> N;
  } else {
    // Read N from standard input
    std::cout << "Input vector length N: ";
    std::cin >> N;
    std::cout << "Using N = " << N << std::endl;
  } // argc, argv

  // Print GPU name
  cudaDeviceProp devprop;
  CUDA_CHECK( cudaGetDeviceProperties( &devprop, 0 ) );
  std::cout << "Using device "
            << devprop.name
            << std::endl;

  // Calculate whether vectors will fit in device memory
  double mem_gib = (double)devprop.totalGlobalMem / (double)(1ULL<<30);
  double sz_gib = (double)(2 * N * sizeof(double)) / (double)(1ULL<<30);
  if (sz_gib > mem_gib) {
    std::cout << "Error: Using vector size " << N
	      << " requires " << sz_gib << " GiB of device memory;"
	      << " this device only has " << mem_gib << " GiB."
	      << std::endl;
    return 1;
  }

  NVTX_CHECK( nvtxRangePushA("init") );

  // Allocate vectors on host
  size_t sz_vec = N * sizeof(double);
  double* h_A = (double*)malloc( sz_vec );
  double* h_B = (double*)malloc( sz_vec );

  // Initialize vectors on host
  for( unsigned long i = 0; i < N; i++ ) {
    h_A[i] = (double)i;
    h_B[i] = (double)(2*i);
  }
  cout << "h_A[" << N-1 << "] = " << h_A[N-1] << endl;
  cout << "h_B[" << N-1 << "] = " << h_B[N-1] << endl;
  
  // Allocate vectors on device
  double* d_A = NULL;
  double* d_B = NULL;
  CUDA_CHECK( cudaMalloc( &d_A, sz_vec ) );
  CUDA_CHECK( cudaMalloc( &d_B, sz_vec ) );

  // Initialize result variable
  double h_res = 0.0;

  // Allocate workspace for blockwise reduction
  int blocksInGrid = NBLOCKS(N, threadsPerBlock);
  int nearest2 = 1;
  while (nearest2 < blocksInGrid) nearest2 <<= 1;
  double* d_work = NULL;
  CUDA_CHECK( cudaMalloc( &d_work, nearest2 * sizeof(double) ) );

  NVTX_CHECK( nvtxRangePushA("h2d") );

  // Transfer data to device (synchronous blocking mode)
  // Note: cudaMemcpy( dest, src, sz, direction )
  CUDA_CHECK( cudaMemcpy( d_A, h_A, sz_vec, cudaMemcpyHostToDevice ) );
  CUDA_CHECK( cudaMemcpy( d_B, h_B, sz_vec, cudaMemcpyHostToDevice ) );

  NVTX_CHECK( nvtxRangePop() ); // h2d

  // Zero workspace and result variable using cudaMemset()
  CUDA_CHECK( cudaMemset( d_work, 0.0, nearest2 * sizeof(double) ) );
  CUDA_CHECK( cudaDeviceSynchronize() );

  NVTX_CHECK( nvtxRangePop() ); // init
  NVTX_CHECK( nvtxRangePushA("calc") );

  // Set up thread block and grid for first kernel
  dim3 grid  = dim3( blocksInGrid, 1, 1 );    // 1-D grid
  dim3 block = dim3( threadsPerBlock, 1, 1 ); // 1-D block
  cout << "Kernel 1 (ddot), workspace size = " << nearest2 << endl;
  cout << "Using grid: " << grid.x << ", " << grid.y << ", " << grid.z << endl;
  cout << "Using block: " << block.x << ", " << block.y << ", " << block.z << endl;
  
  // Call first kernel to calculate and reduce up to block level
  // Note: Arguments are <<< grid, block, LDS_bytes, stream >>>
  size_t s_bytes = (2 * threadsPerBlock + warpsPerBlock) * sizeof(double);
  ddot<<< grid, block, s_bytes, 0 >>>( N, d_A, d_B, d_work );
  CUDA_CHECK( cudaDeviceSynchronize() );

  // Set up thread block and grid for second kernel
  unsigned int worksz = blocksInGrid;
  for (unsigned int extent = nearest2; extent > 1; extent /= threadsPerBlock) {
    int gridsz = NBLOCKS(extent, threadsPerBlock);
    grid  = dim3( gridsz, 1, 1 );          // 1-D grid
    block = dim3( threadsPerBlock, 1, 1 ); // 1-D block
    cout << "Kernel 2 (reduceblocks), workspace size = " << extent << ", filled = " << worksz << endl;
    cout << "Using grid: " << grid.x << ", " << grid.y << ", " << grid.z << endl;
    cout << "Using block: " << block.x << ", " << block.y << ", " << block.z << endl;

    // Call second kernel to reduce across blocks
    s_bytes = (threadsPerBlock + warpsPerBlock) * sizeof(double);
    reduceblocks<<<grid, block, s_bytes, 0>>>( extent, worksz, d_work );
    CUDA_CHECK( cudaDeviceSynchronize() );
    worksz = gridsz;
  }

  NVTX_CHECK( nvtxRangePop() ); // calc
  NVTX_CHECK( nvtxRangePushA("d2h") );
  
  // Transfer data to host (synchronous blocking mode)
  CUDA_CHECK( cudaMemcpy( &h_res, d_work, sizeof(double), cudaMemcpyDeviceToHost ) );

  NVTX_CHECK( nvtxRangePop() ); // d2h

  // Check results
  int errcode = 0;
  double tol = 1.0e-10;
  double check = (double)N * (double)(N - 1) * (double)(2*N - 1) / 3.0;
  if (fabs(h_res/check - 1.0) > tol) {
    cout << "Error! Result = " << h_res
         << " when it should be " << check << endl;
    errcode = 2;
  } else {
    cout << "Success! Result = " << h_res << endl;
  }

  // Clean up
  CUDA_CHECK( cudaFree( d_A ) );
  CUDA_CHECK( cudaFree( d_B ) );
  CUDA_CHECK( cudaFree( d_work ) );
  free( h_A );
  free( h_B );
  
  return errcode;

}
