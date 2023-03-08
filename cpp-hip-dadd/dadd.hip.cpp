#include <iostream>
#include <sstream>
#include "hip/hip_runtime.h"

using namespace std;

// Helper function for error checking
#define HIP_CHECK(command) {                                  \
  hipError_t status = command;                                \
  if( status != hipSuccess ) {                                \
  cerr << "HIP Error: " << hipGetErrorString(status) << endl; \
  abort();                                                    \
  }                                                           \
}

// Kernel to add two vectors
__global__ void dadd( const unsigned long N,
                      const double* d_vecA,
                      const double* d_vecB,
                      double* d_vecC ) {

  // Use 1-D grid and block
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if ( tid < N )
    d_vecC[tid] = d_vecA[tid] + d_vecB[tid];

  __syncthreads();

  return;

}

// Main function
int main( int argc, char* argv[] ) {

  // Set HIP device to device 0
  HIP_CHECK( hipSetDevice( 0 ) );

  unsigned long N;
  // Check argc and read N from argv
  if (argc > 1) {
    std::string str = argv[1];
    std::istringstream s(str);
    s >> N;
  } else {
    // Read N from standard input
    std::cin >> N;
    std::cout << "Using N = " << N << std::endl;
  } // argc, argv

  // Allocate vectors on host
  size_t sz_vec = N * sizeof(double);
  double* h_A = (double*)malloc( sz_vec );
  double* h_B = (double*)malloc( sz_vec );
  double* h_C = (double*)malloc( sz_vec );

  // Initialize vectors on host
  for( unsigned long i = 0; i < N; i++ ) {
    h_A[i] = 1.0;
    h_B[i] = 2.0;
  }
  double chk = 3.0;

  cout << "h_A[" << N-1 << "] = " << h_A[N-1] << endl;
  cout << "h_B[" << N-1 << "] = " << h_B[N-1] << endl;

  // Allocate vectors on device
  double* d_A = NULL;
  double* d_B = NULL;
  double* d_C = NULL;
  HIP_CHECK( hipMalloc( &d_A, sz_vec ) );
  HIP_CHECK( hipMalloc( &d_B, sz_vec ) );
  HIP_CHECK( hipMalloc( &d_C, sz_vec ) );

  // Transfer data to device (synchronous blocking mode)
  // Note: hipMemcpy( dest, src, sz, direction )
  HIP_CHECK( hipMemcpy( d_A, h_A, sz_vec, hipMemcpyHostToDevice ) );
  HIP_CHECK( hipMemcpy( d_B, h_B, sz_vec, hipMemcpyHostToDevice ) );

  // Zero d_C using hipMemset()
  HIP_CHECK( hipMemset( d_C, 0, sz_vec ) );

  // Set up thread block and grid
  int nb = 256;
  dim3 grid  = dim3( (N+nb-1)/nb, 1, 1 ); // 1-D grid
  dim3 block = dim3( nb, 1, 1 );          // 1-D block
  cout << "Using grid: " << grid.x << ", " << grid.y << ", " << grid.z << endl;
  cout << "Using block: " << block.x << ", " << block.y << ", " << block.z << endl;

  // Call kernel using triple chevron syntax
  // Note: Arguments are <<< grid, block, LDS_bytes, stream >>>
  //       For ROCm < 3.6, use hipLaunchKernelGGL()
  dadd<<< grid, block, 0, 0 >>>( N, d_A, d_B, d_C );

  // Transfer data to host (synchronous blocking mode)
  HIP_CHECK( hipMemcpy( h_C, d_C, sz_vec, hipMemcpyDeviceToHost ) );

  // Check results
  int nerr = 0;
  for( unsigned long i = 0; i < N; i++ ) {
    if( h_C[i] != chk ) {
      nerr += 1;
      cerr << "Error: C[" << i << "] = " << h_C[i] << endl;
    }
  }
  if( nerr == 0 ) {
    cout << "Success!" << endl;
    cout << "h_C[" << N-1 << "] = " << h_C[N-1] << endl;
  } else {
    cerr << "Total " << nerr << " errors." << endl;
  }
    
  // Clean up
  HIP_CHECK( hipFree( d_A ) );
  HIP_CHECK( hipFree( d_B ) );
  HIP_CHECK( hipFree( d_C ) );
  free( h_A );
  free( h_B );
  free( h_C );
  
  return nerr;
}
