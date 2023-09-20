#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Global variables
const double tol = 1.0e-10;

int main (int argc, char* argv[]) {

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

  // Allocate vectors and result variables on device (GPU)
  double result;
  #pragma omp target enter data map(to:N) map(alloc: vecA[0:N-1], vecB[0:N-1], result)

  // Initialize vectors on device
  // Note: the order of the clauses matter!
  #pragma omp target teams distribute parallel for shared(vecA, vecB)
  for (unsigned long i = 0; i < N; i++) {
    vecA[i] = (double)i;
    vecB[i] = (double)(2*i);
  }

  // Initialize result variable
#pragma omp target
  {
    result = 0.0;
  }

  // Note: No barrier needed here because omp target is synchronous
  //       (Unless you use nowait clause, then you need omp taskwait to sync)

  time_t t1 = time(NULL);

  // Perform dot product on device
  #pragma omp target teams distribute parallel for shared(vecA, vecB) reduction(+:result)
  for (unsigned long i = 0; i < N; i++) {
    result += vecA[i] * vecB[i];
  }

  // Fetch result
  #pragma omp target update to(result)

  time_t t2 = time(NULL);

  // Close off data region
  #pragma omp target exit data map(delete: N, vecA, vecB, result)

  // Check value ( using relative error ) and print result to standard output
  double check = (double)(N) * (double)(N - 1) * (double)(2*N - 1) / 3.0;
  if ( fabs( result/check - 1.0 ) > tol ) {
    printf("Error! Result = %e when it should be %e\n", result, check);
  } else {
    printf("Success! Result = %e\n", result);
  }

  // Calculate and display timers
  double time_init = difftime(t1, t0);
  double time_ddot = difftime(t2, t1);
  printf("Initialization took %8.3f seconds.\n", time_init);
  printf("Computation took %8.3f seconds.\n", time_ddot);

  // Clean up
  free(vecA);
  free(vecB);
  return 0;
}
