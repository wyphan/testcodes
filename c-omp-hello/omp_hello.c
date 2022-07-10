#include <omp.h>
#include <stdio.h>

int main (int argc, char *argv[]) {

  int nthds, tid;

  #pragma omp parallel private(tid)
  {
    nthds = omp_get_num_threads();
    tid = omp_get_thread_num();
    printf("Hello from thread %i of %i\n", \
           tid, nthds);
  }

  return 0;

}
