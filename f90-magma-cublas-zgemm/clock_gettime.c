#include <time.h>

void my_clock_gettime_( double* seconds ) {
  struct timespec time;
  clock_gettime( CLOCK_THREAD_CPUTIME_ID, &time );
  *seconds = time.tv_sec + (double)time.tv_nsec * 1.0e-9;
  return;
}
