#include <iostream>
#include <cmath>

#ifdef NTHREADS
#include <omp.h>
#endif /* NTHREADS */

#include "quad_trapezoid.hpp"
#include "quad_simpson.hpp"

// Global variables
const double pi = 3.14159265358979;
const double deg2rad = pi/180.0;

// Function to be integrated
double y ( double x ) {
  return sin( x * deg2rad );
}

int main ( int argc, char *argv[] ) {

  double xmin = 0;
  double xmax = pi/2.0;
  double n = 11;

  QuadTrapezoid integrator;
  double res;

  std::cout << "Integrating y = sin(x) from x = " << xmin << " to " << xmax
            << " using " << integrator.quad_name << " with " << n << " points"
            << std::endl;

  res = integrator.integrate( y, xmin, xmax, n );

  std::cout << "Result = " << res << std::endl;

  return 0;
}
