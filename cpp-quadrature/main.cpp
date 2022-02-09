#include <iostream>
#include <cmath>

#ifdef NTHREADS
#include <omp.h>
#endif /* NTHREADS */

#include "quad_trapezoid.hpp"
#include "quad_simpson.hpp"

// Global variables
const double pi = 3.14159265358979;

// Function to be integrated
double y ( double x ) {
  return std::sin( x );
}

int main ( int argc, char *argv[] ) {

  double xmin = 0;
  double xmax = pi/2.0;

  double n = 11;
  double res;

  QuadTrapezoid integ1;

  std::cout << "Integrating y = sin(x) from x = " << xmin << " to " << xmax
            << " using " << integ1.quad_name << " with " << n << " points"
            << std::endl;

  res = integ1.integrate( y, xmin, xmax, n );

  std::cout << "Result = " << res << std::endl;

  QuadSimpson integ2;

  std::cout << "Integrating y = sin(x) from x = " << xmin << " to " << xmax
            << " using " << integ2.quad_name << " with " << n << " points"
            << std::endl;

  res = integ2.integrate( y, xmin, xmax, n );

  std::cout << "Result = " << res << std::endl;

  return 0;
}
