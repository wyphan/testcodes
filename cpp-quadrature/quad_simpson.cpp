#include "quad_simpson.hpp"

// Constructor
QuadSimpson::QuadSimpson ( ) :
  Quadrature( 3, "Simpson's rule" )
{}

// Function to initialize quadrature coefficients
void QuadSimpson::init_coeffs ( int n ) {

  // Round it up to nearest "nice" number
  int r = 2*int(n/2) + 1;
  if (r != coeffs.size()) {
    coeffs.resize(r);
  }

  // End points
  coeffs[0] = coeffs[r-1] = 1.0/3.0;

  // Mid points
#pragma omp parallel for
  for (int i = 1; i < (r-1); i += 2) {
      coeffs[i] = 4.0/3.0;
  }
  if (r > minPoints) {
#pragma omp parallel for
    for (int i = 2; i < (r-2); i += 2) {
      coeffs[i] = 2.0/3.0;
    }
  }

  return;
}
