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
  if (r > minPoints) {
#pragma omp parallel for
    for (int i = 3; i < r-1; i += 2 ) {
      coeffs[i] = 4.0/3.0;
    }
  }

  return;
}
