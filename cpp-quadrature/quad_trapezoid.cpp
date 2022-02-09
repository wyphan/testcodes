#include "quad_trapezoid.hpp"

// Constructor
QuadTrapezoid::QuadTrapezoid () :
  Quadrature( 2, "Trapezoidal rule" )
{}

// Function to initialize quadrature coefficients
void QuadTrapezoid::init_coeffs ( int n ) {

  if (n != coeffs.size()) {
    coeffs.resize(n);
  }

  // End points
  coeffs[0] = coeffs[n-1] = 0.5;

  // Mid points
  if (n > minPoints) {
#pragma omp parallel for
    for (int i = 1; i < n-1; i++ ) {
      coeffs[i] = 1.0;
    }
  }

  return;
}
