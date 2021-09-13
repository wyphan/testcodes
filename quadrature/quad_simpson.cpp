#include "quad_simpson.hpp"

// Constructor
QuadSimpson::QuadSimpson () {

  quad_name = "Simpson's rule";

}

// Function to initialize quadrature coefficients
void QuadSimpson::init_coeffs ( int n ) {

  if (n != coeffs.size()) {
    coeffs.resize(n);
  }

  // End points
  coeffs[0] = coeffs[n-1] = 0.5;

  // Mid points
  if (n > 2) {
#pragma omp parallel for
    for (int i = 1; i < n-1; i++ ) {
      coeffs[i] = 1.0;
    }
  }

  return;
}
