#include <iostream>
#include <cstdlib>

#include "quadrature.hpp"

// Default constructor
Quadrature::Quadrature () {

  // Initialize quadrature coefficients
  int n = 2;
  init_coeffs(n);

}

// Default destructor
Quadrature::~Quadrature () {

  // Clear quadrature coefficients
  coeffs.clear();

}

// Perform numerical integration with bounds and number of points
double Quadrature::integrate ( double& fn(double),
                               double start, double end, int n ) {

  // Quick exit if n < 2
  if (n < 2) {
    std::cerr << "Error[integrate]: cannot integrate with fewer than 2 points"
              << std::endl;
    return 0.0;
  }

  // Initialize quadrature coefficients
  init_coeffs(n);

  // Initialize dx (assuming constant spacing)
  double dx = ( end - start ) / (double)( n - 1 );

  // Initialize x vector
  std::vector<double> x(n);
#pragma omp parallel for
  for( int i = 0; i < n; i++ ) {
    x[i] = start + (double)i * dx;
  }

  // Multiply and accumulate
  double result = 0.0;
#pragma omp parallel for reduction(+:result)
  for( int i = 0; i < n; i++ ) {
    result += fn( x[i] ) * coeffs[i] * dx;
  }
  return result;

}


// Perform numerical integration with vector of points
double Quadrature::integrate ( double& fn(double),
                               std::vector<double> x ) {

  // Quick exit if n < 2
  int n = x.size();
  if (n < 2) {
    std::cerr << "Error[integrate]: cannot integrate with fewer than 2 points"
              << std::endl;
    return 0.0;
  }

  // Initialize quadrature coefficients
  init_coeffs(n);

  // Initialize dx (assuming constant spacing)
  const double dx = x[1] - x[0];

  // Multiply and accumulate
  double result = 0.0;
#pragma omp parallel for reduction(+:result)
  for( int i = 0; i < n; i++ ) {
    result += fn( x[i] ) * coeffs[i] * dx;
  }
  return result;

}
