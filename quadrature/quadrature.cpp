#include <iostream>
#include <cstdlib>

#include "quadrature.hpp"

// Constructor
Quadrature::Quadrature ( int iMinPoints, std::string name ) :
  quad_name(name),
  minPoints(iMinPoints)
{}

// Default destructor
Quadrature::~Quadrature () {

  // Clear quadrature coefficients
  coeffs.clear();

}

// Perform numerical integration with bounds and number of points
double Quadrature::integrate ( func fn,
                               double start, double end, int n ) {

  // Quick exit if n < minPoints
  if (n < minPoints) {
    std::cerr << "Error[integrate]: cannot integrate with fewer than "
              << minPoints << " points"
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
double Quadrature::integrate ( func fn,
                               std::vector<double> x ) {

  // Quick exit if n < 2
  int n = x.size();
  if (n < minPoints) {
    std::cerr << "Error[integrate]: cannot integrate with fewer than "
              << minPoints << " points"
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
