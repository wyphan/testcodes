#ifndef quadrature_hpp
#define quadrature_hpp

#include <vector>
#include <string>

// Class for integral quadratures
class Quadrature {

public:

  // Constructor
  Quadrature ();

  // Destructor
  ~Quadrature ();

  // Name of quadrature method
  std::string quad_name;

  // Function to perform numerical integration with bounds and number of points
  double integrate ( double& fn(double), double start, double end, int n );

  // Function to perform numerical integration with vector of points
  double integrate ( double& fn(double), std::vector<double> x );

private:

  // Vector to hold quadrature coefficients
  std::vector<double> coeffs;

  // Function to initialize quadrature coefficients
  virtual void init_coeffs ( int n );

};

#endif /* quadrature_hpp */
