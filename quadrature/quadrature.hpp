#ifndef quadrature_hpp
#define quadrature_hpp

#include <vector>
#include <string>

// Define function pointer as a typedef
typedef double (*func) ( double );
// typedef double (*pfunc) ( double, struct param );

// Class for integral quadratures
class Quadrature {

public:

  // Constructor with minimum number of points and name of quadrature
  Quadrature ( int iMinPoints, std::string name );

  // Default destructor
  ~Quadrature ();

  // Name of quadrature method
  const std::string quad_name;

  // Function to perform numerical integration with bounds and number of points
  double integrate ( func fn, double start, double end, int n );

  // Function to perform numerical integration with vector of points
  double integrate ( func fn, std::vector<double> x );

protected:

  // Minimum number of points
  const int minPoints;

  // Vector to hold quadrature coefficients
  std::vector<double> coeffs;

  // Function to initialize quadrature coefficients
  virtual void init_coeffs ( int n );

};

#endif /* quadrature_hpp */
