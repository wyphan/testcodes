#ifndef quad_simpson_hpp
#define quad_simpson_hpp

#include "quadrature.hpp"

class QuadSimpson : public Quadrature {

public:

  // Child class constructor is the default constructor
  QuadSimpson ();

protected:

  // Implementation for initializing quadrature coefficients
  void init_coeffs( int n ) override;

};

#endif /* quad_simpson_hpp */
