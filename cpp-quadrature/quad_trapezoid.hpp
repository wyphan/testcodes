#ifndef quad_trapezoid_hpp
#define quad_trapezoid_hpp

#include "quadrature.hpp"

class QuadTrapezoid : public Quadrature {

public:

  // Child class constructor
  QuadTrapezoid ();

protected:

  // Implementation for initializing quadrature coefficients
  void init_coeffs( int n ) override;

};

#endif /* quad_trapezoid_hpp */
