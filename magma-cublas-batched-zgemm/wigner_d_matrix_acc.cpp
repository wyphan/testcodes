#include <cmath>
#include <complex>
#include <iostream>
#include <limits>

#define matD(i,j) matD[(i)*ldd+j]

////////////////////////////////////////////////////////////////////////////////
// Generates the Wigner rotation matrix D(R) using the z-y-z convention,
// where alpha, beta, gamma are the three Euler angles, respectively.
//
// The matrix elements, from Sakurai 2ed, Ch. 3, Equation (5.50), are given by
//
//  (j)                         -i(m' alpha + m gamma)  (j)
// D    (alpha, beta, gamma) = e                       d    (beta)
//  m',m                                                m',m
//
// And from Sakurai 2ed, Ch. 3, Equation (9.33),
//
//              ___              ____________________________
//  (j)         \       k-m+m' \/(j+m)! (j-m)! (j+m')! (j-m)!
// d   (beta) =  >  (-1)       ------------------------------- x
//  m',m        /__            (j+m-k)! k! (j-k-m')! (k-m+m')!
//               k
//                                (2j-2k+m-m')             (2k-m+m')
//                    /     beta \             /     beta \
//                  x ( cos ---- )             ( sin ---- )
//                    \      2   /             \      2   /
//
// where the sum over k runs from 0 and is only for terms when the arguments
// of factorials in the denominator is nonnegative.
//
// Identities of the Wigner d-function, from the Particle Data Group table of
// Clebsh-Gordan coefficients, spherical harmonics, and Wigner d-functions:
//
//  (j)         m-m'  (j)      (j)
// d      = (-1)     d      = d
//  m', m             m, m'    -m, -m'
//
// For integer j, it is related to the spherical harmonics when m or m' is 0:
//            _____
//  (l)      / 4 pi   m   -i m phi
// d    =   / -----  Y   e
//  m,0   \/   2l+1   l
//
// which can be further reduced to Legendre polynomials when m = m' = 0.
//                       _____
//  l                   / 2l+1
// Y  (theta, phi) =   / -----  P  ( cos theta )
//  0                \/   4 pi   l
//
// Due to the way the Euler angles are set up in the z-y-z convention,
// the polar angle (theta) corresponds to the second Euler angle (beta).
////////////////////////////////////////////////////////////////////////////////
int wigner_d_matrix_acc( const bool half,
			 const int jmax,
			 const double alpha,
			 const double beta,
			 const double gamma,
			 const int ldd,
			 std::complex<double>* matD )
{

  // Numerical constants
  const double pi = 2. * asin(1.);
  const double spi = sqrt(pi);
  const std::complex<double> ii( 0., 1. );

  // Convert angles to radians
  const double a = alpha * pi / 180.;
  const double b = beta  * pi / 180.;
  const double g = gamma * pi / 180.;

  // Max value for factorial
  const int fctmax = std::min( 34, 2*jmax );

  // Check half-integer or whole integer
  if( half ) {

    // Half-integer not yet implemented
    std::cerr << "wigner_d_matrix_acc: half-integer jmax not implemented" << std::endl;
    return 1;
    
  } else {

    // Allocate auxiliary quantities on host on the stack
    int blkst[jmax+1];
    double fct[fctmax+1], pn[jmax+1];
    double cosb[jmax+1], cosb2[4*jmax+1], sinb2[4*jmax+1];
    std::complex<double> eia[2*jmax+1], eig[jmax+1];

    // Allocate device vars using OpenACC
#pragma acc data create( blkst, fct, pn, cosb, cosb2, sinb2, eia, eig ) \
                 copyin( a, b, g, jmax, fctmax ) deviceptr(matD)
    {

      // Compute auxiliary quantities on device
#pragma acc kernels
      {

	// Factorial table
	fct[0] = 1.;
#pragma acc loop seq
	for( int j = 1; j <= fctmax; j++ )
	  fct[j] = (double)j * fct[j-1];

#pragma acc loop independent
	for( int j = 0; j <= jmax; j++ ) {

	  // Block start index
	  blkst[j] = j*j;

	  // cos(beta)
	  cosb[j] = pow( cos(b), j );

	  // e^(-i*m'*gamma)
	  // Note: index is positive, but contains values for negative j
	  eig[j] = exp( ii * (double)j * g );

	} // end acc loop

	// e^(-i*m*alpha)
	// Note: index shifted by jmax
	//       since C++ doesn't allow negative array indices
#pragma acc loop independent
	for( int j = -jmax; j <= jmax; j++ )
	  eia[jmax+j] = exp( -ii * (double)j * a );

	// cos(beta/2) and sin(beta/2)
#pragma acc loop
	for( int j = 0; j <= 4*jmax; j++ ) {
          cosb2[j] = pow( cos( 0.5 * b ), j );
	  sinb2[j] = pow( sin( 0.5 * b ), j );
        } // end acc loop

	// Legendre polynomials P_l (cos beta )
	pn[0] = 1.;
	pn[1] = cosb[1];
#pragma acc loop seq
	for( int j = 2; j <= jmax; j++ ) {

          // Use identity for P_(l+1)
          // (l+1) P   (z) - (2l+1) z P (z) + l P   (z) = 0
          //        l+1                l         l-1
	  pn[j] = ( 2. - 1./(double)j ) * cosb[j-1] * pn[j-1] \
	    + ( 1./(double)j - 1. ) * pn[j-2];

        } // end acc loop

      } // end acc kernels

      // Begin main loop
      int posx, posy;
      std::complex<double> val( 0., 0. ), tmp( 0., 0. );
#pragma acc parallel loop gang
      for( int j = 0; j <= jmax; j++ ) {

#pragma acc loop worker independent private( posy )
	for( int m = -j; m <= 0; m++ ) {

	  posy = blkst[j] + j + m;

#pragma acc loop independent private( posx, tmp, val )
	  for( int mp = -m; mp <= m; mp++ ) {

	    posx = blkst[j] + j + mp;

	    if( m == 0 && mp == 0 ) {

	      // Use Legendre polynomials for m=m'=0
	      //  (j)
	      // d    = P ( cos beta )
	      //  0,0    j
	      matD(posy,posx) = std::complex<double>( pn[j] );

	    } else {

	      if( j < 18 ) {

		// The matrix elements start overflowing at j = 18
		// Use Wigner formula for other values of m and m'

		// Sum over k
		double tmpk = 0, sumk = 0.;
#pragma acc loop vector independent reduction(+:sumk) private(tmpk)
		for( int k = 0; k <= 2*j; k++ ) {

		  // Check for negative denominator
		  if( (j+m-k) >= 0 && (j-k-mp) >= 0 && (k+m-mp) >= 0 ) {

		    // Check for overflow of factorial product
		    tmpk = fct[j+m-k] * fct[k] * fct[j-k-mp] * fct[k-m+mp];
		    if( real(tmp) > std::numeric_limits<double>::max() ) {
		      tmpk = 0.; // 1/infinity = 0
		    } else {
		      tmpk = 1. / tmpk;
		    } // factorial product overflow

		    // Multiply by cos(beta/2) and sin(beta/2)
		    tmpk *= cosb2[ 2*j-2*k+m-mp ] * sinb2[ 2*k-m+mp ];

		    // Minus sign
		    if( (k-m+mp)%2 != 0 ) tmpk *= -1.;

		  } // negative denominator

		  // Sum over k
		  sumk += tmpk;

		} // acc loop vector k

		// The rest of the factorials and the exponentials
		// Do multiplication in two stages to avoid overflow
		tmp = sumk * sqrt( fct[j+m] ) * sqrt( fct[j-m] ) * eig[m];
		val = tmp * sqrt( fct[j+mp] ) * sqrt( fct[j-mp] ) * eia[jmax+mp];

	      } else {

		// Use Stirling's formula for j >= 18
		// Obviously the results will be inaccurate;
		// but at least better than having NaNs due to overflow...

		// Sum over k
		double tmpk = 0, sumk = 0.;
#pragma acc loop vector reduction(+:sumk) private(tmpk)
		for( int k = 0; k <= 2*j; k++ ) {

		  // Check for negative denominator
		  if( (j+m-k) >= 0 && (j-k-mp) >= 0 && (k+m-mp) >= 0 ) {

		    // Take ln of each factorial term, then sum them
		    if( (j+m) > 0 )    tmpk += 0.5 * (double)(j+m)    * log( (double)(j+m) );
		    if( (j-m) > 0 )    tmpk += 0.5 * (double)(j-m)    * log( (double)(j-m) );
		    if( (j+mp) > 0 )   tmpk += 0.5 * (double)(j+mp)   * log( (double)(j+mp) );
		    if( (j-mp) > 0 )   tmpk += 0.5 * (double)(j-mp)   * log( (double)(j-mp) );
		    if( (j+m-k) > 0 )  tmpk -=       (double)(j+m-k)  * log( (double)(j+m-k) );
		    if( k > 0 )        tmpk -=       (double)k        * log( (double)k );
		    if( (j-k-mp) > 0 ) tmpk -=       (double)(j-k-mp) * log( (double)(j-k-mp) );
		    if( (k-m+mp) > 0 ) tmpk -=       (double)(k-m+mp) * log( (double)(k-m+mp) );

		    // cos(beta/2) and sin(beta/2)
		    tmpk += (double)(2*j-2*k+m-mp) * log( abs( cosb2[1] ));
		    tmpk += (double)(    2*k-m+mp) * log( abs( sinb2[1] ));

		    // Exponentiate
		    tmpk = exp(tmpk);

		    // Minus signs
		    // cos
		    if( b < -pi || b > pi && (2*j-2*k+m-mp)%2 != 0 ) tmpk *= -1.;
		    // sin
		    if( b < 0.            && (    2*k+m-mp)%2 != 0 ) tmpk *= -1.;
		    // From Wigner's formula
		    if(                      (      k-m+mp)%2 != 0 ) tmpk *= -1.;

		  } // negative denominator

		  // Sum over k
		  sumk += tmpk;

		} // acc loop vector k

		// Multiply with the exponentials
		val = sumk * eig[m] * eia[jmax+mp];

	      } // j >= 18

	      // Fill in matrix element D_(m',m)
	      matD(posy,posx) = val;

	      // Use identity to fill in rest of matrix

	      // D_(-m,-m')
	      if( m != -mp )
		matD(posx-2*m,posy-2*mp) = val;

	      // Minus sign
	      if( (mp-m)%2 != 0 ) val *= -1.;
	      // D_(m,m')
	      if( m != mp )
		matD(posx,posy) = val;
	      // D_(-m',-m)
	      if( m != mp && m != -mp )
		matD(posy-2*mp,posx-2*mp) = val;

	    } // m == m' == 0

	  } // acc loop worker m'
        } // m

      } // acc parallel loop gang j

    } // end acc data

  } // if half

  return 0;
}
