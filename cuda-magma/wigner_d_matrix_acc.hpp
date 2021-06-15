#ifndef WIGNER_D_MATRIX_ACC_HPP
#define WIGNER_D_MATRIX_ACC_HPP 1

int wigner_d_matrix_acc( const bool half,
			 const int jmax,
			 const double alpha,
			 const double beta,
			 const double gamma,
			 const int ldd,
			 std::complex<double>* matD );

#endif /* WIGNER_D_MATRIX_ACC_HPP */
