#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <cstdlib>

#include <Kokkos_Core.hpp>

int main (int argc, char* argv[]) {

  unsigned long N;
  // Check argc and read N from argv
  if (argc > 1) {
    std::string str = argv[1];
    std::istringstream s(str);
    s >> N;
  } else {
    // Read N from standard input
    std::cin >> N;
    std::cout << "Using N = " << N << std::endl;
  } // argc, argv

  // Initialize result on host
  double res = 0.0;

  Kokkos::initialize(argc, argv);
  {

    // Allocate vectors and result variable on device as Kokkos View objects
    Kokkos::View<double*> d_A("A", N);
    Kokkos::View<double*> d_B("B", N);
    Kokkos::View<double> d_res("A \u00b7 B");

    // Initialize vectors on device using Kokkos parallel for
    // Note: result variable is automatically initialized to 0 by parallel_reduce
    Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(const unsigned long i) {
      d_A(i) = (double)i;
      d_B(i) = 2.0 * (double)i;
    });

    // Perform dot product using Kokkos parallel reduce
    Kokkos::parallel_reduce("calc", N, KOKKOS_LAMBDA(const unsigned long i, double& sum) {
      sum += d_A(i) * d_B(i);
    }, d_res);

    // Transfer result to host
    Kokkos::deep_copy(res, d_res);

  }
  Kokkos::finalize();

  // Check value (using relative error) and print to stdout
  double tol = 1.0e-10;
  double check = (double)N * (double)(N - 1) * (double)(2*N - 1) / 3.0;
  if (std::fabs(res/check - 1.0) > tol) {
    std::cout << "Error! Result = " << res
	      << " when it should be " << check << std::endl;
  } else {
    std::cout << "Success! Result = " << res << std::endl;
  }


  return 0;
}
