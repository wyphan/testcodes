#include <iostream>
#include <sstream>
#include <cmath>

#include <CL/sycl.hpp>

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

  // Select GPU device and create queue
  sycl::device mygpu { sycl::gpu_selector_v };
  sycl::queue q(mygpu);

  // Print GPU name
  std::cout << "Using device "
            << mygpu.get_info<sycl::info::device::name>()
            << std::endl;

  // Allocate vectors on GPU and attach buffers
  double* vecA = sycl::malloc_device<double>(N, q);
  double* vecB = sycl::malloc_device<double>(N, q);

  // Initialize vectors on GPU using SYCL lambda kernels
  q.submit( [&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
      vecA[i] = (double)i;
    });
  });
  q.submit( [&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
      vecB[i] = 2.0 * (double)i;
    });
  });

  // Allocate result as shared memory and attach buffer
  double* result = sycl::malloc_shared<double>(1, q);
  sycl::buffer bufres = sycl::buffer<double>(result, 1);

  // Initialize result to 0
  q.submit( [&](sycl::handler& h) {
    sycl::accessor res(bufres, h, sycl::write_only);
    h.single_task([=]() {
      res[0] = 0.0;
    });
  });

  // Manual synchronization
  q.wait();

  // Perform dot product using SYCL lambda kernel and OneAPI built-in reduction
  q.submit( [&](sycl::handler& h) {
    auto red = sycl::reduction(bufres, h, sycl::plus<>());
    h.parallel_for(sycl::range<1>(N), red, [=](sycl::id<1> i, auto &tmp) {
      double prod = vecA[i] * vecB[i];
      tmp += prod;
    });
  });

  // Transfer result to host and synchronize
  // Note: host_accessor is blocking (thus can be used to synchronize)
  sycl::host_accessor res(bufres, sycl::read_only);

  // Check value (using relative error) and print to stdout
  double tol = 1.0e-10;
  double check = (double)N * (double)(N - 1) * (double)(2*N - 1) / 3.0;
  if (std::fabs(res[0]/check - 1.0) > tol) {
    std::cout << "Error! Result = " << res[0]
              << " when it should be " << check << std::endl;
  } else {
    std::cout << "Success! Result = " << res[0] << std::endl;
  }

  // Clean up
  sycl::free(vecA, q);
  sycl::free(vecB, q);
  sycl::free(result, q);

  return 0;
}
