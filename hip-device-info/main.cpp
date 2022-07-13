// Wil's HIP info dumper
// Last modified: Jul 11, 2022

// System includes
#include <iostream>

// HIP includes
#include <hip/hip_runtime.h>

// Project includes
#include "helpers.hip.hpp"

int main (int argc, char* argv[]) {

  // Error code for host-side helper routines
  int ierr = 0;

  // First, initialize HIP runtime by getting number of detected GPUs
  int ngpus = get_ngpus();
  if (ngpus < 0) {
    std::cerr << "Bailing out due to error" << std::endl;
    return 1;
  } else if (ngpus == 0) {
    std::cout << "No HIP-capable devices detected" << std::endl;
    return 0;
  } else {
    std::cout << "Detected " << ngpus << " HIP-capable device(s)." << std::endl;
  }

  // Read GPU table
  std::string tabfile = "gpu_table.txt";
  std::vector<gpurow_t> gputable;
  ierr = read_table(tabfile, gputable);

  // Allocate arrays to hold GPU infos
  device_t gpus[ngpus];

  for (int n = 0; n < ngpus; n++) {
    // Get info for each GPU
    gpus[n].id = n;
    ierr = get_gpu_info(n, gpus[n]);
    if (ierr != 0) {
      std::cerr << "Bailing out due to error" << std::endl;
      return 2;
    }
    // Populate GPU info
    ierr = populate_gpu_fields(n, gpus[n], gputable);
  }

  /*
  // Dump GPU table
  for (int i = 0; i < gputable.size(); i++) {
    std::cout << std::string(80, '=') << std::endl;
    std::cout << i + 1 << ". " << gputable[i].arch
	      << " (" << gputable[i].codename << ")" << std::endl;
    for (int j = 0; j < gputable[i].gpuname.nelems; j++) {
      std::cout << "- " << gputable[i].gpuname.elem[j] << std::endl;
    }
  }
  */

  // List detected GPU names
  ierr = list_gpus(ngpus, gpus);

  // Prompt for GPU selection
  int gpuid;
  bool all_gpus;
  ierr = select_gpu(ngpus, gpuid, all_gpus);
  if (ierr != 0) {
    std::cerr << "Bailing out due to error" << std::endl;
    return 3;
  }

  // Print GPU information
  std::cout << std::string(80, '=') << std::endl;
  if (all_gpus) {
    for (int n = 0; n < ngpus; n++) {
      ierr = print_gpu_info(n, gpus[n]);
      std::cout << std::string(80, '=') << std::endl;
    } // ngpus
  } else {
    ierr = print_gpu_info(gpuid, gpus[gpuid]);
    std::cout << std::string(80, '=') << std::endl;    
  } // all_gpu

  // Clean up
  return 0;
}
