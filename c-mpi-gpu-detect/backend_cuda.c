#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "common.h"

// Helper macro for error checking
#define CUDA_CHECK(command) {             \
  cudaError_t status = command;           \
  if ( status != cudaSuccess ) {          \
    parabort(ERR_TYPE_GPU, status, NULL); \
  }				          \
}

const char* get_gpu_api_name () {
  return "CUDA";
}

// Find CUDA-capable GPUs.
///
// Returns: number of detected GPUs at each rank,
//
// Outputs:
// - char** *addr = PCI domain, bus, and device ID addresses for each GPU
// - char** *name = GPU names
//
int find_GPUs(char*** addr, char*** name) {

  int ngpus = 0;
  CUDA_CHECK( cudaGetDeviceCount(&ngpus) );
  if (ngpus > 0) {
    *addr = (char**)malloc( ngpus * sizeof(char*) );
    *name = (char**)malloc( ngpus * sizeof(char*) );
    for (unsigned int g = 0; g < ngpus; g++) {
      struct cudaDeviceProp prop;
      CUDA_CHECK( cudaGetDeviceProperties(&prop, g) );
      int pci_dom = prop.pciDomainID;
      int pci_bus = prop.pciBusID;
      int pci_dev = prop.pciDeviceID;
      char* pci_addr = (char*) malloc( 8 * sizeof(char) );
      sprintf(pci_addr, "%02x:%02x.%1x\0", pci_dom, pci_bus, pci_dev);
      (*addr)[g] = strdup( pci_addr );
      (*name)[g] = strdup( prop.name );
      free(pci_addr);
    } // g
  } // ngpus

  return ngpus;
}
