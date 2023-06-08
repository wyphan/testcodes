#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <cuda_runtime_api.h>

// Global variables
const int threadsPerBlock = 256;
const int nvWarpSize = 32;
constexpr int warpsPerBlock = threadsPerBlock / nvWarpSize;

#endif /* GLOBALS_HPP */
