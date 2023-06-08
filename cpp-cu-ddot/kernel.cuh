// -*- mode: cuda; -*-
#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
// #include <sm_60_atomic_functions.hpp>

#include "globals.hpp"

__global__ void ddot( const unsigned long N,
		      const double* d_vecA,
		      const double* d_vecB,
		      double* d_work ) {

  // Use 1-D grid and block
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bsz = blockDim.x;
  int gid = bid * bsz + tid;

  // Set up shared memory for each block
  __shared__ double s_vecA[threadsPerBlock];
  __shared__ double s_vecB[threadsPerBlock];
  __shared__ double s_res[warpsPerBlock];
  if (gid < N) {
    s_vecA[tid] = d_vecA[gid];
    s_vecB[tid] = d_vecB[gid];
  } else {
    s_vecA[tid] = 0.0;
    s_vecB[tid] = 0.0;
  }
  __syncthreads();

  // Reduce within the warp
  int laneId = tid % warpSize;
  int bloc = tid / warpSize;
  double t_sum = s_vecA[tid] * s_vecB[tid];
  __syncthreads();
  unsigned int FULL_MASK = 0xFFFFFFFFU;
  for (unsigned int offset = warpSize / 2; offset > 0; offset >>= 1) {
    t_sum += __shfl_down_sync(FULL_MASK, t_sum, offset);
    __syncthreads();
  }

  // Reduce within the block  
  if (laneId == 0) s_res[bloc] = t_sum;
  __syncthreads();
  for (unsigned int offset = warpsPerBlock / 2; offset > 0; offset >>= 1) {
    if (tid < offset) s_res[tid] += s_res[tid + offset];
    __syncthreads();
  }

  // Collect data across blocks
  int blocksInGrid = gridDim.x;
  if (tid == 0 && bid < blocksInGrid) d_work[bid] = s_res[0];
  __syncthreads();

  return;
}

__global__ void reduceblocks(const int extent,
			     const int worksz,
	        	     double* d_work) {

  // Use 1-D grid and block
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bsz = blockDim.x;
  int gid = bid * bsz + tid;

  __shared__ double s_work[threadsPerBlock];
  __shared__ double s_res[warpsPerBlock];
  if (gid < worksz) {
    s_work[tid] = d_work[gid];
  } else {
    s_work[tid] = 0.0;
  }
  __syncthreads();

  // Reduce within the warp
  int laneId = tid % warpSize;
  int bloc = tid / warpSize;
  double t_sum = s_work[tid];
  __syncthreads();
  unsigned int FULL_MASK = 0xFFFFFFFFU;
  for (unsigned int offset = warpSize / 2; offset > 0; offset >>= 1) {
    t_sum += __shfl_down_sync(FULL_MASK, t_sum, offset);
    __syncthreads();
  }

  // Reduce within the block
  if (laneId == 0) s_res[bloc] = t_sum;
  __syncthreads();
  for (unsigned int offset = warpsPerBlock / 2; offset > 0; offset >>= 1) {
    if (tid < offset) s_res[tid] += s_res[tid + offset];
    __syncthreads();
  }

  // Collect data across blocks
  int blocksInGrid = gridDim.x;
  if (tid == 0 && bid < blocksInGrid) d_work[bid] = s_res[0];    
  __syncthreads();

  // Empty rest of workspace
  if (gid >= blocksInGrid && gid < extent) d_work[gid] = 0.0;
  __syncthreads();

  return;
}

#endif /* KERNEL_CUH */
