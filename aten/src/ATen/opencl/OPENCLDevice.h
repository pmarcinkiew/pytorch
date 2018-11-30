#pragma once

//#include "ATen/cuda/Exceptions.h"

//#include "cuda.h"

namespace at {
namespace opencl {

inline Device getDeviceFromPtr(void* ptr) {
/*  struct cudaPointerAttributes attr;
  AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  return {DeviceType::CUDA, attr.device};
*/
}

}} // namespace at::cuda
