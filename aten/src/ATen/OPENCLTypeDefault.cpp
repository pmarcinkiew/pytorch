#include <ATen/OPENCLTypeDefault.h>

#include <ATen/Context.h>
#include <ATen/OPENCLGenerator.h>

namespace at {

Allocator* OPENCLTypeDefault::allocator() const {
  return getOPENCLAllocator();
}

Device OPENCLTypeDefault::getDeviceFromPtr(void * data) const {
  return DeviceType::OPENCL;
}

std::unique_ptr<Generator> OPENCLTypeDefault::generator() const {
  return std::unique_ptr<Generator>(new OPENCLGenerator(&at::globalContext()));
}

} // namespace at
