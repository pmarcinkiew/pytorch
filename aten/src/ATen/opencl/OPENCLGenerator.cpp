#include "ATen/Config.h"

#include "ATen/OPENCLGenerator.h"
#include "ATen/Context.h"
//#include "THCTensorRandom.h"
#include <stdexcept>

// There is only one OPENCLGenerator instance. Calls to seed(), manualSeed(),
// initialSeed(), and unsafeGetTH() refer to the THCGenerator on the current
// device.

//THCGenerator* THCRandom_getGenerator(THCState* state);

namespace at {

OPENCLGenerator::OPENCLGenerator(Context * context_)
  : context(context_)
{
}

OPENCLGenerator::~OPENCLGenerator() {
  // no-op Generator state is global to the program
}

OPENCLGenerator& OPENCLGenerator::copy(const Generator& from) {
  throw std::runtime_error("OPENCLGenerator::copy() not implemented");
}

OPENCLGenerator& OPENCLGenerator::free() {
  //THCRandom_shutdown(context->getTHCState());
  return *this;
}

uint64_t OPENCLGenerator::seed() {
  return 0;//THCRandom_initialSeed(context->getTHCState());
}

uint64_t OPENCLGenerator::initialSeed() {
  return 0;//THCRandom_initialSeed(context->getTHCState());
}

OPENCLGenerator& OPENCLGenerator::manualSeed(uint64_t seed) {
  //THCRandom_manualSeed(context->getTHCState(), seed);
  return *this;
}

OPENCLGenerator& OPENCLGenerator::manualSeedAll(uint64_t seed) {
  //THCRandom_manualSeedAll(context->getTHCState(), seed);
  return *this;
}

void * OPENCLGenerator::unsafeGetTH() {
  return (void*)NULL;//THCRandom_getGenerator(context->getTHCState());
}

} // namespace at
