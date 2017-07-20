
#ifndef CAFFE2_CONTRIB_OPENCL_CL_UTILS_H_
#define CAFFE2_CONTRIB_OPENCL_CL_UTILS_H_

#include "caffe2/core/logging.h"

#include "libopencl.h"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace caffe2 {

inline void cl_utils_log_kernel_times(const std::string &opName, const cl::Event& event) {

  cl_ulong tq, tsub, ts, te;
  cl_int res;

  res = event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &tq);
  CAFFE_ENFORCE_EQ(res, CL_SUCCESS, "result: ", res);

  res = event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &tsub);
  CAFFE_ENFORCE_EQ(res, CL_SUCCESS, "result: ", res);

  res = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
  CAFFE_ENFORCE_EQ(res, CL_SUCCESS, "result: ", res);

  res = event.getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
  CAFFE_ENFORCE_EQ(res, CL_SUCCESS, "result: ", res);

  LOG(INFO)<<"Running "<<opName<<", reached device after: "<<(tsub - tq) / 1000<<" [us], started execution after another: "<<(ts - tsub) / 1000
  <<" [us], then executed in: "<<(te - ts) / 1000<<" [us]";
}

}


#endif /* CAFFE2_CONTRIB_OPENCL_CL_UTILS_H_ */
