/**
 * Copyright (c) 2018-present, Samsung Electronics
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
