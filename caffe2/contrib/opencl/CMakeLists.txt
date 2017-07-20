# ---[ OpenCL
if(USE_OPENCL)
  # OpenCL stub fails with NVIDIA OpenCL 390 driver.
  # A compilation was disabled
  #set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_SOURCE_DIR}/third_party/libopencl-stub/src/libopencl.c)

  # ---[ CPU files.
  file(GLOB tmp *.cc)
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})

  file(GLOB test2 *_test.cc)
  exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${test2})

  add_subdirectory(operators)

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
  set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
else()
  message(STATUS "OpenCL operators skipped due to disabled flag USE_OPENCL")
endif()
