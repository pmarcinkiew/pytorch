#include<vector>
#include<array>
#include<algorithm>
#include<iterator>
#include<utility>
#include<typeinfo>
#include<type_traits>
#include<fstream>
#include<string>
#include<sstream>
#include<ios>
#include<chrono>
#include<gtest/gtest.h>

#include "caffe2/core/init.h"
#include "caffe2/core/context.h"
#include "caffe2/utils/math.h"
#include "caffe2/operators/pool_op.h"
#include "caffe2/contrib/opencl/context.h"

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {


template<typename Context> 
void setDeviceType(OperatorDef& def) {
  def.mutable_device_option()->set_device_type(CPU);
}


template<>
void setDeviceType<OpenCLContext>(OperatorDef& def){
  def.mutable_device_option()->set_device_type(OPENCL);
}


template <typename CtxA, typename CtxB>
struct CLContext{
  using Contex = CtxA;
};


template <typename Ctx>
struct CLContext<OpenCLContext, Ctx> {
  using Context = OpenCLContext;
};


template<typename Ctx>
struct CLContext<Ctx, OpenCLContext> {
  using Context = OpenCLContext;
};


template<typename CtxSrc, typename CtxDst> 
class TensorToTensor {

  public:
    static void transfer(Tensor<CtxSrc> &tensorSrc, Tensor<CtxDst> &tensorDst) {

      tensorDst.ResizeLike(tensorSrc);

      DeviceOption option;

      using Ctx = typename CLContext<CtxSrc, CtxDst>::Context;

      Ctx  context(option);
      context.template Copy<float>(tensorSrc, tensorDst);
    }
};


template <typename Ctx>
class TensorToTensor<Ctx, Ctx> {
  public:
    static void transfer(Tensor<Ctx> &tensorSrc, Tensor<Ctx> &tensorDst) {
      tensorDst.ResizeLike(tensorSrc);
      tensorDst.ShareData(tensorSrc);
    }
};


struct ShapeConfig {
  int N;
  int C;
  int H;
  int W;
  int D;
};


using OperatorLead = std::pair<std::string, TensorCPU&>;
using OperatorLeads = std::vector<OperatorLead>;
using OperatorArgs = std::vector<Argument>;
using OperatorDesc = std::pair<std::string, std::string>;

void showTensorData(TensorCPU &);


template <typename Context>
class OpTest {

  public:
    OpTest(OperatorDesc &operatorDesc, OperatorLeads &inputLeads, OperatorLeads &outputLeads, OperatorArgs &args) 
    : m_context(m_option),
      m_operatorDesc(operatorDesc), m_inputLeads(inputLeads), m_outputLeads(outputLeads), m_args(args)
    {
        setDeviceType<Context>(m_opDef);
    };

  public:

    void operator()() {

      m_opDef.set_name(m_operatorDesc.first);  
      m_opDef.set_type(m_operatorDesc.second);

      copyDataFromInputTensorsCPU();

      addInputLeads();
      addOutputLeads();
      addArguments();
      auto start = std::chrono::system_clock::now();

      runOperator();

      auto end = std::chrono::system_clock::now();
      m_elapsed_time = end - start;

      copyDataToOutputTensorsCPU();
      return;
    }
    std::chrono::duration<double> elapsed_time(){ return m_elapsed_time; }

  private:
    DeviceOption m_option;
    Context  m_context;
    Workspace m_ws;
    OperatorDef m_opDef;

    OperatorDesc& m_operatorDesc;
    OperatorLeads& m_inputLeads;
    OperatorLeads& m_outputLeads;
    OperatorArgs& m_args;
    std::chrono::duration<double> m_elapsed_time;


    void copyDataFromInputTensorsCPU() {

      for(OperatorLead &inputLead: m_inputLeads) {
        auto inputLeadName = inputLead.first;
        auto inputTensor = m_ws.CreateBlob(inputLeadName)->GetMutable<Tensor<Context>>();
        auto & inputTensorCPU = inputLead.second;

        TensorToTensor<CPUContext, Context>::transfer(inputTensorCPU, *inputTensor);
       }
    }

    void addInputLeads() {
      for(OperatorLead &inputLead: m_inputLeads) {
       auto inputLeadName = inputLead.first;
       m_opDef.add_input(inputLeadName);
      }
    }

    void addOutputLeads() {
      for (OperatorLead &outputLead: m_outputLeads) {
        auto outputLeadName = outputLead.first;
        m_ws.CreateBlob(outputLeadName);
        m_opDef.add_output(outputLeadName);
      }
    }

    void addArguments() {
      for(Argument& arg: m_args) {
        m_opDef.add_arg()->CopyFrom(arg);
      }
    }

    void runOperator() {
      unique_ptr<OperatorBase> op(CreateOperator(m_opDef, &m_ws));

      op.get();
      op->Run();
    }

    void copyDataToOutputTensorsCPU() {

      for (OperatorLead &outputLead: m_outputLeads) {
        auto outputLeadName = outputLead.first;
        auto outputTensor = m_ws.GetBlob(outputLeadName)->GetMutable<Tensor<Context>>();
        auto& outputTensorCPU = outputLead.second;

        TensorToTensor<Context, CPUContext>::transfer(*outputTensor, outputTensorCPU);
      }
    }
};


void fillInputTensor(TensorCPU &inputTensor) {
  auto* data = inputTensor.template mutable_data<float>();
  int x = 0;
  for (x=0; x<inputTensor.size(); x++) {
    data[x] = (float)(x%5);
  }
}


void zeroInputTensor(TensorCPU &inputTensor) {
  auto* data = inputTensor.template mutable_data<float>();
  int x = 0;
  for (x=0; x<inputTensor.size(); x++) {
    data[x] = 0.0f;
  }
}


void showTensorData(TensorCPU &tensor) {
  const int cN = 0;
  const int cC = 1;
  const int cH = 2;
  const int cW = 3;
  const int cD = 4;

  int tensor_H = tensor.dim32(cW);
  std::cout<< "-----------==========" << std::endl;
  int tensor_W = tensor.dim32(cH);
  int area_size = tensor_H * tensor_W;
  int tensor_N = tensor.dim32(cN);
  int tensor_C = tensor.dim32(cC);
  std::cout << tensor_C<<std::endl;

  for (int iN = 0; iN < tensor_N; iN++) {
    std::cout<<"iN:---:"<<iN<<std::endl;
      for (int iC = 0; iC < tensor_C; iC++) {
        std::cout<<"iC::"<<iC<<std::endl;

          for(int y = 0; y < tensor_H; y++){

            for (int x = 0; x < tensor_W; x++) {

              int area_idx = iN * tensor_C + iC;

              std::cout<<tensor.data<float>()[   area_idx*(area_size) + (y*tensor_W + x) ]<<" ";

            }
            std::cout<<std::endl;
          }
          std::cout<<std::endl;
       }
       std::cout<<std::endl<<std::endl;
    }
}

using FFPair = std::pair<float, float>;


class TensorsPairedData {

  public:
    TensorsPairedData() = delete;

    TensorsPairedData(const TensorCPU &tensorCpu, const TensorCPU &tensorCl) {
      bool tensors_size_are_not_the_same = (tensorCpu.size() != tensorCl.size());
      bool tensor_size_is_zero = ((tensorCpu.size() == 0) || (tensorCl.size() == 0));

      if (tensors_size_are_not_the_same || tensor_size_is_zero) {
          compareResult.resize(0);
          return;
      }
      fillPair(tensorCpu, tensorCl);
    };
    
   const std::vector<FFPair>& operator()() const  {return compareResult;}

  private:
    std::vector<FFPair> compareResult;

    void fillPair(const TensorCPU &tensorCpu, const TensorCPU &tensorCl) {

      const float* cpuData = tensorCpu.data<float>();

      std::vector<float> vCpu(cpuData, cpuData + tensorCpu.size());

      const float* clData = tensorCl.data<float>();
      std::vector<float> vCl(clData, clData + tensorCl.size());

      std::transform(
         vCpu.begin(),
         vCpu.end(),
         vCl.begin(),
         std::back_inserter(compareResult),
         [](float a, float b) {return std::make_pair(a, b); }
     );
   }
};


using shapeVector = std::vector<TIndex>;


struct AA {
   int a;
   int b;
};



TEST(OpenCL_and_CPU, MaxPool1DNCHWTest) {

  ShapeConfig sh;

  shapeVector sv{
    sh.N=3,
    sh.C=3,
    sh.H=16
  };

  TensorCPU tensorX{sv};

  fillInputTensor(tensorX);

  TensorCPU tensorYCpu;
  TensorCPU tensorYCl;

  OperatorDesc maxPoolDesc{"MaxPoolOp", "MaxPool"};
  OperatorLeads inLeads{{"inX", tensorX}};

  OperatorLeads outLeadsCPU{{"outY", tensorYCpu}};
  OperatorLeads outLeadsCL{{"outY", tensorYCl}};


  std::vector<Argument> argumentList{
    MakeArgument("order", std::string("NCHW")),
    MakeArgument("kernels", std::vector<int>{2}),
    MakeArgument("strides", std::vector<int>{2})
  };

  OpTest<CPUContext> maxPoolTestCPU{maxPoolDesc, inLeads, outLeadsCPU, argumentList};
  maxPoolTestCPU();
 
  OpTest<OpenCLContext> maxPoolTestCL{maxPoolDesc, inLeads, outLeadsCL, argumentList};
  maxPoolTestCL();

  EXPECT_EQ(tensorYCpu.size(), tensorYCl.size());

  TensorsPairedData tensorsPairedData{tensorYCpu, tensorYCl};

  const std::vector<FFPair>& compareOutput = tensorsPairedData();

  for (FFPair compareFFPair: compareOutput) {
    SCOPED_TRACE("Cpu value --- OpenCL value");
    EXPECT_FLOAT_EQ(compareFFPair.first, compareFFPair.second);
  }
  SUCCEED();
}



TEST(OpenCL_and_CPU, MaxPool2DNCHWTest) {

  ShapeConfig sh;

  shapeVector sv{
    sh.N=2,
    sh.C=3,
    sh.H=16,
    sh.W=16
  };
  
  TensorCPU tensorX{sv};

  fillInputTensor(tensorX);

  TensorCPU tensorYCpu;
  TensorCPU tensorYCl;


  OperatorDesc maxPoolDesc{"MaxPoolOp", "MaxPool"};
  OperatorLeads inLeads{{"inX", tensorX}};

  OperatorLeads outLeadsCPU{{"outY", tensorYCpu}};
  OperatorLeads outLeadsCL{{"outY", tensorYCl}};

  std::vector<Argument> argumentList{
    MakeArgument("order", std::string("NCHW")),
    MakeArgument("kernels", std::vector<int>{4,2}),
    MakeArgument("strides", std::vector<int>{2,2})
  };


  OpTest<CPUContext> maxPoolTestCPU{maxPoolDesc, inLeads, outLeadsCPU, argumentList};
  maxPoolTestCPU();

  OpTest<OpenCLContext> maxPoolTestCL{maxPoolDesc, inLeads, outLeadsCL, argumentList};
  maxPoolTestCL();

  EXPECT_EQ(tensorYCpu.size(), tensorYCl.size());

  TensorsPairedData tensorsPairedData{tensorYCpu, tensorYCl};

  const std::vector<FFPair>& compareOutput = tensorsPairedData();

  for (FFPair compareFFPair: compareOutput) {
    SCOPED_TRACE("Cpu value --- OpenCL value");
    EXPECT_FLOAT_EQ(compareFFPair.first, compareFFPair.second);
  }
  SUCCEED();
}


TEST(OpenCL_and_CPU, MaxPool2DNCHWTestLegacy) {

  ShapeConfig sh;

  shapeVector sv{
    sh.N=2,
    sh.C=3,
    sh.H=16,
    sh.W=16
  };

  TensorCPU tensorX{sv};

  fillInputTensor(tensorX);

  TensorCPU tensorYCpu;
  TensorCPU tensorYCl;

  OperatorDesc maxPoolDesc{"MaxPoolOp", "MaxPool"};
  OperatorLeads inLeads{{"inX", tensorX}};

  OperatorLeads outLeadsCPU{{"outY", tensorYCpu}};
  OperatorLeads outLeadsCL{{"outY", tensorYCl}};

  std::vector<Argument> argumentList{
    MakeArgument("order", std::string("NCHW")),
    MakeArgument("kernel", 2),
    MakeArgument("stride", 2)
  };


  OpTest<CPUContext> maxPoolTestCPU{maxPoolDesc, inLeads, outLeadsCPU, argumentList};
  maxPoolTestCPU();
 
  OpTest<OpenCLContext> maxPoolTestCL{maxPoolDesc, inLeads, outLeadsCL, argumentList};
  maxPoolTestCL();
 
  EXPECT_EQ(tensorYCpu.size(), tensorYCl.size());

  TensorsPairedData tensorsPairedData{tensorYCpu, tensorYCl};

  const std::vector<FFPair>& compareOutput = tensorsPairedData();

  for (FFPair compareFFPair: compareOutput) {
    SCOPED_TRACE("Cpu value --- OpenCL value");
    EXPECT_FLOAT_EQ(compareFFPair.first, compareFFPair.second);
  }
  SUCCEED();
}


TEST(OpenCL_and_CPU, MaxPool3DNCHWTest) {

  ShapeConfig sh;

   shapeVector sv{
    sh.N=3,
    sh.C=3,
    sh.H=4,
    sh.W=4,
    sh.D=4
  }; 

  TensorCPU tensorX{sv};

  fillInputTensor(tensorX);

  TensorCPU tensorYCpu;
  TensorCPU tensorYCl;


  OperatorDesc maxPoolDesc{"MaxPoolOp", "MaxPool"};
  OperatorLeads inLeads{{"inX", tensorX}};

  OperatorLeads outLeadsCPU{{"outY", tensorYCpu}};
  OperatorLeads outLeadsCL{{"outY", tensorYCl}};

  std::vector<Argument> argumentList{
     MakeArgument("order", std::string("NCHW")),
     MakeArgument("kernels", std::vector<int>{2,2,2}),
     MakeArgument("strides", std::vector<int>{1,2,2})
   };


  OpTest<CPUContext> maxPoolTestCPU{maxPoolDesc, inLeads, outLeadsCPU, argumentList};
  maxPoolTestCPU();

  OpTest<OpenCLContext> maxPoolTestCL{maxPoolDesc, inLeads, outLeadsCL, argumentList};
  maxPoolTestCL();

  EXPECT_EQ(tensorYCpu.size(), tensorYCl.size());

  TensorsPairedData tensorsPairedData{tensorYCpu, tensorYCl};

  const std::vector<FFPair>& compareOutput = tensorsPairedData();

  for (FFPair compareFFPair: compareOutput) {
    SCOPED_TRACE("Cpu value --- OpenCL value");
    EXPECT_FLOAT_EQ(compareFFPair.first, compareFFPair.second);
  }
  SUCCEED();
}


TEST(OpenCL_and_CPU, MaxPool3DNCHWTestOnePointCube) {

  ShapeConfig sh;

  shapeVector sv{
    sh.N=3,
    sh.C=3,
    sh.H=1,
    sh.W=1,
    sh.D=1
  };

  TensorCPU tensorX{sv};

  fillInputTensor(tensorX);

  TensorCPU tensorYCpu;
  TensorCPU tensorYCl;

  OperatorDesc maxPoolDesc{"MaxPoolOp", "MaxPool"};
  OperatorLeads inLeads{{"inX", tensorX}};

  OperatorLeads outLeadsCPU{{"outY", tensorYCpu}};
  OperatorLeads outLeadsCL{{"outY", tensorYCl}};

  std::vector<Argument> argumentList{
    MakeArgument("order", std::string("NCHW")),
    MakeArgument("kernels", std::vector<int>{1,1,1}),
    MakeArgument("strides", std::vector<int>{1,1,1})
  };

  OpTest<CPUContext> maxPoolTestCPU{maxPoolDesc, inLeads, outLeadsCPU, argumentList};
  maxPoolTestCPU();

  OpTest<OpenCLContext> maxPoolTestCL{maxPoolDesc, inLeads, outLeadsCL, argumentList};
  maxPoolTestCL();

  EXPECT_EQ(tensorYCpu.size(), tensorYCl.size());

  TensorsPairedData tensorsPairedData{tensorYCpu, tensorYCl};

  const std::vector<FFPair>& compareOutput = tensorsPairedData();

  for (FFPair compareFFPair: compareOutput) {
    SCOPED_TRACE("Cpu value --- OpenCL value");
    EXPECT_FLOAT_EQ(compareFFPair.first, compareFFPair.second);
  }
  SUCCEED();

}


TEST(OpenCL_and_CPU, MaxPool2DNCHWTest20x12x12) {

  ShapeConfig sh;

  shapeVector sv{
    sh.N=1,
    sh.C=20,
    sh.H=12,
    sh.W=12,
  };

  TensorCPU tensorX{sv};

  fillInputTensor(tensorX);

  TensorCPU tensorYCpu;
  TensorCPU tensorYCl;

  OperatorDesc maxPoolDesc{"MaxPoolOp", "MaxPool"};
  OperatorLeads inLeads{{"inX", tensorX}};

  OperatorLeads outLeadsCPU{{"outY", tensorYCpu}};
  OperatorLeads outLeadsCL{{"outY", tensorYCl}};

  std::vector<Argument> argumentList{
    MakeArgument("order", std::string("NCHW")), 
    MakeArgument("kernels", std::vector<int>{2,2}),
    MakeArgument("strides", std::vector<int>{2,2})
  };


  OpTest<CPUContext> maxPoolTestCPU{maxPoolDesc, inLeads, outLeadsCPU, argumentList};
  maxPoolTestCPU();

  OpTest<OpenCLContext> maxPoolTestCL{maxPoolDesc, inLeads, outLeadsCL, argumentList};
  maxPoolTestCL();

  EXPECT_EQ(tensorYCpu.size(), tensorYCl.size());

  TensorsPairedData tensorsPairedData{tensorYCpu, tensorYCl};

  const std::vector<FFPair>& compareOutput = tensorsPairedData();

  for (FFPair compareFFPair: compareOutput) {
    SCOPED_TRACE("Cpu value --- OpenCL value");
    EXPECT_FLOAT_EQ(compareFFPair.first, compareFFPair.second);
  }
  SUCCEED();
}



TEST(OpenCL_and_CPU, MaxPool2DNCHWTest50x4x4) {

  ShapeConfig sh;

  shapeVector sv{
    sh.N=1,
    sh.C=50,
    sh.H=4,
    sh.W=4,
  };

  TensorCPU tensorX{sv};

  fillInputTensor(tensorX);

  TensorCPU tensorYCpu;
  TensorCPU tensorYCl;

  OperatorDesc maxPoolDesc{"MaxPoolOp", "MaxPool"};
  OperatorLeads inLeads{{"inX", tensorX}};

  OperatorLeads outLeadsCPU{{"outY", tensorYCpu}};
  OperatorLeads outLeadsCL{{"outY", tensorYCl}};

  std::vector<Argument> argumentList{
    MakeArgument("order", std::string("NCHW")),
    MakeArgument("kernels", std::vector<int>{2,2}),
    MakeArgument("strides", std::vector<int>{2,2})
  };


  OpTest<CPUContext> maxPoolTestCPU{maxPoolDesc, inLeads, outLeadsCPU, argumentList};
  maxPoolTestCPU();

  OpTest<OpenCLContext> maxPoolTestCL{maxPoolDesc, inLeads, outLeadsCL, argumentList};
  maxPoolTestCL();


  EXPECT_EQ(tensorYCpu.size(), tensorYCl.size());

  TensorsPairedData tensorsPairedData{tensorYCpu, tensorYCl};

  const std::vector<FFPair>& compareOutput = tensorsPairedData();

  for (FFPair compareFFPair: compareOutput) {
    SCOPED_TRACE("Cpu value --- OpenCL value");
    EXPECT_FLOAT_EQ(compareFFPair.first, compareFFPair.second);
  }
  SUCCEED();
}


}// namespace caffe2

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe2::GlobalInit(&argc, &argv);
  auto result = RUN_ALL_TESTS();
  google::protobuf::ShutdownProtobufLibrary();
  return result;

}
