#include <gtest/gtest.h>
#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_async_scheduling.h"
#include "caffe2/core/net_dag.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"

#include <google/protobuf/text_format.h>
#include "context.h"

CAFFE2_DECLARE_bool(caffe2_disable_chaining);

namespace caffe2 {

namespace {

static std::atomic<int> counter;

class NetTestDummyOp final : public Operator<OpenCLContext> {
 public:
  using OperatorBase::OperatorBase;

  NetTestDummyOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        fail_(OperatorBase::GetSingleArgument<bool>("fail", false)) {}

  bool RunOnDevice() override {
    if (fail_) {
      return false;
    }
    counter.fetch_add(1);
    return true;
  }

 protected:
  const bool fail_;
};

REGISTER_OPENCL_OPERATOR(NetTestDummy, NetTestDummyOp);
REGISTER_OPENCL_OPERATOR(NetTestDummy2, NetTestDummyOp);

OPERATOR_SCHEMA(NetTestDummy)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});
OPERATOR_SCHEMA(NetTestDummy2)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{1, 0}});

unique_ptr<NetBase> CreateNetTestHelper(
    Workspace* ws,
    const vector<string>& input,
    const vector<string>& output) {
  NetDef net_def;
  {
    auto& op = *(net_def.add_op());
    op.set_type("NetTestDummy");
    op.add_input("in");
    op.add_output("hidden");
  }
  {
    auto& op = *(net_def.add_op());
    op.set_type("NetTestDummy");
    op.add_input("hidden");
    op.add_output("out");
  }

  for (const auto& name : input) {
    net_def.add_external_input(name);
  }
  for (const auto& name : output) {
    net_def.add_external_output(name);
  }
  net_def.mutable_device_option()->set_device_type(OPENCL);
  auto net = CreateNet(net_def, ws);
  return net;
}

} // namespace

TEST(NetTest, ConstructionNoDeclaredInputOutput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>(), vector<string>()));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, ConstructionDeclaredInput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>{"in"}, vector<string>()));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, ConstructionDeclaredOutput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>(), vector<string>{"out"}));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, DeclaredInputInsufficient) {
  Workspace ws;
  ws.CreateBlob("in");
  ASSERT_THROW(
      CreateNetTestHelper(&ws, vector<string>{"unuseful_in"}, vector<string>()),
      EnforceNotMet);
}

TEST(NetDeathTest, DeclaredOutputNotMet) {
  Workspace ws;
  ws.CreateBlob("in");
  ASSERT_THROW(
      CreateNetTestHelper(
          &ws, vector<string>(), vector<string>{"unproduced_out"}),
      EnforceNotMet);
}

void testExecution(std::unique_ptr<NetBase>& net, int num_ops) {
  // Run 100 times
  for (int i = 0; i < 100; i++) {
    counter.exchange(0);
    net.get()->Run();
    ASSERT_EQ(num_ops, counter.load());
  }
}

void checkChainingAndRun(
    const char* spec,
    const dag_utils::ExecutionChains& expected) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(spec, &net_def));
  {
    net_def.set_num_workers(4);
    net_def.mutable_device_option()->set_device_type(OPENCL);
    auto old = FLAGS_caffe2_disable_chaining;
    auto g = MakeGuard([&]() { FLAGS_caffe2_disable_chaining = old; });
    FLAGS_caffe2_disable_chaining = false;

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
    CHECK_NOTNULL(dag);
    const auto& chains = dag->TEST_execution_chains();
    EXPECT_TRUE(chains == expected);
    testExecution(net, net_def.op().size());
  }
}

void checkNumChainsAndRun(const char* spec, const int expected_num_chains) {
  Workspace ws;

  NetDef net_def;
  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(spec, &net_def));
  net_def.set_num_workers(4);
  net_def.mutable_device_option()->set_device_type(OPENCL);

  // Create all external inputs
  for (auto inp : net_def.external_input()) {
    ws.CreateBlob(inp);
  }

  {
    auto old = FLAGS_caffe2_disable_chaining;
    auto g = MakeGuard([&]() { FLAGS_caffe2_disable_chaining = old; });
    FLAGS_caffe2_disable_chaining = false;

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
    CHECK_NOTNULL(dag);
    const auto& chains = dag->TEST_execution_chains();
    EXPECT_EQ(expected_num_chains, chains.size());
    testExecution(net, net_def.op().size());
  }
}

TEST(NetTest, ChainingForLinearModel) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out"
          type: "NetTestDummy"
        }
)DOC";
  checkChainingAndRun(spec, {{0, {0, 1}}});
}

TEST(NetTest, ChainingForFork) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out1"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out2"
          type: "NetTestDummy"
        }
)DOC";
  checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2}}});
}

// TEST(NetTest, ChainingForJoinWithAncestor) {
//   const auto spec = R"DOC(
//         name: "example"
//         type: "dag"
//         external_input: "in"
//         op {
//           input: "in"
//           output: "hidden"
//           type: "NetTestDummy"
//         }
//         op {
//           input: "hidden"
//           output: "out1"
//           type: "NetTestDummy"
//         }
//         op {
//           input: "hidden"
//           output: "out2"
//           type: "NetTestDummy"
//         }
//         op {
//           input: "hidden"
//           input: "out2"
//           type: "NetTestDummy"
//         }
// )DOC";
//   checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
// }

TEST(NetTest, ChainingForForkJoin) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden1"
          type: "NetTestDummy"
        }
        op {
          input: "in"
          output: "hidden2"
          type: "NetTestDummy"
        }
        op {
          input: "hidden1"
          input: "hidden2"
          output: "out"
          type: "NetTestDummy"
        }
        op {
          input: "out"
          output: "out2"
          type: "NetTestDummy"
        }
)DOC";
  checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
}

TEST(NetTest, ChainingForwardBackward) {
  const auto spec = R"DOC(
  name: "gpu_0"
  type: "dag"
  op {
    input: "in"
    input: "fc_0_w"
    input: "fc_0_b"
    output: "fc_0"
    name: "0"
    type: "NetTestDummy"
  }
  op {
    input: "fc_0"
    output: "fc_0"
    name: "1"
    type: "NetTestDummy"
  }
  op {
    input: "fc_0"
    input: "fc_1_w"
    input: "fc_1_b"
    output: "fc_1"
    name: "2"
    type: "NetTestDummy"
  }
  op {
    input: "fc_1"
    output: "fc_1"
    name: "3"
    type: "NetTestDummy"
  }
  op {
    input: "fc_1"
    input: "fc_2_w"
    input: "fc_2_b"
    output: "fc_2"
    name: "4"
    type: "NetTestDummy"
  }
  op {
    input: "fc_2"
    output: "fc_2"
    name: "5"
    type: "NetTestDummy"
  }
  op {
    input: "fc_2"
    input: "fc_3_w"
    input: "fc_3_b"
    output: "fc_3"
    name: "6"
    type: "NetTestDummy"
  }
  op {
    input: "fc_3"
    output: "fc_3"
    name: "7"
    type: "NetTestDummy"
  }
  op {
    input: "fc_3"
    input: "fc_4_w"
    input: "fc_4_b"
    output: "fc_4"
    name: "8"
    type: "NetTestDummy"
  }
  op {
    input: "fc_4"
    output: "fc_4"
    name: "9"
    type: "NetTestDummy"
  }
  op {
    input: "fc_4"
    input: "in2"
    output: "LabelCrossEntropy"
    name: "10"
    type: "NetTestDummy"
  }
  op {
    input: "LabelCrossEntropy"
    output: "AveragedLoss"
    name: "11"
    type: "NetTestDummy"
  }
  op {
    input: "AveragedLoss"
    output: "AveragedLoss_autogen_grad"
    name: "12"
    type: "NetTestDummy"
  }
  op {
    input: "LabelCrossEntropy"
    input: "AveragedLoss_autogen_grad"
    output: "LabelCrossEntropy_grad"
    name: "13"
    type: "NetTestDummy"
  }
  op {
    input: "fc_4"
    input: "label"
    input: "LabelCrossEntropy_grad"
    output: "fc_4_grad"
    name: "14"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_4"
    input: "fc_4_grad"
    output: "fc_4_grad"
    name: "15"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_3"
    input: "fc_4_w"
    input: "fc_4_grad"
    output: "fc_4_w_grad"
    output: "fc_4_b_grad"
    output: "fc_3_grad"
    name: "16"
    type: "NetTestDummy"
  }
  op {
    input: "fc_3"
    input: "fc_3_grad"
    output: "fc_3_grad"
    name: "17"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_2"
    input: "fc_3_w"
    input: "fc_3_grad"
    output: "fc_3_w_grad"
    output: "fc_3_b_grad"
    output: "fc_2_grad"
    name: "18"
    type: "NetTestDummy"
  }
  op {
    input: "fc_2"
    input: "fc_2_grad"
    output: "fc_2_grad"
    name: "19"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_1"
    input: "fc_2_w"
    input: "fc_2_grad"
    output: "fc_2_w_grad"
    output: "fc_2_b_grad"
    output: "fc_1_grad"
    name: "20"
    type: "NetTestDummy"
  }
  op {
    input: "fc_1"
    input: "fc_1_grad"
    output: "fc_1_grad"
    name: "21"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_0"
    input: "fc_1_w"
    input: "fc_1_grad"
    output: "fc_1_w_grad"
    output: "fc_1_b_grad"
    output: "fc_0_grad"
    name: "22"
    type: "NetTestDummy"
  }
  op {
    input: "fc_0"
    input: "fc_0_grad"
    output: "fc_0_grad"
    name: "23"
    type: "NetTestDummy2"
  }
  op {
    input: "in"
    input: "fc_0_w"
    input: "fc_0_grad"
    output: "fc_0_w_grad"
    output: "fc_0_b_grad"
    output: "data_grad"
    name: "24"
    type: "NetTestDummy"
  }
  external_input: "in"
  external_input: "in2"
  external_input: "LR"
  external_input: "fc_0_w"
  external_input: "fc_0_b"
  external_input: "fc_1_w"
  external_input: "fc_1_b"
  external_input: "fc_2_w"
  external_input: "fc_2_b"
  external_input: "fc_3_w"
  external_input: "fc_3_b"
  external_input: "fc_4_w"
  external_input: "fc_4_b"
  external_input: "label"
  )DOC";
  checkNumChainsAndRun(spec, 1);
}

TEST(NetTest, ChainingForHogwildModel) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden1"
          type: "NetTestDummy"
        }
        op {
          input: "hidden1"
          output: "mid1"
          type: "NetTestDummy"
        }
        op {
          input: "mid1"
          output: "out1"
          type: "NetTestDummy"
        }
        op {
          input: "in"
          output: "hidden2"
          type: "NetTestDummy"
        }
        op {
          input: "hidden2"
          output: "mid2"
          type: "NetTestDummy"
        }
        op {
          input: "mid2"
          output: "out2"
          type: "NetTestDummy"
        }
)DOC";
  checkNumChainsAndRun(spec, 2);
}

TEST(NetTest, FailingOperator) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out"
          type: "NetTestDummy"
          arg {
            name: "fail"
            i: 1
          }
        }
)DOC";

  Workspace ws;
  ws.CreateBlob("in");

  NetDef net_def;
  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(spec, &net_def));

  {
    net_def.set_num_workers(4);
    net_def.mutable_device_option()->set_device_type(OPENCL);
    auto old = FLAGS_caffe2_disable_chaining;
    auto g = MakeGuard([&]() { FLAGS_caffe2_disable_chaining = old; });
    FLAGS_caffe2_disable_chaining = false;

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    for (int i = 0; i < 10; i++) {
      counter.exchange(0);
      bool run_result = false;
      try {
        run_result = net->Run();
      } catch (const std::exception&) {
        // async_scheduling would throw
      }
      ASSERT_FALSE(run_result);

      ASSERT_EQ(1, counter.load());
    }
  }
}

const int kTestPoolSize = 4;

class ExecutorHelperDummyOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;

  ExecutorHelperDummyOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws) {}

  bool Run(int /* unused */ /*stream_id*/) override {
    auto helper = GetExecutorHelper();
    CAFFE_ENFORCE(helper);
    auto pool = helper->GetPool(device_option());
    CAFFE_ENFORCE(pool);
    auto pool_size = pool->size();
    CAFFE_ENFORCE_EQ(pool_size, kTestPoolSize);
    return true;
  }
};

REGISTER_OPENCL_OPERATOR(ExecutorHelperDummy, ExecutorHelperDummyOp);

OPERATOR_SCHEMA(ExecutorHelperDummy);

TEST(NetTest, OperatorWithExecutorHelper) {
  const auto spec = R"DOC(
        name: "example"
        type: "async_scheduling"
        op {
          type: "ExecutorHelperDummy"
        }
)DOC";

  NetDef net_def;
  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(spec, &net_def));

  Workspace ws;
  net_def.set_num_workers(kTestPoolSize);
  net_def.mutable_device_option()->set_device_type(OPENCL);
  std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  ASSERT_TRUE(net->Run());
}

TEST(NetTest, OperatorWithDisabledEvent) {
  const auto spec = R"DOC(
        name: "example"
        type: "async_scheduling"
        external_input: "in"
        op {
          input: "in"
          output: "out"
          type: "NetTestDummy"
          arg {
            name: "fail"
            i: 1
          }
        }
)DOC";

  Workspace ws;
  ws.CreateBlob("in");

  NetDef net_def;
  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(spec, &net_def));

  {
    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    net->GetOperators()[0]->DisableEvent();
    // async_scheduling propagates exception
    bool caught_exception = false;
    try {
      net->Run();
    } catch (const std::exception& e) {
      caught_exception = true;
    }
    ASSERT_TRUE(caught_exception);
  }
}

TEST(NetTest, ExecutorOverride) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
  )DOC";

  NetDef net_def;
  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(spec, &net_def));

  {
    Workspace ws;
    auto old = FLAGS_caffe2_override_executor;
    auto g = MakeGuard([&]() { FLAGS_caffe2_override_executor = old; });
    FLAGS_caffe2_override_executor = "dag,async_scheduling";

    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto async_net =
        caffe2::dynamic_cast_if_rtti<AsyncSchedulingNet*>(net.get());
    ASSERT_TRUE(async_net != nullptr);
  }
}

TEST(NetTest, AsyncEmptyNet) {
  const auto spec = R"DOC(
        name: "example"
        type: "async_scheduling"
  )DOC";

  Workspace ws;
  NetDef net_def;
  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(spec, &net_def));

  {
    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    bool caught_exception = false;
    try {
      ASSERT_TRUE(net->Run());
    } catch (const std::exception& e) {
      caught_exception = true;
    }
    ASSERT_FALSE(caught_exception);
  }
}

TEST(NetTest, RunAsyncFailure) {
  const auto spec = R"DOC(
        name: "example"
        type: "async_scheduling"
        op {
          input: "in"
          output: "out"
          type: "NetTestDummy"
          arg {
            name: "fail"
            i: 1
          }
        }
  )DOC";

  Workspace ws;
  ws.CreateBlob("in");

  NetDef net_def;
  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(spec, &net_def));

  {
    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));

    ASSERT_FALSE(net->Run());
  }
}

TEST(NetTest, NoTypeNet) {
  const auto spec = R"DOC(
        name: "no_type_net"
  )DOC";

  Workspace ws;
  NetDef net_def;
  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(spec, &net_def));

  {
    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    ASSERT_TRUE(net);
  }
}

const auto async_dag_multichain_spec = R"DOC(
      name: "example"
      type: "async_dag"
      external_input: "in"
      external_input: "fc1_w"
      external_input: "fc1_b"
      external_input: "fc2_w"
      external_input: "fc2_b"
      external_input: "fc3_w"
      external_input: "fc3_b"
      external_input: "fc4_w"
      external_input: "fc4_b"
      external_input: "fc5_w"
      external_input: "fc5_b"
      external_input: "fc6_w"
      external_input: "fc6_b"

      op {
        input: "in"
        input: "fc1_w"
        input: "fc1_b"
        output: "fc1_out"
        type: "FC"
      }
      op {
        input: "fc1_out"
        input: "fc2_w"
        input: "fc2_b"
        output: "fc2_out"
        type: "FC"
      }
      op {
        input: "fc2_out"
        input: "fc3_w"
        input: "fc3_b"
        output: "fc3_out"
        type: "FC"
      }
      op {
        input: "fc3_out"
        input: "fc4_w"
        input: "fc4_b"
        output: "fc4_out"
        type: "FC"
      }

      op {
        input: "in"
        input: "fc5_w"
        input: "fc5_b"
        output: "fc5_out"
        type: "FC"
      }
      op {
        input: "fc5_out"
        input: "fc6_w"
        input: "fc6_b"
        output: "fc6_out"
        type: "FC"
      }

    )DOC";


TEST(NetTest, FC_async_dag_multichain) {

  Workspace ws;
  ws.CreateBlob("in")->GetMutable<TensorCL>()->Resize(1, 100); //K=100
  ws.GetBlob("in")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc1_w")->GetMutable<TensorCL>()->Resize(10000, 100); //N=10000, K=100
  ws.GetBlob("fc1_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc1_b")->GetMutable<TensorCL>()->Resize(10000); //N=10000
  ws.GetBlob("fc1_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc2_w")->GetMutable<TensorCL>()->Resize(2000, 10000); //N=2000, K=10000
  ws.GetBlob("fc2_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc2_b")->GetMutable<TensorCL>()->Resize(2000); //N=2000
  ws.GetBlob("fc2_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc3_w")->GetMutable<TensorCL>()->Resize(100, 2000); //N=100, K=2000
  ws.GetBlob("fc3_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc3_b")->GetMutable<TensorCL>()->Resize(100); //N=100;
  ws.GetBlob("fc3_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc4_w")->GetMutable<TensorCL>()->Resize(1, 100); //N=1, K=100;
  ws.GetBlob("fc4_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc4_b")->GetMutable<TensorCL>()->Resize(1); //N=1
  ws.GetBlob("fc4_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc5_w")->GetMutable<TensorCL>()->Resize(100, 100); //N=100, K=100;
  ws.GetBlob("fc5_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc5_b")->GetMutable<TensorCL>()->Resize(100); //N=100;
  ws.GetBlob("fc5_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc6_w")->GetMutable<TensorCL>()->Resize(1, 100); //N=1, K=100;
  ws.GetBlob("fc6_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc6_b")->GetMutable<TensorCL>()->Resize(1); //N=1;
  ws.GetBlob("fc6_b")->GetMutable<TensorCL>()->mutable_data<float>();

  NetDef net_def;

  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(async_dag_multichain_spec, &net_def));

  {
    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    ASSERT_TRUE(net);
    ASSERT_TRUE(net->Run());
  }
}

const auto simple_multichain_spec = R"DOC(
      name: "example"
      external_input: "in"
      external_input: "fc1_w"
      external_input: "fc1_b"
      external_input: "fc2_w"
      external_input: "fc2_b"
      external_input: "fc3_w"
      external_input: "fc3_b"
      external_input: "fc4_w"
      external_input: "fc4_b"
      external_input: "fc5_w"
      external_input: "fc5_b"
      external_input: "fc6_w"
      external_input: "fc6_b"

      op {
        input: "in"
        input: "fc1_w"
        input: "fc1_b"
        output: "fc1_out"
        type: "FC"
      }
      op {
        input: "fc1_out"
        input: "fc2_w"
        input: "fc2_b"
        output: "fc2_out"
        type: "FC"
      }
      op {
        input: "fc2_out"
        input: "fc3_w"
        input: "fc3_b"
        output: "fc3_out"
        type: "FC"
      }
      op {
        input: "fc3_out"
        input: "fc4_w"
        input: "fc4_b"
        output: "fc4_out"
        type: "FC"
      }

      op {
        input: "in"
        input: "fc5_w"
        input: "fc5_b"
        output: "fc5_out"
        type: "FC"
      }
      op {
        input: "fc5_out"
        input: "fc6_w"
        input: "fc6_b"
        output: "fc6_out"
        type: "FC"
      }

    )DOC";

TEST(NetTest, FC_simple_multichain) {

  Workspace ws;
  ws.CreateBlob("in")->GetMutable<TensorCL>()->Resize(1, 100); //K=100
  ws.GetBlob("in")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc1_w")->GetMutable<TensorCL>()->Resize(10000, 100); //N=10000, K=100
  ws.GetBlob("fc1_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc1_b")->GetMutable<TensorCL>()->Resize(10000); //N=10000
  ws.GetBlob("fc1_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc2_w")->GetMutable<TensorCL>()->Resize(2000, 10000); //N=2000, K=10000
  ws.GetBlob("fc2_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc2_b")->GetMutable<TensorCL>()->Resize(2000); //N=2000
  ws.GetBlob("fc2_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc3_w")->GetMutable<TensorCL>()->Resize(100, 2000); //N=100, K=2000
  ws.GetBlob("fc3_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc3_b")->GetMutable<TensorCL>()->Resize(100); //N=100;
  ws.GetBlob("fc3_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc4_w")->GetMutable<TensorCL>()->Resize(1, 100); //N=1, K=100;
  ws.GetBlob("fc4_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc4_b")->GetMutable<TensorCL>()->Resize(1); //N=1
  ws.GetBlob("fc4_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc5_w")->GetMutable<TensorCL>()->Resize(100, 100); //N=100, K=100;
  ws.GetBlob("fc5_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc5_b")->GetMutable<TensorCL>()->Resize(100); //N=100;
  ws.GetBlob("fc5_b")->GetMutable<TensorCL>()->mutable_data<float>();

  ws.CreateBlob("fc6_w")->GetMutable<TensorCL>()->Resize(1, 100); //N=1, K=100;
  ws.GetBlob("fc6_w")->GetMutable<TensorCL>()->mutable_data<float>();
  ws.CreateBlob("fc6_b")->GetMutable<TensorCL>()->Resize(1); //N=1;
  ws.GetBlob("fc6_b")->GetMutable<TensorCL>()->mutable_data<float>();

  NetDef net_def;

  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(simple_multichain_spec, &net_def));

  {
    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    ASSERT_TRUE(net);
    ASSERT_TRUE(net->Run());
  }
}


} // namespace caffe2
