/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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
 *******************************************************************************/
#include "TestCaseBuilder.h"
#include "TestUtilities.h"
#include "gtest/gtest.h"

#include "ngraph_utils.h"
#include "tf_graph_writer.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
// #include "tensorflow/core/graph/default_device.h" : should not need this
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {
// check why
#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

TEST_F(BuilderTest, PlaceTest) {
  Scope root = Scope::NewRootScope();
  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto r = ops::RealDiv(root, A, B);
  // vector<float> A_values = {{3.f, 5.f}, {2.f, 0.f}};
  // vector<float> B_values = {{3.f, 2.f}, {.1f, 1.f}};
  Tensor A_values(DT_FLOAT, TensorShape({2, 2}));
  Tensor B_values(DT_FLOAT, TensorShape({2, 2}));

  DummyAssignInputValues(A_values, 2.0f);
  DummyAssignInputValues(B_values, 2.0f);
  vector<DataType> output_datatypes = {DT_FLOAT};

  // Get the graph from the declared scope
  Graph tf_graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&tf_graph));

  // For debug
  GraphToPbTextFile(&tf_graph, "tf_graph_placeholder.pbtxt");

  /*
  // Compute the graph on nGraph and get output as TF Tensors
  vector<Tensor*> ngraph_outputs;
  ComputeOnNGraph(tf_graph, "RealDiv", output_datatypes, ngraph_outputs);
  */

  ClientSession session(root);
  std::vector<Tensor> outputs;

  ASSERT_OK(session.Run({{A, A_values}, {B, B_values}}, {r}, &outputs));

  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.0, mat(0, 0));
  EXPECT_FLOAT_EQ(2.5, mat(0, 1));
  EXPECT_FLOAT_EQ(20.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));
}

TEST_F(BuilderTest, DirectExecution) {
  // Create a tf graph
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  DummyAssignInputValues(A, 2.0f);
  DummyAssignInputValues(B, 2.0f);

  auto R = ops::Add(root, A, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  // Get the graph from the declared scope
  Graph tf_graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&tf_graph));

  // For debug
  GraphToPbTextFile(&tf_graph, "tf_graph.pbtxt");

  // Compute the graph on nGraph and get output as TF Tensors
  vector<Tensor*> ngraph_outputs;
  ComputeOnNGraph(tf_graph, "Add", output_datatypes, ngraph_outputs);
  NGRAPH_VLOG(5) << " printing ops " << ngraph_outputs.size();

  // Run on TF
  DummyDeactivateNGraph();
  ClientSession session(root);
  vector<Tensor> tf_outputs;
  // Run and fetch v
  ASSERT_OK(session.Run({R}, &tf_outputs));

  // Assert nGraph and TF outputs are the same

  ASSERT_EQ(tf_outputs.size(), ngraph_outputs.size());
  DummyAssertTensorEquals(tf_outputs[0], *ngraph_outputs[0]);
}

TEST_F(BuilderTest, TFExec) {
  // Create a tf graph
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  DummyAssignInputValues(A, 2.0f);
  DummyAssignInputValues(B, 2.0f);

  auto R = ops::Add(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  // DummyActivateNGraph();
  DummyActivateNGraph();
  ClientSession session(root);
  vector<Tensor> tf_outputs;
  // Run and fetch v
  ASSERT_OK(session.Run({R}, &tf_outputs));
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
