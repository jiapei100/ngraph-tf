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

#include "TestCaseBuilderSimple.h"
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

TEST(TestSimple, SimpleDEAdd) {
  // Create a tf graph
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  DummyAssignInputValues(A, 2.1f);
  DummyAssignInputValues(B, 4.1f);

  auto R = ops::Add(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  // BuilderTestSimple buildertest(root, "Add", output_datatypes);

  // buildertest.ExecuteOnNgraph();

  /*
  // Get the graph from the declared scope
  Graph tf_graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&tf_graph));

  // For debug
  GraphToPbTextFile(&tf_graph, "tf_graph.pbtxt");

  // Compute the graph on nGraph and get output as TF Tensors
  vector<Tensor> ngraph_outputs;
  ComputeOnNGraph(tf_graph, "Add", output_datatypes, ngraph_outputs);
  NGRAPH_VLOG(5) << " printing ops " << ngraph_outputs.size();

  // Run on TF
  DummyDeactivateNGraph();
  Graph tf_graph_chk(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&tf_graph_chk));

  GraphToPbTextFile(&tf_graph_chk, "tf_graph_again.pbtxt");

  ClientSession session(root);
  vector<Tensor> tf_outputs;
  // Run and fetch v
  ASSERT_OK(session.Run({R}, &tf_outputs));

  // Assert nGraph and TF outputs are the same

  ASSERT_EQ(tf_outputs.size(), ngraph_outputs.size());
  DummyAssertTensorEquals(tf_outputs[0], ngraph_outputs[0]);

  */
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
