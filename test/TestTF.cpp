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

TEST(TestTF, ADD) {
  // Create a tf graph
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  DummyAssignInputValues(A, 2.0f);
  DummyAssignInputValues(B, 2.0f);

  auto R = ops::Subtract(root, A, B);

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