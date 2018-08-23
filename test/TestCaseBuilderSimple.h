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
#ifndef NGRAPH_TF_BRIDGE_BUILDERTEST_H_
#define NGRAPH_TF_BRIDGE_BUILDERTEST_H_

#include "TestUtilities.h"
#include "ngraph/ngraph.hpp"
#include "ngraph_builder.h"
#include "ngraph_utils.h"
#include "tf_graph_writer.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
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

class BuilderTestSimple {
 public:
  // Constructor
  // BuilderTestSimple();

  BuilderTestSimple();

  // Destructor // check private or public
  ~BuilderTestSimple();

  void ComputeOnNGraph(Graph& graph, string test_op_type,
                       vector<Tensor*> tf_inputs,
                       vector<DataType>& output_datatypes,
                       vector<Tensor*>& ngraph_outputs);
  // Compute the tfGraph on nGraph
  //          graph        : Tf Graph to be computed
  //                         Must have only these nodes
  //                          1. n number of "Const" nodes for inputs
  //                          2. 1 node of type test_op_type
  //      test_op_type     : type of the test op ("Add", "ReluGrad")
  //      output_datatypes : vector of expected TF datatypes of the outputs
  //      ngraph_outputs   : vector of computed nGraph outputs as TF Tensors
  void ComputeOnNGraph(Graph& graph, string test_op_type,
                       vector<DataType>& output_datatypes,
                       vector<Tensor>& ngraph_outputs);

  void ExecuteOnNGraph();
  void ExecuteOnTf();
  void CompareNgraphAndTF();

  using NodeMetaData = map<Node*, vector<std::pair<Node*, int>>>;
  using NodeOutEdges = map<Node*, vector<const Edge*>>;

 private:
  Scope tf_scope_;
  string test_op_type_;
  vector<Tensor> tf_inputs_;
  vector<Tensor> ngraph_outputs_;
  vector<DataType> expected_output_datatypes_;

  void GetNodeData(Graph& graph, NodeMetaData& node_inedge_md,
                   NodeMetaData& node_outedge_md, NodeOutEdges& node_outedges);
};

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_BUILDERTEST_H_
