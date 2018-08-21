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

#include "tensorflow/core/framework/tensor.h"
#include "TestUtilities.h"
#include "ngraph_builder.h"
#include "ngraph/ngraph.hpp"
#include "tf_graph_writer.h"
#include "ngraph_utils.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

class BuilderTest : public ::testing::Test {
    public:
    void ComputeOnNGraph(Graph& graph, string test_op_type, vector<Tensor*> tf_inputs,
                      vector<DataType>& output_datatypes, vector<Tensor*>& ngraph_outputs);
    // Compute the tfGraph on nGraph
//          graph        : Tf Graph to be computed
//                         Must have only these nodes
//                          1. n number of "Const" nodes for inputs
//                          2. 1 node of type test_op_type 
//      test_op_type     : type of the test op ("Add", "ReluGrad")
//      output_datatypes : vector of expected TF datatypes of the outputs
//      ngraph_outputs   : vector of computed nGraph outputs as TF Tensors
    void ComputeOnNGraph(Graph& graph, string test_op_type,
                      vector<DataType>& output_datatypes, vector<Tensor*>& ngraph_outputs);
};

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif //NGRAPH_TF_BRIDGE_BUILDERTEST_H_