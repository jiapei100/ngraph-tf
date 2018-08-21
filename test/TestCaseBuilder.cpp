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

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {
// Replace the input nodes to test_op with _Arg
// Add _Retval to test_op

// Compute the tfGraph on nGraph
//          graph        : Tf Graph to be computed
//                         Must have only these nodes
//                          1. n number of "Const" nodes for inputs
//                          2. 1 node of type test_op_type
//      test_op_type     : type of the test op ("Add", "ReluGrad")
//      output_datatypes : vector of expected TF datatypes of the outputs
//      ngraph_outputs   : vector of computed nGraph outputs as TF Tensors
void BuilderTest::ComputeOnNGraph(Graph& graph, string test_op_type,
                                  vector<DataType>& output_datatypes,
                                  vector<Tensor*>& ngraph_outputs) {
  // Check that the graph has n "Const" nodes and 1 test_op_type node
  Node* test_op;
  bool found_test_op = false;
  for (Node* node : graph.nodes()) {
    if (node->IsSource() || node->IsSink()) {
      continue;
    } else if (node->type_string() == test_op_type) {
      // only one node of type test_op
      ASSERT_FALSE(found_test_op);
      found_test_op = true;
      test_op = node;
    } else {
      ASSERT_TRUE(node->type_string() == "Const");
    }
  }

  NGRAPH_VLOG(5) << "Check graph complete";
  for (const Edge* e : graph.edges()) {
    NGRAPH_VLOG(5) << "Edge between, Src: " << e->src()->name()
                   << " ,Dst: " << e->dst()->name();
  }

  // Replace the input nodes ("Const") with _Arg nodes
  // Get Tensor input shapes and values of the const nodes
  int number_of_inputs = test_op->num_inputs();
  vector<TensorShape> input_shapes(number_of_inputs);
  vector<Tensor*> input_tensors(number_of_inputs);
  vector<Node*> input_node(number_of_inputs);

  Tensor ip_tensor;
  for (int i = 0; i < number_of_inputs; i++) {
    Node* ip;
    ASSERT_EQ(Status::OK(), test_op->input_node(i, &ip));
    ASSERT_EQ(Status::OK(), GetNodeAttr(ip->attrs(), "value", &ip_tensor));
    input_shapes[i] = ip_tensor.shape();
    input_tensors[i] = &ip_tensor;
    input_node[i] = ip;
  }

  NGRAPH_VLOG(5) << "Got input nodes and tensors";

  // replace inputs with
  for (int i = 0; i < number_of_inputs; i++) {
    Node* ip_node = input_node[i];
    // For all the input edges (should be 1) to ip_node get the source node and
    // the source_output_index feeding to this node
    // // (TO DO : ) ADD ASSERT
    vector<std::pair<Node*, int>> src_nodes_metadata;

    for (const Edge* e : ip_node->in_edges()) {
      src_nodes_metadata.push_back({e->src(), e->src_output()});
    }

    // Define new _arg node, make function
    string new_node_name = "arg_" + std::to_string(i);  // or ngraph_input_
    NodeDef* new_arg_node_def = new NodeDef();
    new_arg_node_def->set_name(new_node_name);
    new_arg_node_def->set_op("_Arg");
    SetAttrValue(input_tensors[i]->dtype(),
                 &((*(new_arg_node_def->mutable_attr()))["T"]));
    SetAttrValue(i, &((*(new_arg_node_def->mutable_attr()))["index"]));

    // Add node to graph
    Status status;
    Node* arg_node = graph.AddNode(*new_arg_node_def, &status);
    ASSERT_EQ(Status::OK(), status);

    // Removes a node from this graph, including all edges from or to it.
    // *node should not be accessed after calling this function.
    // REQUIRES: node->IsOp()
    graph.RemoveNode(input_node[i]);

    for (int j = 0; j < src_nodes_metadata.size(); j++) {
      graph.AddEdge(src_nodes_metadata[j].first, src_nodes_metadata[j].second,
                    arg_node, 0);
    }
    // Adds an edge that connects the xth output of `source` to the yth input
    // of `dest` and returns it. Does not update dest's NodeDef.
    graph.AddEdge(arg_node, 0, test_op, i);
  }

  NGRAPH_VLOG(5) << "Replaced input nodes with _Arg";

  int number_of_outputs = output_datatypes.size();
  // For all the output edges from test_op (there should be only one)
  // get the dest node and the
  // destination_input_index
  // (TO DO : ) ADD ASSERT
  vector<std::pair<Node*, int>> dest_nodes_metadata;

  for (const Edge* e : test_op->out_edges()) {
    if (e == nullptr) {
      NGRAPH_VLOG(5) << "Found null edge, ";
      continue;
    }
    NGRAPH_VLOG(5) << "Found out edge, ";
    dest_nodes_metadata.push_back({e->dst(), e->dst_input()});
  }

  NGRAPH_VLOG(5) << "Got out edges _Arg";

  for (int i = 0; i < number_of_outputs; i++) {
    // Add new _args node
    string new_node_name = "retval_" + std::to_string(i);  // or ngraph_retval_
    NodeDef* new_ret_node_def = new NodeDef();
    new_ret_node_def->set_name(new_node_name);
    new_ret_node_def->set_op("_Retval");
    SetAttrValue(output_datatypes[i],
                 &((*(new_ret_node_def->mutable_attr()))["T"]));
    SetAttrValue(i, &((*(new_ret_node_def->mutable_attr()))["index"]));

    Status status;
    Node* ret_node = graph.AddNode(*new_ret_node_def, &status);
    ASSERT_EQ(Status::OK(), status);

    for (int j = 0; j < dest_nodes_metadata.size(); j++) {
      graph.AddEdge(ret_node, 0, dest_nodes_metadata[j].first,
                    dest_nodes_metadata[j].second);
    }

    graph.AddEdge(test_op, i, ret_node, 0);
  }

  NGRAPH_VLOG(5) << "Added _Retval nodes ";

  NGRAPH_VLOG(5) << "After rewrite *** ";
   for (const Edge* e : graph.edges()) {
    NGRAPH_VLOG(5) << "Edge between, Src: " << e->src()->name()
                   << " ,Dst: " << e->dst()->name();
  }

  GraphToPbTextFile(&graph, "rewrite_ngraph.pbtxt");
  NGRAPH_VLOG(5) << "num nodes  " << graph.num_nodes();
  // Create nGraph function
  NGRAPH_VLOG(5) << " Create ng function ";
  shared_ptr<ng::Function> ng_function;
  ASSERT_EQ(Status::OK(),
            Builder::TranslateGraph(input_shapes, &graph, ng_function));
  // ng function should get same number of outputs
  ASSERT_EQ(output_datatypes.size(), ng_function->get_output_size());

  // Create nGraph backend
  // Create the nGraph backend
  NGRAPH_VLOG(5) << " Create backend ";
  auto backend = ng::runtime::Backend::create("CPU");

  NGRAPH_VLOG(5) << " backend created ";
  // Allocate tensors for inputs
  vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_ip_tensors(
      number_of_inputs);
  vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_op_tensors(
      number_of_outputs);

  for (int i = 0; i < number_of_inputs; i++) {
    ng::Shape ng_shape;
    ASSERT_EQ(Status::OK(),
              TFTensorShapeToNGraphShape(input_shapes[i], &ng_shape));
    NGRAPH_VLOG(5) << " got op shape " << i;
    ng::element::Type ng_et;
    NGRAPH_VLOG(5) << " for tensor " << input_tensors[i]->DebugString();
    ASSERT_EQ(Status::OK(),
              TFDataTypeToNGraphElementType(input_tensors[i]->dtype(), &ng_et));
    NGRAPH_VLOG(5) << " before casting";
    void* src_ptr = (void*)DMAHelper::base(input_tensors[i]);
    ng_ip_tensors[i] = backend->create_tensor(ng_et, ng_shape, src_ptr);
    // ng_ip_tensors[i]->write(input_tensors[i], 0, sizeof(input_tensors[i]));
  }

  NGRAPH_VLOG(5) << " Creating ng ouptuts ";
  for (int i = 0; i < number_of_outputs; i++) {
    auto ng_op_shape = ng_function->get_output_shape(i);
    auto ng_op_type = ng_function->get_output_element_type(i);

    ng::element::Type ng_et_expected;
    ASSERT_EQ(Status::OK(), TFDataTypeToNGraphElementType(output_datatypes[i],
                                                          &ng_et_expected));

    // Expected element type should match ng_op_type
    // check this comparison/overloades
    // ASSERT_EQ(ng_et_expected, output_datatypes[i]);
    NGRAPH_VLOG(5) << " check shape ";
    TensorShape tf_shape;
    ASSERT_EQ(Status::OK(), NGraphShapeToTFShape(ng_op_shape, &tf_shape));
    Tensor* output_tensor = new Tensor(output_datatypes[i], tf_shape);

    void* dst_ptr = DMAHelper::base(output_tensor);
    NGRAPH_VLOG(5) << " DMA helper ";
    // auto t_result = ng_backend->create_tensor(elem_type, shape, dst_ptr);
    ng_op_tensors[i] = backend->create_tensor(ng_op_type, ng_op_shape, dst_ptr);
    // NGRAPH_VLOG(5) << " backend create ";
    ngraph_outputs.push_back(output_tensor);
  }

  // Executet the nGraph
  NGRAPH_VLOG(5) << " Executing nGraph";
  backend->call(ng_function, ng_op_tensors, ng_ip_tensors);

  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor(cout, ng_function->get_output_op(i)->get_name(),
                 ng_op_tensors[i]);
    cout << endl;
  }

}  // rewrite finish

}  // namespace ngraph_bridge

}  // namespace tensorflow
