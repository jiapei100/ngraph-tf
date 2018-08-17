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
#include "gtest/gtest.h"

#include "ngraph_builder.h"
#include "ngraph_utils.h"
#include "tf_graph_writer.h"

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

// some utility functions copied from tf_exec.cpp
void DummyActivateNGraph() {
  setenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1", 1);
  unsetenv("NGRAPH_TF_DISABLE");
}

void DummyDeactivateNGraph() {
  unsetenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS");
  setenv("NGRAPH_TF_DISABLE", "1", 1);
}

void DummyAssignInputIntValues(Tensor& A, int maxval) {
  auto A_flat = A.flat<int>();
  auto A_flat_data = A_flat.data();
  int counter = 0;
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = counter++;
    if (counter == maxval) {
      counter = 0;
    }
  }
}

void DummyAssignInputValues(Tensor& A, float x) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x;
  }
}

void DummyPrintTensor(const Tensor& T1) {
  LOG(INFO) << "print tensor values" << T1.DebugString();
}

// Rewrite TF Graph to insert _args and _retValue
// make sure there is only one node with inputs, which should be the op
// (test_op) being tested all other nodes must be feeding into this test_op
// Could get the op name as string

void RewriteForNGraph(Graph& graph, string test_op_name,
                      vector<DataType>& output_datatypes) {
  // const DataTypeVector& input_types() const;
  Node* test_op;

  bool found_test_op = false;
  for (Node* node : graph.nodes()) {
    LOG(INFO) << "op " << node->type_string();
    if (node->IsSource() || node->IsSink()) {
      continue;
    } else if (node->type_string() == test_op_name) {
      // only one node of type test_op
      ASSERT_FALSE(found_test_op);
      found_test_op = true;
      test_op = node;
    } else {
      ASSERT_TRUE(node->type_string() == "Const");
    }
  }

  // find input shapes of the node
  // int32 num_inputs() const;
  int number_of_inputs = test_op->num_inputs();
  vector<TensorShape> input_shapes(number_of_inputs);
  vector<Tensor*> input_tensors(number_of_inputs);
  vector<Node*> input_node(number_of_inputs);

  for (int i = 0; i < number_of_inputs; i++) {
    Node* ip;
    ASSERT_EQ(Status::OK(), test_op->input_node(i, &ip));
    NGRAPH_VLOG(5) << "input " << i;
    NGRAPH_VLOG(5) << "input name " << ip->name() << " type "
                   << ip->type_string();
    Tensor ip_tensor;
    ASSERT_EQ(Status::OK(), GetNodeAttr(ip->attrs(), "value", &ip_tensor));
    DummyPrintTensor(ip_tensor);
    NGRAPH_VLOG(5) << "input type " << i <<" :" << ip_tensor.dtype(); 
    input_shapes[i] = ip_tensor.shape();
    input_tensors[i] = &ip_tensor;
    input_node[i] = ip;
  }

  // replace inputs with
  for(int i=0; i<number_of_inputs; i++){
    // Add new _arg s node
    string new_node_name = "arg_" + std::to_string(i);  // or ngraph_input_
    NodeDef* new_arg_node_def = new NodeDef();
    new_arg_node_def->set_name(new_node_name);
    new_arg_node_def->set_op("_Arg");
    //new_arg_node_def->SetAttr("T", input_tensors[i]->dtype());
    //new_arg_node_def->SetAttr("index", i);
    NGRAPH_VLOG(5) << "arg type " << i <<" : " << input_tensors[i]->dtype(); 
    NGRAPH_VLOG(5) << "shape type " << i <<" : " << input_tensors[i]->shape();
    NGRAPH_VLOG(5) << "arg val " << i <<" : " << input_tensors[i]->DebugString();  
    SetAttrValue(input_tensors[i]->dtype(), &((*(new_arg_node_def->mutable_attr()))["T"]));
    SetAttrValue(i,
                  &((*(new_arg_node_def->mutable_attr()))["index"]));

    Status status;
    Node* arg_node = graph.AddNode(*new_arg_node_def, &status);
    ASSERT_EQ(Status::OK(), status);
    
    // Removes a node from this graph, including all edges from or to it.
    // *node should not be accessed after calling this function.
    // REQUIRES: node->IsOp()
    graph.RemoveNode(input_node[i]);

    // Adds an edge that connects the xth output of `source` to the yth input of
    // `dest` and returns it. Does not update dest's NodeDef.
    graph.AddEdge(arg_node, 0, test_op, i);
  }

  int number_of_outputs = output_datatypes.size();
  for (int i = 0; i < number_of_outputs; i++) {
    // Add new _args node
    string new_node_name = "retval_" + std::to_string(i);  // or ngraph_retval_
    NodeDef* new_ret_node_def = new NodeDef();
    new_ret_node_def->set_name(new_node_name);
    new_ret_node_def->set_op("_Retval");
    SetAttrValue(output_datatypes[i], &((*(new_ret_node_def->mutable_attr()))["T"]));
    SetAttrValue(i,
                  &((*(new_ret_node_def->mutable_attr()))["index"]));

    Status status;
    Node* ret_node = graph.AddNode(*new_ret_node_def, &status);
    ASSERT_EQ(Status::OK(), status);

    graph.AddEdge(test_op, i, ret_node, 0);
  }

  GraphToPbTextFile(&graph, "rewrite_ngraph_pbtxtfile.pbtxt");

  //Create nGraph function
  shared_ptr<ng::Function> ng_function;
  ASSERT_EQ(Status::OK(), ngraph_bridge::Builder::TranslateGraph(input_shapes, &graph, ng_function));

  //Create nGraph backend
  // Create the nGraph backend
  auto backend = ng::runtime::Backend::create("CPU");

  // Allocate tensors for inputs
  vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_ip_tensors(number_of_inputs);
  vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_op_tensors(number_of_outputs);

  for(int i=0; i<number_of_inputs; i++){
    ng::Shape ng_shape=ng_function->get_output_shape(i);
    ng::element::Type ng_et;
    ASSERT_EQ(Status::OK(),TFDataTypeToNGraphElementType(input_tensors[i]->dtype(), &ng_et));
    NGRAPH_VLOG(5) << " before casting ";
    ng_ip_tensors[i] = backend->create_tensor(ng_et, ng_shape, (void*) (input_tensors[i]));
  }

  for(int i=0; i<number_of_outputs; i++){
    ng::Shape ng_shape;
    ASSERT_EQ(Status::OK(),TFTensorShapeToNGraphShape(input_shapes[i], &ng_shape));
    ng::element::Type ng_et;
    ASSERT_EQ(Status::OK(),TFDataTypeToNGraphElementType(output_datatypes[i], &ng_et));
    ng_op_tensors[i] = backend->create_tensor(ng_et, ng_shape);
  }

  // Execute the nGraph function.
  cout << "Calling nGraph function\n";
  backend->call(ng_function, ng_op_tensors, ng_ip_tensors);

  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor(cout, ng_function->get_output_op(i)->get_name(), ng_op_tensors[i]);
    cout << endl;
  }
/*
  ng::Shape ng_shape_x(x.shape().dims());
  for (int i = 0; i < x.shape().dims(); ++i) {
    ng_shape_x[i] = x.shape().dim_size(i);
  }

  ng::Shape ng_shape_y(y.shape().dims());
  for (int i = 0; i < y.shape().dims(); ++i) {
    ng_shape_y[i] = y.shape().dim_size(i);
  }

  auto t_x = backend->create_tensor(ng::element::f32, ng_shape_x);
  float v_x[2][3] = {{1, 1, 1}, {1, 1, 1}};
  t_x->write(&v_x, 0, sizeof(v_x));

  auto t_y = backend->create_tensor(ng::element::f32, ng_shape_y);
  t_y->write(&v_x, 0, sizeof(v_x));

  // Allocate tensor for the result(s)
  vector<shared_ptr<ng::runtime::TensorView>> outputs;
  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    auto shape = ng_function->get_output_shape(i);
    auto elem_type = ng_function->get_output_element_type(i);
    auto t_result = backend->create_tensor(elem_type, shape);
    outputs.push_back(t_result);
  }

  // Execute the nGraph function.
  cout << "Calling nGraph function\n";
  backend->call(ng_function, outputs, {t_x, t_y});

  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor(cout, ng_function->get_output_op(i)->get_name(), outputs[i]);
    cout << endl;
  }
  */
}

TEST(TestBuilder, DirectExecution) {
  // Create a tf graph
  Scope root = Scope::NewRootScope();
  int batch_size = 2;
  int num_classes = 2;
  Tensor A(DT_FLOAT, TensorShape({batch_size, num_classes}));
  Tensor B(DT_FLOAT, TensorShape({batch_size, num_classes}));
  //Tensor features(DT_FLOAT, TensorShape({batch_size, num_classes}));
  //Tensor labels(DT_INT32, TensorShape({batch_size}));
  //DummyAssignInputValues(features, 1.0f);
  //DummyAssignInputIntValues(labels, num_classes);
  DummyAssignInputValues(A, 1.0f);
  DummyAssignInputValues(B, 1.0f);
  auto R =ops::Add(root, A, B);
  //auto R = ops::SparseSoftmaxCrossEntropyWithLogits(root, features, labels);
  //vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT};
  vector<DataType> output_datatypes = {DT_FLOAT};
  // convert it to graph def
  // GraphDef def;
  Graph tf_graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&tf_graph));

  GraphToPbTextFile(&tf_graph, "tf_ngraph_pbtxtfile.pbtxt");

  //RewriteForNGraph(tf_graph, "SparseSoftmaxCrossEntropyWithLogits",
                   //output_datatypes);
  RewriteForNGraph(tf_graph, "Add",output_datatypes);
}
//

/*
TEST(TestBuilder, DirectExecution) {
  // Create a tf graph
  Scope root = Scope::NewRootScope();

  // TF Default formats
  // Input NHWC :[batch, in_height, in_width, in_channels]
  std::vector<int64> input_size_NHWC = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  std::vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  // Out_delta :[batch, out_height, out_width, out_channels]
  std::vector<int64> output_del_size_valid = {1, 3, 2, 2};
  std::vector<int64> output_del_size_same = {1, 4, 3, 2};
  Tensor output_delta_valid(DT_FLOAT, TensorShape(output_del_size_valid));
  Tensor output_delta_same(DT_FLOAT, TensorShape(output_del_size_same));
  DummyAssignInputValues(output_delta_valid, -1.1f);
  DummyAssignInputValues(output_delta_same, -1.1f);

  std::map<std::string, Tensor*> out_delta_size_map = {
      {"VALID", &output_delta_valid}, {"SAME", &output_delta_same}};

  std::vector<int> stride = {1, 2, 2, 1};
  Tensor input_data(DT_FLOAT, TensorShape(input_size_NHWC));
  DummyAssignInputValues(input_data, -1.1f);

  auto filter_sizes = ops::Const(root, {3, 3, 2, 2});

  // TEST NHWC : default data format
  // for (auto map_iterator : out_delta_size_map) {
  auto padding_type = "VALID";
  auto output_delta = *(out_delta_size_map[padding_type]);

  auto r = ops::Conv2DBackpropFilter(root, input_data, filter_sizes,
                                     output_delta, stride, padding_type);

  // AssertTensorEquals(outputs_ngraph[0], outputs_cpu[0]);
  // break;
  //}
  // convert it to graph def
  GraphDef def;
  Graph tf_graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&tf_graph));

  GraphToPbTextFile(&tf_graph, "tf_ngraph_pbtxtfile.pbtxt");

  // Compute via nGraph
  // Input : Tf GraphDef
  // Ouput : tf Tensor

  DummyActivateNGraph();
  ClientSession session(root);
  // Run and fetch v
  std::vector<Tensor> outputs;
  ASSERT_OK(session.Run({r}, &outputs));

  // Compute on TF
  // std::vector<Tensor> outputs;
  // ClientSession session(root);
  // Run and fetch v
  // ASSERT_OK(session.Run({v}, &outputs));

  // Compare the outputs
}

*/

}  // namespace ngraph_bridge

}  // namespace tensorflow
