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

#include "TestUtilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {
// some utility functions copied from tf_exec.cpp
void DummyActivateNGraph() {
  setenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1", 1);
  unsetenv("NGRAPH_TF_DISABLE");
}

void DummyDeactivateNGraph() {
  unsetenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS");
  setenv("NGRAPH_TF_DISABLE", "1", 1);
}

void DummyAssertTensorEquals(Tensor& T1, Tensor& T2) {
  auto T_size = T1.flat<float>().size();
  auto T1_data = T1.flat<float>().data();
  auto T2_data = T2.flat<float>().data();
  for (int k = 0; k < T_size; k++) {
    auto a = T1_data[k];
    auto b = T2_data[k];
    EXPECT_FLOAT_EQ(a, b);
  }
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

Status NGraphShapeToTFShape(ng::Shape& ng_op_shape, TensorShape *tf_shape){
  vector<int64> dims;
  for(auto dim : ng_op_shape){
    dims.push_back(dim);
  }
  TensorShape tshape(dims);
  tf_shape=&tshape;
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow