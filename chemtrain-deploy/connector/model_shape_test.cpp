/*
Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "connector/model_shape.h"

#include "gtest/gtest.h"

namespace jcn {
namespace {

TEST(ModelShapeTest, SparseCapacityCarrierIsAbstract) {
  SimpleSparseNeighborList graph({}, false);
  graph.initialize({1.2f});
  const GraphInputSpec spec = graph.input_spec({8, 5, 0});

  ASSERT_EQ(spec.inputs.size(), 3);
  EXPECT_EQ(spec.inputs[0].kind, GraphInputKind::BUFFER);
  EXPECT_EQ(spec.inputs[1].kind, GraphInputKind::BUFFER);
  EXPECT_EQ(spec.inputs[2].kind, GraphInputKind::ABSTRACT);
  EXPECT_EQ(spec.inputs[2].type, xla::PRED);
}

TEST(ModelShapeTest, DenseCapacityCarriersAreAbstract) {
  SimpleDenseNeighborList graph({}, true);
  graph.initialize({1.2f});
  const GraphInputSpec spec = graph.input_spec({8, 0, 5});

  ASSERT_EQ(spec.inputs.size(), 4);
  EXPECT_EQ(spec.inputs[0].kind, GraphInputKind::BUFFER);
  EXPECT_EQ(spec.inputs[1].kind, GraphInputKind::BUFFER);
  EXPECT_EQ(spec.inputs[2].kind, GraphInputKind::ABSTRACT);
  EXPECT_EQ(spec.inputs[3].kind, GraphInputKind::ABSTRACT);
  EXPECT_EQ(spec.inputs[2].type, xla::PRED);
  EXPECT_EQ(spec.inputs[3].type, xla::PRED);
}

}  // namespace
}  // namespace jcn
