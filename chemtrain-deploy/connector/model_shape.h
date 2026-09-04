/*
Copyright 2025 Multiscale Modeling of Fluid Materials, TU Munich

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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_MODEL_SHAPE_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_MODEL_SHAPE_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "connector/jcn_api.h"
#include "connector/utils.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/xla_data.pb.h"

namespace jcn {

enum class GraphInputKind {
  BUFFER,
  ABSTRACT,
};

struct GraphInputDescriptor {
  std::vector<int64_t> shape;
  xla::PrimitiveType type;
  GraphInputKind kind;

  bool operator==(const GraphInputDescriptor& other) const {
    return shape == other.shape && type == other.type && kind == other.kind;
  }
};

struct GraphInputSpec {
  std::vector<GraphInputDescriptor> inputs;
};

/** Translates adapter capacities into the graph buffers expected by a model.
 *
 * Engine adapters own raw atom and neighbor capacities. Concrete graph
 * builders own the pruned edge or triplet capacities learned from statistics
 * returned by the compiled model.
 */
class GraphBuilder {
 public:
  GraphBuilder(std::vector<std::string> statistics, bool include_pair_type);
  virtual ~GraphBuilder() = default;

  virtual void initialize(std::vector<float> multipliers) = 0;

  virtual GraphInputSpec input_spec(
      const jcn_requested_capacities& engine_capacities) const = 0;

  virtual bool evaluate_statistics(
      std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
      bool allow_preemptive_resize,
      const jcn_requested_capacities& engine_capacities) = 0;

  virtual jcn_requested_capacities requested_capacities(
      jcn_requested_capacities current) const {
    return current;
  }

  std::vector<std::string> statistics_keys;
  bool include_pair_type;

 protected:
  Logger logger = Logger::getlogger();
};

/** Builds inputs for a sparse list with a separately pruned edge mask. */
class SimpleSparseNeighborList : public GraphBuilder {
 public:
  SimpleSparseNeighborList(std::vector<std::string> statistics,
                           bool include_pair_type)
      : GraphBuilder(std::move(statistics), include_pair_type) {}

  void initialize(std::vector<float> multipliers) override;

  GraphInputSpec input_spec(
      const jcn_requested_capacities& engine_capacities) const override;

  bool evaluate_statistics(
      std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
      bool allow_preemptive_resize,
      const jcn_requested_capacities& engine_capacities) override;

 private:
  float edge_multiplier;
  int64_t n_valid_edges = 1;
};

/** Builds inputs for a dense list with pruned edge and triplet masks. */
class SimpleDenseNeighborList : public GraphBuilder {
 public:
  SimpleDenseNeighborList(std::vector<std::string> statistics,
                          bool include_pair_type)
      : GraphBuilder(std::move(statistics), include_pair_type) {}

  void initialize(std::vector<float> multipliers) override;

  GraphInputSpec input_spec(
      const jcn_requested_capacities& engine_capacities) const override;

  bool evaluate_statistics(
      std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
      bool allow_preemptive_resize,
      const jcn_requested_capacities& engine_capacities) override;

 private:
  float buffer_multiplier;
  int64_t n_valid_edges_ = 2;
  int64_t n_valid_triplets_ = 1;
};

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_MODEL_SHAPE_H_
