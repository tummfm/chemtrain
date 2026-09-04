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

#include "connector/model_shape.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

#include "absl/status/statusor.h"
#include "xla/literal.h"

namespace jcn {

GraphBuilder::GraphBuilder(std::vector<std::string> statistics,
                           bool include_pair_type)
    : statistics_keys(statistics), include_pair_type(include_pair_type) {}

// Sparse graphs size their pruned edge mask independently of the adapter's
// raw neighbor storage. Statistics returned by the compiled model determine
// whether that model-owned mask must grow.

void SimpleSparseNeighborList::initialize(std::vector<float> multipliers) {
  edge_multiplier = multipliers[0];
}

namespace {

constexpr char kValidEdgesStatistic[] = "max_neighbors";
constexpr char kOverlongStatistic[] = "overlong";

xla::PjRtBuffer* RequireStatistic(
    const std::map<std::string, std::unique_ptr<xla::PjRtBuffer>>& statistics,
    const std::string& key) {
  auto it = statistics.find(key);
  if (it == statistics.end() || it->second == nullptr) {
    throw std::runtime_error("Model statistics are missing required key '" +
                             key + "'");
  }
  return it->second.get();
}

int64_t CheckedProduct(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0) return 1;
  if (lhs > std::numeric_limits<int64_t>::max() / rhs) {
    return std::numeric_limits<int64_t>::max();
  }
  return lhs * rhs;
}

int64_t GrownCapacity(int64_t observed, double multiplier) {
  // Keep capacity at least one and never below the observed requirement.
  observed = std::max<int64_t>(observed, 1);
  return std::max<int64_t>(
      observed, static_cast<int64_t>(std::ceil(observed * multiplier)));
}

bool HalfPaddingConsumed(int64_t capacity, int64_t observed,
                         double multiplier) {
  // Recompile once the observed size has consumed half the configured padding.
  return static_cast<long double>(observed) * 2.0L * multiplier >=
         static_cast<long double>(capacity) * (multiplier + 1.0L);
}

}  // namespace

GraphInputSpec SimpleSparseNeighborList::input_spec(
    const jcn_requested_capacities& engine_capacities) const {
  const int64_t raw_edges = std::max<int64_t>(engine_capacities.raw_edges, 1);
  const int64_t max_pruned_edges =
      std::max<int64_t>(CheckedProduct(2, raw_edges), 1);
  const int64_t valid_edges =
      std::min(std::max<int64_t>(n_valid_edges, 1), max_pruned_edges);

  std::vector<GraphInputDescriptor> inputs = {
      {{raw_edges}, xla::S32, GraphInputKind::BUFFER},
      {{raw_edges}, xla::S32, GraphInputKind::BUFFER},
  };
  if (include_pair_type) {
    inputs.push_back({{raw_edges}, xla::S32, GraphInputKind::BUFFER});
  }
  inputs.push_back({{valid_edges}, xla::PRED, GraphInputKind::ABSTRACT});
  return GraphInputSpec{std::move(inputs)};
}

bool SimpleSparseNeighborList::evaluate_statistics(
    std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
    bool allow_preemptive_resize,
    const jcn_requested_capacities& engine_capacities) {
  // Shape changes are recorded locally after an execution. The embedding
  // runtime decides when every participant retries the same stage.
  absl::StatusOr<std::shared_ptr<xla::Literal>> valid_edges =
      RequireStatistic(statistics, kValidEdgesStatistic)->ToLiteralSync();
  absl::StatusOr<std::shared_ptr<xla::Literal>> overlong =
      RequireStatistic(statistics, kOverlongStatistic)->ToLiteralSync();

  if (!valid_edges.ok() || !overlong.ok()) {
    const absl::Status& status =
        !valid_edges.ok() ? valid_edges.status() : overlong.status();
    throw std::runtime_error("Failed to read sparse graph statistics: " +
                             status.ToString());
  }

  if (valid_edges.value() == nullptr || overlong.value() == nullptr) {
    throw std::runtime_error(
        "Sparse graph statistics returned a null literal.");
  }

  const int64_t required_valid_edges =
      valid_edges.value()->data<int>().data()[0];
  const int64_t raw_edges = std::max<int64_t>(engine_capacities.raw_edges, 1);
  const int64_t max_pruned_edges =
      std::max<int64_t>(CheckedProduct(2, raw_edges), 1);
  const int64_t current_valid_edges =
      std::min(std::max<int64_t>(n_valid_edges, 1), max_pruned_edges);

  const bool buffer_overflow =
      required_valid_edges > current_valid_edges;
  const bool buffer_filled =
      !buffer_overflow && allow_preemptive_resize &&
      current_valid_edges < max_pruned_edges &&
      HalfPaddingConsumed(current_valid_edges, required_valid_edges,
                          edge_multiplier);

  if (buffer_filled) {
    logger.log(
        LogLevel::INFO,
        "SimpleSparseNeighborList: Recompile edge buffer after consuming "
        "half of its padding.");
  }

  if (buffer_overflow || buffer_filled) {
    logger.log(LogLevel::INFO,
               "SimpleSparseNeighborList: Increasing valid edges from " +
                   std::to_string(current_valid_edges) + " to " +
                   std::to_string(required_valid_edges));
    n_valid_edges =
        std::min(GrownCapacity(required_valid_edges, edge_multiplier),
                 max_pruned_edges);
    return false;
  }

  return true;
}

// Dense graphs use separate model-owned capacities for pruned edges and
// triplets. The adapter's atom and neighbor capacities remain upper bounds.

void SimpleDenseNeighborList::initialize(std::vector<float> multipliers) {
  buffer_multiplier = multipliers[0];
}

GraphInputSpec SimpleDenseNeighborList::input_spec(
    const jcn_requested_capacities& engine_capacities) const {
  const int64_t max_atoms = std::max<int64_t>(engine_capacities.max_atoms, 1);
  const int64_t max_neighbors =
      std::max<int64_t>(engine_capacities.max_neighbors_per_atom, 1);
  const int64_t max_pruned_edges =
      std::max<int64_t>(CheckedProduct(max_atoms, max_neighbors), 1);
  const int64_t valid_edges =
      std::min(std::max<int64_t>(n_valid_edges_, 1), max_pruned_edges);
  const int64_t max_pruned_triplets =
      std::max<int64_t>(CheckedProduct(valid_edges, max_neighbors), 1);
  const int64_t valid_triplets =
      std::min(std::max<int64_t>(n_valid_triplets_, 1), max_pruned_triplets);

  std::vector<GraphInputDescriptor> inputs = {
      {{max_atoms, max_neighbors}, xla::S32, GraphInputKind::BUFFER},
  };
  if (include_pair_type) {
    inputs.push_back(
        {{max_atoms, max_neighbors}, xla::S32, GraphInputKind::BUFFER});
  }
  inputs.push_back({{valid_edges}, xla::PRED, GraphInputKind::ABSTRACT});
  inputs.push_back({{valid_triplets}, xla::PRED, GraphInputKind::ABSTRACT});
  return GraphInputSpec{std::move(inputs)};
}

bool SimpleDenseNeighborList::evaluate_statistics(
    std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
    bool allow_preemptive_resize,
    const jcn_requested_capacities& engine_capacities) {
  // Dense models can prune both edges and triplets. The concrete graph class
  // owns those model capacities. The adapter only provides atom and raw
  // dense-neighbor capacities as upper bounds.
  absl::StatusOr<std::shared_ptr<xla::Literal>> valid_edges =
      RequireStatistic(statistics, kValidEdgesStatistic)->ToLiteralSync();
  absl::StatusOr<std::shared_ptr<xla::Literal>> valid_triplets =
      RequireStatistic(statistics, kOverlongStatistic)->ToLiteralSync();

  if (!valid_edges.ok() || !valid_triplets.ok()) {
    const absl::Status& status =
        !valid_edges.ok() ? valid_edges.status() : valid_triplets.status();
    throw std::runtime_error("Failed to read dense graph statistics: " +
                             status.ToString());
  }
  if (valid_edges.value() == nullptr || valid_triplets.value() == nullptr) {
    throw std::runtime_error("Dense graph statistics returned a null literal.");
  }

  const int64_t required_valid_edges =
      valid_edges.value()->data<int>().data()[0];
  const int64_t required_valid_triplets =
      valid_triplets.value()->data<int>().data()[0];
  const int64_t max_atoms = std::max<int64_t>(engine_capacities.max_atoms, 1);
  const int64_t max_neighbors =
      std::max<int64_t>(engine_capacities.max_neighbors_per_atom, 1);
  const int64_t max_pruned_edges =
      std::max<int64_t>(CheckedProduct(max_atoms, max_neighbors), 1);
  const int64_t current_valid_edges =
      std::min(std::max<int64_t>(n_valid_edges_, 1), max_pruned_edges);
  const int64_t max_pruned_triplets =
      std::max<int64_t>(CheckedProduct(current_valid_edges, max_neighbors), 1);
  const int64_t current_valid_triplets =
      std::min(std::max<int64_t>(n_valid_triplets_, 1), max_pruned_triplets);

  const bool edge_buffer_overflow =
      required_valid_edges > current_valid_edges;
  const bool triplet_buffer_overflow =
      required_valid_triplets > current_valid_triplets;
  const bool edge_buffer_filled =
      !edge_buffer_overflow && allow_preemptive_resize &&
      current_valid_edges < max_pruned_edges &&
      HalfPaddingConsumed(current_valid_edges, required_valid_edges,
                          buffer_multiplier);
  const bool triplet_buffer_filled =
      !triplet_buffer_overflow && allow_preemptive_resize &&
      current_valid_triplets < max_pruned_triplets &&
      HalfPaddingConsumed(current_valid_triplets, required_valid_triplets,
                          buffer_multiplier * buffer_multiplier);

  if (edge_buffer_filled || triplet_buffer_filled) {
    logger.log(LogLevel::INFO,
               "Recompiled buffer after consuming half of its padding.");
  }

  bool success = true;
  if (edge_buffer_overflow || edge_buffer_filled) {
    logger.log(LogLevel::INFO,
               "SimpleDenseNeighborList: Increasing valid edges");
    n_valid_edges_ =
        std::min(GrownCapacity(required_valid_edges, buffer_multiplier),
                 max_pruned_edges);
    success = false;
  }
  if (triplet_buffer_overflow || triplet_buffer_filled) {
    logger.log(LogLevel::INFO,
               "SimpleDenseNeighborList: Increasing valid triplets.");
    const int64_t triplet_limit = std::max<int64_t>(
        CheckedProduct(
            std::min(std::max<int64_t>(n_valid_edges_, 1), max_pruned_edges),
            max_neighbors),
        1);
    n_valid_triplets_ =
        std::min(GrownCapacity(required_valid_triplets,
                               buffer_multiplier * buffer_multiplier),
                 triplet_limit);
    success = false;
  }

  return success;
}

}  // namespace jcn
