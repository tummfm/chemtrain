// Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This test intentionally reaches chemtrain-deploy only through jcn_api.h. It
// is a small engine adapter: its host vectors become DLPack buffers, it owns
// concrete dense or sparse neighbor storage, and it retries the public
// capacity protocol. No connector-private header, LAMMPS, MPI, Kokkos, or GPU
// interface is involved.

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "connector/jcn_api.h"
#include "gtest/gtest.h"

namespace {

// Regression configuration.
// ---------------------------------------------------------------------------

constexpr double kOnset = 2.0;
constexpr double kCutoff = 2.5;
constexpr double kTolerance = 2.e-3;

// Test inputs and diagnostics.
// ---------------------------------------------------------------------------

std::string ModelDirectory() {
  const char* directory = std::getenv("JCN_CPU_REGRESSION_MODEL_DIR");
  if (directory == nullptr || *directory == '\0') {
    throw std::runtime_error(
        "JCN_CPU_REGRESSION_MODEL_DIR is unset. Run "
        "regression/run_cpu_regression.py first");
  }
  return directory;
}

std::string ConnectorPath() {
  for (int index = 1; index < ::testing::internal::GetArgvs().size(); ++index) {
    const std::string argument = ::testing::internal::GetArgvs()[index];
    const std::string prefix = "--connector=";
    if (argument.rfind(prefix, 0) == 0) {
      const std::string path = argument.substr(prefix.size());
      if (!path.empty() && path.front() == '/') return path;
      const char* runfiles = std::getenv("TEST_SRCDIR");
      const char* workspace = std::getenv("TEST_WORKSPACE");
      if (runfiles == nullptr || workspace == nullptr) {
        throw std::runtime_error(
            "Bazel did not provide a connector runfiles path");
      }
      return std::string(runfiles) + "/" + workspace + "/" + path;
    }
  }
  throw std::runtime_error("Bazel did not pass --connector to the regression");
}

std::vector<char> ReadFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Could not read model bundle " + path);
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

void ThrowOnStatus(const jcn_status& status, const std::string& operation) {
  if (status.code != JCN_STATUS_OK) {
    throw std::runtime_error(
        operation + ": " +
        (status.message == nullptr ? "unknown error" : status.message));
  }
}

// Host-side DLPack adapter.
// ---------------------------------------------------------------------------

template <typename T>
class HostTensor {
 public:
  HostTensor(std::vector<int64_t> shape, std::vector<T> values)
      : shape_(std::move(shape)), values_(std::move(values)) {
    int64_t elements = 1;
    for (int64_t dimension : shape_) elements *= dimension;
    if (elements != static_cast<int64_t>(values_.size())) {
      throw std::runtime_error("Host tensor data does not match its shape");
    }
  }

  explicit HostTensor(std::vector<int64_t> shape)
      : HostTensor(shape, std::vector<T>(ElementCount(shape))) {}

  std::vector<T>& values() { return values_; }
  const std::vector<T>& values() const { return values_; }

  DLManagedTensor* ReleaseDlpack() {
    auto* tensor = new DLManagedTensor{};
    auto* shape = new int64_t[shape_.size()];
    std::copy(shape_.begin(), shape_.end(), shape);
    tensor->dl_tensor.data = values_.data();
    tensor->dl_tensor.device = {kDLCPU, 0};
    tensor->dl_tensor.ndim = static_cast<int32_t>(shape_.size());
    tensor->dl_tensor.dtype = Dtype();
    tensor->dl_tensor.shape = shape;
    tensor->dl_tensor.strides = nullptr;
    tensor->dl_tensor.byte_offset = 0;
    tensor->deleter = [](DLManagedTensor* released) {
      delete[] released->dl_tensor.shape;
      delete released;
    };
    return tensor;
  }

 private:
  static int64_t ElementCount(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                           std::multiplies<int64_t>());
  }

  static DLDataType Dtype();

  std::vector<int64_t> shape_;
  std::vector<T> values_;
};

template <>
DLDataType HostTensor<float>::Dtype() {
  return {2, 32, 1};
}
template <>
DLDataType HostTensor<int32_t>::Dtype() {
  return {0, 32, 1};
}

struct Result {
  std::vector<float> energy;
  std::vector<float> force;
  std::array<float, 6> virial{};
  jcn_force_result protocol{};
};

// Public connector API harness.
// ---------------------------------------------------------------------------

class JcnSession {
 public:
  JcnSession(const JCN_Api* api, const std::string& model_path) : api_(api) {
    jcn_status status{};
    jcn_client_options client_options{};
    client_options.backend = "cpu";
    client_options.device_ordinal = 0;
    client_ = api_->client_create(&client_options, &status);
    ThrowOnStatus(status, "create CPU client");
    std::vector<char> bundle = ReadFile(model_path);
    model_ = api_->model_load_from_protobuf(bundle.data(), bundle.size(),
                                            &properties_, &status);
    ThrowOnStatus(status, "load model");
    executor_ = api_->executor_create(client_, &status);
    ThrowOnStatus(status, "create executor");
    jcn_engine_abi_options abi{};
    abi.position_dtype = JCN_TENSOR_DTYPE_F32;
    abi.species_dtype = JCN_TENSOR_DTYPE_S32;
    abi.default_output_dtype = JCN_TENSOR_DTYPE_F32;
    abi.communication_dtype = JCN_TENSOR_DTYPE_F32;
    abi.species_encoding = JCN_SPECIES_ZERO_BASED;
    abi.atom_layout = JCN_DENSE_LAYOUT_CONTIGUOUS;
    abi.output_layout = JCN_DENSE_LAYOUT_CONTIGUOUS;
    api_->executor_set_engine_abi_options(executor_, &abi, &status);
    ThrowOnStatus(status, "set CPU engine ABI");
  }

  ~JcnSession() {
    if (executor_ != nullptr) api_->executor_destroy(executor_);
    if (model_ != nullptr) api_->model_destroy(model_);
    if (client_ != nullptr) api_->client_destroy(client_);
  }

  const jcn_model_properties& properties() const { return properties_; }
  jcn_executor* executor() const { return executor_; }

  void SelectModel(bool communication, bool newton = true) {
    jcn_status status{};
    jcn_model_options model_options{};
    model_options.newton_pair = newton ? 1 : 0;
    model_options.use_communication = communication;
    api_->executor_set_model_options(executor_, model_, &model_options,
                                     &properties_, &status);
    ThrowOnStatus(status, "select model variant");
    selected_newton_ = newton;
  }

  void SetCallbacks(const jcn_communication_callbacks& callbacks) {
    jcn_status status{};
    api_->executor_set_communication_callbacks(executor_, &callbacks, &status);
    ThrowOnStatus(status, "install host communication callbacks");
  }

  Result Compute(const std::vector<float>& positions, int32_t owned,
                 int32_t valid, int64_t capacity,
                 const std::vector<std::pair<int32_t, int32_t>>& edges,
                 int64_t raw_edges, int64_t max_neighbors) {
    jcn_status status{};
    jcn_requested_capacities capacities{capacity, raw_edges, max_neighbors};
    api_->executor_set_capacities(executor_, &capacities, &status);
    ThrowOnStatus(status, "set capacities");

    HostTensor<float> position({capacity, 3},
                               PaddedPositions(positions, capacity));
    HostTensor<int32_t> species({capacity}, std::vector<int32_t>(capacity, 0));
    HostTensor<float> energy({capacity});
    HostTensor<float> force({capacity, 3});
    HostTensor<float> virial({6});

    // Import caller-owned host tensors through the public DLPack boundary.
    std::vector<jcn_buffer*> buffers;
    auto import = [&](auto& tensor, jcn_buffer_role role, const char* name) {
      jcn_buffer_import_options options{};
      options.copy_policy = JCN_DLPACK_COPY_IF_NEEDED;
      options.role = role;
      options.debug_name = name;
      options.backend = "cpu";
      options.device_ordinal = 0;
      DLManagedTensor* managed = tensor.ReleaseDlpack();
      jcn_buffer* buffer = api_->buffer_from_dlpack_options(executor_, managed,
                                                            &options, &status);
      if (buffer == nullptr) managed->deleter(managed);
      ThrowOnStatus(status, std::string("import ") + name);
      buffers.push_back(buffer);
      return buffer;
    };

    // Import the atom inputs and built-in outputs before constructing the
    // neighbor-list object that borrows the same buffer ownership boundary.
    jcn_buffer* position_buffer =
        import(position, JCN_BUFFER_ROLE_INPUT, "positions");
    jcn_buffer* species_buffer =
        import(species, JCN_BUFFER_ROLE_INPUT, "species");
    jcn_buffer* energy_buffer = import(energy, JCN_BUFFER_ROLE_OUTPUT, "U");
    jcn_buffer* force_buffer = import(force, JCN_BUFFER_ROLE_OUTPUT, "F");
    jcn_buffer* virial_buffer = import(virial, JCN_BUFFER_ROLE_OUTPUT, "V");

    // Materialize the neighbor payload requested by the exported model.
    // Neighbor DLPack wrappers borrow their backing vectors until the force
    // call returns, so keep the two possible graph columns alive across it.
    std::unique_ptr<HostTensor<int32_t>> first_neighbor_column;
    std::unique_ptr<HostTensor<int32_t>> second_neighbor_column;
    jcn_neighbor_list* neighbors = nullptr;
    if (properties_.neighbor_format == JCN_NEIGHBOR_SIMPLE_SPARSE) {
      first_neighbor_column = std::make_unique<HostTensor<int32_t>>(
          std::vector<int64_t>{raw_edges},
          PaddedEdgeColumn(edges, raw_edges, true, capacity));
      second_neighbor_column = std::make_unique<HostTensor<int32_t>>(
          std::vector<int64_t>{raw_edges},
          PaddedEdgeColumn(edges, raw_edges, false, capacity));
      jcn_sparse_neighbors sparse{
          import(*first_neighbor_column, JCN_BUFFER_ROLE_INPUT, "senders"),
          import(*second_neighbor_column, JCN_BUFFER_ROLE_INPUT, "receivers"),
          nullptr};
      neighbors = api_->neighbor_list_create_simple_sparse(&sparse, &status);
    } else {
      first_neighbor_column = std::make_unique<HostTensor<int32_t>>(
          std::vector<int64_t>{capacity, max_neighbors},
          DenseNeighbors(edges, capacity, max_neighbors));
      jcn_dense_neighbors dense_neighbors{
          import(*first_neighbor_column, JCN_BUFFER_ROLE_INPUT, "neighbors"),
          nullptr};
      neighbors =
          api_->neighbor_list_create_simple_dense(&dense_neighbors, &status);
    }
    ThrowOnStatus(status, "create public neighbor list");

    // Assemble and execute one force request through the versioned JCN API.
    const jcn_named_input inputs[] = {{"species", species_buffer}};
    const jcn_named_output outputs[] = {
        {"U", energy_buffer, owned},
        {"F", force_buffer, selected_newton_ ? valid : owned},
        {"V", virial_buffer, 1},
    };
    jcn_force_request request{};
    request.atoms = {position_buffer, inputs, 1, nullptr, 0, owned, valid};
    request.neighbors = neighbors;
    request.capacities = capacities;
    request.outputs = outputs;
    request.num_outputs = 3;
    request.clear_neighbors = 1;
    request.clear_capacities = 1;
    request.allow_internal_recompile = 1;
    Result result;
    api_->executor_compute_forces(executor_, &request, &result.protocol,
                                  &status);
    if (status.code == JCN_STATUS_OK &&
        result.protocol.code == JCN_COMPUTE_OK) {
      result.energy.assign(energy.values().begin(),
                           energy.values().begin() + owned);
      const int32_t force_rows = selected_newton_ ? valid : owned;
      result.force.assign(force.values().begin(),
                          force.values().begin() + 3 * force_rows);
      std::copy(virial.values().begin(), virial.values().end(),
                result.virial.begin());
    }

    // Release every opaque JCN object while its borrowed host storage lives.
    api_->neighbor_list_destroy(neighbors);
    for (jcn_buffer* buffer : buffers) api_->buffer_destroy(buffer);
    ThrowOnStatus(status, "compute forces through JCN API");
    return result;
  }

  Result ComputeAfterModelCapacityRetries(
      const std::vector<float>& positions, int32_t owned, int32_t valid,
      int64_t capacity,
      const std::vector<std::pair<int32_t, int32_t>>& edges,
      int64_t raw_edges, int64_t max_neighbors) {
    for (int attempt = 0; attempt < 4; ++attempt) {
      Result result = Compute(positions, owned, valid, capacity, edges,
                              raw_edges, max_neighbors);
      if (result.protocol.code != JCN_COMPUTE_NEEDS_CAPACITY_CHANGE) {
        return result;
      }

      const jcn_requested_capacities& required =
          result.protocol.required_capacities;
      if (required.max_atoms > capacity || required.raw_edges > raw_edges ||
          required.max_neighbors_per_atom > max_neighbors) {
        return result;
      }
    }
    throw std::runtime_error(
        "model-owned graph capacities did not settle after four retries");
  }

 private:
  static std::vector<float> PaddedPositions(const std::vector<float>& positions,
                                            int64_t capacity) {
    std::vector<float> padded(3 * capacity, 0.0F);
    std::copy(positions.begin(), positions.end(), padded.begin());
    return padded;
  }
  static std::vector<int32_t> PaddedEdgeColumn(
      const std::vector<std::pair<int32_t, int32_t>>& edges, int64_t raw_edges,
      bool senders, int64_t capacity) {
    std::vector<int32_t> values(raw_edges, static_cast<int32_t>(capacity));
    for (size_t index = 0; index < edges.size() && index < values.size();
         ++index) {
      values[index] = senders ? edges[index].first : edges[index].second;
    }
    return values;
  }
  static std::vector<int32_t> DenseNeighbors(
      const std::vector<std::pair<int32_t, int32_t>>& edges, int64_t capacity,
      int64_t max_neighbors) {
    std::vector<int32_t> values(capacity * max_neighbors,
                                static_cast<int32_t>(capacity));
    std::vector<int64_t> next(capacity, 0);
    for (const auto& [sender, receiver] : edges) {
      if (sender >= 0 && sender < capacity && next[sender] < max_neighbors) {
        values[sender * max_neighbors + next[sender]++] = receiver;
      }
    }
    return values;
  }
  const JCN_Api* api_;
  jcn_client* client_ = nullptr;
  jcn_model* model_ = nullptr;
  jcn_executor* executor_ = nullptr;
  jcn_model_properties properties_{};
  bool selected_newton_ = true;
};

// Lennard-Jones reference calculation.
// ---------------------------------------------------------------------------

double PairEnergy(double squared_distance) {
  if (squared_distance >= kCutoff * kCutoff) return 0.0;
  const double inverse_sixth =
      1.0 / (squared_distance * squared_distance * squared_distance);
  double energy = 4.0 * (inverse_sixth * inverse_sixth - inverse_sixth);
  if (squared_distance <= kOnset * kOnset) return energy;
  const double onset_squared = kOnset * kOnset;
  const double cutoff_squared = kCutoff * kCutoff;
  const double switch_value =
      std::pow(cutoff_squared - squared_distance, 2) *
      (cutoff_squared + 2.0 * squared_distance - 3.0 * onset_squared) /
      std::pow(cutoff_squared - onset_squared, 3);
  return switch_value * energy;
}

double TotalEnergy(const std::vector<float>& positions) {
  double energy = 0.0;
  for (size_t first = 0; first < positions.size() / 3; ++first) {
    for (size_t second = first + 1; second < positions.size() / 3; ++second) {
      double squared = 0.0;
      for (int component = 0; component < 3; ++component) {
        const double delta = positions[3 * first + component] -
                             positions[3 * second + component];
        squared += delta * delta;
      }
      energy += PairEnergy(squared);
    }
  }
  return energy;
}

std::vector<float> ReferenceForces(const std::vector<float>& positions) {
  constexpr double step = 1.e-3;
  std::vector<float> forces(positions.size());
  for (size_t index = 0; index < positions.size(); ++index) {
    auto plus = positions;
    auto minus = positions;
    plus[index] += step;
    minus[index] -= step;
    forces[index] = static_cast<float>(
        -(TotalEnergy(plus) - TotalEnergy(minus)) / (2.0 * step));
  }
  return forces;
}

std::array<float, 6> ReferenceVirial(const std::vector<float>& positions) {
  constexpr double step = 1.e-4;
  std::array<float, 6> virial{};
  for (int component = 0; component < 6; ++component) {
    auto deform = [&](double amount) {
      auto result = positions;
      for (size_t atom = 0; atom < positions.size() / 3; ++atom) {
        const float x = positions[3 * atom];
        const float y = positions[3 * atom + 1];
        const float z = positions[3 * atom + 2];
        if (component == 0) result[3 * atom] = (1.0 + amount) * x;
        if (component == 1) result[3 * atom + 1] = (1.0 + amount) * y;
        if (component == 2) result[3 * atom + 2] = (1.0 + amount) * z;
        if (component == 3) result[3 * atom + 1] = y + amount * x;
        if (component == 4) result[3 * atom + 2] = z + amount * x;
        if (component == 5) result[3 * atom + 2] = z + amount * y;
      }
      return TotalEnergy(result);
    };
    virial[component] =
        static_cast<float>(-(deform(step) - deform(-step)) / (2.0 * step));
  }
  return virial;
}

std::vector<std::pair<int32_t, int32_t>> FullDirectedEdges(int32_t atoms) {
  std::vector<std::pair<int32_t, int32_t>> edges;
  for (int32_t sender = 0; sender < atoms; ++sender) {
    for (int32_t receiver = 0; receiver < atoms; ++receiver) {
      if (sender != receiver) edges.emplace_back(sender, receiver);
    }
  }
  return edges;
}

std::vector<std::pair<int32_t, int32_t>> HalfEdges(int32_t atoms) {
  std::vector<std::pair<int32_t, int32_t>> edges;
  for (int32_t sender = 0; sender < atoms; ++sender) {
    for (int32_t receiver = sender + 1; receiver < atoms; ++receiver) {
      edges.emplace_back(sender, receiver);
    }
  }
  return edges;
}

// Dynamic connector loading.
// ---------------------------------------------------------------------------

class ApiLibrary {
 public:
  ApiLibrary() {
    handle_ = dlopen(ConnectorPath().c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle_ == nullptr) throw std::runtime_error(dlerror());
    auto get_api = reinterpret_cast<const JCN_Api* (*)(uint32_t)>(
        dlsym(handle_, "jcn_get_api"));
    if (get_api == nullptr) {
      throw std::runtime_error("libconnector does not export jcn_get_api");
    }
    api_ = get_api(JCN_API_VERSION);
    if (api_ == nullptr || api_->version != JCN_API_VERSION) {
      throw std::runtime_error("JCN API version mismatch");
    }
  }
  ~ApiLibrary() {
    // PJRT keeps process-global plugin and FFI registrations. Keep their
    // defining connector library loaded until process exit as an embedding
    // simulation engine does.
  }
  const JCN_Api* api() const { return api_; }

 private:
  void* handle_ = nullptr;
  const JCN_Api* api_ = nullptr;
};

// In-process host communication.
// ---------------------------------------------------------------------------

class TwoRankHostCollective {
 public:
  struct RankContext {
    TwoRankHostCollective* collective;
    int rank;
  };

  static int Exchange(void* opaque, void* data, int64_t rows, int64_t columns,
                      jcn_communication_scalar_type type, int32_t reverse,
                      const char** error) {
    auto* context = static_cast<RankContext*>(opaque);
    return context->collective->Exchange(context->rank, data, rows, columns,
                                         type, reverse != 0, error);
  }

  static int Reduce(void* opaque, void* data, int64_t count,
                    jcn_communication_scalar_type type, const char** error) {
    auto* context = static_cast<RankContext*>(opaque);
    return context->collective->Reduce(context->rank, data, count, type, error);
  }

 private:
  enum class Operation { kNone, kGatherForward, kGatherReverse, kReduce };

  int Exchange(int rank, void* opaque_data, int64_t rows, int64_t columns,
               jcn_communication_scalar_type type, bool reverse,
               const char** error) {
    if (type != JCN_COMMUNICATION_F32 || rows < 0 || columns < 1) {
      return Fail("CPU regression expects non-negative f32 gather buffers",
                  error);
    }

    std::unique_lock lock(mutex_);
    const Operation operation =
        reverse ? Operation::kGatherReverse : Operation::kGatherForward;
    if (arrived_ == 0) {
      operation_ = operation;
      columns_ = columns;
    } else if (operation_ != operation || columns_ != columns) {
      return FailLocked("ranks reached different communication calls", error);
    }
    rows_[rank] = rows;
    exchange_data_[rank] = static_cast<float*>(opaque_data);

    if (++arrived_ == 2) {
      // The synthetic topology maps every ghost row to the other rank's
      // first owned row. Multiple ghosts therefore exercise repeated reverse
      // destinations, while different row counts exercise uneven partitions.
      for (int rank_index = 0; rank_index < 2; ++rank_index) {
        const int other = 1 - rank_index;
        for (int64_t row = 1; row < rows_[rank_index]; ++row) {
          for (int64_t column = 0; column < columns_; ++column) {
            const int64_t offset = row * columns_ + column;
            if (reverse) {
              if (rows_[other] > 0) {
                exchange_data_[other][column] +=
                    exchange_data_[rank_index][offset];
              }
            } else {
              exchange_data_[rank_index][offset] =
                  rows_[other] > 0 ? exchange_data_[other][column] : 0.0F;
            }
          }
        }
      }
      return CompleteLocked();
    }

    const int generation = generation_;
    if (!condition_.wait_for(lock, std::chrono::seconds(20), [&] {
          return generation_ != generation || !failure_.empty();
        })) {
      return FailLocked("timed out waiting for the matching gather callback",
                        error);
    }
    if (!failure_.empty()) return FailLocked(failure_, error);
    return 0;
  }

  int Reduce(int rank, void* opaque_data, int64_t count,
             jcn_communication_scalar_type type, const char** error) {
    if (type != JCN_COMMUNICATION_F32 || count < 1) {
      return Fail("CPU regression expects f32 reduction buffers", error);
    }

    std::unique_lock lock(mutex_);
    if (arrived_ == 0) {
      operation_ = Operation::kReduce;
      reduce_count_ = count;
    } else if (operation_ != Operation::kReduce || reduce_count_ != count) {
      return FailLocked("ranks reached different communication calls", error);
    }
    reduce_data_[rank] = static_cast<float*>(opaque_data);

    if (++arrived_ == 2) {
      for (int64_t index = 0; index < count; ++index) {
        const float sum = reduce_data_[0][index] + reduce_data_[1][index];
        reduce_data_[0][index] = sum;
        reduce_data_[1][index] = sum;
      }
      return CompleteLocked();
    }

    const int generation = generation_;
    if (!condition_.wait_for(lock, std::chrono::seconds(20), [&] {
          return generation_ != generation || !failure_.empty();
        })) {
      return FailLocked("timed out waiting for the matching reduce callback",
                        error);
    }
    if (!failure_.empty()) return FailLocked(failure_, error);
    return 0;
  }

  int CompleteLocked() {
    operation_ = Operation::kNone;
    arrived_ = 0;
    ++generation_;
    condition_.notify_all();
    return 0;
  }

  int Fail(const std::string& message, const char** error) {
    std::lock_guard lock(mutex_);
    return FailLocked(message, error);
  }

  int FailLocked(const std::string& message, const char** error) {
    failure_ = message;
    if (error != nullptr) *error = failure_.c_str();
    condition_.notify_all();
    return 1;
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  float* exchange_data_[2]{};
  float* reduce_data_[2]{};
  int64_t rows_[2]{};
  int64_t columns_ = 0;
  int64_t reduce_count_ = 0;
  int arrived_ = 0;
  int generation_ = 0;
  Operation operation_ = Operation::kNone;
  std::string failure_;
};

// Regression scenarios.
// ---------------------------------------------------------------------------

TEST(CpuConnectorRegression, RuntimeReportsJax011XlaRevision) {
  ApiLibrary library;
  jcn_runtime_info runtime{};
  library.api()->get_runtime_info(&runtime);

  ASSERT_NE(runtime.xla_commit, nullptr);
  EXPECT_STREQ(runtime.xla_commit, "131bf41acb4650e4391a640c3f1859c1c86ad74b");
}

TEST(CpuConnectorRegression, SmoothLennardJonesDenseAndSparseMatchReference) {
  ApiLibrary library;
  // The systems cover a pair at the onset and cutoff, a three-particle cluster,
  // and separated two-/three-particle clusters. Padded storage is deliberately
  // larger than the physical systems.
  const std::vector<std::vector<float>> systems = {
      {0.F, 0.F, 0.F, 2.F, 0.F, 0.F},
      {0.F, 0.F, 0.F, 2.5F, 0.F, 0.F},
      {0.F, 0.F, 0.F, 2.6F, 0.F, 0.F},
      {0.F, 0.F, 0.F, 1.2F, 0.F, 0.F, 2.35F, 0.F, 0.F},
      {0.F, 0.F, 0.F, 1.2F, 0.F, 0.F, 5.F, 0.F, 0.F, 6.2F, 0.F, 0.F, 7.35F, 0.F,
       0.F},
  };
  for (const char* format : {"dense", "sparse"}) {
    JcnSession session(library.api(),
                       ModelDirectory() + "/lennard_jones_" + format + ".ptb");
    bool found_virial = false;
    for (int index = 0; index < session.properties().num_outputs; ++index) {
      const jcn_output_descriptor& output = session.properties().outputs[index];
      if (std::string(output.name) != "V") continue;
      found_virial = true;
      EXPECT_EQ(output.scope, JCN_OUTPUT_LOCAL);
      EXPECT_EQ(output.extensive, 1);
    }
    EXPECT_TRUE(found_virial);
    for (bool newton : {true, false}) {
      session.SelectModel(false, newton);
      for (const auto& positions : systems) {
        const int32_t atoms = static_cast<int32_t>(positions.size() / 3);
        const auto edges = !newton && session.properties().half_list
                               ? HalfEdges(atoms)
                               : FullDirectedEdges(atoms);
        const int64_t capacity = atoms + 2;
        const int64_t raw_edges = static_cast<int64_t>(edges.size());
        const int64_t max_neighbors = std::max<int64_t>(1, atoms - 1);
        Result result = session.ComputeAfterModelCapacityRetries(
            positions, atoms, atoms, capacity, edges, raw_edges,
            max_neighbors);
        ASSERT_EQ(result.protocol.code, JCN_COMPUTE_OK);
        EXPECT_NEAR(
            std::accumulate(result.energy.begin(), result.energy.end(), 0.0),
            TotalEnergy(positions), kTolerance);
        const auto forces = ReferenceForces(positions);
        const auto virial = ReferenceVirial(positions);
        for (size_t index = 0; index < forces.size(); ++index) {
          EXPECT_NEAR(result.force[index], forces[index], kTolerance);
        }
        for (size_t index = 0; index < virial.size(); ++index) {
          EXPECT_NEAR(result.virial[index], virial[index], 5 * kTolerance);
        }
      }
    }
  }
}

TEST(CpuConnectorRegression, Float64ModelSupportsFloat32EngineAbi) {
  ApiLibrary library;
  JcnSession session(
      library.api(),
      ModelDirectory() + "/lennard_jones_sparse_x64.ptb");
  session.SelectModel(false);

  const std::vector<float> positions = {
      0.F, 0.F, 0.F, 1.2F, 0.F, 0.F, 5.F, 0.F, 0.F,
      6.2F, 0.F, 0.F, 7.35F, 0.F, 0.F,
  };
  const auto edges = FullDirectedEdges(5);
  Result result = session.ComputeAfterModelCapacityRetries(
      positions, 5, 5, 7, edges, edges.size(), 4);

  ASSERT_EQ(result.protocol.code, JCN_COMPUTE_OK);
  EXPECT_NEAR(
      std::accumulate(result.energy.begin(), result.energy.end(), 0.0),
      TotalEnergy(positions), kTolerance);
  const auto forces = ReferenceForces(positions);
  for (size_t index = 0; index < forces.size(); ++index) {
    EXPECT_NEAR(result.force[index], forces[index], kTolerance);
  }
}

TEST(CpuConnectorRegression, ShortLennardJonesTrajectoryTracksReferenceEnergy) {
  ApiLibrary library;
  JcnSession session(library.api(),
                     ModelDirectory() + "/lennard_jones_sparse.ptb");
  session.SelectModel(false);

  std::vector<float> positions = {
      0.F, 0.F, 0.F, 1.2F, 0.F, 0.F, 0.F, 1.25F, 0.F, 1.2F, 1.25F, 0.F,
  };
  std::vector<float> velocities = {
      0.01F,  0.02F,  0.F, -0.02F,  0.01F,  0.F,
      0.015F, -0.01F, 0.F, -0.005F, -0.02F, 0.F,
  };
  constexpr float time_step = 2.e-4F;
  const auto edges = FullDirectedEdges(4);

  Result state = session.ComputeAfterModelCapacityRetries(
      positions, 4, 4, 6, edges, edges.size(), 3);
  for (int step = 0; step < 6; ++step) {
    ASSERT_EQ(state.protocol.code, JCN_COMPUTE_OK);
    EXPECT_NEAR(std::accumulate(state.energy.begin(), state.energy.end(), 0.0),
                TotalEnergy(positions), kTolerance);

    for (size_t index = 0; index < positions.size(); ++index) {
      velocities[index] += 0.5F * time_step * state.force[index];
      positions[index] += time_step * velocities[index];
    }
    Result next = session.ComputeAfterModelCapacityRetries(
        positions, 4, 4, 6, edges, edges.size(), 3);
    for (size_t index = 0; index < positions.size(); ++index) {
      velocities[index] += 0.5F * time_step * next.force[index];
    }
    state = std::move(next);
  }
}

TEST(CpuConnectorRegression, CapacityRetryUsesPublicRequiredMinima) {
  ApiLibrary library;
  JcnSession session(library.api(),
                     ModelDirectory() + "/lennard_jones_sparse.ptb");
  session.SelectModel(false);
  const std::vector<float> positions = {0.F, 0.F,  0.F, 1.2F, 0.F,
                                        0.F, 2.3F, 0.F, 0.F};
  const auto edges = FullDirectedEdges(3);
  // The connector validates max_atoms before it reads the caller's tensors.
  // The public adapter reports a third atom while its previous allocation
  // still holds two rows, then retries with the concrete required capacity.
  Result first = session.Compute(
      std::vector<float>(positions.begin(), positions.begin() + 6), 3, 3, 2,
      edges, 1, 1);
  ASSERT_EQ(first.protocol.code, JCN_COMPUTE_NEEDS_CAPACITY_CHANGE);
  ASSERT_EQ(first.protocol.required_capacities.max_atoms, 3);
  Result retry = session.ComputeAfterModelCapacityRetries(
      positions, 3, 3, 5, edges, static_cast<int64_t>(edges.size()), 2);
  EXPECT_EQ(retry.protocol.code, JCN_COMPUTE_OK);
}

TEST(CpuConnectorRegression, HostCallbacksCoverUnevenAndEmptyPartitions) {
  TwoRankHostCollective collective;
  TwoRankHostCollective::RankContext contexts[] = {{&collective, 0},
                                                   {&collective, 1}};
  const char* errors[2] = {nullptr, nullptr};

  // Rank zero has two ghosts that both refer to the sole particle on rank one.
  // Forward exchange must fill both rows despite the unequal local sizes.
  std::vector<float> rank_zero = {1.F, 2.F, 0.F, 0.F, 0.F, 0.F};
  std::vector<float> rank_one = {10.F, 20.F};
  int status[2] = {};
  std::thread forward_zero([&] {
    status[0] =
        TwoRankHostCollective::Exchange(&contexts[0], rank_zero.data(), 3, 2,
                                        JCN_COMMUNICATION_F32, 0, &errors[0]);
  });
  std::thread forward_one([&] {
    status[1] =
        TwoRankHostCollective::Exchange(&contexts[1], rank_one.data(), 1, 2,
                                        JCN_COMMUNICATION_F32, 0, &errors[1]);
  });
  forward_zero.join();
  forward_one.join();
  EXPECT_EQ(status[0], 0);
  EXPECT_EQ(status[1], 0);
  EXPECT_EQ(rank_zero, (std::vector<float>{1.F, 2.F, 10.F, 20.F, 10.F, 20.F}));

  // Reverse exchange accumulates both repeated ghost destinations into the
  // same owned row. The connector FFI clears ghost rows after this callback.
  rank_zero = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F};
  rank_one = {10.F, 20.F};
  std::thread reverse_zero([&] {
    status[0] =
        TwoRankHostCollective::Exchange(&contexts[0], rank_zero.data(), 3, 2,
                                        JCN_COMMUNICATION_F32, 1, &errors[0]);
  });
  std::thread reverse_one([&] {
    status[1] =
        TwoRankHostCollective::Exchange(&contexts[1], rank_one.data(), 1, 2,
                                        JCN_COMMUNICATION_F32, 1, &errors[1]);
  });
  reverse_zero.join();
  reverse_one.join();
  EXPECT_EQ(status[0], 0);
  EXPECT_EQ(status[1], 0);
  EXPECT_EQ(rank_one, (std::vector<float>{18.F, 30.F}));

  // Empty active partitions still participate in the same collective order.
  std::vector<float> empty_zero;
  std::vector<float> empty_one;
  std::thread empty_rank_zero([&] {
    status[0] =
        TwoRankHostCollective::Exchange(&contexts[0], empty_zero.data(), 0, 1,
                                        JCN_COMMUNICATION_F32, 0, &errors[0]);
  });
  std::thread empty_rank_one([&] {
    status[1] =
        TwoRankHostCollective::Exchange(&contexts[1], empty_one.data(), 0, 1,
                                        JCN_COMMUNICATION_F32, 0, &errors[1]);
  });
  empty_rank_zero.join();
  empty_rank_one.join();
  EXPECT_EQ(status[0], 0);
  EXPECT_EQ(status[1], 0);

  // A zero-valued rank is still part of a global reduction.
  rank_zero = {0.F, 0.F};
  rank_one = {4.F, -2.F};
  std::thread reduce_zero([&] {
    status[0] = TwoRankHostCollective::Reduce(
        &contexts[0], rank_zero.data(), 2, JCN_COMMUNICATION_F32, &errors[0]);
  });
  std::thread reduce_one([&] {
    status[1] = TwoRankHostCollective::Reduce(
        &contexts[1], rank_one.data(), 2, JCN_COMMUNICATION_F32, &errors[1]);
  });
  reduce_zero.join();
  reduce_one.join();
  EXPECT_EQ(status[0], 0);
  EXPECT_EQ(status[1], 0);
  EXPECT_EQ(rank_zero, (std::vector<float>{4.F, -2.F}));
  EXPECT_EQ(rank_one, (std::vector<float>{4.F, -2.F}));
}

TEST(CpuConnectorRegression, HostCallbacksRejectCollectiveOrderMismatch) {
  TwoRankHostCollective collective;
  TwoRankHostCollective::RankContext contexts[] = {{&collective, 0},
                                                   {&collective, 1}};
  std::vector<float> gather_value = {1.F};
  std::vector<float> reduce_value = {2.F};
  const char* errors[2] = {nullptr, nullptr};
  int status[2] = {};

  const auto start = std::chrono::steady_clock::now();
  std::thread gather([&] {
    status[0] =
        TwoRankHostCollective::Exchange(&contexts[0], gather_value.data(), 1, 1,
                                        JCN_COMMUNICATION_F32, 0, &errors[0]);
  });
  std::thread reduce([&] {
    status[1] =
        TwoRankHostCollective::Reduce(&contexts[1], reduce_value.data(), 1,
                                      JCN_COMMUNICATION_F32, &errors[1]);
  });
  gather.join();
  reduce.join();

  EXPECT_NE(status[0], 0);
  EXPECT_NE(status[1], 0);
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
}

TEST(CpuConnectorRegression, PublicConnectorSupportsAnEmptyRank) {
  ApiLibrary library;
  const std::string model =
      ModelDirectory() + "/asymmetric_message_passing.ptb";
  TwoRankHostCollective collective;
  TwoRankHostCollective::RankContext contexts[] = {{&collective, 0},
                                                   {&collective, 1}};
  JcnSession rank_zero(library.api(), model);
  JcnSession rank_one(library.api(), model);
  rank_zero.SetCallbacks({&contexts[0], &TwoRankHostCollective::Exchange,
                          nullptr, &TwoRankHostCollective::Reduce, nullptr});
  rank_one.SetCallbacks({&contexts[1], &TwoRankHostCollective::Exchange,
                         nullptr, &TwoRankHostCollective::Reduce, nullptr});
  rank_zero.SelectModel(true);
  rank_one.SelectModel(true);

  Result results[2];
  std::exception_ptr failures[2];
  auto run_rank = [&](int rank, JcnSession& session,
                      std::vector<float> positions, int32_t owned,
                      int32_t valid) {
    try {
      results[rank] = session.Compute(positions, owned, valid, 2, {}, 1, 1);
    } catch (...) {
      failures[rank] = std::current_exception();
    }
  };
  std::thread empty_rank(run_rank, 0, std::ref(rank_zero), std::vector<float>{},
                         0, 0);
  std::thread populated_rank(run_rank, 1, std::ref(rank_one),
                             std::vector<float>{1.F, 0.F, 0.F}, 1, 1);
  empty_rank.join();
  populated_rank.join();

  if (failures[0]) std::rethrow_exception(failures[0]);
  if (failures[1]) std::rethrow_exception(failures[1]);
  EXPECT_EQ(results[0].protocol.code, JCN_COMPUTE_OK);
  EXPECT_EQ(results[1].protocol.code, JCN_COMPUTE_OK);
}

TEST(CpuConnectorRegression, ThreadedHostCallbacksPreserveDirectedMessages) {
  ApiLibrary library;
  const std::string model =
      ModelDirectory() + "/asymmetric_message_passing.ptb";
  TwoRankHostCollective collective;
  TwoRankHostCollective::RankContext contexts[] = {{&collective, 0},
                                                   {&collective, 1}};
  JcnSession rank_zero(library.api(), model);
  JcnSession rank_one(library.api(), model);
  rank_zero.SetCallbacks({&contexts[0], &TwoRankHostCollective::Exchange,
                          nullptr, &TwoRankHostCollective::Reduce, nullptr});
  rank_one.SetCallbacks({&contexts[1], &TwoRankHostCollective::Exchange,
                         nullptr, &TwoRankHostCollective::Reduce, nullptr});
  rank_zero.SelectModel(true);
  rank_one.SelectModel(true);
  const std::vector<std::pair<int32_t, int32_t>> edges[] = {
      {{0, 1}, {0, 2}},
      {{0, 1}},
  };
  auto evaluate = [&](float first_position, float second_position) {
    std::array<Result, 2> results;
    for (int attempt = 0; attempt < 4; ++attempt) {
      std::exception_ptr failures[2];
      auto run_rank = [&](int rank, JcnSession& session,
                          std::vector<float> positions, int32_t valid,
                          int64_t capacity) {
        try {
          results[rank] =
              session.Compute(positions, 1, valid, capacity, edges[rank],
                              edges[rank].size(), edges[rank].size());
        } catch (...) {
          failures[rank] = std::current_exception();
        }
      };
      std::thread first(
          run_rank, 0, std::ref(rank_zero),
          std::vector<float>{first_position, 0.F, 0.F, second_position, 0.F,
                             0.F, second_position, 0.F, 0.F},
          3, 4);
      std::thread second(
          run_rank, 1, std::ref(rank_one),
          std::vector<float>{second_position, 0.F, 0.F, first_position, 0.F,
                             0.F},
          2, 3);
      first.join();
      second.join();
      if (failures[0]) std::rethrow_exception(failures[0]);
      if (failures[1]) std::rethrow_exception(failures[1]);

      const bool retry =
          results[0].protocol.code == JCN_COMPUTE_NEEDS_CAPACITY_CHANGE ||
          results[1].protocol.code == JCN_COMPUTE_NEEDS_CAPACITY_CHANGE;
      if (!retry) return results;
    }
    throw std::runtime_error(
        "collective graph capacities did not settle after four retries");
  };

  const auto results = evaluate(0.F, 1.F);
  ASSERT_EQ(results[0].protocol.code, JCN_COMPUTE_OK);
  ASSERT_EQ(results[1].protocol.code, JCN_COMPUTE_OK);

  // Independent directed reference: f_sender <- f_sender + w*f_receiver.
  // Rank zero has two ghost edges to the same remote owner, while rank one
  // has one. This makes repeated reverse destinations and uneven local row
  // counts part of the public connector execution.
  float local[] = {0.5F, 1.5F};
  float ghost[] = {0.F, 0.F};
  for (float weight : {0.25F, 0.5F, 0.75F}) {
    const float old_zero = local[0];
    const float old_one = local[1];
    ghost[0] = old_one;
    ghost[1] = old_zero;
    local[0] = old_zero + 2.F * weight * old_one;
    local[1] = old_one + weight * old_zero;
  }
  const float rank_zero_energy = 0.1F * local[0] * local[0];
  const float rank_one_energy = 0.1F * local[1] * local[1];
  const float reduced = rank_zero_energy + rank_one_energy;
  // Reversing sender and receiver leaves the local central features at 0.5
  // and 1.5, so these independent directed values catch that error directly.
  EXPECT_NEAR(results[0].energy[0], rank_zero_energy + 0.01F * reduced / 2.F,
              kTolerance);
  EXPECT_NEAR(results[1].energy[0], rank_one_energy + 0.01F * reduced / 2.F,
              kTolerance);

  // Finite differences of the complete two-rank energy verify that reverse
  // feature communication transports force derivatives back to the owners.
  // Applying the same deformation to owned and ghost coordinates also checks
  // the LOCAL virial returned by the strain derivative.
  constexpr float step = 1.e-3F;
  auto total_energy = [](const std::array<Result, 2>& values) {
    return values[0].energy[0] + values[1].energy[0];
  };
  const float force_zero = -(total_energy(evaluate(step, 1.F)) -
                             total_energy(evaluate(-step, 1.F))) /
                           (2.F * step);
  const float force_one = -(total_energy(evaluate(0.F, 1.F + step)) -
                            total_energy(evaluate(0.F, 1.F - step))) /
                          (2.F * step);
  EXPECT_NEAR(results[0].force[0], force_zero, 5 * kTolerance);
  EXPECT_NEAR(results[1].force[0], force_one, 5 * kTolerance);

  const float virial_xx = -(total_energy(evaluate(0.F, 1.F + step)) -
                            total_energy(evaluate(0.F, 1.F - step))) /
                          (2.F * step);
  EXPECT_NEAR(results[0].virial[0] + results[1].virial[0], virial_xx,
              5 * kTolerance);
}

TEST(CpuConnectorRegression, CommunicationVariantRejectsMissingCallbacks) {
  ApiLibrary library;
  JcnSession session(library.api(),
                     ModelDirectory() + "/asymmetric_message_passing.ptb");
  EXPECT_THROW(session.SelectModel(true), std::runtime_error);
}

TEST(CpuConnectorRegression, CommunicationRejectsNewtonOffBeforeCallbacks) {
  ApiLibrary library;
  JcnSession session(library.api(),
                     ModelDirectory() + "/asymmetric_message_passing.ptb");
  try {
    session.SelectModel(true, false);
    FAIL() << "comm-on/Newton-off selection unexpectedly succeeded";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("No comm-on/Newton-off"),
              std::string::npos);
  }
}

}  // namespace
