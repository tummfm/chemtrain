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

#include "connector/jcn_api.h"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <vector>

#include "connector/jcn_buffer_internal.h"
#include "connector/jcn_neighbor_internal.h"
#include "connector/model.pb.h"
#include "connector/model_parser.h"
#include "connector/runner.h"
#include "connector/runtime_types.h"

struct jcn_client {
  jcn::ConnectorConfig config;
  std::unique_ptr<xla::PjRtClient> pjrt_client;
  int pjrt_device_index = 0;
};

struct jcn_model {
  jcn::Model proto;
  std::string unit_style;
  std::vector<jcn_particle_field_descriptor> particle_fields;
  std::vector<jcn_global_field_descriptor> global_fields;
  std::vector<std::vector<int64_t>> output_dimensions;
  std::vector<jcn_output_descriptor> outputs;
};

struct jcn_executor {
  jcn_client* client = nullptr;
  jcn_model* model = nullptr;
  jcn_requested_capacities capacities{};
  jcn_communication_callbacks c_communication{};
  jcn::CommunicationCallbacks communication{};
  jcn::EngineAbiSpec engine_abi{};
  std::unique_ptr<jcn::Runner> runner;
};

namespace {

// ---------------------------------------------------------------------------
// Process state and opaque handle storage.
// ---------------------------------------------------------------------------

thread_local std::string g_last_error;
std::mutex g_runtime_initialization_mutex;
std::condition_variable g_runtime_initialization_cv;
enum class RuntimeInitializationState {
  kUninitialized,
  kInitializing,
  kInitialized,
  kFailed,
};
RuntimeInitializationState g_runtime_initialization_state =
    RuntimeInitializationState::kUninitialized;
std::string g_runtime_initialization_error;

class RuntimeInitializationReservation {
 public:
  explicit RuntimeInitializationReservation(bool active) : active_(active) {}
  RuntimeInitializationReservation(const RuntimeInitializationReservation&) =
      delete;
  RuntimeInitializationReservation& operator=(
      const RuntimeInitializationReservation&) = delete;
  ~RuntimeInitializationReservation() {
    if (!active_ || finished_) return;
    std::lock_guard<std::mutex> lock(g_runtime_initialization_mutex);
    g_runtime_initialization_state = RuntimeInitializationState::kUninitialized;
    g_runtime_initialization_cv.notify_all();
  }

  bool active() const { return active_; }

  void MarkInitialized() {
    if (!active_) return;
    std::lock_guard<std::mutex> lock(g_runtime_initialization_mutex);
    g_runtime_initialization_state = RuntimeInitializationState::kInitialized;
    finished_ = true;
    g_runtime_initialization_cv.notify_all();
  }

  void MarkFailed(const std::string& message) {
    if (!active_) return;
    std::lock_guard<std::mutex> lock(g_runtime_initialization_mutex);
    g_runtime_initialization_error = message;
    g_runtime_initialization_state = RuntimeInitializationState::kFailed;
    finished_ = true;
    g_runtime_initialization_cv.notify_all();
  }

 private:
  bool active_;
  bool finished_ = false;
};

RuntimeInitializationReservation ReserveRuntimeInitialization(
    const std::string&) {
  bool should_initialize = false;
  std::unique_lock<std::mutex> lock(g_runtime_initialization_mutex);
  g_runtime_initialization_cv.wait(lock, [] {
    return g_runtime_initialization_state !=
           RuntimeInitializationState::kInitializing;
  });
  if (g_runtime_initialization_state == RuntimeInitializationState::kFailed) {
    throw std::runtime_error("JCN runtime initialization previously failed: " +
                             g_runtime_initialization_error);
  }
  if (g_runtime_initialization_state ==
      RuntimeInitializationState::kUninitialized) {
    g_runtime_initialization_state = RuntimeInitializationState::kInitializing;
    should_initialize = true;
  }
  return RuntimeInitializationReservation(should_initialize);
}

constexpr uint8_t kDlInt = 0;
constexpr uint8_t kDlUInt = 1;
constexpr uint8_t kDlFloat = 2;

// ---------------------------------------------------------------------------
// Status and small conversion helpers.
// ---------------------------------------------------------------------------

jcn_status MakeStatus(jcn_status_code code, const std::string& message) {
  g_last_error = message;
  return jcn_status{code, g_last_error.c_str()};
}

void SetStatus(jcn_status* out, jcn_status_code code,
               const std::string& message = "") {
  if (out == nullptr) return;
  *out = MakeStatus(code, message);
}

std::string Lower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

jcn_communication_scalar_type ToCCommunicationScalar(
    jcn::CommunicationScalarType type) {
  switch (type) {
    case jcn::CommunicationScalarType::F32:
      return JCN_COMMUNICATION_F32;
    case jcn::CommunicationScalarType::F64:
      return JCN_COMMUNICATION_F64;
  }
  return JCN_COMMUNICATION_F64;
}

jcn::TensorDtype ToInternalDtype(jcn_tensor_dtype dtype,
                                 jcn::TensorDtype model_default) {
  switch (dtype) {
    case JCN_TENSOR_DTYPE_MODEL_DEFAULT:
      return model_default;
    case JCN_TENSOR_DTYPE_F32:
      return jcn::TensorDtype::F32;
    case JCN_TENSOR_DTYPE_F64:
      return jcn::TensorDtype::F64;
    case JCN_TENSOR_DTYPE_S32:
      return jcn::TensorDtype::S32;
  }
  return model_default;
}

jcn::EngineAbiSpec ToInternalEngineAbi(const jcn_engine_abi_options& options) {
  jcn::EngineAbiSpec spec;
  spec.position_dtype =
      ToInternalDtype(options.position_dtype, jcn::TensorDtype::F32);
  spec.species_dtype =
      ToInternalDtype(options.species_dtype, jcn::TensorDtype::S32);
  spec.default_output_dtype =
      ToInternalDtype(options.default_output_dtype, jcn::TensorDtype::F32);
  for (int32_t i = 0; i < options.num_output_dtypes; ++i) {
    spec.output_dtypes.emplace_back(
        options.output_dtypes[i].name == nullptr
            ? ""
            : options.output_dtypes[i].name,
        ToInternalDtype(options.output_dtypes[i].dtype,
                        spec.default_output_dtype));
  }
  spec.communication_dtype =
      options.communication_dtype == JCN_TENSOR_DTYPE_MODEL_DEFAULT
          ? jcn::TensorDtype::ModelDefault
          : ToInternalDtype(options.communication_dtype, jcn::TensorDtype::F32);
  spec.species_encoding = options.species_encoding == JCN_SPECIES_ONE_BASED
                              ? jcn::SpeciesEncoding::OneBased
                              : jcn::SpeciesEncoding::ZeroBased;
  return spec;
}

bool IsFloatingOrDefault(jcn_tensor_dtype dtype) {
  return dtype == JCN_TENSOR_DTYPE_MODEL_DEFAULT ||
         dtype == JCN_TENSOR_DTYPE_F32 || dtype == JCN_TENSOR_DTYPE_F64;
}

bool IsSpeciesOrDefault(jcn_tensor_dtype dtype) {
  return dtype == JCN_TENSOR_DTYPE_MODEL_DEFAULT ||
         dtype == JCN_TENSOR_DTYPE_S32;
}

int ExchangeWithAdapter(void* context, void* data, std::int64_t rows,
                        std::int64_t cols, jcn::CommunicationScalarType type,
                        bool reverse, const char** error) {
  auto* callbacks = static_cast<jcn_communication_callbacks*>(context);
  if (callbacks == nullptr || callbacks->exchange == nullptr) {
    if (error != nullptr)
      *error = "engine communication callback is not installed";
    return 1;
  }
  return callbacks->exchange(callbacks->context, data, rows, cols,
                             ToCCommunicationScalar(type), reverse ? 1 : 0,
                             error);
}

int DeviceExchangeWithAdapter(void* context, void* data, std::int64_t rows,
                              std::int64_t cols,
                              jcn::CommunicationScalarType type, bool reverse,
                              const char* backend, int device_ordinal,
                              void* stream, const char** error) {
  auto* callbacks = static_cast<jcn_communication_callbacks*>(context);
  if (callbacks == nullptr || callbacks->device_exchange == nullptr) {
    if (error != nullptr)
      *error = "engine device communication callback is not installed";
    return 1;
  }
  return callbacks->device_exchange(
      callbacks->context, data, rows, cols, ToCCommunicationScalar(type),
      reverse ? 1 : 0, backend, device_ordinal, stream, error);
}

int ReduceWithAdapter(void* context, void* data, std::int64_t count,
                      jcn::CommunicationScalarType type, const char** error) {
  auto* callbacks = static_cast<jcn_communication_callbacks*>(context);
  if (callbacks == nullptr || callbacks->reduce == nullptr) {
    if (error != nullptr) *error = "engine reduce callback is not installed";
    return 1;
  }
  return callbacks->reduce(callbacks->context, data, count,
                           ToCCommunicationScalar(type), error);
}

int DeviceReduceWithAdapter(void* context, void* data, std::int64_t count,
                            jcn::CommunicationScalarType type,
                            const char* backend, int device_ordinal,
                            void* stream, const char** error) {
  auto* callbacks = static_cast<jcn_communication_callbacks*>(context);
  if (callbacks == nullptr || callbacks->device_reduce == nullptr) {
    if (error != nullptr)
      *error = "engine device reduce callback is not installed";
    return 1;
  }
  return callbacks->device_reduce(callbacks->context, data, count,
                                  ToCCommunicationScalar(type), backend,
                                  device_ordinal, stream, error);
}

bool BufferBelongsToExecutor(jcn_executor* executor, const jcn_buffer* buffer,
                             std::string* error) {
  if (buffer == nullptr) return true;
  if (buffer->executor == executor) return true;
  if (error != nullptr) {
    *error =
        "JCN request contains a DLPack buffer imported for a different "
        "executor";
  }
  return false;
}

bool RequestBuffersBelongToExecutor(jcn_executor* executor,
                                    const jcn_force_request* request,
                                    std::string* error) {
  if (request == nullptr) return true;
  if (!BufferBelongsToExecutor(executor, request->atoms.positions, error))
    return false;
  for (int32_t i = 0; i < request->atoms.num_particle_inputs; ++i)
    if (!BufferBelongsToExecutor(
            executor, request->atoms.particle_inputs[i].buffer, error))
      return false;
  for (int32_t i = 0; i < request->atoms.num_global_inputs; ++i)
    if (!BufferBelongsToExecutor(executor,
                                 request->atoms.global_inputs[i].buffer, error))
      return false;
  if (request->neighbors != nullptr) {
    if (!BufferBelongsToExecutor(executor, request->neighbors->sparse.senders,
                                 error) ||
        !BufferBelongsToExecutor(executor, request->neighbors->sparse.receivers,
                                 error) ||
        !BufferBelongsToExecutor(executor, request->neighbors->sparse.pair_type,
                                 error) ||
        !BufferBelongsToExecutor(executor, request->neighbors->dense.neighbors,
                                 error) ||
        !BufferBelongsToExecutor(executor, request->neighbors->dense.pair_type,
                                 error))
      return false;
  }
  for (int32_t i = 0; i < request->num_outputs; ++i)
    if (!BufferBelongsToExecutor(executor, request->outputs[i].buffer, error))
      return false;
  return true;
}

// ---------------------------------------------------------------------------
// Model metadata copied from protobuf into ABI structs.
// ---------------------------------------------------------------------------

void GetRuntimeInfo(jcn_runtime_info* out) {
  if (out == nullptr) return;
  *out = jcn_runtime_info{};
  out->api_version = JCN_API_VERSION;
  out->api_struct_size = sizeof(JCN_Api);
#ifdef JCN_CONNECTOR_BUILD_VERSION
  out->connector_build_version = JCN_CONNECTOR_BUILD_VERSION;
#else
  out->connector_build_version = "uncompiled-draft";
#endif
#ifdef JCN_XLA_COMMIT
  out->xla_commit = JCN_XLA_COMMIT;
#else
  out->xla_commit = "unknown";
#endif
#ifdef JCN_XLA_SHA256
  out->xla_sha256 = JCN_XLA_SHA256;
#else
  out->xla_sha256 = "unknown";
#endif
#ifdef HERMETIC_CUDA_VERSION
  out->build_cuda_version = HERMETIC_CUDA_VERSION;
#else
  out->build_cuda_version = "unknown";
#endif
#ifdef HERMETIC_CUDNN_VERSION
  out->build_cudnn_version = HERMETIC_CUDNN_VERSION;
#else
  out->build_cudnn_version = "unknown";
#endif
#ifdef HERMETIC_CUDA_COMPUTE_CAPABILITIES
  out->build_cuda_compute_capabilities = HERMETIC_CUDA_COMPUTE_CAPABILITIES;
#else
  out->build_cuda_compute_capabilities = "unknown";
#endif
}

jcn_neighbor_format FromProtoNeighbor(jcn::Model::NeighborListType type) {
  switch (type) {
    case jcn::Model::SIMPLE_DENSE:
      return JCN_NEIGHBOR_SIMPLE_DENSE;
    case jcn::Model::SIMPLE_SPARSE:
      return JCN_NEIGHBOR_SIMPLE_SPARSE;
    default:
      return JCN_NEIGHBOR_UNSPECIFIED;
  }
}

jcn_tensor_dtype FromProtoGlobalDtype(jcn::Model::GlobalDtype dtype) {
  switch (dtype) {
    case jcn::Model::GLOBAL_FLOAT32:
      return JCN_TENSOR_DTYPE_F32;
    case jcn::Model::GLOBAL_FLOAT64:
      return JCN_TENSOR_DTYPE_F64;
    case jcn::Model::GLOBAL_INT32:
      return JCN_TENSOR_DTYPE_S32;
    default:
      return JCN_TENSOR_DTYPE_MODEL_DEFAULT;
  }
}

jcn_output_scope FromProtoOutputScope(jcn::Model::OutputScope scope) {
  switch (scope) {
    case jcn::Model::LOCAL:
      return JCN_OUTPUT_LOCAL;
    case jcn::Model::GLOBAL:
      return JCN_OUTPUT_GLOBAL;
    default:
      return JCN_OUTPUT_PARTICLE;
  }
}

void FillPropertiesFromRuntime(const jcn::ModelProperties& in,
                               jcn_neighbor_format format,
                               const jcn_model* model_storage,
                               jcn_model_properties* out) {
  if (out == nullptr) return;
  *out = jcn_model_properties{};
  out->cutoff = in.cutoff;
  out->comm_dist = in.comm_dist;
  out->unit_style = model_storage->unit_style.c_str();
  out->include_ghosts = in.neighbor_list.include_ghosts ? 1 : 0;
  out->half_list = in.neighbor_list.half_list ? 1 : 0;
  out->communication_buffer_width = in.communication_buffer_width;
  out->neighbor_format = format;
  out->particle_fields = model_storage->particle_fields.data();
  out->num_particle_fields =
      static_cast<int32_t>(model_storage->particle_fields.size());
  out->global_fields = model_storage->global_fields.data();
  out->num_global_fields =
      static_cast<int32_t>(model_storage->global_fields.size());
  out->outputs = model_storage->outputs.data();
  out->num_outputs = static_cast<int32_t>(model_storage->outputs.size());
  out->include_pair_type = in.neighbor_list.include_pair_type ? 1 : 0;
  out->requires_communication = in.requires_communication ? 1 : 0;
}

void FillModelProperties(jcn_model* model, jcn_model_properties* out) {
  if (out == nullptr) return;
  const jcn::Model& proto = model->proto;
  out->cutoff = proto.neighbor_list().cutoff();
  out->unit_style = model->unit_style.c_str();
  out->communication_buffer_width = proto.communication_buffer_width();
  out->neighbor_format = FromProtoNeighbor(proto.neighbor_list().type());
  out->half_list = proto.neighbor_list().has_half_list()
                       ? proto.neighbor_list().half_list()
                       : true;

  const int multiplier = proto.neighbor_list().nbr_order_size() == 1
                             ? proto.neighbor_list().nbr_order(0)
                             : 1;
  out->comm_dist = multiplier * proto.neighbor_list().cutoff();
  out->include_ghosts = multiplier > 1 ? 1 : 0;
  out->particle_fields = model->particle_fields.data();
  out->num_particle_fields =
      static_cast<int32_t>(model->particle_fields.size());
  out->global_fields = model->global_fields.data();
  out->num_global_fields = static_cast<int32_t>(model->global_fields.size());
  out->outputs = model->outputs.data();
  out->num_outputs = static_cast<int32_t>(model->outputs.size());
  out->include_pair_type = proto.neighbor_list().include_pair_type() ? 1 : 0;
  out->requires_communication = proto.requires_communication() ? 1 : 0;
}

// ---------------------------------------------------------------------------
// DLPack/PJRT boundary checks.
// ---------------------------------------------------------------------------

bool DeviceMatchesBackend(const jcn_client& client,
                          const DLManagedTensor* tensor, std::string* error) {
  if (tensor == nullptr) {
    *error = "DLPack tensor is null";
    return false;
  }
  const DLDevice device = tensor->dl_tensor.device;
  const std::string backend = Lower(client.config.backend);

  // DLPack reports where caller-owned memory lives. CPU buffers are valid for
  // every backend because PJRT copies them through the explicit host-staging
  // path. Device buffers are shared without copying, so their backend and
  // ordinal must match the selected PJRT device exactly.
  if (device.device_type == kDLCPU) return true;

  bool type_ok = false;
  if (backend == "cpu" || backend.empty()) {
    type_ok = false;
  } else if (backend == "cuda") {
    type_ok =
        device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  } else if (backend == "rocm" || backend == "hip") {
    type_ok = device.device_type == kDLROCM;
  } else if (backend == "sycl" || backend == "oneapi" || backend == "intel") {
    type_ok = device.device_type == kDLOneAPI;
  } else {
    *error = "Unsupported PJRT backend for DLPack validation: " +
             client.config.backend;
    return false;
  }

  if (!type_ok) {
    *error = "DLPack device type does not match PJRT backend " +
             client.config.backend;
    return false;
  }
  if (device.device_id != client.config.device) {
    *error = "DLPack device id does not match selected PJRT device";
    return false;
  }
  return true;
}

bool ValidateDlpackTensor(const DLManagedTensor* tensor, std::string* error) {
  if (tensor == nullptr) {
    *error = "DLPack tensor is null";
    return false;
  }
  const DLTensor& dl = tensor->dl_tensor;
  if (dl.data == nullptr) {
    *error = "DLPack tensor data pointer is null";
    return false;
  }
  if (dl.ndim < 0) {
    *error = "DLPack tensor ndim is negative";
    return false;
  }
  if (dl.ndim > 0 && dl.shape == nullptr) {
    *error = "DLPack tensor shape pointer is null";
    return false;
  }
  if (dl.strides != nullptr) {
    *error = "strided DLPack tensors are not supported";
    return false;
  }
  for (int i = 0; i < dl.ndim; ++i) {
    if (dl.shape[i] < 0) {
      *error = "DLPack tensor has a negative dimension";
      return false;
    }
  }
  if (dl.dtype.lanes != 1) {
    *error = "DLPack tensor lanes other than one are not supported";
    return false;
  }
  const bool dtype_ok = (dl.dtype.code == kDlFloat &&
                         (dl.dtype.bits == 32 || dl.dtype.bits == 64)) ||
                        (dl.dtype.code == kDlInt && dl.dtype.bits == 32) ||
                        (dl.dtype.code == kDlUInt && dl.dtype.bits == 32);
  if (!dtype_ok) {
    *error = "DLPack tensor dtype is not supported";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Neighbor-list pseudo-class helpers.
// ---------------------------------------------------------------------------

jcn_neighbor_list* NeighborListCreateSimpleSparse(
    const jcn_sparse_neighbors* sparse, jcn_status* status) {
  if (sparse == nullptr || sparse->senders == nullptr ||
      sparse->receivers == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "sparse neighbor list is missing sender/receiver buffers");
    return nullptr;
  }
  auto* neighbors = new (std::nothrow) jcn_neighbor_list;
  if (neighbors == nullptr) {
    SetStatus(status, JCN_STATUS_INTERNAL, "failed to allocate neighbor list");
    return nullptr;
  }
  neighbors->format = JCN_NEIGHBOR_SIMPLE_SPARSE;
  neighbors->sparse = *sparse;
  SetStatus(status, JCN_STATUS_OK);
  return neighbors;
}

jcn_neighbor_list* NeighborListCreateSimpleDense(
    const jcn_dense_neighbors* dense, jcn_status* status) {
  if (dense == nullptr || dense->neighbors == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "dense neighbor list is missing neighbor buffer");
    return nullptr;
  }
  auto* neighbors = new (std::nothrow) jcn_neighbor_list;
  if (neighbors == nullptr) {
    SetStatus(status, JCN_STATUS_INTERNAL, "failed to allocate neighbor list");
    return nullptr;
  }
  neighbors->format = JCN_NEIGHBOR_SIMPLE_DENSE;
  neighbors->dense = *dense;
  SetStatus(status, JCN_STATUS_OK);
  return neighbors;
}

jcn_neighbor_format NeighborListFormat(const jcn_neighbor_list* neighbors) {
  return neighbors == nullptr ? JCN_NEIGHBOR_UNSPECIFIED : neighbors->format;
}

void NeighborListDestroy(jcn_neighbor_list* neighbors) { delete neighbors; }

// ---------------------------------------------------------------------------
// Client lifecycle.
// ---------------------------------------------------------------------------

jcn_client* ClientCreate(const jcn_client_options* options,
                         jcn_status* status) {
  if (options == nullptr || options->backend == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "client options/backend is null");
    return nullptr;
  }
  auto* client = new (std::nothrow) jcn_client;
  if (client == nullptr) {
    SetStatus(status, JCN_STATUS_INTERNAL, "failed to allocate client");
    return nullptr;
  }
  client->config.backend = Lower(options->backend);
  client->config.device = options->device_ordinal;
  client->config.memory_fraction = options->memory_fraction;
  try {
    auto initialization = ReserveRuntimeInitialization(client->config.backend);
    if (initialization.active()) {
      try {
        jcn::Runner::initialize();
      } catch (const std::exception& e) {
        initialization.MarkFailed(e.what());
        throw;
      }
      // PJRT plugin registration is process-global and cannot be repeated.
      // Mark initialization complete before client creation so a later option
      // or client failure can retry without registering the plugins again.
      initialization.MarkInitialized();
    }
    client->pjrt_client =
        jcn::CreatePjRtClient(client->config, &client->pjrt_device_index);
    SetStatus(status, JCN_STATUS_OK);
  } catch (const std::exception& e) {
    delete client;
    SetStatus(status, JCN_STATUS_INTERNAL, e.what());
    return nullptr;
  }
  return client;
}

void ClientDestroy(jcn_client* client) { delete client; }

// ---------------------------------------------------------------------------
// Model lifecycle.
// ---------------------------------------------------------------------------

jcn_model* ModelLoadFromProtobuf(const void* data, size_t size,
                                 jcn_model_properties* props_out,
                                 jcn_status* status) {
  if (data == nullptr || size == 0) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT, "model protobuf is empty");
    return nullptr;
  }
  auto* model = new (std::nothrow) jcn_model;
  if (model == nullptr) {
    SetStatus(status, JCN_STATUS_INTERNAL, "failed to allocate model");
    return nullptr;
  }
  try {
    model->proto = jcn::ParseModelProtobuf(data, size);
  } catch (const std::exception& e) {
    delete model;
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT, e.what());
    return nullptr;
  }
  // Copy input descriptors into stable C ABI storage.
  model->particle_fields.reserve(model->proto.particle_fields_size());
  for (const auto& field : model->proto.particle_fields()) {
    model->particle_fields.push_back(
        {field.name().c_str(), JCN_TENSOR_DTYPE_S32});
  }
  model->global_fields.reserve(model->proto.global_fields_size());
  for (int i = 0; i < model->proto.global_fields_size(); ++i) {
    const auto& field = model->proto.global_fields(i);
    model->global_fields.push_back(
        {field.name().c_str(), FromProtoGlobalDtype(field.dtype())});
  }
  // Copy typed output descriptors in executable order.
  const int output_count = model->proto.output_fields_size();
  model->output_dimensions.reserve(output_count);
  model->outputs.reserve(output_count);
  for (int i = 0; i < output_count; ++i) {
    const auto& field = model->proto.output_fields(i);
    model->output_dimensions.emplace_back(field.dimensions().begin(),
                                          field.dimensions().end());
    model->outputs.push_back(
        {field.name().c_str(), FromProtoOutputScope(field.scope()),
         model->output_dimensions.back().data(),
         static_cast<int32_t>(model->output_dimensions.back().size()),
         static_cast<int32_t>(field.components()),
         field.extensive() ? 1 : 0});
  }
  model->unit_style =
      model->proto.has_unit_style() ? model->proto.unit_style() : "real";
  FillModelProperties(model, props_out);
  SetStatus(status, JCN_STATUS_OK);
  return model;
}

void ModelDestroy(jcn_model* model) { delete model; }

// ---------------------------------------------------------------------------
// Executor lifecycle and capacity ownership.
// ---------------------------------------------------------------------------

jcn_executor* ExecutorCreate(jcn_client* client, jcn_status* status) {
  if (client == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT, "client is null");
    return nullptr;
  }
  auto* executor = new (std::nothrow) jcn_executor;
  if (executor == nullptr) {
    SetStatus(status, JCN_STATUS_INTERNAL, "failed to allocate executor");
    return nullptr;
  }
  executor->client = client;
  SetStatus(status, JCN_STATUS_OK);
  return executor;
}

void ExecutorDestroy(jcn_executor* executor) { delete executor; }

void ExecutorSetModelOptions(jcn_executor* executor, jcn_model* model,
                             const jcn_model_options* options,
                             jcn_model_properties* props_out,
                             jcn_status* status) {
  if (executor == nullptr || model == nullptr || options == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "executor/model/options is null");
    return;
  }
  const bool use_communication = options->use_communication != 0;
  const bool newton_pair = options->newton_pair != 0;
  if (use_communication && !newton_pair) {
    SetStatus(status, JCN_STATUS_FAILED_PRECONDITION,
              "Communication requires Newton pair forces. No "
              "comm-on/Newton-off model variant exists.");
    return;
  }
  const bool has_host_communication =
      executor->communication.exchange != nullptr &&
      executor->communication.reduce != nullptr;
  const bool has_device_communication =
      executor->communication.device_exchange != nullptr &&
      executor->communication.device_reduce != nullptr;
  if (use_communication && executor->client->config.backend == "cpu" &&
      !has_host_communication) {
    SetStatus(status, JCN_STATUS_FAILED_PRECONDITION,
              "CPU communication requires host exchange and reduce callbacks");
    return;
  }
  if (use_communication && !has_host_communication &&
      !has_device_communication) {
    SetStatus(status, JCN_STATUS_FAILED_PRECONDITION,
              "communication-aware model requested without engine callbacks");
    return;
  }
  try {
    executor->model = model;
    executor->runner = std::make_unique<jcn::Runner>(
        executor->client->config, false, executor->client->pjrt_client.get(),
        executor->client->pjrt_device_index);

    // Model loading selects the exported variant and prepares its compiler and
    // graph builders. Communication remains an engine option because only the
    // adapter knows whether it can service exchange and reduction on the
    // selected rank and device.
    jcn::ModelConfig config;
    config.model = &model->proto;
    config.newton = newton_pair;
    config.use_communication = use_communication;
    if (use_communication) config.communication = executor->communication;
    config.engine_abi = executor->engine_abi;
    jcn::ModelProperties selected = executor->runner->load_model(config);
    FillPropertiesFromRuntime(selected,
                              executor->runner->selected_neighbor_format(),
                              model, props_out);
    SetStatus(status, JCN_STATUS_OK);
  } catch (const std::exception& e) {
    SetStatus(status, JCN_STATUS_INTERNAL, e.what());
  }
}

void ExecutorSetEngineAbiOptions(jcn_executor* executor,
                                 const jcn_engine_abi_options* options,
                                 jcn_status* status) {
  if (executor == nullptr || options == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "executor/engine ABI options is null");
    return;
  }
  if (options->atom_layout != JCN_DENSE_LAYOUT_CONTIGUOUS ||
      options->output_layout != JCN_DENSE_LAYOUT_CONTIGUOUS) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "engine ABI currently supports only contiguous dense layouts");
    return;
  }
  if (!IsFloatingOrDefault(options->position_dtype) ||
      !IsFloatingOrDefault(options->default_output_dtype) ||
      !IsFloatingOrDefault(options->communication_dtype) ||
      !IsSpeciesOrDefault(options->species_dtype)) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "engine ABI dtypes must use f32/f64 for floating tensors and s32 "
              "for species");
    return;
  }
  if (options->num_output_dtypes < 0 ||
      (options->num_output_dtypes > 0 && options->output_dtypes == nullptr)) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "engine ABI output dtype table is invalid");
    return;
  }
  std::set<std::string> output_names;
  for (int32_t i = 0; i < options->num_output_dtypes; ++i) {
    const jcn_named_tensor_dtype& entry = options->output_dtypes[i];
    if (entry.name == nullptr || entry.name[0] == '\0') {
      SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
                "engine ABI output dtype names must be non-empty");
      return;
    }
    if (!IsFloatingOrDefault(entry.dtype)) {
      SetStatus(
          status, JCN_STATUS_INVALID_ARGUMENT,
          "engine ABI named output dtype must be f32, f64, or model default");
      return;
    }
    if (!output_names.insert(entry.name).second) {
      SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
                "engine ABI output dtype names must be unique");
      return;
    }
  }
  if (options->species_encoding != JCN_SPECIES_ZERO_BASED &&
      options->species_encoding != JCN_SPECIES_ONE_BASED) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "engine ABI species encoding is invalid");
    return;
  }
  executor->engine_abi = ToInternalEngineAbi(*options);
  SetStatus(status, JCN_STATUS_OK);
}

void ExecutorSetCommunicationCallbacks(
    jcn_executor* executor, const jcn_communication_callbacks* callbacks,
    jcn_status* status) {
  if (executor == nullptr || callbacks == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "executor/communication callbacks is null");
    return;
  }
  // Communication models can be serviced by a complete host callback pair or a
  // complete device callback pair. Engines may install only the device pair
  // when their communication buffers already live on the selected backend.
  const bool has_host_communication =
      callbacks->exchange != nullptr && callbacks->reduce != nullptr;
  const bool has_device_communication = callbacks->device_exchange != nullptr &&
                                        callbacks->device_reduce != nullptr;
  if (!has_host_communication && !has_device_communication) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "communication callbacks must provide host exchange/reduce or "
              "device exchange/reduce");
    return;
  }

  // Copy the C callback table into the executor so Runner can keep a stable
  // pointer while PJRT execution is active.  The user context remains owned by
  // the adapter and must outlive the executor.
  executor->c_communication = *callbacks;
  executor->communication.context = &executor->c_communication;
  executor->communication.exchange =
      callbacks->exchange == nullptr ? nullptr : &ExchangeWithAdapter;
  executor->communication.device_exchange =
      callbacks->device_exchange == nullptr ? nullptr
                                            : &DeviceExchangeWithAdapter;
  executor->communication.reduce =
      callbacks->reduce == nullptr ? nullptr : &ReduceWithAdapter;
  executor->communication.device_reduce =
      callbacks->device_reduce == nullptr ? nullptr : &DeviceReduceWithAdapter;
  SetStatus(status, JCN_STATUS_OK);
}

void ExecutorSetCapacities(jcn_executor* executor,
                           const jcn_requested_capacities* capacities,
                           jcn_status* status) {
  if (executor == nullptr || capacities == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "executor/capacities is null");
    return;
  }
  executor->capacities = *capacities;
  SetStatus(status, JCN_STATUS_OK);
}

// ---------------------------------------------------------------------------
// Force execution.
// ---------------------------------------------------------------------------

void ExecutorComputeForces(jcn_executor* executor,
                           const jcn_force_request* request,
                           jcn_force_result* result, jcn_status* status) {
  if (result != nullptr) result->code = JCN_COMPUTE_FAILED;
  if (executor == nullptr || executor->runner == nullptr ||
      request == nullptr || result == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "executor/request/result is null");
    return;
  }

  try {
    // Validate count/table pairs before the ownership walk dereferences any
    // caller-provided C array. Runner repeats the count/table validation for
    // non-C callers, but the C ABI must fail malformed requests with a status
    // instead of allowing an invalid pointer access.
    if (request->atoms.num_particle_inputs < 0 ||
        (request->atoms.num_particle_inputs > 0 &&
         request->atoms.particle_inputs == nullptr) ||
        request->atoms.num_global_inputs < 0 ||
        (request->atoms.num_global_inputs > 0 &&
         request->atoms.global_inputs == nullptr) ||
        request->num_outputs < 0 ||
        (request->num_outputs > 0 && request->outputs == nullptr)) {
      SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
                "request contains an invalid count/table pair");
      return;
    }
    std::string buffer_error;
    if (!RequestBuffersBelongToExecutor(executor, request, &buffer_error)) {
      SetStatus(status, JCN_STATUS_FAILED_PRECONDITION, buffer_error);
      return;
    }
    jcn_force_request effective_request = *request;
    effective_request.capacities = executor->capacities.max_atoms > 0
                                       ? executor->capacities
                                       : request->capacities;

    executor->runner->compute_forces(effective_request, *result);
    SetStatus(status, JCN_STATUS_OK);
  } catch (const std::exception& e) {
    if (result != nullptr) result->code = JCN_COMPUTE_FAILED;
    SetStatus(status, JCN_STATUS_INTERNAL, e.what());
  }
}

// ---------------------------------------------------------------------------
// DLPack buffer ownership.
// ---------------------------------------------------------------------------

jcn_buffer* BufferFromDlpackOptions(jcn_executor* executor,
                                    DLManagedTensor* tensor,
                                    const jcn_buffer_import_options* options,
                                    jcn_status* status) {
  if (executor == nullptr || executor->client == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT, "executor is null");
    return nullptr;
  }
  if (options == nullptr) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "DLPack import options must be provided");
    return nullptr;
  }
  if (options->copy_policy != JCN_DLPACK_COPY_IF_NEEDED &&
      options->copy_policy != JCN_DLPACK_NEVER_COPY &&
      options->copy_policy != JCN_DLPACK_ALWAYS_COPY) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "DLPack import copy policy is invalid");
    return nullptr;
  }
  if (options->role != JCN_BUFFER_ROLE_INPUT &&
      options->role != JCN_BUFFER_ROLE_OUTPUT &&
      options->role != JCN_BUFFER_ROLE_INPUT_OUTPUT) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT,
              "DLPack import buffer role is invalid");
    return nullptr;
  }
  std::string error;
  if (!ValidateDlpackTensor(tensor, &error)) {
    SetStatus(status, JCN_STATUS_INVALID_ARGUMENT, error);
    return nullptr;
  }
  const bool host_tensor = tensor->dl_tensor.device.device_type == kDLCPU;
  if (host_tensor && options->copy_policy == JCN_DLPACK_NEVER_COPY) {
    SetStatus(status, JCN_STATUS_FAILED_PRECONDITION,
              "CPU DLPack imports use host staging and cannot satisfy "
              "JCN_DLPACK_NEVER_COPY");
    return nullptr;
  }
  if (!host_tensor && options->copy_policy == JCN_DLPACK_ALWAYS_COPY) {
    SetStatus(status, JCN_STATUS_FAILED_PRECONDITION,
              "Device DLPack imports do not implement "
              "JCN_DLPACK_ALWAYS_COPY");
    return nullptr;
  }
  if (!DeviceMatchesBackend(*executor->client, tensor, &error)) {
    SetStatus(status, JCN_STATUS_FAILED_PRECONDITION, error);
    return nullptr;
  }
  if (options->backend != nullptr &&
      Lower(options->backend) != Lower(executor->client->config.backend)) {
    SetStatus(status, JCN_STATUS_FAILED_PRECONDITION,
              "DLPack import backend option does not match executor backend");
    return nullptr;
  }
  if (options->device_ordinal >= 0 &&
      options->device_ordinal != executor->client->config.device) {
    SetStatus(status, JCN_STATUS_FAILED_PRECONDITION,
              "DLPack import device option does not match executor device");
    return nullptr;
  }
  auto* buffer = new (std::nothrow) jcn_buffer;
  if (buffer == nullptr) {
    SetStatus(status, JCN_STATUS_INTERNAL, "failed to allocate buffer");
    return nullptr;
  }
  buffer->executor = executor;
  buffer->tensor = tensor;
  buffer->options = *options;
  SetStatus(status, JCN_STATUS_OK);
  return buffer;
}

void BufferDestroy(jcn_buffer* buffer) {
  if (buffer == nullptr) return;
  if (buffer->tensor != nullptr && buffer->tensor->deleter != nullptr) {
    buffer->tensor->deleter(buffer->tensor);
  }
  delete buffer;
}

// ---------------------------------------------------------------------------
// API table export.
// ---------------------------------------------------------------------------

const JCN_Api kApi = {
    JCN_API_VERSION,
    sizeof(JCN_Api),
    &GetRuntimeInfo,
    &ClientCreate,
    &ClientDestroy,
    &ModelLoadFromProtobuf,
    &ModelDestroy,
    &ExecutorCreate,
    &ExecutorDestroy,
    &ExecutorSetModelOptions,
    &ExecutorSetEngineAbiOptions,
    &ExecutorSetCommunicationCallbacks,
    &ExecutorSetCapacities,
    &ExecutorComputeForces,
    &NeighborListCreateSimpleSparse,
    &NeighborListCreateSimpleDense,
    &NeighborListFormat,
    &NeighborListDestroy,
    &BufferFromDlpackOptions,
    &BufferDestroy,
};

}  // namespace

extern "C" const JCN_Api* jcn_get_api(uint32_t requested_version) {
  if (requested_version != JCN_API_VERSION) {
    return nullptr;
  }
  return &kApi;
}
