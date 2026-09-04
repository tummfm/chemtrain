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

#include "connector/runner.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "connector/compiler.h"
#include "connector/dlpack_validation.h"
#include "connector/runtime_loader.h"
#include "connector/ffi/feature_exchange_registration.h"
#include "connector/jcn_buffer_internal.h"
#include "connector/jcn_neighbor_internal.h"
#include "connector/model.pb.h"
#include "connector/model_parser.h"
#include "connector/model_shape.h"
#include "connector/pjrt/buffers.h"
#include "connector/utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/dump.h"

namespace jcn {

// initialize() loads PJRT plugins and registers the internal and external XLA
// FFI handlers. Runner construction then creates the PJRT client and selects an
// addressable device. load_model() prepares the exported model, compiler, and
// neighbor-list state. compute_forces() validates its inputs, executes PJRT,
// services communication callbacks, and reports capacity growth requests.

void Runner::initialize() {
  absl::InitializeLog();

  Logger logger = Logger::getlogger();

  const std::string pjrt_directory = PjrtDirectory();
  const std::vector<std::string> ffi_directories = FfiProviderDirectories();

  std::cout << "[JCN] PJRT plugin directory: " << pjrt_directory << std::endl;
  for (const std::string& directory : ffi_directories) {
    std::cout << "[JCN] FFI provider directory: " << directory << std::endl;
  }

  try {
    struct stat st;
    if (stat(pjrt_directory.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
      throw std::runtime_error("Invalid PJRT plugin directory: " +
                               pjrt_directory);
    }

    DIR* dir = opendir(pjrt_directory.c_str());
    if (!dir) {
      throw std::runtime_error("Failed to open PJRT plugin directory: " +
                               pjrt_directory);
    }

    std::vector<std::string> backends;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
      if (entry->d_name[0] == '.') continue;

      std::string backend(entry->d_name);
      std::string backend_dir = pjrt_directory + "/" + backend;

      struct stat backend_st;
      if (stat(backend_dir.c_str(), &backend_st) != 0 ||
          !S_ISDIR(backend_st.st_mode)) {
        continue;
      }

      backends.push_back(std::move(backend));
    }
    closedir(dir);
    std::sort(backends.begin(), backends.end());

    for (const std::string& backend : backends) {
      const std::string backend_dir = pjrt_directory + "/" + backend;

      std::string plugin_path = backend_dir + "/pjrt_plugin.so";
      if (access(plugin_path.c_str(), R_OK) != 0) {
        continue;
      }

      absl::StatusOr<const PJRT_Api*> status_or_api =
          pjrt::LoadPjrtPlugin(backend, plugin_path);

      if (status_or_api.ok()) {
        logger.log(LogLevel::INFO, "Loaded PJRT plugin " + backend);
        if (backend == "cuda" || backend == "cpu") {
          const XLA_FFI_Api* ffi_api = GetPjrtFfiApi(plugin_path);
          if (RegisterCommunicationFfi(ffi_api, backend) != 0) {
            throw std::runtime_error(
                "Failed to register chemtrain communication XLA FFI handlers "
                "for " + backend);
          }
          RegisterFfiProviders(ffi_api, backend, ffi_directories);
        }
      } else {
        std::cerr << "Failed to load PJRT plugin " << backend << ": "
                  << status_or_api.status().ToString() << std::endl;
      }
    }

  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to load PJRT plugins: ") +
                             e.what());
  }
}

std::unique_ptr<xla::PjRtClient> CreatePjRtClient(const ConnectorConfig& config,
                                                  int* pjrt_device_index) {
  Logger logger = Logger::getlogger();

  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or_status;

  logger.log(LogLevel::INFO, "Initializing PjRtClient for backend '" +
                                 config.backend + "' with options:");
  logger.log(LogLevel::INFO, "  - Device: " + std::to_string(config.device));
  logger.log(LogLevel::INFO,
             "  - Memory fraction: " + std::to_string(config.memory_fraction));

  absl::flat_hash_map<std::string, xla::PjRtValueType> create_options;
  if (config.backend == "cuda") {
    create_options = {
        {"memory_fraction", static_cast<float>(config.memory_fraction)},
        {"visible_devices", std::vector<int64_t>({config.device})},
    };
  }

  // Using the dynamically loaded PJRT C API for CPU and CUDA keeps
  // CPU-only connectors independent of the statically linked TFRT CPU
  // client and gives both backends the same plugin discovery path.
  absl::StatusOr<bool> status_or_success =
      pjrt::IsPjrtPluginInitialized(config.backend);
  if (!status_or_success.ok()) {
    throw std::runtime_error("Failed to initialize PjRtClient: " +
                             status_or_success.status().ToString());
  }

  if (!status_or_success.value()) {
    absl::Status status = pjrt::InitializePjrtPlugin(config.backend);
    if (!status.ok()) {
      throw std::runtime_error("Failed to initialize PjRtClient: " +
                               status.ToString());
    }
  }

  client_or_status = xla::GetCApiClient(config.backend, create_options);

  if (!client_or_status.ok()) {
    throw std::runtime_error("Failed to initialize PjRtClient: " +
                             client_or_status.status().ToString());
  }

  std::unique_ptr<xla::PjRtClient> client = std::move(client_or_status).value();

  // Determine the index into addressable_devices() to use for buffer
  // allocation. When visible_devices filtering is applied,
  // addressable_devices() may be remapped.
  int selected_device_index = -1;
  absl::Span<xla::PjRtDevice* const> addressable =
      client->addressable_devices();
  if (addressable.empty()) {
    throw std::runtime_error("PjRtClient has no addressable devices");
  }

  for (int i = 0; i < addressable.size(); ++i) {
    if (addressable[i]->id() == config.device) {
      selected_device_index = i;
      break;
    }
  }
  if (selected_device_index < 0) {
    throw std::runtime_error(
        "Requested PJRT device id " + std::to_string(config.device) +
        " is not addressable for backend '" + config.backend + "'");
  }

  logger.log(
      LogLevel::INFO,
      "Using addressable device index " +
          std::to_string(selected_device_index) +
          " (requested id=" + std::to_string(config.device) + ", actual id=" +
          std::to_string(addressable[selected_device_index]->id()) + ")");
  if (pjrt_device_index != nullptr) *pjrt_device_index = selected_device_index;

  absl::Span<xla::PjRtDevice* const> devices = client->devices();
  std::string device_list = "";
  for (int i = 0; i < devices.size(); i++) {
    device_list += std::string(devices[i]->ToString()) + ",";
  }
  logger.log(LogLevel::INFO, "Found devices [" + device_list + "]");

  return client;
}

Runner::Runner(ConnectorConfig connector_config, bool initialize)
    : config(std::move(connector_config)) {
  if (initialize) {
    Runner::initialize();
  }
  owned_client = CreatePjRtClient(config, &pjrt_device_index_);
  client = owned_client.get();
}

Runner::Runner(ConnectorConfig connector_config, bool initialize,
               xla::PjRtClient* shared_client, int pjrt_device_index)
    : config(std::move(connector_config)),
      client(shared_client),
      pjrt_device_index_(pjrt_device_index) {
  if (initialize) {
    Runner::initialize();
  }
  if (client == nullptr) {
    throw std::runtime_error("Runner shared PJRT client is null");
  }
}

// -----------------------------------------------------------------------
// Model selection and runtime setup.
// -----------------------------------------------------------------------

void Runner::configure_selected_model(SelectedModel selected) {
  // Store the selected executable and its input fields.
  selected_platforms_ = std::move(selected.platforms);
  model = std::make_unique<jcn::Model>(std::move(selected.model));
  particle_fields_.clear();
  for (const auto& field : model->particle_fields()) {
    particle_fields_.push_back({field.name(), TensorDtype::S32});
  }
  global_fields_.clear();
  for (const auto& field : model->global_fields()) {
    TensorDtype dtype = TensorDtype::ModelDefault;
    switch (field.dtype()) {
      case Model::GLOBAL_FLOAT32:
        dtype = TensorDtype::F32;
        break;
      case Model::GLOBAL_FLOAT64:
        dtype = TensorDtype::F64;
        break;
      case Model::GLOBAL_INT32:
        dtype = TensorDtype::S32;
        break;
      default:
        throw std::runtime_error(
            "Selected model has unsupported global field dtype.");
    }
    global_fields_.push_back({field.name(), dtype});
  }

  output_fields_.clear();
  for (const auto& field : model->output_fields()) {
    ModelProperties::OutputScope scope;
    switch (field.scope()) {
      case Model::PARTICLE:
        scope = ModelProperties::OutputScope::PARTICLE;
        break;
      case Model::LOCAL:
        scope = ModelProperties::OutputScope::LOCAL;
        break;
      case Model::GLOBAL:
        scope = ModelProperties::OutputScope::GLOBAL;
        break;
      default:
        throw std::runtime_error(
            "Selected model has unsupported output scope.");
    }
    output_fields_.push_back({field.name(), scope,
                              std::vector<int64_t>(field.dimensions().begin(),
                                                   field.dimensions().end()),
                              static_cast<int>(field.components()),
                              field.extensive()});
  }
  include_pair_type_ = model->neighbor_list().include_pair_type();
}

void Runner::configure_neighbor_list(const ModelConfig& config) {
  std::vector<std::string> statistics_keys;
  for (const auto& key : model->neighbor_list().statistics_keys()) {
    statistics_keys.push_back(key);
  }
  switch (model->neighbor_list().type()) {
    case jcn::Model::SIMPLE_SPARSE:
      neighbor_list = std::make_unique<SimpleSparseNeighborList>(
          statistics_keys, include_pair_type_);
      neighbor_list->initialize(config.neighbor_list_multipliers);
      Logger::getlogger().log(LogLevel::INFO,
                              "Initialize SimpleSparseNeighborList");
      break;
    case jcn::Model::SIMPLE_DENSE:
      neighbor_list = std::make_unique<SimpleDenseNeighborList>(
          statistics_keys, include_pair_type_);
      neighbor_list->initialize(config.neighbor_list_multipliers);
      Logger::getlogger().log(LogLevel::INFO,
                              "Initialize SimpleDenseNeighborList");
      break;
    case jcn::Model::DEVICE_SPARSE:
      throw std::runtime_error(
          "DeviceSparseNeighborList is not supported by the current "
          "engine adapter. Re-export the model with a host "
          "neighbor-list variant.");
    default:
      throw std::runtime_error("Unknown neighbor list type: " +
                               std::to_string(model->neighbor_list().type()));
  }
}

ModelProperties Runner::load_model(ModelConfig config) {
  Logger logger = Logger::getlogger();

  // Save the Newton, communication, and dtype policies used by graph
  // construction and compilation.
  newton = config.newton;
  communication_callbacks = config.communication;
  engine_abi_ = config.engine_abi;

  // Select the executable whose communication and Newton policies match the
  // engine. Selection copies its metadata into one normalized model view used
  // by compilation, graph construction, and runtime properties. A request
  // must never fall back to a variant with different force semantics.
  // Reverse halo exchange returns ghost-row contributions to their owning
  // ranks, so the current communication ABI requires Newton-on ownership.
  if (config.use_communication && !config.newton) {
    throw std::runtime_error(
        "Communication requires Newton pair forces. Use 'newton on' "
        "or select 'comm off' for the standard Newton-off model.");
  }

  // Select and configure the requested model variant.
  if (config.model == nullptr) {
    throw std::runtime_error("Cannot load model: parsed model is null.");
  }
  SelectedModel selected = SelectModelVariant(
      *config.model, config.use_communication, config.newton);
  configure_selected_model(std::move(selected));
  if (std::find(selected_platforms_.begin(), selected_platforms_.end(),
                this->config.backend) == selected_platforms_.end()) {
    throw std::runtime_error(
        "Selected model variant does not contain backend '" +
        this->config.backend + "'. Re-export the model for that backend.");
  }

  logger.log(LogLevel::DEBUG,
             "Model communication: enabled=" +
                 std::to_string(model->uses_communication()) + ", width=" +
                 std::to_string(model->communication_buffer_width()) +
                 ", newton=" + std::to_string(newton) + ", neighbor_order=" +
                 (model->neighbor_list().nbr_order_size() > 0
                      ? std::to_string(model->neighbor_list().nbr_order(0))
                      : "missing"));

  // Prepare compilation and graph storage.
  compiler = std::make_unique<Compiler>(
      model->mlir_module_serialized(),
      static_cast<int>(model->calling_convention_version()),
      model->communication_buffer_width(), selected_platforms_,
      this->config.backend);

  configure_neighbor_list(config);

  return get_model_properties();
}

jcn_neighbor_format Runner::selected_neighbor_format() const {
  if (!model) return JCN_NEIGHBOR_UNSPECIFIED;
  switch (model->neighbor_list().type()) {
    case jcn::Model::SIMPLE_DENSE:
      return JCN_NEIGHBOR_SIMPLE_DENSE;
    case jcn::Model::SIMPLE_SPARSE:
      return JCN_NEIGHBOR_SIMPLE_SPARSE;
    default:
      return JCN_NEIGHBOR_UNSPECIFIED;
  }
}

// -----------------------------------------------------------------------
// Force execution phases.
// -----------------------------------------------------------------------

bool Runner::needs_compilation(const jcn_force_request& request,
                               const GraphInputSpec& graph_spec) const {
  return !executable || request.clear_capacities != 0 ||
         compiled_max_atoms_ != request.capacities.max_atoms ||
         compiled_graph_inputs_ != graph_spec.inputs ||
         compiled_neighbor_format_ != request.neighbors->format ||
         compiled_engine_abi_ != engine_abi_;
}

void Runner::compile_for_request(const jcn_force_request& request,
                                 const GraphInputSpec& graph_spec,
                                 jcn_force_result& result) {
  const bool initial_compilation = !executable;
  const bool atom_capacity_changed =
      !initial_compilation &&
      compiled_max_atoms_ != request.capacities.max_atoms;
  const bool edge_capacity_changed =
      !initial_compilation && compiled_graph_inputs_ != graph_spec.inputs;

  // Collect the particle ABI fields.
  std::vector<std::string> particle_names;
  std::vector<xla::PrimitiveType> particle_types;
  for (const auto& field : particle_fields_) {
    particle_names.push_back(field.name);
    switch (field.dtype) {
      case TensorDtype::S32:
        particle_types.push_back(xla::S32);
        break;
      default:
        throw std::runtime_error("Unsupported compiled particle field dtype.");
    }
  }
  // Collect the global ABI fields.
  std::vector<std::string> global_names;
  std::vector<xla::PrimitiveType> global_types;
  for (const auto& field : global_fields_) {
    global_names.push_back(field.name);
    switch (field.dtype) {
      case TensorDtype::F32:
        global_types.push_back(xla::F32);
        break;
      case TensorDtype::F64:
        global_types.push_back(xla::F64);
        break;
      case TensorDtype::S32:
        global_types.push_back(xla::S32);
        break;
      default:
        throw std::runtime_error("Unsupported compiled global field dtype.");
    }
  }
  // Compile and load the capacity-specific executable.
  compiler->compile(static_cast<int>(request.capacities.max_atoms),
                    graph_spec.inputs, particle_types, particle_names,
                    global_types, global_names, engine_abi_, output_fields_);
  xla::MaybeOwningMlirModule module(compiler->module());
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> loaded =
      client->CompileAndLoad(std::move(module), compile_options);
  if (!loaded.ok()) {
    throw std::runtime_error("Failed to compile: " +
                             loaded.status().ToString());
  }
  // Record the inputs that define the compiled executable.
  executable = std::move(loaded).value();
  compiled_max_atoms_ = request.capacities.max_atoms;
  compiled_graph_inputs_ = graph_spec.inputs;
  compiled_neighbor_format_ = request.neighbors->format;
  compiled_engine_abi_ = engine_abi_;
  result.compilations = 1;

  // Regression tooling consumes this stable record. Keep the fields numeric
  // so records from independent MPI ranks can be aggregated directly.
  Logger::getlogger().log(
      LogLevel::INFO,
      "JCN_STATS compilation initial=" +
          std::to_string(initial_compilation ? 1 : 0) + " atom=" +
          std::to_string(atom_capacity_changed ? 1 : 0) + " edge=" +
          std::to_string(edge_capacity_changed ? 1 : 0));
}

Runner::ExecutionResults Runner::execute_prepared_inputs(
    const std::vector<std::vector<xla::PjRtBuffer*>>& arg_handles,
    int64_t owned_atoms, int64_t requested_atoms) {
  const bool uses_communication = model->uses_communication();
  xla::ExecuteContext execute_context;
  xla::ExecuteOptions execute_options;
  CommunicationContext communication_context(
      communication_callbacks, uses_communication, &communication_workspace_,
      owned_atoms, requested_atoms, model->communication_buffer_width());
  HostCommunicationContext host_communication_context{&communication_context};
  CudaCommunicationContext cuda_communication_context{&communication_context};
  if (uses_communication) {
    absl::Status status =
        config.backend == "cpu"
            ? AddHostCommunicationContextToExecuteContext(
                  &execute_context, &host_communication_context)
            : AddCudaCommunicationContextToExecuteContext(
                  &execute_context, &cuda_communication_context);
    if (!status.ok()) {
      throw std::runtime_error(
          "Failed to initialize communication execution context: " +
          status.ToString());
    }
    execute_options.context = &execute_context;
    communication_context.BeginExecution();
  }

  std::future<absl::StatusOr<ExecutionResults>> future_results =
      std::async(std::launch::async, [&]() {
        try {
          auto results = executable->Execute(
              absl::Span<const std::vector<xla::PjRtBuffer*>>(arg_handles),
              execute_options);
          if (results.ok()) {
            std::vector<xla::Future<>> ready_futures;
            for (const auto& replica : results.value()) {
              for (const auto& buffer : replica) {
                ready_futures.push_back(buffer->GetReadyFuture());
              }
            }
            absl::Status ready =
                xla::JoinFutures(absl::MakeConstSpan(ready_futures)).Await();
            if (!ready.ok()) {
              results = decltype(results)(ready);
            }
          }
          if (uses_communication) {
            communication_context.NotifyExecutionComplete();
          }
          return results;
        } catch (...) {
          if (uses_communication) {
            communication_context.NotifyExecutionComplete();
          }
          throw;
        }
      });
  if (uses_communication) {
    communication_context.ServiceUntilExecutionComplete();
  }

  absl::StatusOr<ExecutionResults> results = future_results.get();
  if (!results.ok()) {
    throw std::runtime_error("Failed to execute: " +
                             results.status().ToString());
  }
  if (uses_communication) {
    absl::Status status = communication_context.ValidateExecution();
    if (!status.ok()) {
      throw std::runtime_error("Communication execution validation failed: " +
                               status.ToString());
    }
  }
  return std::move(results).value();
}

bool Runner::evaluate_statistics(ExecutionResults& results,
                                 const jcn_force_request& request,
                                 jcn_force_result& result) {
  std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics;
  const int offset = static_cast<int>(output_fields_.size());
  for (int i = 0; i < neighbor_list->statistics_keys.size(); ++i) {
    statistics.emplace(neighbor_list->statistics_keys[i],
                       std::move(results[0][i + offset]));
  }
  if (neighbor_list->evaluate_statistics(std::move(statistics),
                                         request.allow_internal_recompile != 0,
                                         request.capacities)) {
    return true;
  }
  result.code = JCN_COMPUTE_NEEDS_CAPACITY_CHANGE;
  result.required_capacities =
      neighbor_list->requested_capacities(request.capacities);
  return false;
}

Runner::ValidatedRequest Runner::validate_request(
    const jcn_force_request& request, jcn_force_result& result) const {
  // Validate the model and named input tables.
  ValidatedRequest validated;
  if (!model) {
    throw std::runtime_error("Model is not initialized");
  }
  if (request.atoms.positions == nullptr) {
    throw std::runtime_error(
        "JCN atom request is missing required DLPack buffers.");
  }
  if (request.atoms.num_particle_inputs < 0 ||
      (request.atoms.num_particle_inputs > 0 &&
       request.atoms.particle_inputs == nullptr)) {
    throw std::runtime_error(
        "JCN atom request has an invalid named particle input table.");
  }
  for (int i = 0; i < request.atoms.num_particle_inputs; ++i) {
    const jcn_named_input& input = request.atoms.particle_inputs[i];
    if (input.name == nullptr || input.name[0] == '\0' ||
        input.buffer == nullptr ||
        !validated.particle_inputs.emplace(input.name, input.buffer).second) {
      throw std::runtime_error(
          "JCN named particle inputs contain an empty, "
          "duplicate, or null entry.");
    }
  }
  if (validated.particle_inputs.size() != particle_fields_.size()) {
    throw std::runtime_error(
        "JCN named particle inputs do not match the selected model fields.");
  }
  for (const auto& field : particle_fields_) {
    if (validated.particle_inputs.find(field.name) ==
        validated.particle_inputs.end()) {
      throw std::runtime_error(
          "JCN atom request is missing named particle input '" + field.name +
          "'.");
    }
  }
  if (request.atoms.num_global_inputs < 0 ||
      (request.atoms.num_global_inputs > 0 &&
       request.atoms.global_inputs == nullptr)) {
    throw std::runtime_error(
        "JCN atom request has an invalid named global input table.");
  }
  for (int i = 0; i < request.atoms.num_global_inputs; ++i) {
    const jcn_named_input& input = request.atoms.global_inputs[i];
    if (input.name == nullptr || input.name[0] == '\0' ||
        input.buffer == nullptr ||
        !validated.global_inputs.emplace(input.name, input.buffer).second) {
      throw std::runtime_error(
          "JCN named global inputs contain an empty, "
          "duplicate, or null entry.");
    }
  }
  if (validated.global_inputs.size() != global_fields_.size()) {
    throw std::runtime_error(
        "JCN named global inputs do not match the selected model fields.");
  }
  for (const auto& field : global_fields_) {
    if (validated.global_inputs.find(field.name) ==
        validated.global_inputs.end()) {
      throw std::runtime_error(
          "JCN atom request is missing named global input '" + field.name +
          "'.");
    }
  }

  // Validate atom counts, capacities, and neighbor storage.
  validated.owned_atoms = request.atoms.num_owned_atoms;
  validated.requested_atoms = request.atoms.num_valid_atoms;
  if (validated.owned_atoms < 0 ||
      validated.requested_atoms < validated.owned_atoms) {
    throw std::runtime_error(
        "JCN atom request has invalid owned/valid atom counts.");
  }
  if (request.capacities.max_atoms < validated.requested_atoms) {
    result.code = JCN_COMPUTE_NEEDS_CAPACITY_CHANGE;
    result.required_capacities = request.capacities;
    result.required_capacities.max_atoms = validated.requested_atoms;
    validated.needs_capacity_change = true;
    return validated;
  }
  if (request.neighbors == nullptr) {
    throw std::runtime_error("JCN request is missing neighbor-list handle.");
  }
  switch (request.neighbors->format) {
    case JCN_NEIGHBOR_SIMPLE_SPARSE:
      if (request.neighbors->sparse.senders == nullptr ||
          request.neighbors->sparse.receivers == nullptr) {
        throw std::runtime_error(
            "Sparse JCN request is missing neighbor buffers.");
      }
      if ((request.neighbors->sparse.pair_type != nullptr) !=
          include_pair_type_) {
        throw std::runtime_error(
            "Sparse pair_type buffer presence does not match model metadata.");
      }
      break;
    case JCN_NEIGHBOR_SIMPLE_DENSE:
      if (request.neighbors->dense.neighbors == nullptr) {
        throw std::runtime_error(
            "Dense JCN request is missing neighbor buffers.");
      }
      if ((request.neighbors->dense.pair_type != nullptr) !=
          include_pair_type_) {
        throw std::runtime_error(
            "Dense pair_type buffer presence does not match model metadata.");
      }
      break;
    default:
      throw std::runtime_error("Unsupported JCN neighbor format.");
  }

  validated.max_atoms = request.capacities.max_atoms;
  validated.raw_edges = request.capacities.raw_edges;
  validated.max_neighbors =
      std::max<int64_t>(request.capacities.max_neighbors_per_atom, 1);
  validated.graph_spec = neighbor_list->input_spec(request.capacities);

  // Validate model inputs and select one staging path.
  const xla::PrimitiveType position_type =
      PrimitiveForDtype(engine_abi_.position_dtype, xla::F32);
  const xla::PrimitiveType species_type =
      PrimitiveForDtype(engine_abi_.species_dtype, xla::S32);
  if (position_type != xla::F32 && position_type != xla::F64) {
    throw std::runtime_error("Engine ABI position dtype must be f32 or f64.");
  }
  if (species_type != xla::S32) {
    throw std::runtime_error("Engine ABI species dtype must be s32.");
  }
  ValidateDlpackInput(request.atoms.positions, {validated.max_atoms, 3},
                      DlpackTypeForPrimitive(position_type), "positions");

  const bool inputs_are_cpu = IsCpuDlpack(request.atoms.positions);
  auto require_same_staging = [&](const jcn_buffer* buffer, const char* role) {
    if (IsCpuDlpack(buffer) != inputs_are_cpu) {
      throw std::runtime_error(
          std::string(role) +
          " and positions must use the same host/device staging path.");
    }
  };
  for (const auto& field : particle_fields_) {
    const std::string role = "particle field '" + field.name + "'";
    jcn_buffer* buffer = validated.particle_inputs.at(field.name);
    const xla::PrimitiveType type =
        field.name == "species" ? species_type
                                : PrimitiveForDtype(field.dtype, xla::S32);
    ValidateDlpackInput(buffer, {validated.max_atoms},
                        DlpackTypeForPrimitive(type), role.c_str());
    require_same_staging(buffer, role.c_str());
  }
  for (const auto& field : global_fields_) {
    const std::string role = "global field '" + field.name + "'";
    jcn_buffer* buffer = validated.global_inputs.at(field.name);
    const xla::PrimitiveType type = PrimitiveForDtype(field.dtype, xla::F32);
    if (type != xla::F32 && type != xla::F64 && type != xla::S32) {
      throw std::runtime_error("Global field '" + field.name +
                               "' has unsupported dtype.");
    }
    ValidateDlpackInput(buffer, {}, DlpackTypeForPrimitive(type), role.c_str());
    require_same_staging(buffer, role.c_str());
  }

  // Match runtime neighbor buffers with graph descriptors.
  std::vector<std::pair<jcn_buffer*, const char*>> graph_buffers;
  if (request.neighbors->format == JCN_NEIGHBOR_SIMPLE_SPARSE) {
    graph_buffers.emplace_back(request.neighbors->sparse.senders,
                               "sparse senders");
    graph_buffers.emplace_back(request.neighbors->sparse.receivers,
                               "sparse receivers");
    if (include_pair_type_) {
      graph_buffers.emplace_back(request.neighbors->sparse.pair_type,
                                 "sparse pair_type");
    }
  } else {
    graph_buffers.emplace_back(request.neighbors->dense.neighbors,
                               "dense neighbors");
    if (include_pair_type_) {
      graph_buffers.emplace_back(request.neighbors->dense.pair_type,
                                 "dense pair_type");
    }
  }
  size_t graph_buffer_index = 0;
  for (const auto& input : validated.graph_spec.inputs) {
    if (input.kind != GraphInputKind::BUFFER) continue;
    if (graph_buffer_index >= graph_buffers.size()) {
      throw std::runtime_error(
          "Model graph descriptors exceed runtime neighbor buffers.");
    }
    const auto& [buffer, role] = graph_buffers[graph_buffer_index++];
    ValidateDlpackInput(buffer, input.shape, DlpackTypeForPrimitive(input.type),
                        role);
    require_same_staging(buffer, role);
  }
  if (graph_buffer_index != graph_buffers.size()) {
    throw std::runtime_error(
        "Runtime neighbor buffers exceed model graph descriptors.");
  }

  // Validate requested outputs against the model ABI.
  if (request.num_outputs < 0 ||
      (request.num_outputs > 0 && request.outputs == nullptr)) {
    throw std::runtime_error(
        "JCN force request has invalid named output table.");
  }
  std::map<std::string, const ModelProperties::OutputField*> fields;
  for (const auto& field : output_fields_) fields.emplace(field.name, &field);
  auto force = fields.find("F");
  if (force == fields.end() ||
      force->second->scope != ModelProperties::OutputScope::PARTICLE ||
      force->second->dimensions != std::vector<int64_t>{3}) {
    throw std::runtime_error(
        "Model does not declare required particle output 'F' with shape [3].");
  }
  std::set<std::string> requested_names;
  for (int i = 0; i < request.num_outputs; ++i) {
    const jcn_named_output& output = request.outputs[i];
    if (output.name == nullptr || output.name[0] == '\0' ||
        output.buffer == nullptr ||
        !requested_names.insert(output.name).second) {
      throw std::runtime_error(
          "JCN force request has an empty, duplicate, or null output entry.");
    }
    auto field_it = fields.find(output.name);
    if (field_it == fields.end()) {
      throw std::runtime_error(
          std::string("Model does not declare requested output '") +
          output.name + "'.");
    }
    const auto& field = *field_it->second;
    std::vector<int64_t> buffer_shape = field.dimensions;
    if (field.scope == ModelProperties::OutputScope::PARTICLE) {
      if (output.rows < 0 || output.rows > validated.max_atoms) {
        throw std::runtime_error(std::string("Particle output '") +
                                 output.name + "' has invalid row count.");
      }
      if (field.name == "F") {
        const int64_t expected_force_rows =
            newton ? validated.requested_atoms : validated.owned_atoms;
        if (output.rows != expected_force_rows) {
          throw std::runtime_error(
              "Particle output 'F' row count does not match the selected "
              "Newton variant.");
        }
      } else if (field.name == "U" &&
                 output.rows != validated.owned_atoms) {
        throw std::runtime_error(
            "Particle output 'U' requires one row per owned atom.");
      }
      buffer_shape.insert(buffer_shape.begin(), validated.max_atoms);
    } else if (output.rows != 1) {
      throw std::runtime_error(std::string("Configuration output '") +
                               output.name +
                               "' requires exactly one logical record.");
    }
    TensorDtype dtype = engine_abi_.default_output_dtype;
    for (const auto& entry : engine_abi_.output_dtypes) {
      if (entry.first == field.name) dtype = entry.second;
    }
    const xla::PrimitiveType output_type = PrimitiveForDtype(dtype, xla::F32);
    if (output_type != xla::F32 && output_type != xla::F64) {
      throw std::runtime_error(std::string("Output '") + output.name +
                               "' requires a floating-point engine ABI dtype.");
    }
    ValidateDlpackInput(output.buffer, buffer_shape,
                        DlpackTypeForPrimitive(output_type), output.name);
    require_same_staging(output.buffer, output.name);
  }
  if (requested_names.find("F") == requested_names.end()) {
    throw std::runtime_error("JCN force request is missing named output 'F'.");
  }
  if (requested_names.find("U") == requested_names.end()) {
    throw std::runtime_error("JCN force request is missing named output 'U'.");
  }
  if (requested_names.find("V") == requested_names.end()) {
    throw std::runtime_error("JCN force request is missing named output 'V'.");
  }
  validated.use_device_inputs = !inputs_are_cpu;
  return validated;
}

Runner::PreparedInputs Runner::prepare_inputs(const jcn_force_request& request,
                                              const ValidatedRequest& validated,
                                              bool needs_compile) {
  const auto& particle_inputs = validated.particle_inputs;
  const auto& global_inputs = validated.global_inputs;
  const int64_t owned_atoms = validated.owned_atoms;
  const int64_t requested_atoms = validated.requested_atoms;
  const int64_t max_atoms = validated.max_atoms;
  const int64_t raw_edges = validated.raw_edges;
  const int64_t max_neighbors = validated.max_neighbors;
  const size_t expected_graph_buffers = static_cast<size_t>(std::count_if(
      validated.graph_spec.inputs.begin(), validated.graph_spec.inputs.end(),
      [](const GraphInputDescriptor& input) {
        return input.kind == GraphInputKind::BUFFER;
      }));

  // Host buffers are copied into connector-owned literals. Device buffers
  // become non-owning same-device PJRT views, so caller memory and
  // producer streams must remain valid until execution completes. Scalar
  // counts are copied as literals in both paths.
  std::vector<std::unique_ptr<xla::PjRtBuffer>> input_buffers;
  std::vector<xla::PjRtBuffer*> buffer_ptrs;
  std::vector<std::unique_ptr<xla::Literal>> literals;
  const bool use_device_inputs = !IsCpuDlpack(request.atoms.positions);
  const xla::PrimitiveType position_type =
      PrimitiveForDtype(engine_abi_.position_dtype, xla::F32);
  const xla::PrimitiveType species_type =
      PrimitiveForDtype(engine_abi_.species_dtype, xla::S32);
  if (position_type != xla::F32 && position_type != xla::F64) {
    throw std::runtime_error("Engine ABI position dtype must be f32 or f64.");
  }
  if (species_type != xla::S32) {
    throw std::runtime_error("Engine ABI species dtype must be s32.");
  }

  auto push_literal = [&](std::unique_ptr<xla::Literal> literal) {
    input_buffers.push_back(
        CreatePjRtBufferFromLiteral(client, pjrt_device_index_, literal.get()));
    buffer_ptrs.push_back(input_buffers.back().get());
    literals.push_back(std::move(literal));
  };

  auto make_device_view =
      [&](jcn_buffer* buffer, const std::vector<int64_t>& shape,
          xla::PrimitiveType type,
          const char* role) -> std::unique_ptr<xla::PjRtBuffer> {
    if (buffer->options.copy_policy == JCN_DLPACK_ALWAYS_COPY) {
      throw std::runtime_error(
          std::string("Device DLPack import does not implement an explicit "
                      "copy for ") +
          role + ".");
    }
    xla::Shape xla_shape = xla::ShapeUtil::MakeShape(
        type, absl::Span<const int64_t>(shape.data(), shape.size()));
    absl::StatusOr<xla::PjRtMemorySpace*> memory_space =
        client->addressable_devices()[pjrt_device_index_]
            ->default_memory_space();
    if (!memory_space.ok()) {
      throw std::runtime_error("Failed to get PJRT memory space: " +
                               memory_space.status().ToString());
    }
    std::optional<std::intptr_t> producer_stream =
        ExternalReadyStreamForProducer(
            client->addressable_devices()[pjrt_device_index_],
            reinterpret_cast<std::intptr_t>(buffer->options.producer_stream),
            role);
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> view =
        client->CreateViewOfDeviceBuffer(
            MutableDlData(buffer), xla_shape, memory_space.value(), []() {},
            producer_stream);
    if (!view.ok()) {
      throw std::runtime_error(
          std::string("Failed to create PJRT DLPack device view for ") + role +
          ": " + view.status().ToString());
    }
    return std::move(view).value();
  };

  auto push_device_view = [&](jcn_buffer* buffer,
                              const std::vector<int64_t>& shape,
                              xla::PrimitiveType type, const char* role) {
    input_buffers.push_back(make_device_view(buffer, shape, type, role));
    buffer_ptrs.push_back(input_buffers.back().get());
  };

  auto append_neighbor_inputs = [&]() {
    // Device neighbor arrays are non-owning views of adapter-managed Kokkos
    // allocations. Keep those views local to one execution so a LAMMPS
    // neighbor rebuild can replace its storage without leaving cached PJRT
    // buffers pointing at freed device memory.
    if (use_device_inputs) {
      const size_t first_graph_buffer = buffer_ptrs.size();
      if (request.neighbors->format == JCN_NEIGHBOR_SIMPLE_SPARSE) {
        push_device_view(request.neighbors->sparse.senders, {raw_edges},
                         xla::S32, "sparse senders");
        push_device_view(request.neighbors->sparse.receivers, {raw_edges},
                         xla::S32, "sparse receivers");
        if (include_pair_type_) {
          push_device_view(request.neighbors->sparse.pair_type, {raw_edges},
                           xla::S32, "sparse pair_type");
        }
      } else {
        push_device_view(request.neighbors->dense.neighbors,
                         {max_atoms, max_neighbors}, xla::S32,
                         "dense neighbors");
        if (include_pair_type_) {
          push_device_view(request.neighbors->dense.pair_type,
                           {max_atoms, max_neighbors}, xla::S32,
                           "dense pair_type");
        }
      }
      if (buffer_ptrs.size() - first_graph_buffer != expected_graph_buffers) {
        throw std::runtime_error(
            "Prepared graph buffers do not match model input descriptors.");
      }
      return;
    }

    // Host graph inputs are owned PJRT literals, so they remain safe to cache.
    // clear_neighbors refreshes their copied values after a list rebuild.
    const bool host_neighbor_cache_invalid =
        needs_compile || request.clear_neighbors != 0 ||
        host_neighbor_request_ != request.neighbors ||
        host_neighbor_max_atoms_ != max_atoms ||
        host_neighbor_raw_edges_ != raw_edges ||
        host_neighbor_max_neighbors_ != max_neighbors ||
        host_neighbor_input_format_ != request.neighbors->format;

    if (host_neighbor_cache_invalid) {
      host_neighbor_input_buffers_.clear();
      host_neighbor_input_ptrs_.clear();
      host_neighbor_literals_.clear();

      auto cache_literal = [&](std::unique_ptr<xla::Literal> literal) {
        host_neighbor_input_buffers_.push_back(CreatePjRtBufferFromLiteral(
            client, pjrt_device_index_, literal.get()));
        host_neighbor_input_ptrs_.push_back(
            host_neighbor_input_buffers_.back().get());
        host_neighbor_literals_.push_back(std::move(literal));
      };
      if (request.neighbors->format == JCN_NEIGHBOR_SIMPLE_SPARSE) {
        cache_literal(MakeIntLiteral(request.neighbors->sparse.senders,
                                     {raw_edges}, raw_edges));
        cache_literal(MakeIntLiteral(request.neighbors->sparse.receivers,
                                     {raw_edges}, raw_edges));
        if (include_pair_type_) {
          cache_literal(MakeIntLiteral(request.neighbors->sparse.pair_type,
                                       {raw_edges}, raw_edges));
        }
      } else {
        cache_literal(MakeIntLiteral(request.neighbors->dense.neighbors,
                                     {max_atoms, max_neighbors},
                                     max_atoms * max_neighbors));
        if (include_pair_type_) {
          cache_literal(MakeIntLiteral(request.neighbors->dense.pair_type,
                                       {max_atoms, max_neighbors},
                                       max_atoms * max_neighbors));
        }
      }

      host_neighbor_request_ = request.neighbors;
      host_neighbor_max_atoms_ = max_atoms;
      host_neighbor_raw_edges_ = raw_edges;
      host_neighbor_max_neighbors_ = max_neighbors;
      host_neighbor_input_format_ = request.neighbors->format;
    }

    if (host_neighbor_input_ptrs_.size() != expected_graph_buffers) {
      throw std::runtime_error(
          "Prepared graph buffers do not match model input descriptors.");
    }

    buffer_ptrs.insert(buffer_ptrs.end(), host_neighbor_input_ptrs_.begin(),
                       host_neighbor_input_ptrs_.end());
  };

  if (use_device_inputs) {
    // Device imports intentionally mirror JAX's copy=False behavior:
    // if same-device sharing cannot be honored, fail rather than
    // silently staging through host or allocating a hidden copy.
    push_device_view(request.atoms.positions, {max_atoms, 3}, position_type,
                     "positions");
    for (const auto& field : particle_fields_) {
      const xla::PrimitiveType type =
          field.name == "species" ? species_type : xla::S32;
      push_device_view(particle_inputs.at(field.name), {max_atoms}, type,
                       field.name.c_str());
    }
    for (const auto& field : global_fields_) {
      const xla::PrimitiveType type = PrimitiveForDtype(field.dtype, xla::F32);
      push_device_view(global_inputs.at(field.name), {}, type,
                       field.name.c_str());
    }
    push_literal(MakeScalarIntLiteral(static_cast<int>(owned_atoms)));
    push_literal(
        MakeScalarIntLiteral(static_cast<int>(requested_atoms - owned_atoms)));
    append_neighbor_inputs();
  } else {
    // Host DLPack follows the existing staging path: exact-ABI CPU
    // buffers are copied into PJRT buffers and outputs are copied back to
    // caller-provided DLPack output buffers after execution.
    push_literal(MakeFloatLiteral(request.atoms.positions, {max_atoms, 3},
                                  requested_atoms * 3, position_type));
    for (const auto& field : particle_fields_) {
      push_literal(MakeIntLiteral(particle_inputs.at(field.name), {max_atoms},
                                  requested_atoms));
    }
    for (const auto& field : global_fields_) {
      const xla::PrimitiveType type = PrimitiveForDtype(field.dtype, xla::F32);
      if (type == xla::S32) {
        push_literal(MakeIntLiteral(global_inputs.at(field.name), {}, 1));
      } else {
        push_literal(
            MakeFloatLiteral(global_inputs.at(field.name), {}, 1, type));
      }
    }
    push_literal(MakeScalarIntLiteral(static_cast<int>(owned_atoms)));
    push_literal(
        MakeScalarIntLiteral(static_cast<int>(requested_atoms - owned_atoms)));
    append_neighbor_inputs();
  }
  PreparedInputs prepared;
  prepared.input_buffers = std::move(input_buffers);
  prepared.buffer_ptrs = std::move(buffer_ptrs);
  prepared.literals = std::move(literals);
  prepared.argument_handles = {prepared.buffer_ptrs};
  return prepared;
}

void Runner::copy_outputs(const jcn_force_request& request,
                          ExecutionResults& result_buffers,
                          bool use_device_inputs) {
  // Index executable results by declared output name. PARTICLE outputs
  // copy a requested leading prefix. LOCAL and GLOBAL outputs copy one
  // complete configuration tensor. MPI reduction is an adapter concern:
  // the adapter reduces LOCAL values once and leaves GLOBAL values alone.
  std::map<std::string, int> output_index;
  for (int i = 0; i < static_cast<int>(output_fields_.size()); ++i) {
    output_index.emplace(output_fields_[i].name, i);
  }
  if (output_index.find("F") == output_index.end()) {
    throw std::runtime_error("Model proto does not list required output 'F'.");
  }

  // Validate a PARTICLE result and restrict its leading dimension to the
  // rows requested by the adapter.
  auto particle_copy_shape =
      [](xla::PjRtBuffer* buffer, int64_t rows,
         const std::string& name) -> std::pair<std::vector<int64_t>, int64_t> {
    std::vector<int64_t> shape(buffer->dimensions().begin(),
                               buffer->dimensions().end());
    if (shape.empty()) {
      if (rows > 1) {
        throw std::runtime_error("Scalar model output '" + name +
                                 "' cannot copy more than one row.");
      }
      return {std::vector<int64_t>{}, rows == 0 ? 0 : 1};
    }
    if (rows > shape[0]) {
      throw std::runtime_error(
          "Named output '" + name +
          "' requested more rows than the model produced.");
    }
    int64_t row_elements = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
      if (shape[i] < 0) {
        throw std::runtime_error("Named output '" + name +
                                 "' has an unsupported dynamic shape.");
      }
      row_elements *= shape[i];
    }
    shape[0] = rows;
    return {shape, rows * row_elements};
  };

  for (int i = 0; i < request.num_outputs; ++i) {
    const jcn_named_output& output = request.outputs[i];
    auto it = output_index.find(output.name);
    if (it == output_index.end()) {
      throw std::runtime_error(
          std::string("Model proto does not list requested output '") +
          output.name + "'.");
    }
    xla::PjRtBuffer* pjrt_output = result_buffers[0][it->second].get();
    const auto& field = output_fields_[it->second];
    std::vector<int64_t> shape;
    int64_t copied_values;
    if (field.scope == ModelProperties::OutputScope::PARTICLE) {
      std::tie(shape, copied_values) =
          particle_copy_shape(pjrt_output, output.rows, output.name);
    } else {
      shape.assign(pjrt_output->dimensions().begin(),
                   pjrt_output->dimensions().end());
      if (shape != field.dimensions || output.rows != 1) {
        throw std::runtime_error(std::string("Configuration output '") +
                                 output.name +
                                 "' does not match its declared dimensions.");
      }
      copied_values = field.components;
    }
    if (use_device_inputs) {
      CopyDeviceOutputToDlpack(pjrt_output, output.buffer, shape, copied_values,
                               pjrt_output->element_type(), output.name);
    } else {
      (void)shape;
      absl::StatusOr<std::shared_ptr<xla::Literal>> literal =
          pjrt_output->ToLiteralSync();
      if (!literal.ok()) {
        throw std::runtime_error(std::string("Failed to copy output '") +
                                 output.name + "' from PJRT.");
      }
      CopyFloatLiteralToDlpack(*literal.value(), output.buffer, copied_values);
    }
  }
}

void Runner::compute_forces(const jcn_force_request& request,
                            jcn_force_result& result) {
  result = jcn_force_result{};
  result.code = JCN_COMPUTE_FAILED;

  ValidatedRequest validated = validate_request(request, result);
  if (validated.needs_capacity_change) return;

  const bool needs_compile = needs_compilation(request, validated.graph_spec);
  if (needs_compile) {
    compile_for_request(request, validated.graph_spec, result);
  }
  PreparedInputs prepared = prepare_inputs(request, validated, needs_compile);
  ExecutionResults result_buffers =
      execute_prepared_inputs(prepared.argument_handles, validated.owned_atoms,
                              validated.requested_atoms);
  if (!evaluate_statistics(result_buffers, request, result)) return;
  copy_outputs(request, result_buffers, validated.use_device_inputs);

  result.code = JCN_COMPUTE_OK;
  result.flops = 0.0;
}

ModelProperties Runner::get_model_properties() {
  Logger logger = Logger::getlogger();

  ModelProperties properties;

  if (!model) {
    throw std::runtime_error("Model is not initialized");
  }

  // The exported neighbor order is the maximum graph distance needed to
  // evaluate owned outputs in the selected Newton mode. Multiplying the
  // order by the model cutoff gives the required communication distance.
  const int multiplier = model->neighbor_list().nbr_order()[0];
  properties.comm_dist = multiplier * model->neighbor_list().cutoff();
  properties.communication_buffer_width = model->communication_buffer_width();
  properties.particle_fields = particle_fields_;
  properties.global_fields = global_fields_;
  properties.outputs = output_fields_;
  properties.requires_communication = model->requires_communication();
  properties.neighbor_list.include_pair_type = include_pair_type_;

  if (model->has_unit_style()) {
    properties.unit_style = model->unit_style().c_str();
  } else {
    properties.unit_style = "real";
  }

  switch (model->neighbor_list().type()) {
    case jcn::Model::SIMPLE_SPARSE:
    case jcn::Model::SIMPLE_DENSE:
      // Neighbor list cutoff must be larger than the model cutoff
      properties.cutoff = model->neighbor_list().cutoff();

      if (multiplier > 1) {
        // Orders above one require ghost-central neighbor rows so
        // graph pruning can retain dependencies beyond the first
        // cutoff shell.
        properties.neighbor_list.include_ghosts = true;
        logger.log(LogLevel::INFO,
                   "Include ghosts: " +
                       std::to_string(properties.neighbor_list.include_ghosts));
      }
      if (model->neighbor_list().has_half_list()) {
        properties.neighbor_list.half_list = model->neighbor_list().half_list();
        logger.log(LogLevel::INFO,
                   "Use half list only " +
                       std::to_string(properties.neighbor_list.half_list));
      }

      break;
    case jcn::Model::DEVICE_SPARSE:
      throw std::runtime_error(
          "DeviceSparseNeighborList is not supported by the current "
          "engine adapter.");
  }

  logger.log(LogLevel::INFO,
             std::string("Model properties:") +
                 "\n\t-Cutoff: " + std::to_string(properties.cutoff) +
                 "\n\t-Com. distance: " + std::to_string(properties.comm_dist) +
                 "\n\t-Unit style: " + properties.unit_style);

  return properties;
}

}  // namespace jcn
