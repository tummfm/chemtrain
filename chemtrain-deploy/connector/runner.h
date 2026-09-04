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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_RUNNER_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_RUNNER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "connector/communication/runtime.h"
#include "connector/compiler.h"
#include "connector/jcn_api.h"
#include "connector/model.pb.h"
#include "connector/model_parser.h"
#include "connector/model_shape.h"
#include "connector/runtime_types.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/dump.h"

namespace jcn {

// Creates a client from the dynamically loaded PJRT backend and returns the
// selected index in addressable_devices().
std::unique_ptr<xla::PjRtClient> CreatePjRtClient(const ConnectorConfig& config,
                                                  int* pjrt_device_index);

class Runner {
 public:
  Runner(ConnectorConfig connector_config, bool initialize);
  Runner(ConnectorConfig connector_config, bool initialize,
         xla::PjRtClient* shared_client, int pjrt_device_index);
  ~Runner() = default;

  ModelProperties load_model(ModelConfig config);
  jcn_neighbor_format selected_neighbor_format() const;

  // Executes one JCN force request. The engine adapter has already converted
  // atom and neighbor storage into DLPack buffers and chosen explicit static
  // capacities. Runner validates and executes those shapes instead of
  // discovering sizes from engine arrays.
  void compute_forces(const jcn_force_request& request,
                      jcn_force_result& result);

  static void initialize();

 private:
  struct ValidatedRequest {
    bool needs_capacity_change = false;
    std::map<std::string, jcn_buffer*> particle_inputs;
    std::map<std::string, jcn_buffer*> global_inputs;
    int64_t owned_atoms = 0;
    int64_t requested_atoms = 0;
    int64_t max_atoms = 0;
    int64_t raw_edges = 0;
    int64_t max_neighbors = 0;
    bool use_device_inputs = false;
    GraphInputSpec graph_spec;
  };
  struct PreparedInputs {
    std::vector<std::unique_ptr<xla::PjRtBuffer>> input_buffers;
    std::vector<xla::PjRtBuffer*> buffer_ptrs;
    std::vector<std::unique_ptr<xla::Literal>> literals;
    std::vector<std::vector<xla::PjRtBuffer*>> argument_handles;
  };

  ModelProperties get_model_properties();
  void configure_selected_model(SelectedModel selected);
  void configure_neighbor_list(const ModelConfig& config);
  ValidatedRequest validate_request(const jcn_force_request& request,
                                    jcn_force_result& result) const;
  bool needs_compilation(const jcn_force_request& request,
                         const GraphInputSpec& graph_spec) const;
  void compile_for_request(const jcn_force_request& request,
                           const GraphInputSpec& graph_spec,
                           jcn_force_result& result);
  using ExecutionResults =
      std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>;
  ExecutionResults execute_prepared_inputs(
      const std::vector<std::vector<xla::PjRtBuffer*>>& arg_handles,
      int64_t owned_atoms, int64_t requested_atoms);
  bool evaluate_statistics(ExecutionResults& results,
                           const jcn_force_request& request,
                           jcn_force_result& result);
  PreparedInputs prepare_inputs(const jcn_force_request& request,
                                const ValidatedRequest& validated,
                                bool needs_compile);
  void copy_outputs(const jcn_force_request& request, ExecutionResults& results,
                    bool use_device_inputs);

  std::unique_ptr<jcn::Model> model;

  std::unique_ptr<xla::PjRtClient> owned_client;
  xla::PjRtClient* client = nullptr;
  std::unique_ptr<xla::PjRtLoadedExecutable> executable;

  std::unique_ptr<GraphBuilder> neighbor_list;

  std::unique_ptr<Compiler> compiler;

  xla::CompileOptions compile_options;
  ConnectorConfig config;

  // Index into client->addressable_devices() used for buffer allocation.
  int pjrt_device_index_ = 0;

  bool newton;

  CommunicationCallbacks communication_callbacks;
  CommunicationWorkspace communication_workspace_;
  int64_t compiled_max_atoms_ = -1;
  std::vector<GraphInputDescriptor> compiled_graph_inputs_;
  std::vector<ModelProperties::ParticleField> particle_fields_;
  std::vector<ModelProperties::GlobalField> global_fields_;
  std::vector<ModelProperties::OutputField> output_fields_;
  std::vector<std::string> selected_platforms_;
  bool include_pair_type_ = false;
  EngineAbiSpec engine_abi_;
  EngineAbiSpec compiled_engine_abi_;
  jcn_neighbor_format compiled_neighbor_format_ = JCN_NEIGHBOR_UNSPECIFIED;
  // CPU graph inputs are copied into owned literals and can be reused safely.
  // CUDA graph inputs are transient zero-copy views owned by PreparedInputs.
  std::vector<std::unique_ptr<xla::PjRtBuffer>> host_neighbor_input_buffers_;
  std::vector<xla::PjRtBuffer*> host_neighbor_input_ptrs_;
  std::vector<std::unique_ptr<xla::Literal>> host_neighbor_literals_;
  const jcn_neighbor_list* host_neighbor_request_ = nullptr;
  int64_t host_neighbor_max_atoms_ = -1;
  int64_t host_neighbor_raw_edges_ = -1;
  int64_t host_neighbor_max_neighbors_ = -1;
  jcn_neighbor_format host_neighbor_input_format_ = JCN_NEIGHBOR_UNSPECIFIED;
};

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_RUNNER_H_
