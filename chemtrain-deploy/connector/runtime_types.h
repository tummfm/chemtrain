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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_RUNTIME_TYPES_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_RUNTIME_TYPES_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace jcn {

class Model;

enum class CommunicationScalarType {
  F32,
  F64,
};

enum class TensorDtype {
  ModelDefault,
  F32,
  F64,
  S32,
};

enum class SpeciesEncoding {
  ZeroBased,
  OneBased,
};

struct EngineAbiSpec {
  TensorDtype position_dtype = TensorDtype::F32;
  TensorDtype species_dtype = TensorDtype::S32;
  TensorDtype default_output_dtype = TensorDtype::F32;
  std::vector<std::pair<std::string, TensorDtype>> output_dtypes;
  TensorDtype communication_dtype = TensorDtype::ModelDefault;
  SpeciesEncoding species_encoding = SpeciesEncoding::ZeroBased;

  bool operator==(const EngineAbiSpec& other) const {
    return position_dtype == other.position_dtype &&
           species_dtype == other.species_dtype &&
           default_output_dtype == other.default_output_dtype &&
           output_dtypes == other.output_dtypes &&
           communication_dtype == other.communication_dtype &&
           species_encoding == other.species_encoding;
  }

  bool operator!=(const EngineAbiSpec& other) const {
    return !(*this == other);
  }
};

struct CommunicationCallbacks {
  void* context = nullptr;
  int (*exchange)(void* context, void* data, std::int64_t rows,
                  std::int64_t cols, CommunicationScalarType type, bool reverse,
                  const char** error) = nullptr;
  int (*device_exchange)(void* context, void* data, std::int64_t rows,
                         std::int64_t cols, CommunicationScalarType type,
                         bool reverse, const char* backend, int device_ordinal,
                         void* stream, const char** error) = nullptr;
  int (*reduce)(void* context, void* data, std::int64_t count,
                CommunicationScalarType type, const char** error) = nullptr;
  int (*device_reduce)(void* context, void* data, std::int64_t count,
                       CommunicationScalarType type, const char* backend,
                       int device_ordinal, void* stream,
                       const char** error) = nullptr;
};

struct ConnectorConfig {
  std::string backend;
  int device = 0;
  float memory_fraction = 0.75f;
};

struct ModelConfig {
  // Borrowed for load_model(). The caller retains ownership. A single PJRT
  // client may own several model executors, one for each engine-side model
  // context. The engine adapter is responsible for passing the same callbacks
  // and ABI policy to every context that participates in one pair style.
  const Model* model = nullptr;
  std::vector<float> neighbor_list_multipliers = {1.5};
  bool newton = true;
  bool use_communication = false;
  CommunicationCallbacks communication;
  EngineAbiSpec engine_abi;
};

struct ModelProperties {
  double cutoff = 0.0;
  double comm_dist = 0.0;
  int communication_buffer_width = 0;
  const char* unit_style = nullptr;
  struct {
    bool include_ghosts = false;
    bool half_list = true;
    bool include_pair_type = false;
  } neighbor_list;
  struct ParticleField {
    std::string name;
    TensorDtype dtype = TensorDtype::S32;
  };
  struct GlobalField {
    std::string name;
    TensorDtype dtype = TensorDtype::F32;
  };
  enum class OutputScope {
    PARTICLE,
    LOCAL,
    GLOBAL,
  };
  struct OutputField {
    std::string name;
    OutputScope scope = OutputScope::PARTICLE;
    std::vector<int64_t> dimensions;
    int components = 1;
    bool extensive = false;
  };
  std::vector<ParticleField> particle_fields;
  std::vector<GlobalField> global_fields;
  std::vector<OutputField> outputs;
  bool requires_communication = false;
};

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_RUNTIME_TYPES_H_
