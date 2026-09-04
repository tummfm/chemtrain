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

#include "connector/model_parser.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace jcn {
namespace {

// Descriptor validation.
// ---------------------------------------------------------------------------

void ValidateParticleFields(
    const google::protobuf::RepeatedPtrField<Model::ParticleField>& fields,
    const char* owner) {
  if (fields.empty()) {
    throw std::runtime_error(std::string(owner) +
                             " has no named particle fields.");
  }
  if (fields.Get(0).name() != "species" || fields.Get(0).components() != 1 ||
      fields.Get(0).dtype() != Model::INT32) {
    throw std::runtime_error(std::string(owner) +
                             " must declare zero-based scalar INT32 species as "
                             "its first particle field.");
  }
  const std::regex portable_name("[A-Za-z_][A-Za-z0-9_]*");
  std::set<std::string> names;
  for (const auto& field : fields) {
    if (!std::regex_match(field.name(), portable_name) ||
        !names.insert(field.name()).second) {
      throw std::runtime_error(
          std::string(owner) +
          " contains an invalid or duplicate particle field name.");
    }
    if (field.components() != 1 || field.dtype() != Model::INT32) {
      throw std::runtime_error(
          "Particle field '" + field.name() +
          "' must be scalar INT32 in the current model schema.");
    }
  }
}

void ValidateGlobalFields(
    const google::protobuf::RepeatedPtrField<Model::GlobalField>& fields,
    const char* owner) {
  const std::regex portable_name("[A-Za-z_][A-Za-z0-9_]*");
  std::set<std::string> names;
  for (const auto& field : fields) {
    if (!std::regex_match(field.name(), portable_name) ||
        !names.insert(field.name()).second) {
      throw std::runtime_error(
          std::string(owner) +
          " contains an invalid or duplicate global field name.");
    }
    if (field.dtype() != Model::GLOBAL_FLOAT32 &&
        field.dtype() != Model::GLOBAL_FLOAT64 &&
        field.dtype() != Model::GLOBAL_INT32) {
      throw std::runtime_error("Global field '" + field.name() +
                               "' must be scalar FLOAT32, FLOAT64, or INT32.");
    }
  }
}

// Output descriptors.
// ---------------------------------------------------------------------------

void ValidateOutputs(const Model& model) {
  // Validate every typed descriptor and retain the required built-ins.
  const std::regex portable_name("[A-Za-z_][A-Za-z0-9_]*");
  std::set<std::string> names;
  const Model::OutputField* energy = nullptr;
  const Model::OutputField* force = nullptr;
  const Model::OutputField* virial = nullptr;
  for (const auto& field : model.output_fields()) {
    if (!std::regex_match(field.name(), portable_name) ||
        !names.insert(field.name()).second) {
      throw std::runtime_error(
          "Model contains an invalid or duplicate output field name.");
    }
    if (field.scope() != Model::PARTICLE && field.scope() != Model::LOCAL &&
        field.scope() != Model::GLOBAL) {
      throw std::runtime_error("Model output '" + field.name() +
                               "' has an invalid scope.");
    }
    if (field.scope() == Model::PARTICLE && field.extensive()) {
      throw std::runtime_error("Particle output '" + field.name() +
                               "' cannot be marked as extensive.");
    }
    uint64_t components = 1;
    for (uint64_t dimension : field.dimensions()) {
      if (dimension == 0 ||
          components > std::numeric_limits<uint64_t>::max() / dimension) {
        throw std::runtime_error("Model output '" + field.name() +
                                 "' has an invalid shape.");
      }
      components *= dimension;
    }
    if (field.components() != components || components == 0 ||
        components >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error("Model output '" + field.name() +
                               "' has inconsistent components.");
    }
    if (field.name() == "U") energy = &field;
    if (field.name() == "F") force = &field;
    if (field.name() == "V") virial = &field;
  }

  // The connector ABI requires U as PARTICLE scalar, F as PARTICLE [3], and V
  // as LOCAL [6].
  if (energy == nullptr) {
    throw std::runtime_error("Model is missing built-in output 'U'.");
  }
  if (energy->scope() != Model::PARTICLE || energy->dimensions_size() != 0 ||
      energy->components() != 1 || energy->extensive()) {
    throw std::runtime_error("Built-in output 'U' has invalid metadata.");
  }
  if (force == nullptr) {
    throw std::runtime_error("Model is missing built-in output 'F'.");
  }
  if (force->scope() != Model::PARTICLE || force->dimensions_size() != 1 ||
      force->dimensions(0) != 3 || force->components() != 3 ||
      force->extensive()) {
    throw std::runtime_error("Built-in output 'F' has invalid metadata.");
  }
  if (virial == nullptr) {
    throw std::runtime_error("Model is missing built-in output 'V'.");
  }
  if (virial->scope() != Model::LOCAL || virial->dimensions_size() != 1 ||
      virial->dimensions(0) != 6 || virial->components() != 6 ||
      !virial->extensive()) {
    throw std::runtime_error("Built-in output 'V' has invalid metadata.");
  }
}

// Neighbor metadata and execution variants.
// ---------------------------------------------------------------------------

void ValidateNeighborMetadata(const Model::NeighborList& neighbors) {
  if (neighbors.type() != Model::SIMPLE_SPARSE &&
      neighbors.type() != Model::SIMPLE_DENSE) {
    throw std::runtime_error(
        "Model variant uses an unsupported neighbor-list type.");
  }
  if (neighbors.nbr_order_size() != 1) {
    throw std::runtime_error(
        "Each model variant must provide exactly one neighbor order.");
  }
  if (neighbors.nbr_order(0) < 1) {
    throw std::runtime_error("Model neighbor orders must be positive.");
  }
  if (!std::isfinite(neighbors.cutoff()) || neighbors.cutoff() <= 0.0) {
    throw std::runtime_error(
        "Model neighbor cutoff must be finite and positive.");
  }
}

std::vector<std::string> ValidatePlatforms(
    const google::protobuf::RepeatedPtrField<std::string>& platforms,
    const char* owner) {
  if (platforms.empty()) {
    throw std::runtime_error(std::string(owner) +
                             " has no supported platforms.");
  }
  std::set<std::string> unique;
  for (const auto& platform : platforms) {
    if ((platform != "cpu" && platform != "cuda") ||
        !unique.insert(platform).second) {
      throw std::runtime_error(std::string(owner) +
                               " has an invalid or duplicate platform.");
    }
  }
  return {platforms.begin(), platforms.end()};
}

std::string VariantName(bool use_communication, bool newton_pair) {
  if (use_communication && !newton_pair) {
    throw std::runtime_error(
        "Communication requires Newton pair on; comm on/newton off has no "
        "model variant.");
  }
  if (use_communication) return "comm_on_newton_on";
  return newton_pair ? "comm_off_newton_on" : "comm_off_newton_off";
}

}  // namespace

Model ParseModelProtobuf(const void* data, std::size_t size) {
  // Decode the wire format.
  if (data == nullptr || size == 0) {
    throw std::runtime_error("Cannot load model: Model file is empty.");
  }
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Cannot load model: Model file is too large.");
  }
  Model model;
  if (!model.ParseFromArray(data, size)) {
    throw std::runtime_error(
        "Cannot load model: Model file is invalid or corrupted.");
  }
  if (model.format_version() != 5) {
    throw std::runtime_error(
        "Unsupported exported model schema. Re-export the model with the "
        "current chemtrain-deploy exporter.");
  }

  // Validate the metadata shared by all execution variants.
  ValidateParticleFields(model.particle_fields(), "Model");
  ValidateGlobalFields(model.global_fields(), "Model");
  ValidateOutputs(model);
  ValidateNeighborMetadata(model.neighbor_list());
  ValidatePlatforms(model.platforms(), "Model");
  if (model.quantities_size() != model.output_fields_size() ||
      model.quantity_components_size() != model.output_fields_size()) {
    throw std::runtime_error(
        "Flattened output metadata does not match typed output metadata.");
  }
  for (int i = 0; i < model.output_fields_size(); ++i) {
    if (model.quantities(i) != model.output_fields(i).name() ||
        model.quantity_components(i) != model.output_fields(i).components()) {
      throw std::runtime_error(
          "Flattened output metadata does not match typed output metadata.");
    }
  }

  // Validate every executable variant and record the supported selections.
  bool has_newton_off = false;
  bool has_newton_on = false;
  bool has_communication = false;
  std::set<std::string> names;
  for (const auto& variant : model.variants()) {
    const std::string expected =
        VariantName(variant.uses_communication(), variant.newton_pair());
    if (variant.name() != expected || !names.insert(variant.name()).second) {
      throw std::runtime_error(
          "Model contains an invalid, inconsistent, or duplicate variant.");
    }
    ValidateParticleFields(variant.particle_fields(), "Model variant");
    ValidateGlobalFields(variant.global_fields(), "Model variant");
    ValidateNeighborMetadata(variant.neighbor_list());
    ValidatePlatforms(variant.platforms(), "Model variant");
    if (variant.particle_fields_size() != model.particle_fields_size() ||
        variant.global_fields_size() != model.global_fields_size()) {
      throw std::runtime_error(
          "Model variant inputs do not match top-level model metadata.");
    }
    for (int i = 0; i < variant.particle_fields_size(); ++i) {
      if (variant.particle_fields(i).SerializeAsString() !=
          model.particle_fields(i).SerializeAsString()) {
        throw std::runtime_error(
            "Model variant particle fields do not match top-level metadata.");
      }
    }
    for (int i = 0; i < variant.global_fields_size(); ++i) {
      if (variant.global_fields(i).SerializeAsString() !=
          model.global_fields(i).SerializeAsString()) {
        throw std::runtime_error(
            "Model variant global fields do not match top-level metadata.");
      }
    }
    if (variant.mlir_module_serialized().empty() ||
        variant.calling_convention_version() == 0) {
      throw std::runtime_error(
          "Model variant is missing its executable or calling convention.");
    }
    if (variant.uses_communication()) {
      if (variant.communication_buffer_width() <= 0 ||
          variant.neighbor_list().nbr_order(0) != 1) {
        throw std::runtime_error(
            "Communication variant requires positive communication width "
            "and neighbor order one.");
      }
    } else if (variant.communication_buffer_width() != 0) {
      throw std::runtime_error(
          "Non-communication variant contains communication-only metadata.");
    }
    if (variant.neighbor_list().type() == Model::SIMPLE_SPARSE) {
      if (!variant.neighbor_list().has_half_list() ||
          variant.neighbor_list().half_list() == variant.newton_pair()) {
        throw std::runtime_error(
            "Sparse variant has inconsistent Newton and half-list metadata.");
      }
    } else if (!variant.neighbor_list().has_half_list() ||
               variant.neighbor_list().half_list()) {
      throw std::runtime_error(
          "Dense model variants require full neighbor lists.");
    }
    has_newton_off |= !variant.uses_communication() && !variant.newton_pair();
    has_newton_on |= !variant.uses_communication() && variant.newton_pair();
    has_communication |= variant.uses_communication();
  }
  // Every format-5 bundle must support both non-communication modes.
  if (!has_newton_off || !has_newton_on) {
    throw std::runtime_error(
        "Model must contain both comm-off Newton variants.");
  }
  if (model.requires_communication() && !has_communication) {
    throw std::runtime_error(
        "Model requires communication but has no comm-on Newton-on variant.");
  }
  // The stable variant order makes the top-level Newton-on mirror unambiguous.
  if ((model.variants_size() != 2 && model.variants_size() != 3) ||
      model.variants(0).name() != "comm_off_newton_off" ||
      model.variants(1).name() != "comm_off_newton_on" ||
      (model.variants_size() == 3 &&
       model.variants(2).name() != "comm_on_newton_on")) {
    throw std::runtime_error(
        "Model variants are not in canonical execution order.");
  }
  // Metadata-only readers see the comm-off, Newton-on variant at the top level.
  const auto& mirrored = model.variants(1);
  if (mirrored.name() != "comm_off_newton_on" ||
      model.mlir_module_serialized() != mirrored.mlir_module_serialized() ||
      model.calling_convention_version() !=
          mirrored.calling_convention_version() ||
      model.uses_communication() || model.communication_buffer_width() != 0 ||
      model.neighbor_list().SerializeAsString() !=
          mirrored.neighbor_list().SerializeAsString() ||
      std::vector<std::string>(model.platforms().begin(),
                               model.platforms().end()) !=
          std::vector<std::string>(mirrored.platforms().begin(),
                                   mirrored.platforms().end())) {
    throw std::runtime_error(
        "Top-level model metadata must mirror comm_off_newton_on.");
  }
  return model;
}

SelectedModel SelectModelVariant(const Model& source, bool use_communication,
                                 bool newton_pair) {
  // Find the executable for the requested communication and Newton settings.
  if (source.requires_communication() && !use_communication) {
    throw std::runtime_error("Model requires communication. Select comm on.");
  }
  const std::string requested_name =
      VariantName(use_communication, newton_pair);
  const Model::ModelVariant* selected = nullptr;
  for (const auto& variant : source.variants()) {
    if (variant.name() != requested_name) continue;
    if (selected != nullptr) {
      throw std::runtime_error("Model contains duplicate '" + requested_name +
                               "' variants.");
    }
    selected = &variant;
  }
  if (selected == nullptr) {
    throw std::runtime_error("Model does not contain the requested '" +
                             requested_name + "' variant.");
  }
  if (selected->uses_communication() != use_communication ||
      selected->newton_pair() != newton_pair) {
    throw std::runtime_error("Model variant '" + requested_name +
                             "' has inconsistent selection metadata.");
  }

  // Match the selected input descriptors with the shared schema.
  if (selected->particle_fields_size() != source.particle_fields_size()) {
    throw std::runtime_error(
        "Selected model variant particle fields do not "
        "match top-level model metadata.");
  }
  for (int i = 0; i < selected->particle_fields_size(); ++i) {
    const auto& variant_field = selected->particle_fields(i);
    const auto& model_field = source.particle_fields(i);
    if (variant_field.name() != model_field.name() ||
        variant_field.dtype() != model_field.dtype() ||
        variant_field.components() != model_field.components()) {
      throw std::runtime_error(
          "Selected model variant particle fields do not "
          "match top-level model metadata.");
    }
  }
  if (selected->global_fields_size() != source.global_fields_size()) {
    throw std::runtime_error(
        "Selected model variant global fields do not "
        "match top-level model metadata.");
  }
  for (int i = 0; i < selected->global_fields_size(); ++i) {
    const auto& variant_field = selected->global_fields(i);
    const auto& model_field = source.global_fields(i);
    if (variant_field.name() != model_field.name() ||
        variant_field.dtype() != model_field.dtype()) {
      throw std::runtime_error(
          "Selected model variant global fields do not "
          "match top-level model metadata.");
    }
  }
  ValidateParticleFields(selected->particle_fields(), "Model variant");
  ValidateGlobalFields(selected->global_fields(), "Model variant");
  ValidateNeighborMetadata(selected->neighbor_list());

  // Check the selected executable and communication metadata.
  if (selected->uses_communication()) {
    if (selected->communication_buffer_width() <= 0) {
      throw std::runtime_error(
          "Communication variant must provide a positive buffer width.");
    }
  } else if (selected->communication_buffer_width() != 0) {
    throw std::runtime_error(
        "Non-communication variant contains communication-only metadata.");
  }
  if (selected->mlir_module_serialized().empty()) {
    throw std::runtime_error(
        "Selected model variant has no serialized StableHLO module.");
  }
  if (selected->calling_convention_version() == 0) {
    throw std::runtime_error(
        "Selected model variant has no calling convention version.");
  }
  if (selected->calling_convention_version() >
      static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "Selected model variant calling convention version is too large.");
  }

  // Preserve the selected variant's supported platform order.
  std::vector<std::string> platforms =
      ValidatePlatforms(selected->platforms(), "Selected model variant");
  const std::vector<std::string> source_platforms(source.platforms().begin(),
                                                  source.platforms().end());
  if (!use_communication && newton_pair && source_platforms != platforms) {
    throw std::runtime_error(
        "Newton-on variant platforms do not match top-level model metadata.");
  }

  // Copy selected runtime metadata into the model object returned to callers.
  SelectedModel result;
  result.model = source;
  result.model.set_mlir_module_serialized(selected->mlir_module_serialized());
  result.model.mutable_neighbor_list()->CopyFrom(selected->neighbor_list());
  result.model.set_uses_communication(selected->uses_communication());
  result.model.set_communication_buffer_width(
      selected->communication_buffer_width());
  result.model.set_calling_convention_version(
      selected->calling_convention_version());
  result.model.clear_platforms();
  for (const auto& platform : platforms) {
    result.model.add_platforms(platform);
  }
  result.platforms = std::move(platforms);
  return result;
}

}  // namespace jcn
