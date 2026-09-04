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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_MODEL_PARSER_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_MODEL_PARSER_H_

#include <cstddef>
#include <string>
#include <vector>

#include "connector/model.pb.h"

namespace jcn {

// Decodes and validates metadata shared by every executable variant.
Model ParseModelProtobuf(const void* data, std::size_t size);

struct SelectedModel {
  Model model;
  std::vector<std::string> platforms;
};

// Returns an owned runtime projection containing the requested executable.
SelectedModel SelectModelVariant(const Model& model, bool use_communication,
                                 bool newton_pair);

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_MODEL_PARSER_H_
