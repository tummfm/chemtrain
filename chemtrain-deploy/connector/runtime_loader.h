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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_RUNTIME_LOADER_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_RUNTIME_LOADER_H_

#include <string>
#include <vector>

#include "xla/ffi/api/c_api.h"

namespace jcn {

std::string PjrtDirectory();

std::vector<std::string> FfiProviderDirectories();

const char* XlaFfiPlatformForBackend(const std::string& backend);

void RegisterFfiProviders(const XLA_FFI_Api* api, const std::string& backend,
                          const std::vector<std::string>& search_directories);

const XLA_FFI_Api* GetPjrtFfiApi(const std::string& pjrt_plugin_path);

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_RUNTIME_LOADER_H_
