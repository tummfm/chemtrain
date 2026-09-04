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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_JCN_BUFFER_INTERNAL_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_JCN_BUFFER_INTERNAL_H_

#include "connector/jcn_api.h"

/*
 * Internal view of the opaque jcn_buffer handle.
 *
 * The public ABI deliberately exposes only `jcn_buffer*`.  Connector internals
 * need the wrapped DLManagedTensor to build PJRT inputs and to copy model
 * outputs back into caller-owned memory.  Keeping this definition out of
 * jcn_api.h preserves the C ABI boundary while giving Runner one shared
 * implementation of DLPack ownership.
 */
struct jcn_buffer {
  jcn_executor* executor = nullptr;
  DLManagedTensor* tensor = nullptr;
  jcn_buffer_import_options options{};
};

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_JCN_BUFFER_INTERNAL_H_
