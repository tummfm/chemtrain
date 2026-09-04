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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_FFI_FEATURE_EXCHANGE_REGISTRATION_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_FFI_FEATURE_EXCHANGE_REGISTRATION_H_

#include <string>

#include "absl/status/status.h"
#include "connector/ffi/feature_exchange.h"
#include "xla/pjrt/pjrt_executable.h"

struct XLA_FFI_Api;

namespace jcn {

int RegisterCommunicationFfi(const XLA_FFI_Api* api,
                             const std::string& backend);

absl::Status AddHostCommunicationContextToExecuteContext(
    xla::ExecuteContext* execute_context,
    HostCommunicationContext* communication_context);

absl::Status AddCudaCommunicationContextToExecuteContext(
    xla::ExecuteContext* execute_context,
    CudaCommunicationContext* communication_context);

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_FFI_FEATURE_EXCHANGE_REGISTRATION_H_
