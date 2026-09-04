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

#include "connector/ffi/feature_exchange_registration.h"

#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "connector/ffi/feature_exchange.h"
#include "connector/runtime_loader.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/type_registry.h"

namespace jcn {
namespace {

struct RegisteredCommunicationContextType {
  const XLA_FFI_Api* api;
  xla::ffi::TypeRegistry::TypeId type_id;
};

std::map<std::string, RegisteredCommunicationContextType>
    g_communication_context_type_ids;
std::map<std::string, const XLA_FFI_Api*> g_registered_handlers;
std::recursive_mutex g_communication_registration_mutex;

absl::Status ConsumeFfiError(const XLA_FFI_Api* api, XLA_FFI_Error* error,
                             absl::string_view operation) {
  if (error == nullptr) return absl::OkStatus();

  XLA_FFI_Error_GetMessage_Args message_args{};
  message_args.struct_size = XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE;
  message_args.error = error;
  api->XLA_FFI_Error_GetMessage(&message_args);
  const std::string message = message_args.message == nullptr
                                  ? "unknown XLA FFI error"
                                  : std::string(message_args.message);

  XLA_FFI_Error_Destroy_Args destroy_args{};
  destroy_args.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.error = error;
  api->XLA_FFI_Error_Destroy(&destroy_args);
  return absl::InternalError(std::string(operation) + ": " + message);
}

template <typename ContextBinding>
absl::Status RegisterCommunicationContextType(const XLA_FFI_Api* api,
                                              const char* platform_name) {
  std::lock_guard<std::recursive_mutex> lock(
      g_communication_registration_mutex);
  const std::string platform(platform_name);
  const auto existing = g_communication_context_type_ids.find(platform);
  if (existing != g_communication_context_type_ids.end()) {
    if (existing->second.api != api) {
      return absl::FailedPreconditionError(
          "CommunicationContext FFI type for " + platform +
          " is already bound to a different XLA FFI API");
    }
    return absl::OkStatus();
  }

  absl::string_view type_name =
      xla::ffi::TypeRegistry::GetTypeName<ContextBinding>();
  xla::ffi::TypeRegistry::TypeInfo cpp_type_info =
      xla::ffi::TypeRegistry::GetTypeInfo<ContextBinding>();
  XLA_FFI_TypeInfo type_info{};
  type_info.struct_size = XLA_FFI_TypeInfo_STRUCT_SIZE;
  type_info.deleter = cpp_type_info.deleter;

  XLA_FFI_TypeId type_id = XLA_FFI_UNKNOWN_TYPE_ID;
  XLA_FFI_Type_Register_Args args{};
  args.struct_size = XLA_FFI_Type_Register_Args_STRUCT_SIZE;
  args.name = {type_name.data(), type_name.size()};
  args.type_id = &type_id;
  args.type_info = &type_info;
  absl::Status status = ConsumeFfiError(
      api, api->XLA_FFI_Type_Register(&args),
      "Failed to register CommunicationContext FFI type");
  if (!status.ok()) return status;

  const xla::ffi::TypeRegistry::TypeId registered_type_id(type_id.type_id);
  if (registered_type_id == xla::ffi::TypeRegistry::kUnknownTypeId) {
    return absl::InternalError(
        "XLA FFI registered CommunicationContext with unknown type id");
  }
  g_communication_context_type_ids.emplace(
      platform, RegisteredCommunicationContextType{api, registered_type_id});
  return absl::OkStatus();
}

absl::Status RegisterOne(const XLA_FFI_Api* api, const char* name,
                         XLA_FFI_Handler_Bundle bundle,
                         const char* platform_name) {
  const std::string key = std::string(platform_name) + "\n" + name;
  const auto existing = g_registered_handlers.find(key);
  if (existing != g_registered_handlers.end()) {
    if (existing->second != api) {
      return absl::FailedPreconditionError(
          std::string("FFI handler ") + name + " for " + platform_name +
          " is already bound to a different XLA FFI API");
    }
    return absl::OkStatus();
  }

  XLA_FFI_Handler_Register_Args args{};
  args.struct_size = XLA_FFI_Handler_Register_Args_STRUCT_SIZE;
  args.name = {name, std::strlen(name)};
  args.platform = {platform_name, std::strlen(platform_name)};
  args.bundle = bundle;
  args.traits = 0;
  absl::Status status = ConsumeFfiError(
      api, api->XLA_FFI_Handler_Register(&args),
      std::string("Failed to register FFI handler ") + name);
  if (status.ok()) g_registered_handlers.emplace(key, api);
  return status;
}

struct RegisteredHandler {
  const char* name;
  XLA_FFI_Handler_Bundle bundle;
};

const std::array<RegisteredHandler, 4>& CommunicationBundles(bool host) {
  static const std::array<RegisteredHandler, 4> kHost = {{
      {"chemtrain_deploy.gather_forward",
       {HostInstantiateHandler(), PrepareHandler(), InitializeHandler(),
        HostGatherForwardHandler()}},
      {"chemtrain_deploy.gather_reverse",
       {HostInstantiateHandler(), PrepareHandler(), InitializeHandler(),
        HostGatherReverseHandler()}},
      {"chemtrain_deploy.reduce",
       {HostInstantiateHandler(), PrepareHandler(), InitializeHandler(),
        HostReduceHandler()}},
      {"chemtrain_deploy.reduce_transpose",
       {HostInstantiateHandler(), PrepareHandler(), InitializeHandler(),
        HostReduceTransposeHandler()}},
  }};
  static const std::array<RegisteredHandler, 4> kCuda = {{
      {"chemtrain_deploy.gather_forward",
       {GatherInstantiateHandler(), PrepareHandler(), InitializeHandler(),
        GatherForwardHandler()}},
      {"chemtrain_deploy.gather_reverse",
       {GatherInstantiateHandler(), PrepareHandler(), InitializeHandler(),
        GatherReverseHandler()}},
      {"chemtrain_deploy.reduce",
       {ReduceInstantiateHandler(), PrepareHandler(), InitializeHandler(),
        ReduceHandler()}},
      {"chemtrain_deploy.reduce_transpose",
       {ReduceInstantiateHandler(), PrepareHandler(), InitializeHandler(),
        ReduceTransposeHandler()}},
  }};
  return host ? kHost : kCuda;
}

absl::Status AddRegisteredCommunicationContext(
    xla::ExecuteContext* execute_context, void* communication_context,
    const char* platform_name) {
  if (execute_context == nullptr || communication_context == nullptr) {
    return absl::InvalidArgumentError(
        "communication execution context and data must not be null");
  }
  std::lock_guard<std::recursive_mutex> lock(
      g_communication_registration_mutex);
  const auto registered = g_communication_context_type_ids.find(platform_name);
  if (registered == g_communication_context_type_ids.end()) {
    return absl::FailedPreconditionError(
        std::string("CommunicationContext FFI type was not registered for ") +
        platform_name);
  }
  return execute_context->ffi_context().Insert(registered->second.type_id,
                                               communication_context);
}

}  // namespace

int RegisterCommunicationFfi(const XLA_FFI_Api* api,
                             const std::string& backend) {
  std::lock_guard<std::recursive_mutex> lock(
      g_communication_registration_mutex);
  const char* platform_name = XlaFfiPlatformForBackend(backend);
  if (api == nullptr || platform_name == nullptr ||
      api->struct_size < XLA_FFI_Api_STRUCT_SIZE ||
      api->api_version.struct_size < XLA_FFI_Api_Version_STRUCT_SIZE ||
      api->api_version.major_version != XLA_FFI_API_MAJOR ||
      api->XLA_FFI_Handler_Register == nullptr ||
      api->XLA_FFI_Type_Register == nullptr ||
      api->XLA_FFI_Error_GetMessage == nullptr ||
      api->XLA_FFI_Error_Destroy == nullptr || api->internal_api == nullptr ||
      api->internal_api->XLA_FFI_INTERNAL_Error_Forward == nullptr ||
      api->internal_api->XLA_FFI_INTERNAL_Future_Forward == nullptr ||
      api->internal_api->XLA_FFI_Internal_TypeRegistrationMap_Get == nullptr ||
      api->internal_api->XLA_FFI_INTERNAL_ExecutionContext_Get == nullptr ||
      api->internal_api->XLA_FFI_INTERNAL_Stream_Get == nullptr) {
    std::cerr << "XLA FFI registration API is incomplete" << std::endl;
    return 1;
  }

  const bool host = backend == "cpu";
  absl::Status type_status =
      host ? RegisterCommunicationContextType<HostCommunicationContext>(
                 api, platform_name)
           : RegisterCommunicationContextType<CudaCommunicationContext>(
                 api, platform_name);
  if (!type_status.ok()) {
    std::cerr << type_status.ToString() << std::endl;
    return 1;
  }
  for (const RegisteredHandler& handler : CommunicationBundles(host)) {
    absl::Status status =
        RegisterOne(api, handler.name, handler.bundle, platform_name);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
      return 1;
    }
  }
  return 0;
}

absl::Status AddHostCommunicationContextToExecuteContext(
    xla::ExecuteContext* execute_context,
    HostCommunicationContext* communication_context) {
  return AddRegisteredCommunicationContext(execute_context,
                                           communication_context, "Host");
}

absl::Status AddCudaCommunicationContextToExecuteContext(
    xla::ExecuteContext* execute_context,
    CudaCommunicationContext* communication_context) {
  return AddRegisteredCommunicationContext(execute_context,
                                           communication_context, "CUDA");
}

}  // namespace jcn
