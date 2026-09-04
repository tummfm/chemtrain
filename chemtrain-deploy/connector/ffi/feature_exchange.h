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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_FFI_FEATURE_EXCHANGE_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_FFI_FEATURE_EXCHANGE_H_

#include "xla/ffi/api/c_api.h"

namespace jcn {

class CommunicationContext;

// Host and CUDA PJRT plugins keep independent external FFI type registries.
// Distinct binding types prevent XLA's per-C++-type cache from reusing a type
// ID that belongs to the other plugin.
struct HostCommunicationContext {
  CommunicationContext* context;
};

struct CudaCommunicationContext {
  CommunicationContext* context;
};

XLA_FFI_Handler* GatherInstantiateHandler();
XLA_FFI_Handler* ReduceInstantiateHandler();
XLA_FFI_Handler* HostInstantiateHandler();
XLA_FFI_Handler* PrepareHandler();
XLA_FFI_Handler* InitializeHandler();

XLA_FFI_Handler* GatherForwardHandler();
XLA_FFI_Handler* GatherReverseHandler();
XLA_FFI_Handler* ReduceHandler();
XLA_FFI_Handler* ReduceTransposeHandler();

XLA_FFI_Handler* HostGatherForwardHandler();
XLA_FFI_Handler* HostGatherReverseHandler();
XLA_FFI_Handler* HostReduceHandler();
XLA_FFI_Handler* HostReduceTransposeHandler();

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_FFI_FEATURE_EXCHANGE_H_
