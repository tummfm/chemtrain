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

#include "xla/ffi/api/c_api.h"

// XLA_FFI_GetApi is linked with hidden visibility in the pinned XLA build.
// Export a generic plugin-local forwarding anchor without changing XLA itself.
extern "C" __attribute__((visibility("default"))) const XLA_FFI_Api*
XLA_FFI_GetPluginApi() {
  return XLA_FFI_GetApi();
}
