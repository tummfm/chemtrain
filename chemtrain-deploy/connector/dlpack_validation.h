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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_DLPACK_VALIDATION_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_DLPACK_VALIDATION_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dlpack/dlpack.h"

struct jcn_buffer;

namespace jcn {

// Validates a contiguous input tensor against an exact refined ABI contract.
void ValidateDlpackInput(const jcn_buffer* buffer,
                         const std::vector<int64_t>& shape, DLDataType dtype,
                         const char* role);

// Host-staging and output-copy helpers share these DLPack layout checks.
void RequireHostStagingDlpack(const jcn_buffer* buffer, const char* role);
void ValidateDlpackShape(const jcn_buffer* buffer,
                         const std::vector<int64_t>& minimum_shape,
                         const char* role);
bool IsCpuDlpack(const jcn_buffer* buffer);
const void* DlData(const jcn_buffer* buffer);
void* MutableDlData(jcn_buffer* buffer);
void WriteDlScalar(jcn_buffer* buffer, int64_t index, double value);
std::size_t DlpackElementBytes(const DLTensor& tensor);

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_DLPACK_VALIDATION_H_
