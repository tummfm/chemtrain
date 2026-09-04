/*
Copyright 2025-2026 Multiscale Modeling of Fluid Materials, TU Munich

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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_PJRT_BUFFERS_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_PJRT_BUFFERS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "connector/runtime_types.h"
#include "dlpack/dlpack.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"

struct jcn_buffer;

namespace jcn {

std::unique_ptr<xla::PjRtBuffer> CreatePjRtBufferFromLiteral(
    xla::PjRtClient* client, int device_id, xla::Literal* literal);

xla::PrimitiveType PrimitiveForDtype(TensorDtype dtype,
                                     xla::PrimitiveType model_default);
DLDataType DlpackTypeForPrimitive(xla::PrimitiveType type);

std::unique_ptr<xla::Literal> MakeFloatLiteral(
    const jcn_buffer* buffer, const std::vector<int64_t>& shape,
    int64_t copied_values, xla::PrimitiveType type = xla::F32);
std::unique_ptr<xla::Literal> MakeIntLiteral(const jcn_buffer* buffer,
                                             const std::vector<int64_t>& shape,
                                             int64_t copied_values);
std::unique_ptr<xla::Literal> MakeScalarIntLiteral(int value);

void CopyFloatLiteralToDlpack(const xla::Literal& literal, jcn_buffer* buffer,
                              int64_t copied_values);
std::optional<std::intptr_t> ExternalReadyStreamForProducer(
    xla::PjRtDevice* device, std::intptr_t producer_stream, const char* role);
void CopyDeviceOutputToDlpack(xla::PjRtBuffer* output, jcn_buffer* destination,
                              const std::vector<int64_t>& shape,
                              int64_t copied_values, xla::PrimitiveType type,
                              const char* role);

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_PJRT_BUFFERS_H_
