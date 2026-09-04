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

#include "connector/dlpack_validation.h"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "connector/jcn_buffer_internal.h"

namespace jcn {

namespace {

constexpr uint8_t kDLFloat = 2;

void RequireDlpack(const jcn_buffer* buffer, const char* role) {
  if (buffer == nullptr || buffer->tensor == nullptr) {
    throw std::runtime_error(std::string("Missing DLPack buffer for ") + role);
  }
}

}  // namespace

void ValidateDlpackInput(const jcn_buffer* buffer,
                         const std::vector<int64_t>& shape,
                         DLDataType expected_dtype, const char* role) {
  if (buffer == nullptr || buffer->tensor == nullptr) {
    throw std::runtime_error(std::string("Missing DLPack buffer for ") + role);
  }
  const DLTensor& tensor = buffer->tensor->dl_tensor;
  if (tensor.data == nullptr) {
    throw std::runtime_error(std::string("Missing DLPack data for ") + role);
  }
  if (tensor.strides != nullptr) {
    throw std::runtime_error(
        std::string("Strided DLPack tensors are not supported for ") + role);
  }
  if (tensor.ndim != static_cast<int32_t>(shape.size())) {
    throw std::runtime_error(std::string("Unexpected DLPack rank for ") + role);
  }
  if (tensor.ndim > 0 && tensor.shape == nullptr) {
    throw std::runtime_error(std::string("Missing DLPack shape for ") + role);
  }
  for (int i = 0; i < tensor.ndim; ++i) {
    if (tensor.shape[i] != shape[i]) {
      throw std::runtime_error(std::string("Unexpected DLPack shape for ") +
                               role);
    }
  }
  if (tensor.dtype.code != expected_dtype.code ||
      tensor.dtype.bits != expected_dtype.bits ||
      tensor.dtype.lanes != expected_dtype.lanes) {
    throw std::runtime_error(std::string("Unexpected DLPack dtype for ") +
                             role);
  }
}

void RequireHostStagingDlpack(const jcn_buffer* buffer, const char* role) {
  RequireDlpack(buffer, role);
  const DLTensor& tensor = buffer->tensor->dl_tensor;
  if (tensor.device.device_type != kDLCPU) {
    throw std::runtime_error(
        std::string(
            "The current PJRT staging path needs a CPU DLPack tensor for ") +
        role +
        ". Native device DLPack must enter through the PJRT "
        "device-buffer adapter, not the host literal path.");
  }
  if (tensor.strides != nullptr) {
    throw std::runtime_error(
        std::string("Strided DLPack tensors are not supported for ") + role);
  }
}

void ValidateDlpackShape(const jcn_buffer* buffer,
                         const std::vector<int64_t>& minimum_shape,
                         const char* role) {
  RequireDlpack(buffer, role);
  const DLTensor& tensor = buffer->tensor->dl_tensor;
  if (tensor.strides != nullptr) {
    throw std::runtime_error(
        std::string("Strided DLPack tensors are not supported for ") + role);
  }
  if (tensor.ndim != static_cast<int32_t>(minimum_shape.size())) {
    throw std::runtime_error(std::string("Unexpected DLPack rank for ") + role);
  }
  if (tensor.ndim > 0 && tensor.shape == nullptr) {
    throw std::runtime_error(std::string("Missing DLPack shape for ") + role);
  }
  for (int index = 0; index < tensor.ndim; ++index) {
    if (tensor.shape[index] < minimum_shape[index]) {
      throw std::runtime_error(
          std::string("DLPack shape is smaller than requested shape for ") +
          role);
    }
  }
}

bool IsCpuDlpack(const jcn_buffer* buffer) {
  return buffer != nullptr && buffer->tensor != nullptr &&
         buffer->tensor->dl_tensor.device.device_type == kDLCPU;
}

const void* DlData(const jcn_buffer* buffer) {
  const DLTensor& tensor = buffer->tensor->dl_tensor;
  return static_cast<const char*>(tensor.data) + tensor.byte_offset;
}

void* MutableDlData(jcn_buffer* buffer) {
  DLTensor& tensor = buffer->tensor->dl_tensor;
  return static_cast<char*>(tensor.data) + tensor.byte_offset;
}

void WriteDlScalar(jcn_buffer* buffer, int64_t index, double value) {
  DLTensor& tensor = buffer->tensor->dl_tensor;
  void* data = MutableDlData(buffer);
  if (tensor.dtype.code == kDLFloat && tensor.dtype.bits == 64) {
    static_cast<double*>(data)[index] = value;
    return;
  }
  if (tensor.dtype.code == kDLFloat && tensor.dtype.bits == 32) {
    static_cast<float*>(data)[index] = static_cast<float>(value);
    return;
  }
  throw std::runtime_error("Unsupported DLPack output dtype.");
}

std::size_t DlpackElementBytes(const DLTensor& tensor) {
  return static_cast<std::size_t>((tensor.dtype.bits * tensor.dtype.lanes + 7) /
                                  8);
}

}  // namespace jcn
