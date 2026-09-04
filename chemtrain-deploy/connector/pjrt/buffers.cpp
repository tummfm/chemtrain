/*
Copyright 2017 The OpenXLA Authors.
Modifications Copyright 2025 Multiscale Modeling of Fluid Materials,
TU Munich.

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

#include "connector/pjrt/buffers.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "connector/dlpack_validation.h"
#include "connector/jcn_buffer_internal.h"
#include "xla/shape_util.h"

namespace jcn {
namespace {

constexpr uint8_t kDLInt = 0;
constexpr uint8_t kDLFloat = 2;

struct CudaRuntimeCopy {
  using MemcpyAsync = int (*)(void*, const void*, size_t, int, void*);
  using StreamSynchronize = int (*)(void*);
  using EventCreateWithFlags = int (*)(void**, unsigned int);
  using EventRecord = int (*)(void*, void*);
  using StreamWaitEvent = int (*)(void*, void*, unsigned int);
  using EventDestroy = int (*)(void*);
  using GetErrorString = const char* (*)(int);

  void* handle = nullptr;
  MemcpyAsync memcpy_async = nullptr;
  StreamSynchronize stream_synchronize = nullptr;
  EventCreateWithFlags event_create_with_flags = nullptr;
  EventRecord event_record = nullptr;
  StreamWaitEvent stream_wait_event = nullptr;
  EventDestroy event_destroy = nullptr;
  GetErrorString get_error_string = nullptr;

  bool ok() const {
    return handle != nullptr && memcpy_async != nullptr &&
           stream_synchronize != nullptr &&
           event_create_with_flags != nullptr && event_record != nullptr &&
           stream_wait_event != nullptr && event_destroy != nullptr &&
           get_error_string != nullptr;
  }
};

CudaRuntimeCopy& CudaRuntime() {
  static CudaRuntimeCopy runtime = [] {
    CudaRuntimeCopy result;
    // CUDA remains an optional runtime dependency. Loading cudart lazily lets
    // CPU and future backend builds fail only when CUDA interop is requested.
    const char* names[] = {"libcudart.so", "libcudart.so.13", "libcudart.so.12",
                           "libcudart.so.11.0"};
    for (const char* name : names) {
      result.handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
      if (result.handle != nullptr) break;
    }
    if (result.handle == nullptr) return result;

    result.memcpy_async = reinterpret_cast<CudaRuntimeCopy::MemcpyAsync>(
        dlsym(result.handle, "cudaMemcpyAsync"));
    result.stream_synchronize =
        reinterpret_cast<CudaRuntimeCopy::StreamSynchronize>(
            dlsym(result.handle, "cudaStreamSynchronize"));
    result.event_create_with_flags =
        reinterpret_cast<CudaRuntimeCopy::EventCreateWithFlags>(
            dlsym(result.handle, "cudaEventCreateWithFlags"));
    result.event_record = reinterpret_cast<CudaRuntimeCopy::EventRecord>(
        dlsym(result.handle, "cudaEventRecord"));
    result.stream_wait_event =
        reinterpret_cast<CudaRuntimeCopy::StreamWaitEvent>(
            dlsym(result.handle, "cudaStreamWaitEvent"));
    result.event_destroy = reinterpret_cast<CudaRuntimeCopy::EventDestroy>(
        dlsym(result.handle, "cudaEventDestroy"));
    result.get_error_string = reinterpret_cast<CudaRuntimeCopy::GetErrorString>(
        dlsym(result.handle, "cudaGetErrorString"));
    return result;
  }();
  return runtime;
}

void CheckDeviceOutputDlpack(jcn_buffer* buffer,
                             const std::vector<int64_t>& shape,
                             xla::PrimitiveType type, const char* role) {
  ValidateDlpackShape(buffer, shape, role);
  const DLTensor& tensor = buffer->tensor->dl_tensor;
  const bool dtype_matches =
      (type == xla::F32 && tensor.dtype.code == kDLFloat &&
       tensor.dtype.bits == 32) ||
      (type == xla::F64 && tensor.dtype.code == kDLFloat &&
       tensor.dtype.bits == 64);
  if (!dtype_matches) {
    throw std::runtime_error(
        std::string("Device PJRT output dtype cannot be copied into caller "
                    "DLPack buffer for ") +
        role);
  }
  if (buffer->options.copy_policy == JCN_DLPACK_ALWAYS_COPY) {
    throw std::runtime_error(
        std::string("Device output buffer for ") + role +
        " requested ALWAYS_COPY, but output copy-back writes the "
        "caller-owned DLPack memory directly.");
  }
}

}  // namespace

std::unique_ptr<xla::PjRtBuffer> CreatePjRtBufferFromLiteral(
    xla::PjRtClient* client, int device_id, xla::Literal* literal) {
  // Adapted from OpenXLA's cross-client literal-copy path in
  // PjRtStreamExecutorBuffer::CopyToDeviceMemorySpace at
  // https://github.com/openxla/xla/blob/ee9ee727b533dbd14698c9eda979a8c83ed86e11/xla/pjrt/pjrt_stream_executor_client.cc#L1699
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space =
      client->addressable_devices()[device_id]->default_memory_space();
  if (!memory_space.ok()) {
    throw std::runtime_error("Failed to get memory space: " +
                             memory_space.status().ToString());
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> input_buffer =
      client->BufferFromHostBuffer(
          literal->untyped_data(), literal->shape().element_type(),
          literal->shape().dimensions(),
          std::optional<absl::Span<const int64_t>>{},
          xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy, []() {},
          memory_space.value(), nullptr);
  if (!input_buffer.ok()) {
    throw std::runtime_error("Failed to create buffer: " +
                             input_buffer.status().ToString());
  }
  return std::move(input_buffer).value();
}

xla::PrimitiveType PrimitiveForDtype(TensorDtype dtype,
                                     xla::PrimitiveType model_default) {
  switch (dtype) {
    case TensorDtype::ModelDefault:
      return model_default;
    case TensorDtype::F32:
      return xla::F32;
    case TensorDtype::F64:
      return xla::F64;
    case TensorDtype::S32:
      return xla::S32;
  }
  return model_default;
}

DLDataType DlpackTypeForPrimitive(xla::PrimitiveType type) {
  switch (type) {
    case xla::F32:
      return {kDLFloat, 32, 1};
    case xla::F64:
      return {kDLFloat, 64, 1};
    case xla::S32:
      return {kDLInt, 32, 1};
    default:
      throw std::runtime_error("Unsupported refined DLPack input dtype.");
  }
}

std::unique_ptr<xla::Literal> MakeFloatLiteral(
    const jcn_buffer* buffer, const std::vector<int64_t>& shape,
    int64_t copied_values, xla::PrimitiveType type) {
  if (type != xla::F32 && type != xla::F64) {
    throw std::runtime_error("Floating input literal requires f32 or f64.");
  }
  xla::Shape xla_shape = xla::ShapeUtil::MakeShape(
      type, absl::Span<const int64_t>(shape.data(), shape.size()));
  auto literal =
      std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(xla_shape));
  if (type == xla::F32) {
    float* data = literal->data<float>().data();
    std::fill(data, data + xla::ShapeUtil::ElementsIn(literal->shape()), 0.0F);
    std::memcpy(data, DlData(buffer),
                static_cast<size_t>(copied_values) * sizeof(float));
  } else {
    double* data = literal->data<double>().data();
    std::fill(data, data + xla::ShapeUtil::ElementsIn(literal->shape()), 0.0);
    std::memcpy(data, DlData(buffer),
                static_cast<size_t>(copied_values) * sizeof(double));
  }
  return literal;
}

std::unique_ptr<xla::Literal> MakeIntLiteral(const jcn_buffer* buffer,
                                             const std::vector<int64_t>& shape,
                                             int64_t copied_values) {
  xla::Shape xla_shape = xla::ShapeUtil::MakeShape(
      xla::S32, absl::Span<const int64_t>(shape.data(), shape.size()));
  auto literal =
      std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(xla_shape));
  int32_t* data = literal->data<int32_t>().data();
  std::fill(data, data + xla::ShapeUtil::ElementsIn(literal->shape()), 0);
  std::memcpy(data, DlData(buffer),
              static_cast<size_t>(copied_values) * sizeof(int32_t));
  return literal;
}

std::unique_ptr<xla::Literal> MakeScalarIntLiteral(int value) {
  return std::make_unique<xla::Literal>(
      xla::LiteralUtil::CreateR0<int32_t>(value));
}

void CopyFloatLiteralToDlpack(const xla::Literal& literal, jcn_buffer* buffer,
                              int64_t copied_values) {
  RequireHostStagingDlpack(buffer, "floating output");
  if (literal.shape().element_type() == xla::F32) {
    const float* data = literal.data<float>().data();
    for (int64_t index = 0; index < copied_values; ++index) {
      WriteDlScalar(buffer, index, data[index]);
    }
    return;
  }
  if (literal.shape().element_type() == xla::F64) {
    const double* data = literal.data<double>().data();
    for (int64_t index = 0; index < copied_values; ++index) {
      WriteDlScalar(buffer, index, data[index]);
    }
    return;
  }
  throw std::runtime_error("Expected floating PJRT output literal.");
}

std::optional<std::intptr_t> ExternalReadyStreamForProducer(
    xla::PjRtDevice* device, std::intptr_t producer_stream, const char* role) {
  if (producer_stream == 0) return std::nullopt;

  absl::StatusOr<std::intptr_t> external_stream =
      device->GetStreamForExternalReadyEvents();
  if (!external_stream.ok()) {
    throw std::runtime_error(
        std::string("Failed to get PJRT external-ready stream for ") + role +
        ": " + external_stream.status().ToString());
  }

  CudaRuntimeCopy& cuda = CudaRuntime();
  if (!cuda.ok()) {
    throw std::runtime_error(
        "CUDA runtime library is required for device DLPack stream bridging "
        "but could not be loaded.");
  }

  void* event = nullptr;
  constexpr unsigned int kCudaEventDisableTiming = 2;
  int error = cuda.event_create_with_flags(&event, kCudaEventDisableTiming);
  if (error != 0) {
    throw std::runtime_error(std::string("CUDA event creation failed for ") +
                             role + ": " + cuda.get_error_string(error));
  }
  error = cuda.event_record(event, reinterpret_cast<void*>(producer_stream));
  if (error != 0) {
    cuda.event_destroy(event);
    throw std::runtime_error(std::string("CUDA event record failed for ") +
                             role + ": " + cuda.get_error_string(error));
  }
  error = cuda.stream_wait_event(
      reinterpret_cast<void*>(external_stream.value()), event, 0);
  if (error != 0) {
    cuda.event_destroy(event);
    throw std::runtime_error(
        std::string("CUDA external-ready stream wait failed for ") + role +
        ": " + cuda.get_error_string(error));
  }
  error = cuda.event_destroy(event);
  if (error != 0) {
    throw std::runtime_error(std::string("CUDA event destroy failed for ") +
                             role + ": " + cuda.get_error_string(error));
  }
  return external_stream.value();
}

void CopyDeviceOutputToDlpack(xla::PjRtBuffer* output, jcn_buffer* destination,
                              const std::vector<int64_t>& shape,
                              int64_t copied_values, xla::PrimitiveType type,
                              const char* role) {
  CheckDeviceOutputDlpack(destination, shape, type, role);
  const absl::string_view platform_name = output->client()->platform_name();
  const std::string platform(platform_name.data(), platform_name.size());
  if (platform.find("CUDA") == std::string::npos &&
      platform.find("cuda") == std::string::npos) {
    throw std::runtime_error(
        std::string("Device output copy-back for ") + role +
        " currently supports CUDA PJRT only. Non-CUDA backends must add "
        "backend-native copy and stream-ordering helpers before they can "
        "share caller-owned DLPack output buffers.");
  }

  absl::Status ready = output->GetReadyFuture().Await();
  if (!ready.ok()) {
    throw std::runtime_error(
        std::string("PJRT output buffer is not ready for ") + role + ": " +
        ready.ToString());
  }
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
      reference = output->AcquireExternalReference();
  if (!reference.ok()) {
    throw std::runtime_error(
        std::string("Failed to acquire PJRT output device pointer for ") +
        role + ": " + reference.status().ToString());
  }

  std::intptr_t stream =
      reinterpret_cast<std::intptr_t>(destination->options.producer_stream);
  if (stream != 0) {
    absl::Status stream_status =
        reference.value()->WaitUntilBufferReadyOnStream(stream);
    if (!stream_status.ok()) {
      throw std::runtime_error(
          std::string("PJRT output is not ready on caller stream for ") + role +
          ": " + stream_status.ToString());
    }
  }

  CudaRuntimeCopy& cuda = CudaRuntime();
  if (!cuda.ok()) {
    throw std::runtime_error(
        "CUDA runtime library is required for device DLPack output copy-back "
        "but could not be loaded.");
  }
  const size_t bytes = static_cast<size_t>(copied_values) *
                       DlpackElementBytes(destination->tensor->dl_tensor);
  constexpr int kCudaMemcpyDeviceToDevice = 3;
  void* cuda_stream = destination->options.producer_stream;
  int error =
      cuda.memcpy_async(MutableDlData(destination),
                        reference.value()->OpaqueDeviceMemoryDataPointer(),
                        bytes, kCudaMemcpyDeviceToDevice, cuda_stream);
  if (error != 0) {
    throw std::runtime_error(
        std::string("CUDA device output copy failed for ") + role + ": " +
        cuda.get_error_string(error));
  }

  // Synchronization keeps the adapter boundary synchronous while preserving
  // ordering on a caller-provided CUDA stream.
  error = cuda.stream_synchronize(cuda_stream);
  if (error != 0) {
    throw std::runtime_error(
        std::string("CUDA device output synchronization failed for ") + role +
        ": " + cuda.get_error_string(error));
  }
}

}  // namespace jcn
