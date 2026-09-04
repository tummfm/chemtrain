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

#include "connector/ffi/feature_exchange.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "connector/communication/runtime.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace jcn {

namespace {

// CPU handlers receive host buffers directly because Host FFI has no stream
// context. CUDA handlers retain the stream-aware staging path so PJRT kernels,
// adapter callbacks, and reusable workspaces remain correctly ordered.

namespace ffi = xla::ffi;
namespace se = stream_executor;

bool IsCudaPlatform(const std::string& name) {
  std::string lower = name;
  std::transform(
      lower.begin(), lower.end(), lower.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lower.find("cuda") != std::string::npos;
}

absl::Status WaitForCommunicationCopy(se::Stream* stream) {
  auto copy_ready = stream->parent()->CreateEvent();
  if (!copy_ready.ok()) return copy_ready.status();
  std::unique_ptr<se::Event> event = std::move(copy_ready).value();
  absl::Status status = stream->RecordEvent(event.get());
  if (!status.ok()) return status;
  return event->Synchronize();
}

absl::Status ValidateExchangeSignature(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output) {
  if (input.dimensions().size() != 2 || output->dimensions().size() != 2 ||
      input.dimensions()[0] != output->dimensions()[0] ||
      input.dimensions()[1] != output->dimensions()[1] ||
      input.element_type() != output->element_type()) {
    return absl::InvalidArgumentError(
        "chemtrain exchange expects matching rank-2 input and output buffers");
  }
  if (token_input.element_type() != xla::F32 ||
      token_output->element_type() != xla::F32 ||
      token_input.dimensions().size() != 1 ||
      token_output->dimensions().size() != 1 ||
      token_input.dimensions()[0] != 1 || token_output->dimensions()[0] != 1) {
    return absl::InvalidArgumentError(
        "chemtrain exchange expects a matching float32[1] ordering token");
  }
  if (input.element_type() != xla::F32 && input.element_type() != xla::F64) {
    return absl::InvalidArgumentError(
        "chemtrain exchange supports only f32 and f64 buffers");
  }
  return absl::OkStatus();
}

absl::Status ValidateReduceSignature(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output) {
  if (input.dimensions().size() != 1 || output->dimensions().size() != 1 ||
      input.dimensions()[0] != output->dimensions()[0] ||
      input.element_type() != output->element_type()) {
    return absl::InvalidArgumentError(
        "chemtrain reduce expects matching rank-1 input and output buffers");
  }
  if (token_input.element_type() != xla::F32 ||
      token_output->element_type() != xla::F32 ||
      token_input.dimensions().size() != 1 ||
      token_output->dimensions().size() != 1 ||
      token_input.dimensions()[0] != 1 || token_output->dimensions()[0] != 1) {
    return absl::InvalidArgumentError(
        "chemtrain reduce expects a matching float32[1] ordering token");
  }
  if (input.element_type() != xla::F32 && input.element_type() != xla::F64) {
    return absl::InvalidArgumentError(
        "chemtrain reduce supports only f32 and f64 buffers");
  }
  if (input.dimensions()[0] <= 0 ||
      input.dimensions()[0] > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        "chemtrain reduce expects a positive buffer size within integer range");
  }
  return absl::OkStatus();
}

absl::Status InstantiateGather(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output) {
  return ValidateExchangeSignature(input, token_input, output, token_output);
}

absl::Status InstantiateReduce(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output) {
  return ValidateReduceSignature(input, token_input, output, token_output);
}

absl::Status InstantiateHost() {
  // Pinned CPU XLA supplies a zero-arity call frame during instantiation. The
  // Host execute handlers therefore retain the complete signature validation.
  return absl::OkStatus();
}

absl::Status NoopLifecycleStage() { return absl::OkStatus(); }

tsl::AsyncValueRef<tsl::Chain> RunExchange(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output, se::Stream* stream,
    CommunicationContext* context, bool reverse) {
  // XLA calls this FFI from a PJRT worker thread, but engine communication must
  // run on the caller thread. "Forward" communication exchanges owned
  // atom values into ghost rows. "Reverse" communication is the transpose:
  // ghost-row contributions are accumulated back into owned rows and ghost
  // rows are cleared afterward. The token buffers are not model data. They
  // create an ordinary XLA dependency so consecutive communication calls
  // execute in program order.
  auto done = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();

  auto fail = [&done](std::string message) {
    done.SetError(absl::InternalError(std::move(message)));
  };

  if (stream == nullptr) {
    fail("chemtrain communication FFI called without stream");
    return done;
  }

  if (context == nullptr) {
    fail("chemtrain communication FFI called without CommunicationContext");
    return done;
  }

  if (context->workspace() == nullptr) {
    fail("chemtrain communication FFI called without workspace");
    return done;
  }

  if (input.dimensions().size() != 2 || output->dimensions().size() != 2 ||
      input.dimensions()[0] != output->dimensions()[0] ||
      input.dimensions()[1] != output->dimensions()[1] ||
      input.element_type() != output->element_type()) {
    fail("chemtrain exchange expects matching rank-2 input and output buffers");
    return done;
  }

  if (token_input.element_type() != xla::F32 ||
      token_output->element_type() != xla::F32 ||
      token_input.dimensions().size() != 1 ||
      token_output->dimensions().size() != 1 ||
      token_input.dimensions()[0] != 1 || token_output->dimensions()[0] != 1) {
    fail("chemtrain exchange expects a matching float32[1] ordering token");
    return done;
  }

  CommunicationScalarType scalar_type;
  if (input.element_type() == xla::F32) {
    scalar_type = CommunicationScalarType::F32;
  } else if (input.element_type() == xla::F64) {
    scalar_type = CommunicationScalarType::F64;
  } else {
    fail("chemtrain exchange supports only f32 and f64 buffers");
    return done;
  }

  const std::int64_t rows = input.dimensions()[0];
  const std::int64_t cols = input.dimensions()[1];
  const std::size_t bytes = input.size_bytes();
  auto row_bounds = context->ValidateRowBounds(rows);
  if (!row_bounds.ok()) {
    done.SetError(row_bounds.status());
    return done;
  }
  const std::int64_t active_rows = row_bounds->active_rows;
  const std::int64_t owned_rows = row_bounds->owned_rows;
  const std::size_t element_bytes = scalar_type == CommunicationScalarType::F32
                                        ? sizeof(float)
                                        : sizeof(double);
  const std::size_t owned_bytes = static_cast<std::size_t>(owned_rows) *
                                  static_cast<std::size_t>(cols) *
                                  element_bytes;
  const std::size_t active_bytes = static_cast<std::size_t>(active_rows) *
                                   static_cast<std::size_t>(cols) *
                                   element_bytes;
  const std::size_t ghost_bytes = active_bytes - owned_bytes;
  const bool compact_staging = owned_rows < active_rows;

  void* input_data = input.untyped_data();
  void* output_data = output->untyped_data();
  void* token_input_data = token_input.untyped_data();
  void* token_output_data = token_output->untyped_data();
  CommunicationWorkspace* workspace = context->workspace();

  const std::string platform_name = stream->parent()->GetPlatform()->Name();
  const bool use_device_exchange =
      context->has_device_exchange() && IsCudaPlatform(platform_name);
  if (use_device_exchange) {
    workspace->ScheduleDevice(
        stream->parent(), active_bytes,
        [=](void* scratch) -> absl::Status {
          se::DeviceAddressBase src(input_data, bytes);
          se::DeviceAddressBase dst(output_data, bytes);
          se::DeviceAddressBase token_src(token_input_data, sizeof(float));
          se::DeviceAddressBase token_dst(token_output_data, sizeof(float));
          se::DeviceAddressBase scratch_dst(scratch, active_bytes);

          absl::Status status;
          if (!context->enabled()) {
            status = stream->Memcpy(&dst, src, bytes);
            if (!status.ok()) return status;
            status = stream->Memcpy(&token_dst, token_src, sizeof(float));
            if (!status.ok()) return status;
            return absl::OkStatus();
          }

          status = stream->Memcpy(&dst, src, bytes);
          if (!status.ok()) return status;

          se::DeviceAddressBase inbound_src(input_data, active_bytes);
          status = stream->Memcpy(&scratch_dst, inbound_src, active_bytes);
          if (!status.ok()) return status;

          status = WaitForCommunicationCopy(stream);
          if (!status.ok()) return status;

          status = context->DeviceExchange(
              scratch, active_rows, cols, scalar_type, reverse,
              platform_name.c_str(), stream->parent()->device_ordinal(),
              stream->platform_specific_handle().stream);
          if (!status.ok()) return status;

          if (compact_staging) {
            if (reverse) {
              se::DeviceAddressBase owned_dst(output_data, owned_bytes);
              se::DeviceAddressBase owned_src(scratch, owned_bytes);
              status = stream->Memcpy(&owned_dst, owned_src, owned_bytes);
              if (!status.ok()) return status;
              if (ghost_bytes > 0) {
                se::DeviceAddressBase ghost_dst(
                    static_cast<char*>(output_data) + owned_bytes, ghost_bytes);
                status = stream->MemZero(&ghost_dst, ghost_bytes);
                if (!status.ok()) return status;
              }
            } else if (ghost_bytes > 0) {
              se::DeviceAddressBase ghost_dst(
                  static_cast<char*>(output_data) + owned_bytes, ghost_bytes);
              se::DeviceAddressBase ghost_src(
                  static_cast<char*>(scratch) + owned_bytes, ghost_bytes);
              status = stream->Memcpy(&ghost_dst, ghost_src, ghost_bytes);
              if (!status.ok()) return status;
            }
          } else {
            status = stream->Memcpy(&dst, scratch_dst, active_bytes);
            if (!status.ok()) return status;
            if (reverse && ghost_bytes > 0) {
              se::DeviceAddressBase ghost_dst(
                  static_cast<char*>(output_data) + owned_bytes, ghost_bytes);
              status = stream->MemZero(&ghost_dst, ghost_bytes);
              if (!status.ok()) return status;
            }
          }

          status = stream->Memcpy(&token_dst, token_src, sizeof(float));
          if (!status.ok()) return status;
          // Device communication uses a reusable connector-owned scratch
          // buffer. The stream copies above only enqueue reads from scratch, so
          // keep the worker job alive until the PJRT stream has consumed it.
          status = WaitForCommunicationCopy(stream);
          if (!status.ok()) return status;
          return absl::OkStatus();
        },
        [done](absl::Status status) mutable {
          if (status.ok()) {
            done.SetStateConcrete();
          } else {
            done.SetError(std::move(status));
          }
        });
    return done;
  }

  workspace->Schedule(
      stream->parent(), active_bytes,
      [=](void* host) -> absl::Status {
        se::DeviceAddressBase src(input_data, bytes);
        se::DeviceAddressBase dst(output_data, bytes);
        se::DeviceAddressBase token_src(token_input_data, sizeof(float));
        se::DeviceAddressBase token_dst(token_output_data, sizeof(float));

        absl::Status status;

        // The identity path preserves exact values and is not performance
        // critical.
        if (!context->enabled()) {
          status = stream->Memcpy(&dst, src, bytes);
          if (!status.ok()) return status;

          status = stream->Memcpy(&token_dst, token_src, sizeof(float));
          if (!status.ok()) return status;

          return absl::OkStatus();
        }

        // Preserve exact identity semantics for inactive padding rows. The
        // communication callbacks below only redefine active rows.
        status = stream->Memcpy(&dst, src, bytes);
        if (!status.ok()) return status;

        se::DeviceAddressBase inbound_src(input_data, active_bytes);
        // Data crosses the host boundary in a fixed order: copy the active
        // device rows to pinned host storage, let the main engine thread run
        // its callback, copy the exchanged rows back to device memory, and
        // finally copy the token to release the next XLA communication call.
        status = stream->Memcpy(host, inbound_src, active_bytes);
        if (!status.ok()) return status;

        status = WaitForCommunicationCopy(stream);
        if (!status.ok()) return status;

        status =
            context->Exchange(host, active_rows, cols, scalar_type, reverse);
        if (!status.ok()) return status;

        if (compact_staging) {
          if (reverse) {
            // Reverse communication updates owned rows and defines ghost rows
            // as zero after their contributions have been returned.
            se::DeviceAddressBase owned_dst(output_data, owned_bytes);
            status = stream->Memcpy(&owned_dst, host, owned_bytes);
            if (!status.ok()) return status;
            if (ghost_bytes > 0) {
              se::DeviceAddressBase ghost_dst(
                  static_cast<char*>(output_data) + owned_bytes, ghost_bytes);
              status = stream->MemZero(&ghost_dst, ghost_bytes);
              if (!status.ok()) return status;
            }
          } else if (ghost_bytes > 0) {
            // Owned rows already came from the D2D identity copy. The adapter
            // has overwritten every active non-owned row during forward
            // exchange.
            se::DeviceAddressBase ghost_dst(
                static_cast<char*>(output_data) + owned_bytes, ghost_bytes);
            status = stream->Memcpy(&ghost_dst,
                                    static_cast<char*>(host) + owned_bytes,
                                    ghost_bytes);
            if (!status.ok()) return status;
          }
        } else {
          status = stream->Memcpy(&dst, host, active_bytes);
          if (!status.ok()) return status;
          if (reverse && ghost_bytes > 0) {
            // The adapter callback deliberately leaves staged non-owned rows
            // untouched. Define the transpose result on the device in both
            // compact and full-staging modes without an extra host traversal.
            se::DeviceAddressBase ghost_dst(
                static_cast<char*>(output_data) + owned_bytes, ghost_bytes);
            status = stream->MemZero(&ghost_dst, ghost_bytes);
            if (!status.ok()) return status;
          }
        }
        // The token copy is deliberately enqueued last. Its output cannot
        // become ready until engine communication and the value copy-back
        // finish, so the next FFI call has an ordinary XLA data dependency.
        status = stream->Memcpy(&token_dst, token_src, sizeof(float));
        if (!status.ok()) return status;
        // The host workspace owns a reusable pinned scratch buffer. Because the
        // stream copies only enqueue reads from the buffer, the worker job must
        // remain alive until the stream has consumed the data. The next
        // communication job can then reuse the host storage safely.
        status = WaitForCommunicationCopy(stream);
        if (!status.ok()) return status;
        return absl::OkStatus();
      },
      [done](absl::Status status) mutable {
        // Completing this async value tells XLA that both the host callback
        // and the final host-to-device copy are finished. Dependent kernels
        // must not consume the output before this point.
        if (status.ok()) {
          done.SetStateConcrete();
        } else {
          done.SetError(std::move(status));
        }
      });
  return done;
}

tsl::AsyncValueRef<tsl::Chain> GatherForward(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output, se::Stream* stream,
    CudaCommunicationContext* binding) {
  return RunExchange(input, token_input, output, token_output, stream,
                     binding == nullptr ? nullptr : binding->context, false);
}

tsl::AsyncValueRef<tsl::Chain> GatherReverse(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output, se::Stream* stream,
    CudaCommunicationContext* binding) {
  return RunExchange(input, token_input, output, token_output, stream,
                     binding == nullptr ? nullptr : binding->context, true);
}

tsl::AsyncValueRef<tsl::Chain> RunReduce(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output, se::Stream* stream,
    CommunicationContext* context, bool transpose) {
  auto done = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();

  auto fail = [&done](std::string message) {
    done.SetError(absl::InternalError(std::move(message)));
  };

  if (stream == nullptr) {
    fail("chemtrain reduce FFI called without stream");
    return done;
  }
  if (context == nullptr) {
    fail("chemtrain reduce FFI called without CommunicationContext");
    return done;
  }
  if (context->workspace() == nullptr) {
    fail("chemtrain reduce FFI called without workspace");
    return done;
  }
  if (input.dimensions().size() != 1 || output->dimensions().size() != 1 ||
      input.dimensions()[0] != output->dimensions()[0] ||
      input.element_type() != output->element_type()) {
    fail("chemtrain reduce expects matching rank-1 input and output buffers");
    return done;
  }
  if (token_input.element_type() != xla::F32 ||
      token_output->element_type() != xla::F32 ||
      token_input.dimensions().size() != 1 ||
      token_output->dimensions().size() != 1 ||
      token_input.dimensions()[0] != 1 || token_output->dimensions()[0] != 1) {
    fail("chemtrain reduce expects a matching float32[1] ordering token");
    return done;
  }

  CommunicationScalarType scalar_type;
  if (input.element_type() == xla::F32) {
    scalar_type = CommunicationScalarType::F32;
  } else if (input.element_type() == xla::F64) {
    scalar_type = CommunicationScalarType::F64;
  } else {
    fail("chemtrain reduce supports only f32 and f64 buffers");
    return done;
  }

  const std::int64_t count = input.dimensions()[0];
  if (count <= 0 || count > std::numeric_limits<int>::max()) {
    fail(
        "chemtrain reduce expects a positive buffer size within integer range");
    return done;
  }
  const std::size_t bytes = input.size_bytes();
  void* input_data = input.untyped_data();
  void* output_data = output->untyped_data();
  void* token_input_data = token_input.untyped_data();
  void* token_output_data = token_output->untyped_data();
  CommunicationWorkspace* workspace = context->workspace();

  const std::string platform_name = stream->parent()->GetPlatform()->Name();
  const bool use_device_reduce =
      context->has_device_reduce() && IsCudaPlatform(platform_name);
  if (use_device_reduce) {
    workspace->ScheduleDevice(
        stream->parent(), bytes,
        [=](void* scratch) -> absl::Status {
          se::DeviceAddressBase src(input_data, bytes);
          se::DeviceAddressBase dst(output_data, bytes);
          se::DeviceAddressBase token_src(token_input_data, sizeof(float));
          se::DeviceAddressBase token_dst(token_output_data, sizeof(float));
          se::DeviceAddressBase scratch_dst(scratch, bytes);

          absl::Status status;
          if (!context->enabled()) {
            status = stream->Memcpy(&dst, src, bytes);
            if (!status.ok()) return status;
            status = stream->Memcpy(&token_dst, token_src, sizeof(float));
            if (!status.ok()) return status;
            return WaitForCommunicationCopy(stream);
          }

          status = stream->Memcpy(&scratch_dst, src, bytes);
          if (!status.ok()) return status;

          status = WaitForCommunicationCopy(stream);
          if (!status.ok()) return status;

          status = context->DeviceReduce(
              scratch, count, scalar_type, transpose, platform_name.c_str(),
              stream->parent()->device_ordinal(),
              stream->platform_specific_handle().stream);
          if (!status.ok()) return status;

          status = stream->Memcpy(&dst, scratch_dst, bytes);
          if (!status.ok()) return status;
          status = stream->Memcpy(&token_dst, token_src, sizeof(float));
          if (!status.ok()) return status;
          return WaitForCommunicationCopy(stream);
        },
        [done](absl::Status status) mutable {
          if (status.ok()) {
            done.SetStateConcrete();
          } else {
            done.SetError(std::move(status));
          }
        });
    return done;
  }

  workspace->Schedule(
      stream->parent(), bytes,
      [=](void* host) -> absl::Status {
        se::DeviceAddressBase src(input_data, bytes);
        se::DeviceAddressBase dst(output_data, bytes);
        se::DeviceAddressBase token_src(token_input_data, sizeof(float));
        se::DeviceAddressBase token_dst(token_output_data, sizeof(float));

        if (!context->enabled()) {
          absl::Status status = stream->Memcpy(&dst, src, bytes);
          if (!status.ok()) return status;
          status = stream->Memcpy(&token_dst, token_src, sizeof(float));
          if (!status.ok()) return status;
          return WaitForCommunicationCopy(stream);
        }

        absl::Status status = stream->Memcpy(host, src, bytes);
        if (!status.ok()) return status;

        status = WaitForCommunicationCopy(stream);
        if (!status.ok()) return status;

        status = context->Reduce(host, count, scalar_type, transpose);
        if (!status.ok()) return status;

        status = stream->Memcpy(&dst, host, bytes);
        if (!status.ok()) return status;
        status = stream->Memcpy(&token_dst, token_src, sizeof(float));
        if (!status.ok()) return status;
        return WaitForCommunicationCopy(stream);
      },
      [done](absl::Status status) mutable {
        if (status.ok()) {
          done.SetStateConcrete();
        } else {
          done.SetError(std::move(status));
        }
      });
  return done;
}

tsl::AsyncValueRef<tsl::Chain> Reduce(ffi::AnyBuffer input,
                                      ffi::AnyBuffer token_input,
                                      ffi::Result<ffi::AnyBuffer> output,
                                      ffi::Result<ffi::AnyBuffer> token_output,
                                      se::Stream* stream,
                                      CudaCommunicationContext* binding) {
  return RunReduce(input, token_input, output, token_output, stream,
                   binding == nullptr ? nullptr : binding->context, false);
}

tsl::AsyncValueRef<tsl::Chain> ReduceTranspose(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output, se::Stream* stream,
    CudaCommunicationContext* binding) {
  return RunReduce(input, token_input, output, token_output, stream,
                   binding == nullptr ? nullptr : binding->context, true);
}

absl::Status RunHostExchange(ffi::AnyBuffer input, ffi::AnyBuffer token_input,
                             ffi::Result<ffi::AnyBuffer> output,
                             ffi::Result<ffi::AnyBuffer> token_output,
                             CommunicationContext* context, bool reverse) {
  if (context == nullptr) {
    return absl::InvalidArgumentError(
        "chemtrain host exchange called without CommunicationContext");
  }
  if (input.dimensions().size() != 2 || output->dimensions().size() != 2 ||
      input.dimensions()[0] != output->dimensions()[0] ||
      input.dimensions()[1] != output->dimensions()[1] ||
      input.element_type() != output->element_type()) {
    return absl::InvalidArgumentError(
        "chemtrain host exchange expects matching rank-2 buffers");
  }
  if (token_input.element_type() != xla::F32 ||
      token_output->element_type() != xla::F32 ||
      token_input.dimensions().size() != 1 ||
      token_output->dimensions().size() != 1 ||
      token_input.dimensions()[0] != 1 || token_output->dimensions()[0] != 1) {
    return absl::InvalidArgumentError(
        "chemtrain host exchange expects a float32[1] ordering token");
  }

  CommunicationScalarType scalar_type;
  std::size_t element_bytes;
  if (input.element_type() == xla::F32) {
    scalar_type = CommunicationScalarType::F32;
    element_bytes = sizeof(float);
  } else if (input.element_type() == xla::F64) {
    scalar_type = CommunicationScalarType::F64;
    element_bytes = sizeof(double);
  } else {
    return absl::InvalidArgumentError(
        "chemtrain host exchange supports only f32 and f64 buffers");
  }

  const std::int64_t rows = input.dimensions()[0];
  const std::int64_t cols = input.dimensions()[1];
  auto row_bounds = context->ValidateRowBounds(rows);
  if (!row_bounds.ok()) return row_bounds.status();

  std::memcpy(output->untyped_data(), input.untyped_data(), input.size_bytes());
  if (context->enabled()) {
    absl::Status status =
        context->Exchange(output->untyped_data(), row_bounds->active_rows, cols,
                          scalar_type, reverse);
    if (!status.ok()) return status;

    if (reverse && row_bounds->owned_rows < row_bounds->active_rows) {
      const std::size_t owned_bytes =
          static_cast<std::size_t>(row_bounds->owned_rows) *
          static_cast<std::size_t>(cols) * element_bytes;
      const std::size_t ghost_bytes =
          static_cast<std::size_t>(row_bounds->active_rows -
                                   row_bounds->owned_rows) *
          static_cast<std::size_t>(cols) * element_bytes;
      std::memset(static_cast<char*>(output->untyped_data()) + owned_bytes, 0,
                  ghost_bytes);
    }
  }
  std::memcpy(token_output->untyped_data(), token_input.untyped_data(),
              sizeof(float));
  return absl::OkStatus();
}

absl::Status HostGatherForward(ffi::AnyBuffer input, ffi::AnyBuffer token_input,
                               ffi::Result<ffi::AnyBuffer> output,
                               ffi::Result<ffi::AnyBuffer> token_output,
                               HostCommunicationContext* binding) {
  return RunHostExchange(input, token_input, output, token_output,
                         binding == nullptr ? nullptr : binding->context, false);
}

absl::Status HostGatherReverse(ffi::AnyBuffer input, ffi::AnyBuffer token_input,
                               ffi::Result<ffi::AnyBuffer> output,
                               ffi::Result<ffi::AnyBuffer> token_output,
                               HostCommunicationContext* binding) {
  return RunHostExchange(input, token_input, output, token_output,
                         binding == nullptr ? nullptr : binding->context, true);
}

absl::Status RunHostReduce(ffi::AnyBuffer input, ffi::AnyBuffer token_input,
                           ffi::Result<ffi::AnyBuffer> output,
                           ffi::Result<ffi::AnyBuffer> token_output,
                           CommunicationContext* context, bool transpose) {
  if (context == nullptr) {
    return absl::InvalidArgumentError(
        "chemtrain host reduce called without CommunicationContext");
  }
  if (input.dimensions().size() != 1 || output->dimensions().size() != 1 ||
      input.dimensions()[0] != output->dimensions()[0] ||
      input.element_type() != output->element_type()) {
    return absl::InvalidArgumentError(
        "chemtrain host reduce expects matching rank-1 buffers");
  }
  if (token_input.element_type() != xla::F32 ||
      token_output->element_type() != xla::F32 ||
      token_input.dimensions().size() != 1 ||
      token_output->dimensions().size() != 1 ||
      token_input.dimensions()[0] != 1 || token_output->dimensions()[0] != 1) {
    return absl::InvalidArgumentError(
        "chemtrain host reduce expects a float32[1] ordering token");
  }

  CommunicationScalarType scalar_type;
  if (input.element_type() == xla::F32) {
    scalar_type = CommunicationScalarType::F32;
  } else if (input.element_type() == xla::F64) {
    scalar_type = CommunicationScalarType::F64;
  } else {
    return absl::InvalidArgumentError(
        "chemtrain host reduce supports only f32 and f64 buffers");
  }

  const std::int64_t count = input.dimensions()[0];
  if (count <= 0 || count > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        "chemtrain host reduce expects a positive buffer size");
  }
  std::memcpy(output->untyped_data(), input.untyped_data(), input.size_bytes());
  if (context->enabled()) {
    absl::Status status =
        context->Reduce(output->untyped_data(), count, scalar_type, transpose);
    if (!status.ok()) return status;
  }
  std::memcpy(token_output->untyped_data(), token_input.untyped_data(),
              sizeof(float));
  return absl::OkStatus();
}

absl::Status HostReduce(ffi::AnyBuffer input, ffi::AnyBuffer token_input,
                        ffi::Result<ffi::AnyBuffer> output,
                        ffi::Result<ffi::AnyBuffer> token_output,
                        HostCommunicationContext* binding) {
  return RunHostReduce(input, token_input, output, token_output,
                       binding == nullptr ? nullptr : binding->context, false);
}

absl::Status HostReduceTranspose(ffi::AnyBuffer input,
                                 ffi::AnyBuffer token_input,
                                 ffi::Result<ffi::AnyBuffer> output,
                                 ffi::Result<ffi::AnyBuffer> token_output,
                                 HostCommunicationContext* binding) {
  return RunHostReduce(input, token_input, output, token_output,
                       binding == nullptr ? nullptr : binding->context, true);
}

XLA_FFI_DEFINE_HANDLER(kGatherInstantiate, InstantiateGather,
                       ffi::Ffi::BindInstantiate()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER(kReduceInstantiate, InstantiateReduce,
                       ffi::Ffi::BindInstantiate()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER(kHostInstantiate, InstantiateHost,
                       ffi::Ffi::BindInstantiate());

XLA_FFI_DEFINE_HANDLER(kPrepare, NoopLifecycleStage,
                       ffi::Ffi::BindPrepare());

XLA_FFI_DEFINE_HANDLER(kInitialize, NoopLifecycleStage,
                       ffi::Ffi::BindInitialize());

XLA_FFI_DEFINE_HANDLER(kGatherForward, GatherForward,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::UserData<CudaCommunicationContext>>());

XLA_FFI_DEFINE_HANDLER(kGatherReverse, GatherReverse,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::UserData<CudaCommunicationContext>>());

XLA_FFI_DEFINE_HANDLER(kReduce, Reduce,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::UserData<CudaCommunicationContext>>());

XLA_FFI_DEFINE_HANDLER(kReduceTranspose, ReduceTranspose,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::UserData<CudaCommunicationContext>>());

XLA_FFI_DEFINE_HANDLER(kHostGatherForward, HostGatherForward,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<HostCommunicationContext>>());

XLA_FFI_DEFINE_HANDLER(kHostGatherReverse, HostGatherReverse,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<HostCommunicationContext>>());

XLA_FFI_DEFINE_HANDLER(kHostReduce, HostReduce,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<HostCommunicationContext>>());

XLA_FFI_DEFINE_HANDLER(kHostReduceTranspose, HostReduceTranspose,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<HostCommunicationContext>>());

// The StableHLO modules exported today call targets named
// `chemtrain_deploy.gather_forward` and `chemtrain_deploy.gather_reverse`.
// Keep the handler symbols aligned with those stable target names even though
// the implementation is now documented as a generic per-atom exchange.

}  // namespace

XLA_FFI_Handler* GatherInstantiateHandler() { return kGatherInstantiate; }

XLA_FFI_Handler* ReduceInstantiateHandler() { return kReduceInstantiate; }

XLA_FFI_Handler* HostInstantiateHandler() { return kHostInstantiate; }

XLA_FFI_Handler* PrepareHandler() { return kPrepare; }

XLA_FFI_Handler* InitializeHandler() { return kInitialize; }

XLA_FFI_Handler* GatherForwardHandler() { return kGatherForward; }

XLA_FFI_Handler* GatherReverseHandler() { return kGatherReverse; }

XLA_FFI_Handler* ReduceHandler() { return kReduce; }

XLA_FFI_Handler* ReduceTransposeHandler() { return kReduceTranspose; }

XLA_FFI_Handler* HostGatherForwardHandler() { return kHostGatherForward; }

XLA_FFI_Handler* HostGatherReverseHandler() { return kHostGatherReverse; }

XLA_FFI_Handler* HostReduceHandler() { return kHostReduce; }

XLA_FFI_Handler* HostReduceTransposeHandler() { return kHostReduceTranspose; }

}  // namespace jcn
