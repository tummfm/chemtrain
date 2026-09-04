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

#include "connector/communication/runtime.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"

namespace jcn {

// ---------------------------------------------------------------------------
// Workspace: serialized staging for FFI callbacks
// ---------------------------------------------------------------------------

CommunicationWorkspace::CommunicationWorkspace()
    : worker_(&CommunicationWorkspace::WorkerLoop, this) {}

CommunicationWorkspace::~CommunicationWorkspace() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  ready_.notify_one();
  if (worker_.joinable()) worker_.join();
  if (device_buffer_executor_ != nullptr &&
      device_buffer_.opaque() != nullptr) {
    device_buffer_executor_->Deallocate(&device_buffer_);
  }
}

void CommunicationWorkspace::Schedule(
    stream_executor::StreamExecutor* executor, std::size_t bytes, Task task,
    Completion completion) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_.push(Job{executor, bytes, std::move(task), std::move(completion),
                   false});
  }
  ready_.notify_one();
}

void CommunicationWorkspace::ScheduleDevice(
    stream_executor::StreamExecutor* executor, std::size_t bytes, Task task,
    Completion completion) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_.push(Job{executor, bytes, std::move(task), std::move(completion),
                   true});
  }
  ready_.notify_one();
}

void CommunicationWorkspace::WorkerLoop() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
      if (stopping_ && jobs_.empty()) return;
      job = std::move(jobs_.front());
      jobs_.pop();
    }

    absl::Status status;
    try {
      if (job.executor == nullptr) {
        status = absl::InternalError(
            "communication job has no StreamExecutor");
      } else {
        const std::size_t allocation_bytes =
            std::max<std::size_t>(job.bytes, 1);
        if (job.device) {
          if (device_buffer_executor_ != job.executor) {
            if (device_buffer_.opaque() != nullptr) {
              device_buffer_executor_->Deallocate(&device_buffer_);
            }
            device_buffer_executor_ = job.executor;
            device_buffer_capacity_ = 0;
          }
          if (device_buffer_.opaque() == nullptr ||
              device_buffer_capacity_ < allocation_bytes) {
            if (device_buffer_.opaque() != nullptr) {
              job.executor->Deallocate(&device_buffer_);
            }
            device_buffer_ = job.executor->Allocate(allocation_bytes);
            if (device_buffer_.opaque() == nullptr) {
              status = absl::InternalError(
                  "failed to allocate device communication scratch buffer");
            } else {
              device_buffer_capacity_ = allocation_bytes;
            }
          }
          if (status.ok()) status = job.task(device_buffer_.opaque());
        } else {
          if (buffer_ == nullptr || buffer_executor_ != job.executor ||
              buffer_capacity_ < allocation_bytes) {
            // One worker serializes all jobs, so a pinned allocation can
            // be reused safely. Keep a non-null one-byte allocation for empty
            // ranks with no active rows, which still participate in adapter
            // communication.
            auto allocation = job.executor->HostMemoryAllocate(allocation_bytes);
            if (!allocation.ok()) {
              status = allocation.status();
            } else {
              buffer_ = std::move(allocation).value();
              buffer_executor_ = job.executor;
              buffer_capacity_ = allocation_bytes;
            }
          }
          if (status.ok()) {
            status = job.task(buffer_->address().opaque());
          }
        }
      }
    } catch (const std::exception& error) {
      status = absl::InternalError(error.what());
    } catch (...) {
      status = absl::InternalError("unknown communication worker failure");
    }
    job.completion(std::move(status));
  }
}

// ---------------------------------------------------------------------------
// CommunicationContext: worker-thread FFI requests serviced by adapter thread
// ---------------------------------------------------------------------------

CommunicationContext::CommunicationContext(CommunicationCallbacks callbacks,
                                           bool enabled,
                                           CommunicationWorkspace* workspace,
                                           std::int64_t owned_rows,
                                           std::int64_t active_rows,
                                           int communication_buffer_width)
    : callbacks_(callbacks),
      enabled_(enabled),
      workspace_(workspace),
      owned_rows_(owned_rows),
      active_rows_(active_rows),
      communication_buffer_width_(communication_buffer_width) {}

absl::StatusOr<CommunicationContext::RowBounds>
CommunicationContext::ValidateRowBounds(std::int64_t capacity) const {
  if (capacity < 0) {
    return absl::InvalidArgumentError(
        "communication buffer row capacity must be non-negative");
  }
  if (!enabled_) return RowBounds{capacity, capacity};

  // The adapter stores owned atoms first and any additional valid rows directly
  // afterward. Runner supplies both prefixes so FFI transfers can skip inactive
  // padding while preserving identity semantics for rows outside active_rows.
  if (active_rows_ < 0 || active_rows_ > capacity) {
    return absl::InvalidArgumentError(
        "active communication rows are outside the communication buffer capacity");
  }
  if (owned_rows_ < 0 || owned_rows_ > active_rows_) {
    return absl::InvalidArgumentError(
        "owned communication rows are outside the active communication rows");
  }
  return RowBounds{owned_rows_, active_rows_};
}

absl::Status CommunicationContext::ValidateCommunicationWidth(
    std::int64_t width, const char* operation) const {
  if (!enabled_ || communication_buffer_width_ <= 0) {
    return absl::OkStatus();
  }
  if (width > communication_buffer_width_) {
    return absl::FailedPreconditionError(
        std::string(operation) + " communication width " +
        std::to_string(width) + " exceeds exported buffer width " +
        std::to_string(communication_buffer_width_));
  }
  return absl::OkStatus();
}

absl::Status CommunicationContext::Exchange(
    void* data, std::int64_t rows, std::int64_t cols,
    CommunicationScalarType type, bool reverse) {
  if (!enabled_) return absl::OkStatus();

  if (callbacks_.exchange == nullptr) {
    return absl::FailedPreconditionError(
        "communicating model executed without engine communication callbacks");
  }

  std::unique_lock<std::mutex> lock(mutex_);

  if (cols <= 0) {
    return absl::InvalidArgumentError(
        "communication width must be positive");
  }
  if (cols > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        "communication width exceeds integer range");
  }
  absl::Status validation = ValidateCommunicationWidth(
      cols, reverse ? "transpose/reverse exchange" : "primal/forward exchange");
  if (!validation.ok()) return validation;

  request_done_.wait(lock, [this] { return !pending_; });

  data_ = data;
  rows_ = rows;
  cols_ = cols;
  type_ = type;
  reverse_ = reverse;
  request_kind_ = RequestKind::kExchange;
  backend_.clear();
  device_ordinal_ = -1;
  stream_ = nullptr;
  error_.clear();
  completed_ = false;
  pending_ = true;

  request_ready_.notify_one();
  request_done_.wait(lock, [this] { return completed_; });

  std::string error = error_;

  pending_ = false;
  completed_ = false;
  request_done_.notify_all();

  if (!error.empty()) return absl::InternalError(error);
  return absl::OkStatus();
}

absl::Status CommunicationContext::Reduce(
    void* data, std::int64_t count, CommunicationScalarType type,
    bool transpose) {
  if (!enabled_) return absl::OkStatus();

  if (callbacks_.reduce == nullptr) {
    return absl::FailedPreconditionError(
        "communicating model executed without engine reduce callbacks");
  }

  if (data == nullptr || count <= 0) {
    return absl::InvalidArgumentError(
        "reduce expects a non-empty scalar/vector buffer");
  }
  if (count > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        "reduce buffer width exceeds integer range");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  absl::Status validation = ValidateCommunicationWidth(
      count, transpose ? "transpose/reverse reduce" : "primal/forward reduce");
  if (!validation.ok()) return validation;
  request_done_.wait(lock, [this] { return !pending_; });

  data_ = data;
  rows_ = count;
  cols_ = 1;
  type_ = type;
  reverse_ = false;
  request_kind_ = RequestKind::kReduce;
  backend_.clear();
  device_ordinal_ = -1;
  stream_ = nullptr;
  error_.clear();
  completed_ = false;
  pending_ = true;

  request_ready_.notify_one();
  request_done_.wait(lock, [this] { return completed_; });

  std::string error = error_;

  pending_ = false;
  completed_ = false;
  request_done_.notify_all();

  if (!error.empty()) return absl::InternalError(error);
  return absl::OkStatus();
}

absl::Status CommunicationContext::DeviceExchange(
    void* data, std::int64_t rows, std::int64_t cols,
    CommunicationScalarType type, bool reverse, const char* backend,
    int device_ordinal, void* stream) {
  if (!enabled_) return absl::OkStatus();

  if (callbacks_.device_exchange == nullptr) {
    return absl::FailedPreconditionError(
        "communicating model executed without engine device communication callbacks");
  }

  std::unique_lock<std::mutex> lock(mutex_);

  if (cols <= 0) {
    return absl::InvalidArgumentError(
        "device communication width must be positive");
  }
  if (cols > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        "device communication width exceeds integer range");
  }
  absl::Status validation = ValidateCommunicationWidth(
      cols,
      reverse ? "device transpose/reverse exchange"
              : "device primal/forward exchange");
  if (!validation.ok()) return validation;

  request_done_.wait(lock, [this] { return !pending_; });

  data_ = data;
  rows_ = rows;
  cols_ = cols;
  type_ = type;
  reverse_ = reverse;
  request_kind_ = RequestKind::kDeviceExchange;
  backend_ = backend == nullptr ? "" : backend;
  device_ordinal_ = device_ordinal;
  stream_ = stream;
  error_.clear();
  completed_ = false;
  pending_ = true;

  request_ready_.notify_one();
  request_done_.wait(lock, [this] { return completed_; });

  std::string error = error_;

  pending_ = false;
  completed_ = false;
  request_done_.notify_all();

  if (!error.empty()) return absl::InternalError(error);
  return absl::OkStatus();
}

absl::Status CommunicationContext::DeviceReduce(
    void* data, std::int64_t count, CommunicationScalarType type,
    bool transpose, const char* backend, int device_ordinal, void* stream) {
  if (!enabled_) return absl::OkStatus();

  if (callbacks_.device_reduce == nullptr) {
    return absl::FailedPreconditionError(
        "communicating model executed without engine device reduce callbacks");
  }

  if (data == nullptr || count <= 0) {
    return absl::InvalidArgumentError(
        "device reduce expects a non-empty scalar/vector buffer");
  }
  if (count > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        "device reduce buffer width exceeds integer range");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  absl::Status validation = ValidateCommunicationWidth(
      count,
      transpose ? "device transpose/reverse reduce"
                : "device primal/forward reduce");
  if (!validation.ok()) return validation;
  request_done_.wait(lock, [this] { return !pending_; });

  data_ = data;
  rows_ = count;
  cols_ = 1;
  type_ = type;
  reverse_ = false;
  request_kind_ = RequestKind::kDeviceReduce;
  backend_ = backend == nullptr ? "" : backend;
  device_ordinal_ = device_ordinal;
  stream_ = stream;
  error_.clear();
  completed_ = false;
  pending_ = true;

  request_ready_.notify_one();
  request_done_.wait(lock, [this] { return completed_; });

  std::string error = error_;

  pending_ = false;
  completed_ = false;
  request_done_.notify_all();

  if (!error.empty()) return absl::InternalError(error);
  return absl::OkStatus();
}

bool CommunicationContext::ServiceOne() {
  void* data;
  std::int64_t rows;
  std::int64_t cols;
  CommunicationScalarType type;
  bool reverse;
  RequestKind request_kind;
  std::string backend;
  int device_ordinal;
  void* stream;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_ || servicing_ || completed_) return false;

    servicing_ = true;
    data = data_;
    rows = rows_;
    cols = cols_;
    type = type_;
    reverse = reverse_;
    request_kind = request_kind_;
    backend = backend_;
    device_ordinal = device_ordinal_;
    stream = stream_;
  }

  const char* callback_error = nullptr;
  int rc = 1;
  switch (request_kind) {
    case RequestKind::kReduce:
      rc = callbacks_.reduce(callbacks_.context, data, rows, type,
                             &callback_error);
      break;
    case RequestKind::kDeviceReduce:
      rc = callbacks_.device_reduce(callbacks_.context, data, rows, type,
                                    backend.c_str(), device_ordinal, stream,
                                    &callback_error);
      break;
    case RequestKind::kDeviceExchange:
      rc = callbacks_.device_exchange(callbacks_.context, data, rows, cols, type,
                                      reverse, backend.c_str(), device_ordinal,
                                      stream, &callback_error);
      break;
    case RequestKind::kExchange:
      rc = callbacks_.exchange(callbacks_.context, data, rows, cols, type,
                               reverse, &callback_error);
      break;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    error_.clear();

    if (rc != 0) {
      error_ = callback_error == nullptr
                   ? "engine communication callback failed"
                   : callback_error;
    }

    servicing_ = false;
    completed_ = true;
  }

  request_done_.notify_all();
  return true;
}

bool CommunicationContext::HasPending() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pending_ && !completed_;
}

void CommunicationContext::BeginExecution() {
  std::lock_guard<std::mutex> lock(mutex_);
  execution_complete_ = false;
}

absl::Status CommunicationContext::ValidateExecution() const {
  return absl::OkStatus();
}

void CommunicationContext::NotifyExecutionComplete() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    execution_complete_ = true;
  }
  request_ready_.notify_all();
}

void CommunicationContext::ServiceUntilExecutionComplete() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // PJRT executes on a worker, but engine communication must remain on the
      // caller thread. Wake the caller thread only for a callback or after all
      // device work has completed. No periodic polling is needed.
      request_ready_.wait(lock, [this] {
        return execution_complete_ ||
               (pending_ && !servicing_ && !completed_);
      });

      // A communication request takes priority over execution completion.
      // In practice PJRT cannot finish while an async FFI result is pending,
      // The ordering also makes shutdown robust to future FFI changes.
      if (!(pending_ && !servicing_ && !completed_) &&
          execution_complete_) {
        return;
      }
    }
    ServiceOne();
  }
}
}  // namespace jcn
