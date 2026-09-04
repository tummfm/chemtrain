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

#ifndef CHEMTRAIN_DEPLOY_COMMUNICATION_RUNTIME_H_
#define CHEMTRAIN_DEPLOY_COMMUNICATION_RUNTIME_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "connector/runtime_types.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"

namespace stream_executor {
class MemoryAllocation;
class StreamExecutor;
}  // namespace stream_executor

namespace jcn {

// Owns one persistent pinned host buffer and a worker that executes ordered
// communication jobs. A Runner has one workspace, so forward and reverse
// calls reuse storage safely across timesteps.
class CommunicationWorkspace {
 public:
  CommunicationWorkspace();
  ~CommunicationWorkspace();

  CommunicationWorkspace(const CommunicationWorkspace&) = delete;
  CommunicationWorkspace& operator=(const CommunicationWorkspace&) = delete;

  using Task = std::function<absl::Status(void*)>;
  using Completion = std::function<void(absl::Status)>;

  void Schedule(stream_executor::StreamExecutor* executor, std::size_t bytes,
                Task task, Completion completion);
  void ScheduleDevice(stream_executor::StreamExecutor* executor,
                      std::size_t bytes, Task task, Completion completion);

 private:
  struct Job {
    stream_executor::StreamExecutor* executor;
    std::size_t bytes;
    Task task;
    Completion completion;
    bool device = false;
  };

  void WorkerLoop();

  std::mutex mutex_;
  std::condition_variable ready_;
  std::queue<Job> jobs_;
  bool stopping_ = false;
  std::thread worker_;

  stream_executor::StreamExecutor* buffer_executor_ = nullptr;
  std::size_t buffer_capacity_ = 0;
  std::unique_ptr<stream_executor::MemoryAllocation> buffer_;

  stream_executor::StreamExecutor* device_buffer_executor_ = nullptr;
  std::size_t device_buffer_capacity_ = 0;
  stream_executor::DeviceAddressBase device_buffer_;
};

// Per-execution rendezvous between PJRT's asynchronous FFI worker and the
// caller thread that owns engine communication. It checks every communication
// call against the maximum packed width recorded during export.
class CommunicationContext {
 public:
  struct RowBounds {
    std::int64_t owned_rows;
    std::int64_t active_rows;
  };

  CommunicationContext(CommunicationCallbacks callbacks, bool enabled,
                       CommunicationWorkspace* workspace,
                       std::int64_t owned_rows,
                       std::int64_t active_rows,
                       int communication_buffer_width = 0);

  absl::Status Exchange(void* data, std::int64_t rows, std::int64_t cols,
                        CommunicationScalarType type, bool reverse);
  absl::Status Reduce(void* data, std::int64_t count,
                      CommunicationScalarType type, bool transpose);
  absl::Status DeviceExchange(void* data, std::int64_t rows,
                              std::int64_t cols,
                              CommunicationScalarType type, bool reverse,
                              const char* backend, int device_ordinal,
                              void* stream);
  absl::Status DeviceReduce(void* data, std::int64_t count,
                            CommunicationScalarType type, bool transpose,
                            const char* backend, int device_ordinal,
                            void* stream);
  bool ServiceOne();
  bool HasPending() const;
  void BeginExecution();
  void NotifyExecutionComplete();
  void ServiceUntilExecutionComplete();
  absl::Status ValidateExecution() const;
  bool enabled() const { return enabled_; }
  absl::StatusOr<RowBounds> ValidateRowBounds(std::int64_t capacity) const;
  CommunicationWorkspace* workspace() const { return workspace_; }
  bool has_device_exchange() const {
    return callbacks_.device_exchange != nullptr;
  }
  bool has_device_reduce() const {
    return callbacks_.device_reduce != nullptr;
  }

 private:
  enum class RequestKind {
    kExchange,
    kDeviceExchange,
    kReduce,
    kDeviceReduce,
  };

  CommunicationCallbacks callbacks_;
  bool enabled_;
  CommunicationWorkspace* workspace_;
  std::int64_t owned_rows_;
  std::int64_t active_rows_;

  mutable std::mutex mutex_;
  std::condition_variable request_ready_;
  std::condition_variable request_done_;

  bool pending_ = false;
  bool servicing_ = false;
  bool completed_ = false;
  bool execution_complete_ = false;
  int communication_buffer_width_ = 0;

  absl::Status ValidateCommunicationWidth(std::int64_t width,
                                          const char* operation) const;

  void* data_ = nullptr;
  std::int64_t rows_ = 0;
  std::int64_t cols_ = 0;
  CommunicationScalarType type_ = CommunicationScalarType::F32;
  bool reverse_ = false;
  RequestKind request_kind_ = RequestKind::kExchange;
  std::string backend_;
  int device_ordinal_ = -1;
  void* stream_ = nullptr;
  std::string error_;
};

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_COMMUNICATION_RUNTIME_H_
