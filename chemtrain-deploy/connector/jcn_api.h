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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_JCN_API_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_JCN_API_H_

#include <stddef.h>
#include <stdint.h>

#include "dlpack/dlpack.h"

#if defined(_WIN32)
#define JCN_EXPORT __declspec(dllexport)
#else
#define JCN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define JCN_API_VERSION 10

/*
 * JCN is the stable C boundary between engine adapters and chemtrain-deploy:
 *
 *   client -> model -> executor -> capacities -> DLPack buffers -> force call.
 *
 * The engine adapter owns atom storage, device views, neighbor-list conversion,
 * row/image/alias mapping, communication, collective retry decisions, and the
 * lifetime of the underlying memory.  The connector owns PJRT
 * client/executable state and validates that every imported DLPack handle
 * matches that PJRT backend.  If static capacities are too small, the force
 * call reports rank-local minima and the adapter performs the collective retry.
 *
 * CPU DLPack buffers are staged for each compute call. Device buffers use a
 * same-device PJRT view and do not fall back to a device copy. Engine memory
 * may be updated between calls, but it must not be mutated while a compute call
 * is using it.
 */

/* Opaque handles.  Callers use the method table below instead of touching C++
 * implementation objects directly. */
typedef struct jcn_client jcn_client;
typedef struct jcn_model jcn_model;
typedef struct jcn_executor jcn_executor;
typedef struct jcn_buffer jcn_buffer;
typedef struct jcn_neighbor_list jcn_neighbor_list;

/* Status and compute outcomes. */
typedef enum jcn_status_code {
  JCN_STATUS_OK = 0,
  JCN_STATUS_INVALID_ARGUMENT = 1,
  JCN_STATUS_NOT_FOUND = 2,
  JCN_STATUS_FAILED_PRECONDITION = 3,
  JCN_STATUS_INTERNAL = 4,
} jcn_status_code;

typedef struct jcn_status {
  jcn_status_code code;
  // Pointer is valid until the next JCN API call on the same thread.
  const char* message;
} jcn_status;

typedef enum jcn_neighbor_format {
  JCN_NEIGHBOR_UNSPECIFIED = 0,
  JCN_NEIGHBOR_SIMPLE_DENSE = 2,
  JCN_NEIGHBOR_SIMPLE_SPARSE = 3,
} jcn_neighbor_format;

typedef enum jcn_compute_code {
  JCN_COMPUTE_OK = 0,
  JCN_COMPUTE_NEEDS_CAPACITY_CHANGE = 1,
  JCN_COMPUTE_FAILED = 2,
} jcn_compute_code;

typedef enum jcn_communication_scalar_type {
  JCN_COMMUNICATION_F32 = 0,
  JCN_COMMUNICATION_F64 = 1,
} jcn_communication_scalar_type;

typedef enum jcn_dlpack_copy_policy {
  JCN_DLPACK_COPY_IF_NEEDED = 0,
  JCN_DLPACK_NEVER_COPY = 1,
  JCN_DLPACK_ALWAYS_COPY = 2,
} jcn_dlpack_copy_policy;

typedef enum jcn_buffer_role {
  JCN_BUFFER_ROLE_INPUT = 0,
  JCN_BUFFER_ROLE_OUTPUT = 1,
  JCN_BUFFER_ROLE_INPUT_OUTPUT = 2,
} jcn_buffer_role;

typedef enum jcn_tensor_dtype {
  // The meaning of MODEL_DEFAULT depends on the ABI field. Positions, species,
  // and default outputs currently resolve to float32, int32, and float32,
  // respectively. Communication keeps the dtype chosen by the exported
  // communication operation.
  JCN_TENSOR_DTYPE_MODEL_DEFAULT = 0,
  JCN_TENSOR_DTYPE_F32 = 1,
  JCN_TENSOR_DTYPE_F64 = 2,
  JCN_TENSOR_DTYPE_S32 = 3,
} jcn_tensor_dtype;

typedef enum jcn_species_encoding {
  JCN_SPECIES_ZERO_BASED = 0,
  JCN_SPECIES_ONE_BASED = 1,
} jcn_species_encoding;

typedef enum jcn_dense_layout {
  JCN_DENSE_LAYOUT_CONTIGUOUS = 0,
} jcn_dense_layout;

typedef struct jcn_named_tensor_dtype {
  const char* name;
  jcn_tensor_dtype dtype;
} jcn_named_tensor_dtype;

typedef struct jcn_particle_field_descriptor {
  // Exact exported input name. Names are case-sensitive and are not aliases.
  // Descriptor zero is always canonical zero-based species. Remaining fields
  // have model-defined semantics documented with the exported model.
  const char* name;
  // Particle fields are scalar int32 arrays with shape [max_atoms].
  jcn_tensor_dtype dtype;
} jcn_particle_field_descriptor;

typedef struct jcn_global_field_descriptor {
  // Exact exported scalar input name. Names are case-sensitive and are not
  // aliases.
  const char* name;
  jcn_tensor_dtype dtype;
} jcn_global_field_descriptor;

typedef enum jcn_output_scope {
  JCN_OUTPUT_PARTICLE = 1,
  JCN_OUTPUT_LOCAL = 2,
  JCN_OUTPUT_GLOBAL = 3,
} jcn_output_scope;

typedef struct jcn_output_descriptor {
  // Exact exported output name and reduction scope. PARTICLE values carry a
  // leading atom dimension. LOCAL values are additive rank-local quantities,
  // while GLOBAL values are already complete and must not be reduced again.
  const char* name;
  jcn_output_scope scope;
  // Logical value shape without the particle axis. The engine ABI selects the
  // dtype, and the compiler wrapper applies the selection. Output metadata does
  // not store a dtype.
  const int64_t* dimensions;
  int32_t rank;
  int32_t components;
  // Nonzero when a configuration value scales with system size. This controls
  // engine-side normalization, not MPI reduction.
  int32_t extensive;
} jcn_output_descriptor;

/* PJRT client setup.  Backend/device must match the DLPack device imported
 * later for atom and neighbor buffers. */
typedef struct jcn_client_options {
  // Backend name used by PJRT, e.g. "cpu", "cuda", "rocm", or "sycl".
  const char* backend;
  int32_t device_ordinal;
  float memory_fraction;
  int32_t visible_device_count;
  const int32_t* visible_devices;
} jcn_client_options;

/* Build/runtime metadata. Engine adapters log this once at initialization so
 * mismatched connector/PJRT installations are easier to diagnose. String
 * pointers are owned by libconnector and remain valid for the process
 * lifetime. */
typedef struct jcn_runtime_info {
  uint32_t api_version;
  size_t api_struct_size;
  const char* connector_build_version;
  const char* xla_commit;
  const char* xla_sha256;
  const char* build_cuda_version;
  const char* build_cudnn_version;
  const char* build_cuda_compute_capabilities;
} jcn_runtime_info;

/* Model attachment options. The adapter chooses whether the deployed
 * executable should use the communication-aware StableHLO variant. The
 * connector verifies that the loaded model contains a matching variant
 * and that callbacks were registered before execution can enter adapter-owned
 * communication. */
typedef struct jcn_model_options {
  int32_t newton_pair;
  int32_t use_communication;
} jcn_model_options;

/* Engine ABI settings for the compiled entrypoint. The exported model keeps
 * its canonical ABI internally, but the compiler may wrap it so adapters can
 * pass dense engine-native buffers directly. The current implementation
 * supports only contiguous dense tensors. Communication dtype records the
 * preferred adapter precision for communication FFI operands and results. */
typedef struct jcn_engine_abi_options {
  jcn_tensor_dtype position_dtype;
  jcn_tensor_dtype species_dtype;
  jcn_tensor_dtype default_output_dtype;
  const jcn_named_tensor_dtype* output_dtypes;
  int32_t num_output_dtypes;
  jcn_tensor_dtype communication_dtype;
  jcn_species_encoding species_encoding;
  jcn_dense_layout atom_layout;
  jcn_dense_layout output_layout;
} jcn_engine_abi_options;

/* Communication callbacks installed by the engine adapter. The connector calls
 * these from its PJRT/FFI service loop whenever the model reaches a per-atom
 * exchange or global reduction. `exchange` receives a host-staged `[rows,
 * cols]` value buffer. `device_exchange` receives connector-owned device
 * scratch plus the backend/device metadata needed by a device-native adapter.
 * `reduce` performs a host-staged sum over a rank-1 buffer. `device_reduce`
 * performs the same sum on connector-owned device scratch. Device scratch is a
 * borrowed, contiguous, read-write view that is valid only until the callback
 * returns. Callbacks must not retain the pointer and must order their device
 * work with the supplied backend-native stream before returning. */
typedef struct jcn_communication_callbacks {
  void* context;
  int (*exchange)(void* context, void* data, int64_t rows, int64_t cols,
                  jcn_communication_scalar_type type, int32_t reverse,
                  const char** error);
  int (*device_exchange)(void* context, void* data, int64_t rows, int64_t cols,
                         jcn_communication_scalar_type type, int32_t reverse,
                         const char* backend, int32_t device_ordinal,
                         void* stream, const char** error);
  int (*reduce)(void* context, void* data, int64_t count,
                jcn_communication_scalar_type type, const char** error);
  int (*device_reduce)(void* context, void* data, int64_t count,
                       jcn_communication_scalar_type type, const char* backend,
                       int32_t device_ordinal, void* stream,
                       const char** error);
} jcn_communication_callbacks;

/* CPU COPY_IF_NEEDED and ALWAYS_COPY buffers are staged for each compute call;
 * CPU NEVER_COPY is rejected. Exact-backend, exact-device COPY_IF_NEEDED and
 * NEVER_COPY buffers share the device allocation without a copy fallback;
 * device ALWAYS_COPY is rejected. Successful import transfers wrapper
 * ownership to jcn_buffer, while the underlying engine memory remains
 * caller-owned and must remain valid until buffer_destroy. The engine may
 * update that memory between compute calls, but it must not mutate it while a
 * call is active. producer_stream is an optional backend-native stream handle
 * that orders PJRT access after engine-side device work. For output buffers,
 * the policy controls how caller storage is bound; model results are still
 * copied back into that storage. */
typedef struct jcn_buffer_import_options {
  jcn_dlpack_copy_policy copy_policy;
  jcn_buffer_role role;
  const char* debug_name;
  const char* backend;
  int32_t device_ordinal;
  void* producer_stream;
} jcn_buffer_import_options;

/* Model properties that adapters need before they can request neighbor lists
 * and communication buffers. String and particle descriptor pointers are
 * borrowed from jcn_model and remain valid until model_destroy. A model
 * attached to an executor must therefore outlive that executor and every
 * borrowed pointer. */
typedef struct jcn_model_properties {
  double cutoff;
  double comm_dist;
  const char* unit_style;
  int32_t include_ghosts;
  int32_t half_list;
  int32_t communication_buffer_width;
  jcn_neighbor_format neighbor_format;
  const jcn_particle_field_descriptor* particle_fields;
  int32_t num_particle_fields;
  const jcn_global_field_descriptor* global_fields;
  int32_t num_global_fields;
  const jcn_output_descriptor* outputs;
  int32_t num_outputs;
  int32_t include_pair_type;
  int32_t requires_communication;
} jcn_model_properties;

/* Explicit rank-local capacities chosen by the adapter for caller-owned
 * buffers. Abstract model capacities are connector-owned and are not part of
 * this runtime request. */
typedef struct jcn_requested_capacities {
  // Capacities are concrete rank-local sizes chosen by the adapter after it has
  // converted its atom and raw neighbor-list storage into the deployed model
  // format. The connector may report larger adapter-owned required minima, but
  // the adapter decides when all ranks enter the resize/recompile path.
  int64_t max_atoms;
  int64_t raw_edges;
  int64_t max_neighbors_per_atom;
} jcn_requested_capacities;

typedef struct jcn_named_input {
  // Must exactly match one descriptor returned in model properties. Every
  // descriptor must occur exactly once. Table order is irrelevant. Values are
  // read on every force call and may be updated between evaluations. The name
  // pointer and containing table need only remain valid for the call.
  const char* name;
  jcn_buffer* buffer;
} jcn_named_input;

/* DLPack-backed atom buffers. Positions and named particle fields are inputs.
 * Species is always present as a zero-based scalar S32 field. Buffer handles
 * and their underlying storage must remain valid for the force call. Every
 * host or device input must exactly match the shape and dtype of the refined
 * engine ABI. The connector does not cast host inputs. Forces and named
 * outputs are caller-owned output buffers written by the connector after PJRT
 * execution. The connector does not export a separate connector-owned result
 * buffer back to DLPack. */
typedef struct jcn_atoms {
  // All tensors are jcn_buffer handles imported from DLPack. DLPack carries the
  // pointer, dtype, shape, device, and ownership policy. The jcn_atoms
  // structure only names the physical role of each tensor.
  jcn_buffer* positions;
  const jcn_named_input* particle_inputs;
  int32_t num_particle_inputs;
  const jcn_named_input* global_inputs;
  int32_t num_global_inputs;
  // Rows [0, num_owned_atoms) belong to this rank/adapter instance for output
  // ownership. Rows [0, num_valid_atoms) are real model inputs. Rows
  // [num_valid_atoms, capacity) are padding. Engine-specific halo/image/alias
  // categories are intentionally not represented in the connector ABI.
  int32_t num_owned_atoms;
  int32_t num_valid_atoms;
} jcn_atoms;

typedef struct jcn_named_output {
  // Name from the exported model output list.
  const char* name;
  jcn_buffer* buffer;
  // Number of requested output values. For PARTICLE output, one value is one
  // particle row. F uses all valid atoms with Newton on and owned atoms with
  // Newton off. U uses owned atoms. LOCAL and GLOBAL each use one complete
  // configuration value.
  int64_t rows;
} jcn_named_output;

/* Concrete neighbor buffers used to create an opaque jcn_neighbor_list. The
 * C ABI keeps these records small and explicit for the supported formats,
 * while the force request itself only sees the pseudo-class handle. */
typedef struct jcn_sparse_neighbors {
  jcn_buffer* senders;
  jcn_buffer* receivers;
  // Optional int32 [raw_edges] topology category, required exactly when the
  // model declares include_pair_type. Values are 0 unclassified, 1 directly
  // bonded (1-2), 2 one intermediate bond (1-3), and 3 two intermediate bonds
  // (1-4), and 4 three intermediate bonds (1-5). Padding is category 0 and is
  // excluded by the graph builder.
  jcn_buffer* pair_type;
} jcn_sparse_neighbors;

typedef struct jcn_dense_neighbors {
  jcn_buffer* neighbors;
  // Optional int32 [max_atoms, max_neighbors_per_atom] topology category,
  // required exactly when include_pair_type is set. Category meanings match
  // the sparse pair_type field above.
  jcn_buffer* pair_type;
} jcn_dense_neighbors;

/* One force evaluation. `allow_internal_recompile` permits the connector to
 * resize an internal shape once its configured padding threshold is reached.
 * The embedding runtime remains responsible for retrying all participants. */
typedef struct jcn_force_request {
  jcn_atoms atoms;
  const jcn_neighbor_list* neighbors;
  jcn_requested_capacities capacities;
  const jcn_named_output* outputs;
  int32_t num_outputs;
  // Invalidates cached host copies of neighbor and pair-type data. Set this
  // whenever their values or storage change. Device views are per execution.
  int32_t clear_neighbors;
  int32_t clear_capacities;
  int32_t allow_internal_recompile;
} jcn_force_request;

typedef struct jcn_force_result {
  // NEEDS_CAPACITY_CHANGE means required_capacities contains this rank's local
  // lower bounds. The adapter performs the collective decision, applies local
  // capacities through executor_set_capacities, and retries the same request.
  jcn_compute_code code;
  jcn_requested_capacities required_capacities;
  double flops;
  int32_t compilations;
} jcn_force_result;

/* PJRT-like pseudo-class API.  All functions are reached through this table so
 * the shared library can keep a stable C ABI while changing C++ internals. */
typedef struct JCN_Api {
  uint32_t version;
  size_t struct_size;

  void (*get_runtime_info)(jcn_runtime_info*);

  jcn_client* (*client_create)(const jcn_client_options*, jcn_status*);
  void (*client_destroy)(jcn_client*);

  jcn_model* (*model_load_from_protobuf)(const void*, size_t,
                                         jcn_model_properties*, jcn_status*);
  void (*model_destroy)(jcn_model*);

  jcn_executor* (*executor_create)(jcn_client*, jcn_status*);
  void (*executor_destroy)(jcn_executor*);
  void (*executor_set_model_options)(jcn_executor*, jcn_model*,
                                     const jcn_model_options*,
                                     jcn_model_properties*, jcn_status*);
  void (*executor_set_engine_abi_options)(jcn_executor*,
                                          const jcn_engine_abi_options*,
                                          jcn_status*);
  void (*executor_set_communication_callbacks)(
      jcn_executor*, const jcn_communication_callbacks*, jcn_status*);
  void (*executor_set_capacities)(jcn_executor*,
                                  const jcn_requested_capacities*, jcn_status*);
  void (*executor_compute_forces)(jcn_executor*, const jcn_force_request*,
                                  jcn_force_result*, jcn_status*);

  // Neighbor lists are pseudo-class handles. The adapter owns conversion from
  // its native neighbor-list object into one of the supported buffer layouts.
  // The connector only consumes the role-labeled DLPack buffers behind the
  // handle.
  jcn_neighbor_list* (*neighbor_list_create_simple_sparse)(
      const jcn_sparse_neighbors*, jcn_status*);
  jcn_neighbor_list* (*neighbor_list_create_simple_dense)(
      const jcn_dense_neighbors*, jcn_status*);
  jcn_neighbor_format (*neighbor_list_format)(const jcn_neighbor_list*);
  void (*neighbor_list_destroy)(jcn_neighbor_list*);

  // Import succeeds only if the DLPack metadata is valid and the device matches
  // the selected PJRT addressable device. On success, jcn_buffer owns the
  // DLManagedTensor wrapper and will call its deleter from buffer_destroy. The
  // caller keeps ownership of the underlying memory. On failure, ownership of
  // the wrapper remains with the caller.
  jcn_buffer* (*buffer_from_dlpack_options)(jcn_executor*, DLManagedTensor*,
                                            const jcn_buffer_import_options*,
                                            jcn_status*);
  void (*buffer_destroy)(jcn_buffer*);
} JCN_Api;

JCN_EXPORT const JCN_Api* jcn_get_api(uint32_t requested_version);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_JCN_API_H_
