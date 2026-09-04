# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@xla//xla/tsl/platform:build_config.bzl", "tf_proto_library")

config_setting(
    name = "cuda_connector",
    define_values = {"chemtrain_connector_cuda": "true"},
)

cc_library(
    name = "jcn_api_headers",
    hdrs = ["jcn_api.h"],
    visibility = ["//regression:__pkg__"],
    deps = [
        "//third_party/dlpack:dlpack",
    ],
)

cc_library(
    name = "runtime_types",
    hdrs = ["runtime_types.h"],
)

cc_binary(
    name = "libconnector.so",
    additional_linker_inputs = ["connector_version_script.lds"],
    srcs = [
        "jcn_api.cpp",
        "jcn_api.h",
        "jcn_buffer_internal.h",
        "jcn_neighbor_internal.h",
        "pjrt/cpu_api.cpp",
        "runtime_types.h",
    ],
    linkopts = [
        "-Wl,--no-undefined",
        "-Wl,--version-script,$(location :connector_version_script.lds)",
        "-ldl",
    ],
    linkshared = True,
    tags = ["manual"],
    visibility = ["//regression:__pkg__"],
    deps = [
        ":jcn_api_headers",
        ":model_parser",
        ":runner",
        ":runtime_types",
        "//third_party/dlpack:dlpack",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/ffi:ffi_api",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_cpu_internal",
        "@xla//xla/pjrt/c:pjrt_c_api_macros",
    ],
)

# The CPU runtime remains in libconnector.so to avoid loading a second XLA
# copy and registering LLVM command-line options twice. This shim forwards the
# uniform plugin ABI to the connector-owned CPU runtime and FFI API.
cc_binary(
    name = "pjrt_c_api_cpu_plugin.so",
    srcs = ["pjrt/cpu_plugin.cpp"],
    linkopts = ["-ldl"],
    linkshared = True,
    tags = ["manual"],
    deps = [
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_macros",
    ],
)

cc_binary(
    name = "pjrt_c_api_gpu_plugin.so",
    additional_linker_inputs = ["pjrt/plugin.lds"],
    srcs = ["pjrt/gpu_plugin.cpp"],
    linkopts = [
        "-Wl,--version-script,$(location :pjrt/plugin.lds)",
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    tags = ["manual"],
    deps = [
        "@xla//xla/pjrt/c:pjrt_c_api_gpu",
        "@xla//xla/service:gpu_plugin",
    ] + select({
        ":cuda_connector": ["@xla//xla/stream_executor:cuda_platform"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "model_parser",
    srcs = ["model_parser.cpp"],
    hdrs = ["model_parser.h"],
    deps = [":model_proto_cc"],
)

cc_test(
    name = "model_shape_test",
    srcs = ["model_shape_test.cpp"],
    linkstatic = True,
    deps = [
        ":model_shape",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_parser_test",
    srcs = ["model_parser_test.cpp"],
    linkstatic = True,
    deps = [
        ":model_parser",
        ":model_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dlpack_validation",
    srcs = ["dlpack_validation.cpp"],
    hdrs = [
        "dlpack_validation.h",
        "jcn_buffer_internal.h",
    ],
    deps = [
        ":jcn_api_headers",
        "//third_party/dlpack:dlpack",
    ],
)

cc_test(
    name = "dlpack_validation_test",
    srcs = ["dlpack_validation_test.cpp"],
    linkstatic = True,
    deps = [
        ":dlpack_validation",
        "//third_party/dlpack:dlpack",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "runner",
    srcs = [
        "jcn_buffer_internal.h",
        "jcn_neighbor_internal.h",
        "runtime_types.h",
        "runner.cpp",
        "runner.h",
    ],
    deps = [
        ":compiler",
        ":communication_runtime",
        ":dlpack_validation",
        ":feature_exchange_registration",
        ":runtime_loader",
        ":jcn_api_headers",
        ":model_shape",
        ":model_parser",
        ":model_proto_cc",
        ":pjrt_buffers",
        ":runtime_types",
        ":utils",
        ":xla_call_module_loader",
        "//third_party/dlpack:dlpack",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:future",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:maybe_owning_mlir_module",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
    ],
)

cc_library(
    name = "runtime_loader",
    srcs = ["runtime_loader.cpp"],
    hdrs = ["runtime_loader.h"],
    defines = ["_GNU_SOURCE"],
    linkopts = ["-ldl"],
    deps = [
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "communication_runtime",
    srcs = ["communication/runtime.cpp"],
    hdrs = [
        "communication/runtime.h",
        "runtime_types.h",
    ],
    deps = [
        ":runtime_types",
        "@com_google_absl//absl/status",
        "@xla//xla/stream_executor:device_memory",
        "@xla//xla/stream_executor:memory_allocation",
        "@xla//xla/stream_executor:stream_executor_h",
    ],
)

cc_library(
    name = "feature_exchange_ffi",
    srcs = ["ffi/feature_exchange.cpp"],
    hdrs = ["ffi/feature_exchange.h"],
    deps = [
        ":communication_runtime",
        "@com_google_absl//absl/status",
        "@xla//xla/backends/gpu:ffi",
        "@xla//xla/ffi:ffi",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/stream_executor:device_memory",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
    ],
)

cc_library(
    name = "feature_exchange_registration",
    srcs = ["ffi/feature_exchange_registration.cpp"],
    hdrs = ["ffi/feature_exchange_registration.h"],
    deps = [
        ":feature_exchange_ffi",
        ":runtime_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@xla//xla/ffi:ffi",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:c_api_internal",
        "@xla//xla/ffi:type_registry",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

cc_library(
    name = "pjrt_buffers",
    srcs = [
        "pjrt/buffers.cpp",
        "pjrt/buffers.h",
    ],
    deps = [
        ":dlpack_validation",
        ":jcn_api_headers",
        ":runtime_types",
        "//third_party/dlpack:dlpack",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
    ],
)

cc_library(
    name = "model_shape",
    srcs = ["model_shape.cpp"],
    hdrs = ["model_shape.h"],
    deps = [
        ":jcn_api_headers",
        ":utils",
        "//third_party/dlpack:dlpack",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cpp"],
    hdrs = ["compiler.h"],
    deps = [
        ":model_shape",
        ":runtime_types",
        ":utils",
        ":xla_call_module_loader",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MLProgramDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@shardy//shardy/dialect/sdy/ir:register",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:version",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla/mlir/utils:error_util",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "xla_call_module_loader",
    srcs = ["xla_call_module_loader.cpp"],
    hdrs = ["xla_call_module_loader.h"],
    deps = [
        ":runtime_types",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/import:passes",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:vhlo_ops",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla/mlir/utils:type_util",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "xla_call_module_loader_test",
    srcs = ["xla_call_module_loader_test.cpp"],
    linkstatic = True,
    deps = [
        ":xla_call_module_loader",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:version",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h"],
)

proto_library(
    name = "model_proto",
    srcs = ["model.proto"],
)

cc_proto_library(
    name = "model_proto_cc",
    deps = [":model_proto"],
)
