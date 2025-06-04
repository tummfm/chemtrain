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

load("@tsl//tsl/platform:build_config.bzl", "tf_proto_library")

cc_binary(
    name = "libconnector.so",
    srcs = [
        "libconnector.cpp",
        "libconnector.h",
    ],
    linkopts = [
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    tags = ["manual"],
    deps = [
        ":runner",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt/cpu:cpu_client",
        "@xla//xla/service:dump",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "runner",
    srcs = [
        "libconnector.h",
        "runner.cpp",
        "runner.h",
    ],
    deps = [
        ":buffer",
        ":compiler",
        ":domain",
        ":graph_builder",
        ":model_proto_cc",
        ":utils",
        ":xla_call_module_loader",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/service:dump",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "buffer",
    srcs = [
        "buffer.cpp",
        "buffer.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/service:dump",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "graph_builder",
    srcs = ["graph_builder.cpp"],
    hdrs = ["graph_builder.h"],
    deps = [
        ":buffer",
        ":utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/service:dump",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "domain",
    srcs = ["domain.cpp"],
    hdrs = ["domain.h"],
    deps = [
        ":buffer",
        ":utils",
        "@com_google_absl//absl/types:span",
        "@xla//xla:literal_util",
    ],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cpp"],
    hdrs = ["compiler.h"],
    deps = [
        ":utils",
        ":xla_call_module_loader",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLProgramDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
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
        "@xla//xla/client:xla_computation",
        "@xla//xla/mlir_hlo:hlo_dialect_registration",
        "@xla//xla/mlir_hlo:mhlo_passes",
        "@xla//xla/mlir_hlo:stablehlo_extension_passes",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
    ],
)

cc_library(
    name = "xla_call_module_loader",
    srcs = ["xla_call_module_loader.cpp"],
    hdrs = ["xla_call_module_loader.h"],
    deps = [
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
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:vhlo_ops",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:shape_util",
        "@xla//xla/client:xla_computation",
        "@xla//xla/mlir/utils:type_util",
        "@xla//xla/mlir_hlo",
        "@xla//xla/mlir_hlo:mhlo_passes",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h"],
)

tf_proto_library(
    name = "model_proto",
    srcs = ["model.proto"],
)
