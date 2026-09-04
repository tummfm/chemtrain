# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
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

cc_test(
    name = "cpu_connector_regression_test",
    srcs = ["cpu_connector_regression_test.cpp"],
    args = [
        "--connector=$(rootpath //connector:libconnector.so)",
    ],
    data = ["//connector:libconnector.so"],
    linkopts = ["-ldl"],
    tags = ["manual", "cpu", "no_gpu", "no_kokkos", "no_lammps", "no_mpi"],
    deps = [
        "//connector:jcn_api_headers",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files([
    "export_cpu_models.py",
    "run_cpu_regression.py",
])
